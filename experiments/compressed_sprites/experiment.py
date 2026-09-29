"""Controlled detection from pixels, raw indices or complete encoded file bytes."""
import argparse
import copy
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def indexed(array, palette):
    image = Image.fromarray(array).convert("P")
    image.putpalette(palette.reshape(-1).tolist())
    return image


def read_sequence(path, mode, limit):
    """Return source bytes without decoding; raw indices must describe a 64x64 scene."""
    raw = path.read_bytes()
    if mode == "raw_bytes" and len(raw) != 64 * 64:
        raise ValueError(f"{path}: raw indices must contain exactly 4096 bytes, got {len(raw)}")
    if len(raw) > limit:
        raise ValueError(f"{path}: {len(raw)} bytes exceeds --max-bytes {limit}; no truncation allowed")
    return raw


def generate(args):
    root = Path(args.data)
    if root.exists():
        raise FileExistsError(f"Use a new dataset directory: {root}")
    root.mkdir(parents=True)
    rng = np.random.default_rng(args.seed)
    # A shared, explicit 256-colour palette: 3 red, 3 green, 2 blue bits.
    palette = np.array([(r * 255 // 7, g * 255 // 7, b * 255 // 3)
                        for r in range(8) for g in range(8) for b in range(4)], dtype=np.uint8)
    records = []
    for split, count in (("train", args.train), ("val", args.val), ("test", args.test)):
        for i in range(count):
            folder = root / split / f"{i:06d}"
            folder.mkdir(parents=True)
            background, foreground = rng.choice(256, 2, replace=False).tolist()
            array = np.full((64, 64), background, dtype=np.uint8)
            mask = Image.new("L", (64, 64))
            # Class 0 is background; colours are independent of shape and presence.
            label = 0 if rng.random() < 0.15 else int(rng.integers(1, 4))
            if label:
                width, height = rng.integers(4, 33, 2).tolist()
                x, y = int(rng.integers(0, 65 - width)), int(rng.integers(0, 65 - height))
                bounds = (x, y, x + width - 1, y + height - 1)
                draw = ImageDraw.Draw(mask)
                if label == 1:
                    draw.rectangle(bounds, fill=255)
                elif label == 2:
                    draw.ellipse(bounds, fill=255)
                else:
                    draw.polygon([(x + width // 2, y), (x, y + height - 1),
                                  (x + width - 1, y + height - 1)], fill=255)
            selected = np.asarray(mask) != 0
            array[selected] = foreground
            image = indexed(array, palette)
            image.save(folder / "image.png", bits=8, compress_level=6)
            image.save(folder / "image_level0.png", bits=8, compress_level=0)
            image.save(folder / "image_level9.png", bits=8, compress_level=9)
            image.save(folder / "image.gif", optimize=False)
            # Same RGB image, different index assignment and palette ordering.
            permutation = rng.permutation(256)
            remapped_palette = np.empty_like(palette)
            remapped_palette[permutation] = palette
            indexed(permutation[array].astype(np.uint8), remapped_palette).save(
                folder / "image_reindexed.png", bits=8, compress_level=6)
            array.tofile(folder / "indices.bin")
            palette.tofile(folder / "palette.rgb")
            altered = array.copy()
            altered[selected] = (foreground + 1) % 256
            recoloured = indexed(altered, palette)
            recoloured.save(folder / "recoloured.png", bits=8)
            # The decoded RGB diff, not a diff of compressed byte offsets.
            diff = np.any(np.asarray(image.convert("RGB")) !=
                          np.asarray(recoloured.convert("RGB")), axis=2)
            if not np.array_equal(diff, selected):
                raise AssertionError("Recolouring diff must reproduce the annotation")
            Image.fromarray((diff * 255).astype(np.uint8)).save(folder / "diff.png")
            mask.save(folder / "mask.png")
            box = mask.getbbox() or (0, 0, 0, 0)  # exclusive right/bottom
            record = {"path": str(folder.relative_to(root)), "split": split,
                      "class": label, "box": [v / 64 for v in box]}
            (folder / "label.json").write_text(json.dumps(record, indent=2))
            records.append(record)
    manifest = {"seed": args.seed, "classes": ["background", "rectangle", "ellipse", "triangle"],
                "size": 64, "records": records}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Generated {len(records)} independent scenes in {root}")


def train(args):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(args.threads)
    root, out = Path(args.data), Path(args.output)
    if out.exists():
        raise FileExistsError(f"Use a new output directory: {out}")
    out.mkdir(parents=True)
    records = json.loads((root / "manifest.json").read_text())["records"]
    device = torch.device(args.device)
    variants = ["image.png", "image_level0.png", "image_level9.png", "image_reindexed.png"]
    results = {}

    class Scenes(Dataset):
        def __init__(self, split, mode, variant, limit):
            self.items = [r for r in records if r["split"] == split]
            self.mode, self.variant, self.limit = mode, variant, limit

        def __len__(self):
            return len(self.items)

        def __getitem__(self, index):
            record = self.items[index]
            path = root / record["path"] / self.variant
            if self.mode == "pixels":
                with Image.open(path) as im:
                    x = torch.from_numpy(np.array(im.convert("RGB"))).permute(2, 0, 1).float() / 255
            else:
                raw = read_sequence(path, self.mode, self.limit)
                x = torch.full((self.limit,), 256, dtype=torch.long)
                x[:len(raw)] = torch.tensor(list(raw), dtype=torch.long)
            return x, torch.tensor(record["class"]), torch.tensor(record["box"], dtype=torch.float32)

    class Detector(nn.Module):
        def __init__(self, mode):
            super().__init__()
            self.mode = mode
            if mode == "pixels":
                self.encoder = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2), nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
                    nn.AdaptiveAvgPool2d((8, 8)), nn.Flatten())
                features = 32 * 8 * 8
            else:
                self.embedding = nn.Embedding(257, 8, padding_idx=256)
                self.encoder = nn.Sequential(nn.Conv1d(8, 32, 9, stride=4, padding=4), nn.ReLU(),
                    nn.Conv1d(32, 32, 9, stride=4, padding=4), nn.ReLU(),
                    nn.Conv1d(32, 32, 9, stride=4, padding=4), nn.ReLU(),
                    nn.AdaptiveAvgPool1d(32), nn.Flatten())
                features = 32 * 32
            self.head = nn.Sequential(nn.Linear(features, 128), nn.ReLU(), nn.Linear(128, 8))

        def forward(self, x):
            if self.mode != "pixels":
                x = self.embedding(x).transpose(1, 2)
            prediction = self.head(self.encoder(x))
            coordinates = prediction[:, 4:].sigmoid()
            # Produce valid xyxy boxes throughout training.
            low = torch.minimum(coordinates[:, :2], coordinates[:, 2:])
            high = torch.maximum(coordinates[:, :2], coordinates[:, 2:])
            return prediction[:, :4], torch.cat((low, high), dim=1)

    def iou(a, b):
        intersection = (torch.minimum(a[:, 2:], b[:, 2:]) -
                        torch.maximum(a[:, :2], b[:, :2])).clamp(min=0).prod(1)
        union = (a[:, 2:] - a[:, :2]).prod(1) + (b[:, 2:] - b[:, :2]).prod(1) - intersection
        return intersection / union.clamp(min=1e-8)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    def evaluate(model, loader):
        model.eval()
        total = correct = positives = detected = negatives = false_positive = 0
        overlap_sum = model_seconds = 0.0
        sync()
        start = time.perf_counter()
        with torch.inference_mode():
            for x, labels, boxes in loader:
                x, labels, boxes = x.to(device), labels.to(device), boxes.to(device)
                sync()
                before = time.perf_counter()
                logits, predicted_boxes = model(x)
                sync()
                model_seconds += time.perf_counter() - before
                predicted_class = logits.argmax(1)
                positive = labels > 0
                overlaps = iou(predicted_boxes, boxes)
                total += len(labels)
                correct += int((predicted_class == labels).sum())
                positives += int(positive.sum())
                negatives += int((~positive).sum())
                false_positive += int(((predicted_class > 0) & ~positive).sum())
                detected += int(((predicted_class == labels) & positive & (overlaps >= .5)).sum())
                overlap_sum += float(overlaps[positive].sum())
        elapsed = time.perf_counter() - start
        return {"images": total, "class_accuracy": correct / total,
                "positive_mean_iou": overlap_sum / max(positives, 1),
                "correct_class_recall_iou50": detected / max(positives, 1),
                "background_false_positive_rate": false_positive / max(negatives, 1),
                "model_ms_per_image": 1000 * model_seconds / total,
                "pipeline_ms_per_image": 1000 * elapsed / total}

    for mode in args.modes:
        torch.manual_seed(args.seed)
        variant = {"raw_bytes": "indices.bin", "gif_bytes": "image.gif"}.get(mode, "image.png")
        def loader(split, name=variant, shuffle=False):
            return DataLoader(Scenes(split, mode, name, args.max_bytes), batch_size=args.batch_size,
                              shuffle=shuffle, num_workers=0)
        model = Detector(mode).to(device)
        optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
        training = loader("train", shuffle=True)
        validation = loader("val")
        best, best_state, best_epoch = -1, None, None
        history = []
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sync()
        started = time.perf_counter()
        for epoch in range(args.epochs):
            model.train()
            loss_sum = 0
            for x, labels, boxes in training:
                x, labels, boxes = x.to(device), labels.to(device), boxes.to(device)
                logits, predicted_boxes = model(x)
                positive = labels > 0
                loss = nn.functional.cross_entropy(logits, labels)
                if positive.any():
                    loss = loss + 5 * nn.functional.l1_loss(predicted_boxes[positive], boxes[positive])
                    loss = loss + (1 - iou(predicted_boxes[positive], boxes[positive])).mean()
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                loss_sum += float(loss.detach()) * len(labels)
            metrics = evaluate(model, validation)
            # Continuous score avoids arbitrary selection among zero-recall early epochs.
            score = metrics["correct_class_recall_iou50"] + .01 * metrics["positive_mean_iou"]
            if score > best:
                best, best_state, best_epoch = score, copy.deepcopy(model.state_dict()), epoch + 1
            history.append({"epoch": epoch + 1, "loss": loss_sum / len(training.dataset), "val": metrics})
            print(f"{mode} epoch {epoch + 1}: loss={history[-1]['loss']:.4f} val recall@.5={metrics['correct_class_recall_iou50']:.3f}", flush=True)
        sync()
        training_seconds = time.perf_counter() - started
        peak = torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
        model.load_state_dict(best_state)
        torch.save({"model": best_state, "mode": mode, "args": vars(args),
                    "best_epoch": best_epoch}, out / f"{mode}.pt")
        tests = {}
        for name in ([variant] if mode in ("raw_bytes", "gif_bytes") else variants):
            # Warm-up outside reported timing; test labels never select weights.
            with torch.inference_mode():
                model(next(iter(loader("test", name)))[0].to(device))
            tests[name] = evaluate(model, loader("test", name))
        lengths = [(root / r["path"] / variant).stat().st_size for r in records if r["split"] == "test"]
        results[mode] = {"parameters": sum(p.numel() for p in model.parameters()),
                         "best_epoch": best_epoch, "training_seconds_including_validation": training_seconds,
                         "peak_training_cuda_allocated_bytes": peak,
                         "mean_test_file_bytes": float(np.mean(lengths)),
                         "input_tensor_bytes_per_image": 3 * 64 * 64 * 4 if mode == "pixels" else args.max_bytes * 8,
                         "test": tests, "history": history}
        (out / "results.json").write_text(json.dumps({"args": vars(args), "results": results}, indent=2))
    print(f"Results: {out / 'results.json'}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    gen = sub.add_parser("generate")
    gen.add_argument("--data", default="experiments/compressed_sprites/data")
    gen.add_argument("--train", type=int, default=3000)
    gen.add_argument("--val", type=int, default=500)
    gen.add_argument("--test", type=int, default=500)
    gen.add_argument("--seed", type=int, default=42)
    run = sub.add_parser("train")
    run.add_argument("--data", default="experiments/compressed_sprites/data")
    run.add_argument("--output", default="experiments/compressed_sprites/runs/seed42")
    run.add_argument("--modes", nargs="+", choices=["pixels", "raw_bytes", "png_bytes", "gif_bytes"],
                     default=["pixels", "raw_bytes", "png_bytes", "gif_bytes"])
    run.add_argument("--epochs", type=int, default=30)
    run.add_argument("--batch-size", type=int, default=32)
    run.add_argument("--max-bytes", type=int, default=8192)
    run.add_argument("--lr", type=float, default=.001)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--device", default="cpu")
    run.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    for name in ("train", "val", "test", "epochs", "batch_size", "max_bytes", "threads"):
        value = getattr(args, name, None)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    (generate if args.command == "generate" else train)(args)


if __name__ == "__main__":
    main()
