"""Train RT-DETR-R18 from scratch on the same YOLO-format VOC data.

Expected layout
---------------
ROOT/
    images/train/*.jpg
    images/val/*.jpg
    labels/train/*.txt
    labels/val/*.txt

Each label row is standard normalized YOLO format:

    class_id x_center y_center width height

The console and checkpoint layout intentionally follow the author's YOLOv26
repository so training behaviour can be compared quickly.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance
from scipy.optimize import linear_sum_assignment
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from model import RTDETR, box_cxcywh_to_xyxy


ROOT = "/mnt/scratch2/users/40464858/VOC_dataset/voc_yolo"
DEFAULT_IMG_SIZE = 416
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 100
DEFAULT_NUM_CLASSES = 20
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-4


# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------


class YoloTxtDataset(Dataset):
    def __init__(
        self,
        root: str = ROOT,
        split: str = "train",
        img_size: int = DEFAULT_IMG_SIZE,
        num_classes: int = DEFAULT_NUM_CLASSES,
        augment: bool = False,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.img_size = img_size
        self.num_classes = num_classes
        self.augment = augment
        self.image_dir = self.root / "images" / split
        self.label_dir = self.root / "labels" / split
        extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp")
        self.image_paths = sorted(
            path for pattern in extensions for path in self.image_dir.glob(pattern)
        )
        if not self.image_paths:
            raise FileNotFoundError(f"No images found in {self.image_dir}")

    def __len__(self) -> int:
        return len(self.image_paths)

    def _label_path(self, image_path: Path) -> Path:
        return self.label_dir / f"{image_path.stem}.txt"

    def _read_labels(self, path: Path) -> Tuple[np.ndarray, np.ndarray]:
        if not path.exists() or path.stat().st_size == 0:
            return np.zeros((0,), np.int64), np.zeros((0, 4), np.float32)
        array = np.loadtxt(path, ndmin=2, dtype=np.float32)
        if array.shape[1] != 5:
            raise ValueError(f"Bad label shape in {path}: {array.shape}, expected Nx5")
        if not np.isfinite(array).all():
            raise ValueError(f"NaN or Inf in {path}")
        labels = array[:, 0].astype(np.int64)
        boxes = array[:, 1:5].astype(np.float32)
        if (labels < 0).any() or (labels >= self.num_classes).any():
            raise ValueError(f"Class id outside 0..{self.num_classes - 1} in {path}")
        boxes[:, :2] = np.clip(boxes[:, :2], 0.0, 1.0)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 1e-6, 1.0)
        return labels, boxes

    def __getitem__(self, index: int) -> Tuple[Tensor, Dict[str, Tensor]]:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        labels, boxes = self._read_labels(self._label_path(image_path))

        if self.augment:
            if random.random() < 0.5:
                image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                if len(boxes):
                    boxes[:, 0] = 1.0 - boxes[:, 0]
            if random.random() < 0.5:
                image = ImageEnhance.Brightness(image).enhance(random.uniform(0.8, 1.2))
                image = ImageEnhance.Contrast(image).enhance(random.uniform(0.8, 1.2))
                image = ImageEnhance.Color(image).enhance(random.uniform(0.8, 1.2))

        image = image.resize((self.img_size, self.img_size), Image.Resampling.BILINEAR)
        image_tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).copy())
        image_tensor = image_tensor.permute(2, 0, 1) / 255.0
        target = {
            "labels": torch.from_numpy(labels).long(),
            "boxes": torch.from_numpy(boxes).float(),
            "image_id": torch.tensor(index, dtype=torch.long),
            "path": str(image_path),
        }
        return image_tensor, target


def collate_fn(
    batch: Sequence[Tuple[Tensor, Dict[str, Tensor]]],
) -> Tuple[Tensor, List[Dict[str, Tensor]]]:
    images, targets = zip(*batch)
    return torch.stack(images, dim=0), list(targets)


# -----------------------------------------------------------------------------
# Box metrics and Hungarian matching
# -----------------------------------------------------------------------------


def box_area(boxes: Tensor) -> Tensor:
    return (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=0)


def box_iou(boxes1: Tensor, boxes2: Tensor) -> Tuple[Tensor, Tensor]:
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)
    left_top = torch.maximum(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.minimum(boxes1[:, None, 2:], boxes2[:, 2:])
    width_height = (right_bottom - left_top).clamp(min=0)
    intersection = width_height[:, :, 0] * width_height[:, :, 1]
    union = area1[:, None] + area2 - intersection
    return intersection / union.clamp(min=1e-7), union


def generalized_box_iou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    iou, union = box_iou(boxes1, boxes2)
    left_top = torch.minimum(boxes1[:, None, :2], boxes2[:, :2])
    right_bottom = torch.maximum(boxes1[:, None, 2:], boxes2[:, 2:])
    width_height = (right_bottom - left_top).clamp(min=0)
    enclosing_area = width_height[:, :, 0] * width_height[:, :, 1]
    return iou - (enclosing_area - union) / enclosing_area.clamp(min=1e-7)


class HungarianMatcher(nn.Module):
    def __init__(
        self,
        cost_class: float = 2.0,
        cost_bbox: float = 5.0,
        cost_giou: float = 2.0,
        alpha: float = 0.25,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        self.alpha = alpha
        self.gamma = gamma

    @torch.no_grad()
    def forward(
        self,
        outputs: Mapping[str, Tensor],
        targets: Sequence[Mapping[str, Tensor]],
    ) -> List[Tuple[Tensor, Tensor]]:
        logits = outputs["pred_logits"].float()
        boxes = outputs["pred_boxes"].float()
        batch_size, query_count = logits.shape[:2]
        sizes = [len(target["boxes"]) for target in targets]
        if sum(sizes) == 0:
            empty = torch.empty(0, dtype=torch.int64)
            return [(empty, empty) for _ in targets]

        probabilities = logits.flatten(0, 1).sigmoid()
        flat_boxes = boxes.flatten(0, 1)
        target_labels = torch.cat([target["labels"] for target in targets])
        target_boxes = torch.cat([target["boxes"] for target in targets]).float()

        class_probability = probabilities[:, target_labels]
        negative_cost = (
            (1.0 - self.alpha)
            * class_probability.pow(self.gamma)
            * -(1.0 - class_probability + 1e-8).log()
        )
        positive_cost = (
            self.alpha
            * (1.0 - class_probability).pow(self.gamma)
            * -(class_probability + 1e-8).log()
        )
        classification_cost = positive_cost - negative_cost
        bbox_cost = torch.cdist(flat_boxes, target_boxes, p=1)
        giou_cost = -generalized_box_iou(
            box_cxcywh_to_xyxy(flat_boxes),
            box_cxcywh_to_xyxy(target_boxes),
        )
        total_cost = (
            self.cost_class * classification_cost
            + self.cost_bbox * bbox_cost
            + self.cost_giou * giou_cost
        )
        total_cost = total_cost.view(batch_size, query_count, -1).cpu()
        split_costs = total_cost.split(sizes, dim=-1)
        indices: List[Tuple[Tensor, Tensor]] = []
        for batch_index, cost in enumerate(split_costs):
            if sizes[batch_index] == 0:
                empty = torch.empty(0, dtype=torch.int64)
                indices.append((empty, empty))
                continue
            prediction_index, target_index = linear_sum_assignment(cost[batch_index])
            indices.append(
                (
                    torch.as_tensor(prediction_index, dtype=torch.int64),
                    torch.as_tensor(target_index, dtype=torch.int64),
                )
            )
        return indices


# -----------------------------------------------------------------------------
# Official-style RT-DETR set criterion: VFL + L1 box + GIoU
# -----------------------------------------------------------------------------


class SetCriterion(nn.Module):
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        weight_vfl: float = 1.0,
        weight_bbox: float = 5.0,
        weight_giou: float = 2.0,
        alpha: float = 0.75,
        gamma: float = 2.0,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.weight_vfl = weight_vfl
        self.weight_bbox = weight_bbox
        self.weight_giou = weight_giou
        self.alpha = alpha
        self.gamma = gamma

    @staticmethod
    def _permutation_indices(
        indices: Sequence[Tuple[Tensor, Tensor]],
        device: torch.device,
    ) -> Tuple[Tensor, Tensor]:
        batch_parts: List[Tensor] = []
        source_parts: List[Tensor] = []
        for batch_index, (source, _) in enumerate(indices):
            if source.numel():
                batch_parts.append(
                    torch.full_like(source, batch_index, device=device)
                )
                source_parts.append(source.to(device))
        if not source_parts:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return empty, empty
        return torch.cat(batch_parts), torch.cat(source_parts)

    def _loss_for_indices(
        self,
        outputs: Mapping[str, Tensor],
        targets: Sequence[Mapping[str, Tensor]],
        indices: Sequence[Tuple[Tensor, Tensor]],
        num_boxes: float,
    ) -> Dict[str, Tensor]:
        logits = outputs["pred_logits"].float()
        predicted_boxes = outputs["pred_boxes"].float()
        device = logits.device
        batch_indices, source_indices = self._permutation_indices(indices, device)

        target_classes = torch.full(
            logits.shape[:2],
            self.num_classes,
            dtype=torch.long,
            device=device,
        )
        target_boxes_list: List[Tensor] = []
        target_labels_list: List[Tensor] = []
        for target, (_, target_index) in zip(targets, indices):
            if target_index.numel():
                target_boxes_list.append(target["boxes"][target_index].to(device))
                target_labels_list.append(target["labels"][target_index].to(device))
        if target_labels_list:
            matched_labels = torch.cat(target_labels_list)
            matched_target_boxes = torch.cat(target_boxes_list).float()
            target_classes[batch_indices, source_indices] = matched_labels
            matched_predicted_boxes = predicted_boxes[batch_indices, source_indices]
            pairwise_iou, _ = box_iou(
                box_cxcywh_to_xyxy(matched_predicted_boxes),
                box_cxcywh_to_xyxy(matched_target_boxes),
            )
            matched_iou = torch.diag(pairwise_iou).detach()
        else:
            matched_labels = torch.empty(0, dtype=torch.long, device=device)
            matched_target_boxes = torch.empty(0, 4, device=device)
            matched_predicted_boxes = torch.empty(0, 4, device=device)
            matched_iou = torch.empty(0, device=device)

        one_hot = F.one_hot(
            target_classes,
            num_classes=self.num_classes + 1,
        )[..., :-1].to(logits.dtype)
        target_score_per_query = torch.zeros_like(target_classes, dtype=logits.dtype)
        if matched_iou.numel():
            target_score_per_query[batch_indices, source_indices] = matched_iou
        target_scores = target_score_per_query.unsqueeze(-1) * one_hot
        predicted_scores = logits.sigmoid().detach()
        varifocal_weight = (
            self.alpha * predicted_scores.pow(self.gamma) * (1.0 - one_hot)
            + target_scores
        )
        loss_vfl = F.binary_cross_entropy_with_logits(
            logits,
            target_scores,
            weight=varifocal_weight,
            reduction="none",
        )
        loss_vfl = loss_vfl.mean(dim=1).sum() * logits.shape[1] / num_boxes

        if matched_predicted_boxes.numel():
            loss_bbox = F.l1_loss(
                matched_predicted_boxes,
                matched_target_boxes,
                reduction="none",
            ).sum() / num_boxes
            loss_giou = (
                1.0
                - torch.diag(
                    generalized_box_iou(
                        box_cxcywh_to_xyxy(matched_predicted_boxes),
                        box_cxcywh_to_xyxy(matched_target_boxes),
                    )
                )
            ).sum() / num_boxes
        else:
            loss_bbox = predicted_boxes.sum() * 0.0
            loss_giou = predicted_boxes.sum() * 0.0

        return {
            "vfl": loss_vfl,
            "bbox": loss_bbox,
            "giou": loss_giou,
        }

    def _weighted_total(self, components: Mapping[str, Tensor]) -> Tensor:
        return (
            self.weight_vfl * components["vfl"]
            + self.weight_bbox * components["bbox"]
            + self.weight_giou * components["giou"]
        )

    def forward(
        self,
        outputs: Mapping[str, object],
        targets: Sequence[Mapping[str, Tensor]],
    ) -> Tuple[Tensor, Dict[str, float]]:
        main_outputs = {
            "pred_logits": outputs["pred_logits"],
            "pred_boxes": outputs["pred_boxes"],
        }
        if not isinstance(main_outputs["pred_logits"], Tensor) or not isinstance(
            main_outputs["pred_boxes"], Tensor
        ):
            raise TypeError("RT-DETR outputs must contain tensors")
        num_boxes = float(max(1, sum(len(target["labels"]) for target in targets)))
        main_indices = self.matcher(main_outputs, targets)
        main_components = self._loss_for_indices(
            main_outputs,
            targets,
            main_indices,
            num_boxes,
        )
        main_total = self._weighted_total(main_components)

        auxiliary_total = main_total.new_zeros(())
        auxiliary_outputs = outputs.get("aux_outputs", [])
        if isinstance(auxiliary_outputs, list):
            for auxiliary in auxiliary_outputs:
                if not isinstance(auxiliary, Mapping):
                    continue
                auxiliary_indices = self.matcher(auxiliary, targets)
                components = self._loss_for_indices(
                    auxiliary,
                    targets,
                    auxiliary_indices,
                    num_boxes,
                )
                auxiliary_total = auxiliary_total + self._weighted_total(components)

        denoising_total = main_total.new_zeros(())
        denoising_outputs = outputs.get("dn_aux_outputs", [])
        metadata = outputs.get("dn_meta")
        if isinstance(denoising_outputs, list) and isinstance(metadata, Mapping):
            positive_indices = metadata.get("dn_positive_idx")
            group_count = int(metadata.get("dn_num_group", 1))
            if isinstance(positive_indices, tuple):
                dn_indices: List[Tuple[Tensor, Tensor]] = []
                for target, source_index in zip(targets, positive_indices):
                    target_count = len(target["labels"])
                    target_index = torch.arange(target_count, dtype=torch.long).repeat(
                        group_count
                    )
                    dn_indices.append((source_index.cpu(), target_index))
                for denoising in denoising_outputs:
                    components = self._loss_for_indices(
                        denoising,
                        targets,
                        dn_indices,
                        num_boxes * group_count,
                    )
                    denoising_total = denoising_total + self._weighted_total(components)

        total = main_total + auxiliary_total + denoising_total
        metrics = {
            "loss": float(total.detach()),
            "vfl": float(main_components["vfl"].detach()),
            "bbox": float(main_components["bbox"].detach()),
            "giou": float(main_components["giou"].detach()),
            "aux": float(auxiliary_total.detach()),
            "dn": float(denoising_total.detach()),
        }
        return total, metrics


# -----------------------------------------------------------------------------
# Training utilities
# -----------------------------------------------------------------------------


class MetricAverager:
    def __init__(self) -> None:
        self.totals: Dict[str, float] = {}
        self.count = 0

    def update(self, metrics: Mapping[str, float]) -> None:
        for key, value in metrics.items():
            self.totals[key] = self.totals.get(key, 0.0) + float(value)
        self.count += 1

    def averages(self) -> Dict[str, float]:
        divisor = max(self.count, 1)
        return {key: value / divisor for key, value in self.totals.items()}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def choose_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def create_grad_scaler(enabled: bool) -> torch.amp.GradScaler:
    return torch.amp.GradScaler("cuda", enabled=enabled)


def autocast_context(device: torch.device, enabled: bool):
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    return torch.amp.autocast(device_type=device.type, dtype=dtype, enabled=enabled)


def move_targets_to_device(
    targets: Sequence[Dict[str, Tensor]],
    device: torch.device,
) -> List[Dict[str, Tensor]]:
    moved: List[Dict[str, Tensor]] = []
    for target in targets:
        moved.append(
            {
                key: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
                for key, value in target.items()
            }
        )
    return moved


def set_batchnorm_eval(module: nn.Module) -> None:
    if isinstance(module, nn.BatchNorm2d):
        module.eval()


def run_training_epoch(
    model: RTDETR,
    loader: DataLoader,
    criterion: SetCriterion,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    max_grad_norm: float,
    amp_enabled: bool,
    log_interval: int,
) -> Dict[str, float]:
    model.train()
    averages = MetricAverager()
    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        targets = move_targets_to_device(targets, device)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_enabled):
            outputs = model(images, targets)
        loss, metrics = criterion(outputs, targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(
                f"Non-finite loss at batch {batch_index}: {float(loss)}; {metrics}"
            )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        averages.update(metrics)
        if batch_index % log_interval == 0 or batch_index == len(loader):
            current = averages.averages()
            print(
                f"  batch {batch_index:4d}/{len(loader):4d} "
                f"loss={current.get('loss', math.nan):.4f} "
                f"vfl={current.get('vfl', math.nan):.4f} "
                f"box={current.get('bbox', math.nan):.4f} "
                f"giou={current.get('giou', math.nan):.4f}"
            )
    return averages.averages()


@torch.no_grad()
def run_validation_epoch(
    model: RTDETR,
    loader: DataLoader,
    criterion: SetCriterion,
    device: torch.device,
    amp_enabled: bool,
) -> Dict[str, float]:
    # Training mode returns decoder auxiliary outputs needed for comparable loss.
    # BatchNorm is forced into evaluation mode so validation does not update it.
    model.train()
    model.apply(set_batchnorm_eval)
    averages = MetricAverager()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = move_targets_to_device(targets, device)
        with autocast_context(device, amp_enabled):
            # Keep decoder auxiliary outputs but omit random denoising queries,
            # so validation loss is deterministic between epochs.
            outputs = model(images, None)
        _, metrics = criterion(outputs, targets)
        averages.update(metrics)
    return averages.averages()


def format_epoch_metrics(prefix: str, metrics: Mapping[str, float]) -> str:
    return (
        f"{prefix}: loss={metrics.get('loss', math.nan):.4f}, "
        f"vfl={metrics.get('vfl', math.nan):.4f}, "
        f"bbox={metrics.get('bbox', math.nan):.4f}, "
        f"giou={metrics.get('giou', math.nan):.4f}, "
        f"aux={metrics.get('aux', math.nan):.4f}, "
        f"dn={metrics.get('dn', math.nan):.4f}"
    )


@dataclass
class ExperimentConfig:
    root: str
    img_size: int
    batch_size: int
    epochs: int
    num_classes: int
    learning_rate: float
    backbone_learning_rate: float
    weight_decay: float
    num_queries: int
    hidden_dim: int
    decoder_layers: int
    num_denoising: int
    seed: int


def save_checkpoint(
    path: Path,
    model: RTDETR,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    epoch: int,
    best_validation_loss: float,
    config: ExperimentConfig,
    train_metrics: Mapping[str, float],
    validation_metrics: Mapping[str, float],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": scaler.state_dict(),
            "best_validation_loss": best_validation_loss,
            "config": asdict(config),
            "train_metrics": dict(train_metrics),
            "validation_metrics": dict(validation_metrics),
        },
        temporary,
    )
    os.replace(temporary, path)


def load_checkpoint(
    path: Path,
    model: RTDETR,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    device: torch.device,
) -> Tuple[int, float]:
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    if "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return int(checkpoint["epoch"]) + 1, float(
        checkpoint.get("best_validation_loss", math.inf)
    )


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = choose_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Dataset root: {args.root}")
    train_dataset = YoloTxtDataset(
        args.root,
        "train",
        args.img_size,
        args.num_classes,
        augment=not args.no_augment,
    )
    validation_dataset = YoloTxtDataset(
        args.root,
        "val",
        args.img_size,
        args.num_classes,
        augment=False,
    )
    print(f"Training images: {len(train_dataset)}")
    print(f"Validation images: {len(validation_dataset)}")

    generator = torch.Generator().manual_seed(args.seed)
    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": collate_fn,
        "worker_init_fn": worker_init_fn,
        "generator": generator,
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=len(train_dataset) >= args.batch_size,
        **loader_options,
    )
    validation_loader = DataLoader(
        validation_dataset,
        shuffle=False,
        drop_last=False,
        **loader_options,
    )

    model = RTDETR(
        input_channels=3,
        num_classes=args.num_classes,
        num_queries=args.num_queries,
        hidden_dim=args.hidden_dim,
        num_decoder_layers=args.decoder_layers,
        num_denoising=args.num_denoising,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(f"Model parameters: {parameter_count:,}")
    print(f"Trainable parameters: {trainable_count:,}")

    matcher = HungarianMatcher(cost_class=2.0, cost_bbox=5.0, cost_giou=2.0)
    criterion = SetCriterion(args.num_classes, matcher)
    backbone_parameters = list(model.backbone.parameters())
    backbone_ids = {id(parameter) for parameter in backbone_parameters}
    other_parameters = [
        parameter for parameter in model.parameters() if id(parameter) not in backbone_ids
    ]
    optimizer = torch.optim.AdamW(
        [
            {"params": backbone_parameters, "lr": args.backbone_learning_rate},
            {"params": other_parameters, "lr": args.learning_rate},
        ],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(args.epochs, 1),
        eta_min=args.learning_rate * 0.01,
    )
    scaler = create_grad_scaler(amp_enabled)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig(
        root=args.root,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        num_classes=args.num_classes,
        learning_rate=args.learning_rate,
        backbone_learning_rate=args.backbone_learning_rate,
        weight_decay=args.weight_decay,
        num_queries=args.num_queries,
        hidden_dim=args.hidden_dim,
        decoder_layers=args.decoder_layers,
        num_denoising=args.num_denoising,
        seed=args.seed,
    )
    with (output_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(asdict(config), file, indent=2)

    start_epoch = 0
    best_validation_loss = math.inf
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"Resume checkpoint not found: {resume_path}")
        start_epoch, best_validation_loss = load_checkpoint(
            resume_path,
            model,
            optimizer,
            scheduler,
            scaler,
            device,
        )
        print(f"Resumed from epoch {start_epoch}")

    history_path = output_dir / "history.jsonl"
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.perf_counter()
        learning_rate = optimizer.param_groups[-1]["lr"]
        print(f"\nEpoch {epoch + 1}/{args.epochs} lr={learning_rate:.6g}")
        train_metrics = run_training_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            args.max_grad_norm,
            amp_enabled,
            args.log_interval,
        )
        validation_metrics = run_validation_epoch(
            model,
            validation_loader,
            criterion,
            device,
            amp_enabled,
        )
        scheduler.step()
        elapsed = time.perf_counter() - epoch_start
        print(format_epoch_metrics("Train", train_metrics))
        print(format_epoch_metrics("Val  ", validation_metrics))
        print(f"Epoch time: {elapsed:.1f} seconds")

        validation_loss = validation_metrics["loss"]
        improved = validation_loss < best_validation_loss
        if improved:
            best_validation_loss = validation_loss
        save_checkpoint(
            output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            epoch,
            best_validation_loss,
            config,
            train_metrics,
            validation_metrics,
        )
        if improved:
            save_checkpoint(
                output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                best_validation_loss,
                config,
                train_metrics,
                validation_metrics,
            )
            print(f"Saved new best checkpoint: val loss {best_validation_loss:.4f}")
        with history_path.open("a", encoding="utf-8") as file:
            file.write(
                json.dumps(
                    {
                        "epoch": epoch + 1,
                        "learning_rate": learning_rate,
                        "seconds": elapsed,
                        "train": train_metrics,
                        "validation": validation_metrics,
                        "best_validation_loss": best_validation_loss,
                    }
                )
                + "\n"
            )

    print(f"\nTraining complete. Checkpoints are in: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train RT-DETR-R18 from scratch")
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--img-size", type=int, default=DEFAULT_IMG_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--num-queries", type=int, default=300)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--decoder-layers", type=int, default=3)
    parser.add_argument("--num-denoising", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument(
        "--backbone-learning-rate",
        type=float,
        default=DEFAULT_LEARNING_RATE,
        help="Same as main LR by default because the backbone is not pretrained.",
    )
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output-dir", default="runs/rtdetr_voc")
    parser.add_argument("--resume", default="")
    parser.add_argument("--max-grad-norm", type=float, default=0.1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-augment", action="store_true")
    args = parser.parse_args()

    if args.img_size % 32 != 0:
        parser.error("--img-size must be divisible by 32")
    if args.batch_size <= 0 or args.epochs <= 0:
        parser.error("--batch-size and --epochs must be positive")
    if args.num_classes <= 0 or args.num_queries <= 0:
        parser.error("--num-classes and --num-queries must be positive")
    if args.hidden_dim % 8 != 0:
        parser.error("--hidden-dim must be divisible by 8")
    if args.decoder_layers <= 0 or args.num_denoising < 0:
        parser.error("--decoder-layers must be positive and --num-denoising non-negative")
    return args


if __name__ == "__main__":
    train(parse_args())
