#!/usr/bin/env python3
"""
RT-DETR parameter report
========================

Scans RT-DETR experiment folders, loads checkpoints, and reports model-size
statistics without running inference.

Typical use:
    python rtdetr_parameter_report.py \
        --runs /users/40464858/Research/RT-DETR/runs

Single checkpoint:
    python rtdetr_parameter_report.py \
        --checkpoint /users/40464858/Research/RT-DETR/runs/rtdetr_coco_1/best.pt

Outputs:
    model_parameter_report.csv
    model_parameter_report.json

Notes:
- If the checkpoint contains a serialized nn.Module, parameter counts are exact,
  including trainable/non-trainable status.
- If the checkpoint contains only a state_dict, learned-parameter count is
  reconstructed from weight tensors after excluding standard registered buffers
  such as BatchNorm running_mean/running_var/num_batches_tracked.
- FLOPs are deliberately not estimated here. Parameter count and inference cost
  are different quantities.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn


COMMON_BUFFER_SUFFIXES = (
    "running_mean",
    "running_var",
    "num_batches_tracked",
)

COMMON_BUFFER_NAMES = {
    "anchor_grid",
    "anchors",
    "position_ids",
}


def human_int(n: Optional[int]) -> str:
    if n is None:
        return "-"
    return f"{n:,}"


def human_m(n: Optional[int]) -> str:
    if n is None:
        return "-"
    return f"{n / 1_000_000:.3f}"


def bytes_to_mib(n: Optional[int]) -> str:
    if n is None:
        return "-"
    return f"{n / (1024 ** 2):.2f}"


def safe_torch_load(path: Path) -> Any:
    """
    Prefer weights_only=True when possible. Fall back to normal checkpoint
    loading for locally created/trusted checkpoints.
    """
    errors = []

    # First try the safer loader.
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        # Older PyTorch without weights_only.
        pass
    except Exception as e:
        errors.append(f"weights_only=True: {e}")

    # Fall back for project checkpoints that contain custom objects.
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        try:
            return torch.load(path, map_location="cpu")
        except Exception as e:
            errors.append(f"default load: {e}")
    except Exception as e:
        errors.append(f"weights_only=False: {e}")

    raise RuntimeError(
        f"Could not load checkpoint {path}\n" + "\n".join(errors)
    )


def looks_like_state_dict(obj: Any) -> bool:
    if not isinstance(obj, dict) or not obj:
        return False

    tensor_items = sum(torch.is_tensor(v) for v in obj.values())
    # A real model state_dict should be overwhelmingly tensor-valued.
    return tensor_items >= max(1, int(0.8 * len(obj)))


def extract_model_or_state_dict(
    checkpoint: Any,
) -> Tuple[Optional[nn.Module], Optional[Dict[str, torch.Tensor]], str]:
    """
    Returns:
        model, state_dict, source_description
    """
    if isinstance(checkpoint, nn.Module):
        return checkpoint, checkpoint.state_dict(), "serialized_module"

    if looks_like_state_dict(checkpoint):
        return None, checkpoint, "state_dict"

    if not isinstance(checkpoint, dict):
        return None, None, "unknown"

    # Prefer exact model objects if present.
    module_keys = ("model", "ema", "module", "net")
    for key in module_keys:
        value = checkpoint.get(key)
        if isinstance(value, nn.Module):
            return value, value.state_dict(), f"serialized_module:{key}"

    # Common state-dict keys, ordered by likely intent.
    state_keys = (
        "model_state_dict",
        "state_dict",
        "model_state",
        "model",
        "ema_state_dict",
        "ema",
        "net",
    )
    for key in state_keys:
        value = checkpoint.get(key)
        if looks_like_state_dict(value):
            return None, value, f"state_dict:{key}"

    return None, None, "unknown"


def normalize_state_dict_keys(
    state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    sd = dict(state_dict)

    # Strip DataParallel prefix if every key has it.
    prefixes = ("module.", "_orig_mod.")
    for prefix in prefixes:
        if sd and all(k.startswith(prefix) for k in sd):
            sd = {k[len(prefix):]: v for k, v in sd.items()}

    return sd


def is_probable_buffer(name: str) -> bool:
    last = name.split(".")[-1]

    if last in COMMON_BUFFER_SUFFIXES:
        return True

    if last in COMMON_BUFFER_NAMES:
        return True

    # Common non-learned lookup/grid buffers.
    lower = name.lower()
    if lower.endswith(".anchor_grid") or lower.endswith(".anchors"):
        return True

    return False


def tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def state_dict_parameter_estimate(
    state_dict: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """
    Estimate learned parameter count from state_dict.

    Standard registered non-learned buffers are excluded. This is exact for
    ordinary Conv/Linear/Norm-heavy networks unless the model defines unusual
    custom buffers that are not covered by is_probable_buffer().
    """
    state_dict = normalize_state_dict_keys(state_dict)

    learned = {}
    buffers = {}

    for name, tensor in state_dict.items():
        if not torch.is_tensor(tensor):
            continue
        if is_probable_buffer(name):
            buffers[name] = tensor
        else:
            learned[name] = tensor

    learned_params = sum(t.numel() for t in learned.values())
    learned_bytes = sum(tensor_bytes(t) for t in learned.values())

    buffer_elems = sum(t.numel() for t in buffers.values())
    buffer_bytes = sum(tensor_bytes(t) for t in buffers.values())

    return {
        "learned": learned,
        "buffers": buffers,
        "learned_params": learned_params,
        "learned_bytes": learned_bytes,
        "buffer_elements": buffer_elems,
        "buffer_bytes": buffer_bytes,
    }


def exact_module_stats(model: nn.Module) -> Dict[str, Any]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable

    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    return {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
        "param_bytes": param_bytes,
    }


def split_parts(name: str) -> List[str]:
    return [p for p in name.split(".") if p]


def is_backbone_name(name: str, backbone_prefix: str) -> bool:
    parts = split_parts(name)
    target_parts = split_parts(backbone_prefix)

    if not target_parts:
        return False

    # Exact prefix match.
    if name == backbone_prefix or name.startswith(backbone_prefix + "."):
        return True

    # Match the requested path as a subsequence of path components.
    n = len(target_parts)
    for i in range(0, len(parts) - n + 1):
        if parts[i:i+n] == target_parts:
            return True

    return False


def group_name(name: str) -> str:
    """
    Produce a useful top-level group while preserving backbone visibility.
    """
    parts = split_parts(name)

    if "backbone" in parts:
        i = parts.index("backbone")
        return ".".join(parts[: i + 1])

    if not parts:
        return "<root>"

    # If all model parameters are nested below "model.", the next component is
    # usually more informative.
    if parts[0] in {"model", "net", "module"} and len(parts) > 1:
        return ".".join(parts[:2])

    return parts[0]


def parameter_breakdown_from_state(
    learned: Dict[str, torch.Tensor],
    backbone_prefix: str,
) -> Tuple[int, int, Dict[str, int]]:
    backbone = 0
    groups = defaultdict(int)

    for name, tensor in learned.items():
        n = tensor.numel()

        if is_backbone_name(name, backbone_prefix):
            backbone += n

        groups[group_name(name)] += n

    total = sum(groups.values())
    return total, backbone, dict(groups)


def parameter_breakdown_from_model(
    model: nn.Module,
    backbone_prefix: str,
) -> Tuple[int, int, Dict[str, int]]:
    total = 0
    backbone = 0
    groups = defaultdict(int)

    for name, param in model.named_parameters():
        n = param.numel()
        total += n

        if is_backbone_name(name, backbone_prefix):
            backbone += n

        groups[group_name(name)] += n

    return total, backbone, dict(groups)


def read_config(run_dir: Path) -> Tuple[Optional[Path], Dict[str, Any]]:
    candidates = []

    exact = run_dir / "config.json"
    if exact.exists():
        candidates.append(exact)

    candidates.extend(sorted(run_dir.glob("config*.json")))

    # Remove duplicates while preserving order.
    unique = []
    seen = set()
    for p in candidates:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)

    for path in unique:
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                return path, data
        except Exception:
            continue

    return None, {}


def extract_epoch(checkpoint: Any) -> Optional[int]:
    if not isinstance(checkpoint, dict):
        return None

    for key in (
        "checkpoint_epoch",
        "best_epoch",
        "epoch",
        "start_epoch",
    ):
        value = checkpoint.get(key)
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)

    return None


def find_checkpoints(
    runs: Optional[Path],
    checkpoint_args: List[Path],
    checkpoint_name: str,
) -> List[Path]:
    found = []

    for p in checkpoint_args:
        if p.is_file():
            found.append(p)

    if runs is not None:
        if not runs.exists():
            raise FileNotFoundError(f"Runs directory does not exist: {runs}")

        # Prefer one standard checkpoint per experiment directory.
        found.extend(sorted(runs.rglob(checkpoint_name)))

    # Unique paths, stable order.
    unique = []
    seen = set()
    for p in found:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            unique.append(p)

    return unique


def short_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "backbone": config.get("backbone"),
        "seed": config.get("seed"),
        "use_p2": config.get("use_p2"),
        "use_spd": config.get("use_spd"),
        "use_s2_fusion": config.get("use_s2_fusion"),
        "use_spd_detr": config.get("use_spd_detr"),
        "img_size": config.get("img_size"),
        "num_queries": config.get("num_queries"),
        "decoder_layers": config.get("decoder_layers"),
    }


def analyse_checkpoint(
    checkpoint_path: Path,
    backbone_prefix: str,
) -> Dict[str, Any]:
    run_dir = checkpoint_path.parent
    run_name = run_dir.name

    config_path, config = read_config(run_dir)
    cfg = short_config(config)

    checkpoint = safe_torch_load(checkpoint_path)
    epoch = extract_epoch(checkpoint)

    model, state_dict, source = extract_model_or_state_dict(checkpoint)

    if state_dict is None:
        raise RuntimeError(
            "No model/state_dict found. Known keys were not present in checkpoint."
        )

    state_dict = normalize_state_dict_keys(state_dict)
    state_stats = state_dict_parameter_estimate(state_dict)

    if model is not None:
        exact = exact_module_stats(model)
        total_params, backbone_params, groups = parameter_breakdown_from_model(
            model, backbone_prefix
        )
        trainable_params = exact["trainable_params"]
        non_trainable_params = exact["non_trainable_params"]
        param_bytes = exact["param_bytes"]
        count_mode = "exact"
    else:
        total_params, backbone_params, groups = parameter_breakdown_from_state(
            state_stats["learned"], backbone_prefix
        )
        trainable_params = None
        non_trainable_params = None
        param_bytes = state_stats["learned_bytes"]
        count_mode = "state_dict_estimate"

    non_backbone_params = (
        total_params - backbone_params if backbone_params is not None else None
    )

    row = {
        "run": run_name,
        "checkpoint": str(checkpoint_path),
        "config": str(config_path) if config_path else None,
        "checkpoint_epoch": epoch,
        "count_mode": count_mode,
        "checkpoint_source": source,
        **cfg,
        "total_params": total_params,
        "total_params_m": total_params / 1_000_000,
        "backbone_params": backbone_params,
        "backbone_params_m": backbone_params / 1_000_000,
        "non_backbone_params": non_backbone_params,
        "non_backbone_params_m": non_backbone_params / 1_000_000,
        "trainable_params": trainable_params,
        "non_trainable_params": non_trainable_params,
        "parameter_memory_mib": param_bytes / (1024 ** 2),
        "checkpoint_size_mib": checkpoint_path.stat().st_size / (1024 ** 2),
        "registered_buffer_elements": state_stats["buffer_elements"],
        "groups": dict(sorted(groups.items(), key=lambda kv: kv[1], reverse=True)),
    }

    return row


def print_table(rows: List[Dict[str, Any]]) -> None:
    if not rows:
        print("No successful model analyses.")
        return

    headers = [
        "Run",
        "Backbone",
        "P2",
        "SPD",
        "Seed",
        "Epoch",
        "Params(M)",
        "Backbone(M)",
        "Other(M)",
        "Param MiB",
        "CKPT MiB",
        "Mode",
    ]

    table_rows = []
    for r in rows:
        table_rows.append([
            str(r["run"]),
            str(r.get("backbone") if r.get("backbone") is not None else "-"),
            str(r.get("use_p2") if r.get("use_p2") is not None else "-"),
            str(r.get("use_spd") if r.get("use_spd") is not None else "-"),
            str(r.get("seed") if r.get("seed") is not None else "-"),
            str(r.get("checkpoint_epoch") if r.get("checkpoint_epoch") is not None else "-"),
            f'{r["total_params_m"]:.3f}',
            f'{r["backbone_params_m"]:.3f}',
            f'{r["non_backbone_params_m"]:.3f}',
            f'{r["parameter_memory_mib"]:.2f}',
            f'{r["checkpoint_size_mib"]:.2f}',
            str(r["count_mode"]),
        ])

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in table_rows))
        for i in range(len(headers))
    ]

    def print_row(row: Iterable[str]) -> None:
        print(" | ".join(str(v).ljust(widths[i]) for i, v in enumerate(row)))

    print()
    print_row(headers)
    print("-+-".join("-" * w for w in widths))
    for row in table_rows:
        print_row(row)
    print()


def print_breakdowns(rows: List[Dict[str, Any]], top_n: int) -> None:
    for r in rows:
        print(f"\n[{r['run']}] parameter groups")
        print("-" * 72)

        groups = list(r["groups"].items())
        for name, n in groups[:top_n]:
            pct = (100.0 * n / r["total_params"]) if r["total_params"] else 0.0
            print(f"{name:40s} {n:12,d}  {n/1e6:8.3f} M  {pct:6.2f}%")

        if len(groups) > top_n:
            remainder = sum(n for _, n in groups[top_n:])
            pct = 100.0 * remainder / r["total_params"]
            print(f"{'<remaining groups>':40s} {remainder:12,d}  "
                  f"{remainder/1e6:8.3f} M  {pct:6.2f}%")


def write_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    fields = [
        "run",
        "checkpoint",
        "config",
        "checkpoint_epoch",
        "count_mode",
        "checkpoint_source",
        "backbone",
        "seed",
        "use_p2",
        "use_spd",
        "use_s2_fusion",
        "use_spd_detr",
        "img_size",
        "num_queries",
        "decoder_layers",
        "total_params",
        "total_params_m",
        "backbone_params",
        "backbone_params_m",
        "non_backbone_params",
        "non_backbone_params_m",
        "trainable_params",
        "non_trainable_params",
        "parameter_memory_mib",
        "checkpoint_size_mib",
        "registered_buffer_elements",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        for r in rows:
            writer.writerow({k: r.get(k) for k in fields})


def write_json(rows: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Scan RT-DETR checkpoints and report parameter counts."
    )

    p.add_argument(
        "--runs",
        type=Path,
        default=None,
        help="Root runs directory. All matching checkpoints below it are scanned.",
    )
    p.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=[],
        help="Specific checkpoint. May be supplied multiple times.",
    )
    p.add_argument(
        "--checkpoint-name",
        default="best.pt",
        help="Filename to search for below --runs (default: best.pt).",
    )
    p.add_argument(
        "--backbone-prefix",
        default="backbone",
        help="Module path/name used to identify backbone parameters "
             "(default: backbone).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for CSV/JSON output (default: current directory).",
    )
    p.add_argument(
        "--top-groups",
        type=int,
        default=12,
        help="Number of parameter groups to print per model (default: 12).",
    )
    p.add_argument(
        "--no-breakdown",
        action="store_true",
        help="Do not print per-model module breakdowns.",
    )

    return p


def main() -> int:
    args = build_parser().parse_args()

    if args.runs is None and not args.checkpoint:
        print(
            "ERROR: supply --runs <directory> or --checkpoint <file>.",
            file=sys.stderr,
        )
        return 2

    checkpoints = find_checkpoints(
        args.runs,
        args.checkpoint,
        args.checkpoint_name,
    )

    if not checkpoints:
        print("No checkpoints found.", file=sys.stderr)
        return 1

    print(f"Found {len(checkpoints)} checkpoint(s).")

    rows = []
    failures = []

    for i, checkpoint in enumerate(checkpoints, 1):
        print(f"[{i}/{len(checkpoints)}] {checkpoint}")

        try:
            row = analyse_checkpoint(
                checkpoint,
                backbone_prefix=args.backbone_prefix,
            )
            rows.append(row)
        except Exception as e:
            failures.append((str(checkpoint), str(e)))
            print(f"  FAILED: {e}", file=sys.stderr)

    rows.sort(key=lambda r: r["run"])

    print_table(rows)

    if not args.no_breakdown:
        print_breakdowns(rows, max(1, args.top_groups))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "model_parameter_report.csv"
    json_path = args.output_dir / "model_parameter_report.json"

    if rows:
        write_csv(rows, csv_path)
        write_json(rows, json_path)

        print(f"\nCSV : {csv_path.resolve()}")
        print(f"JSON: {json_path.resolve()}")

    if failures:
        print("\nFailures:")
        for path, error in failures:
            print(f"- {path}: {error}")

    # Non-zero only when every checkpoint failed.
    return 0 if rows else 1


if __name__ == "__main__":
    raise SystemExit(main())
