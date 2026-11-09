#!/usr/bin/env python3
"""
Profile RT-DETR checkpoints: parameters, GFLOPs, activations, peak VRAM and latency.

Designed for:
    https://github.com/iaforek/RT-DETR

Important design choice:
    This script deliberately reuses validate.build_model_from_checkpoint()
    from the repository instead of duplicating RT-DETR construction logic.
    Therefore baseline / P2 / SPD / SpatialUNet variants are reconstructed
    from the checkpoint config in the same way as validate.py.

Examples
--------
Single checkpoint:

    python profile_model.py \
      --checkpoint runs/rtdetr_spatial_unet_p2_coco_1/best.pt \
      --amp

All best.pt checkpoints below runs/:

    python profile_model.py \
      --runs runs \
      --amp

Restrict a runs scan:

    python profile_model.py \
      --runs runs \
      --include "rtdetr_spatial_unet*" \
      --amp

More stable latency benchmark:

    python profile_model.py \
      --checkpoint runs/rtdetr_spatial_unet_p2_coco_1/best.pt \
      --amp \
      --warmup 50 \
      --repeats 300

Dependencies
------------
PyTorch plus the repository dependencies, and fvcore for FLOP/activation analysis:

    pip install fvcore

Outputs
-------
For each model:
  - exact total/trainable parameters from the instantiated model
  - parameter breakdown by top-level module and by leaf module
  - fvcore FLOP estimate (GFLOPs)
  - fvcore activation count
  - unsupported operators reported by fvcore
  - CUDA peak allocated/reserved memory
  - incremental forward allocated memory above model+input baseline
  - CUDA-event latency: mean / median / std / p95
  - throughput (images/s)

When --output-dir is supplied (or by default current directory):
  - model_profile_report.csv
  - model_profile_report.json
  - model_profile_layers.csv

Notes
-----
1. FLOPs:
   fvcore counts one fused multiply-add as one FLOP. State this convention
   when reporting results. Unsupported operators are printed and saved;
   if present, GFLOPs may be an underestimate.

2. VRAM:
   "peak allocated" is the maximum tensor memory allocated by PyTorch.
   "peak reserved" is the maximum CUDA caching-allocator reservation.
   "forward extra" is peak allocated minus the allocated baseline after
   the model and input tensor are already resident on the GPU.

3. Latency:
   This script measures pure model forward time with CUDA events. It does
   not include image loading, preprocessing, AP calculation or annotation.
   Benchmark all models in the same GPU allocation/session for fair comparison.

4. Batch size:
   Default is 1, appropriate for deployment-style per-image latency.
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import json
import math
import os
import statistics
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

try:
    from validate import build_model_from_checkpoint
except Exception as exc:
    raise SystemExit(
        "Could not import build_model_from_checkpoint from validate.py.\n"
        "Run this script from the RT-DETR repository root, for example:\n"
        "    cd /users/40464858/Research/RT-DETR\n"
        "    python profile_model.py ...\n"
        f"Import error: {exc}"
    ) from exc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mib(value: int) -> float:
    return value / (1024.0 ** 2)


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def load_checkpoint(path: Path) -> Mapping[str, Any]:
    """
    Load a trusted locally-created research checkpoint on CPU.

    RT-DETR checkpoints in this repository contain dictionaries/state_dicts and
    config metadata. weights_only=False is explicit for PyTorch versions that
    changed the default behaviour.
    """
    try:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(path, map_location="cpu")

    if not isinstance(obj, Mapping):
        raise TypeError(
            f"Expected checkpoint mapping/dict, got {type(obj).__name__}"
        )
    return obj


def checkpoint_epoch_display(checkpoint: Mapping[str, Any]) -> Optional[int]:
    # Repository checkpoints store zero-based epoch numbers.
    epoch = checkpoint.get("epoch")
    if isinstance(epoch, int):
        return epoch + 1
    return None


def config_bool(config: Mapping[str, Any], key: str) -> bool:
    return bool(config.get(key, False))


def autocast_context(device: torch.device, enabled: bool):
    if device.type == "cuda":
        return torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=enabled,
        )
    return nullcontext()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def empty_cuda_cache(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        synchronize(device)


# ---------------------------------------------------------------------------
# Parameter analysis
# ---------------------------------------------------------------------------

def exact_parameter_stats(model: nn.Module) -> Dict[str, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable = total - trainable
    bytes_total = sum(p.numel() * p.element_size() for p in model.parameters())

    return {
        "total_params": total,
        "trainable_params": trainable,
        "non_trainable_params": non_trainable,
        "parameter_bytes": bytes_total,
    }


def top_level_parameter_breakdown(model: nn.Module) -> Dict[str, int]:
    """
    Count parameters according to the first component of each named parameter.

    In this repository this normally produces groups such as:
        backbone, encoder, decoder
    """
    groups: Dict[str, int] = {}

    for name, param in model.named_parameters():
        group = name.split(".", 1)[0] if name else "<root>"
        groups[group] = groups.get(group, 0) + param.numel()

    return dict(sorted(groups.items(), key=lambda item: item[1], reverse=True))


def leaf_parameter_breakdown(model: nn.Module) -> List[Dict[str, Any]]:
    """
    One row per module that directly owns parameters (recurse=False).
    """
    rows: List[Dict[str, Any]] = []

    for module_name, module in model.named_modules():
        own = list(module.named_parameters(recurse=False))
        if not own:
            continue

        total = sum(p.numel() for _, p in own)
        trainable = sum(
            p.numel() for _, p in own if p.requires_grad
        )

        rows.append(
            {
                "module": module_name or "<root>",
                "type": module.__class__.__name__,
                "parameters": total,
                "trainable_parameters": trainable,
            }
        )

    rows.sort(key=lambda row: row["parameters"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# fvcore FLOPs / activation analysis
# ---------------------------------------------------------------------------

def profile_fvcore(
    model: nn.Module,
    sample: torch.Tensor,
) -> Dict[str, Any]:
    try:
        from fvcore.nn import ActivationCountAnalysis, FlopCountAnalysis
    except ImportError as exc:
        raise RuntimeError(
            "fvcore is required for GFLOPs/activation profiling.\n"
            "Install it in your RT-DETR environment with:\n"
            "    pip install fvcore"
        ) from exc

    # fvcore/JIT tracing is most reproducible in FP32. Do not wrap this in AMP.
    model.eval()

    with torch.inference_mode():
        flop_analysis = FlopCountAnalysis(model, sample)
        flop_total = int(flop_analysis.total())
        flop_unsupported = {
            str(k): int(v)
            for k, v in flop_analysis.unsupported_ops().items()
        }
        flop_by_module = {
            str(k): int(v)
            for k, v in flop_analysis.by_module().items()
            if int(v) > 0
        }

        activation_analysis = ActivationCountAnalysis(model, sample)
        activation_total = int(activation_analysis.total())
        activation_unsupported = {
            str(k): int(v)
            for k, v in activation_analysis.unsupported_ops().items()
        }
        activation_by_module = {
            str(k): int(v)
            for k, v in activation_analysis.by_module().items()
            if int(v) > 0
        }

    return {
        "flops": flop_total,
        "gflops": flop_total / 1e9,
        "flop_unsupported_ops": flop_unsupported,
        "flops_by_module": flop_by_module,
        "activations": activation_total,
        "activations_m": activation_total / 1e6,
        "activation_unsupported_ops": activation_unsupported,
        "activations_by_module": activation_by_module,
    }


# ---------------------------------------------------------------------------
# CUDA memory and latency
# ---------------------------------------------------------------------------

@torch.inference_mode()
def warm_up(
    model: nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    amp: bool,
    iterations: int,
) -> None:
    for _ in range(iterations):
        with autocast_context(device, amp):
            _ = model(sample)
    synchronize(device)


@torch.inference_mode()
def profile_cuda_memory(
    model: nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    amp: bool,
    iterations: int,
) -> Dict[str, Optional[float]]:
    if device.type != "cuda":
        return {
            "baseline_allocated_mib": None,
            "baseline_reserved_mib": None,
            "peak_allocated_mib": None,
            "peak_reserved_mib": None,
            "forward_extra_allocated_mib": None,
        }

    empty_cuda_cache(device)

    # Model and input are already resident. This baseline represents persistent
    # memory before the measured forward passes.
    baseline_allocated = torch.cuda.memory_allocated(device)
    baseline_reserved = torch.cuda.memory_reserved(device)

    torch.cuda.reset_peak_memory_stats(device)

    for _ in range(iterations):
        with autocast_context(device, amp):
            _ = model(sample)

    synchronize(device)

    peak_allocated = torch.cuda.max_memory_allocated(device)
    peak_reserved = torch.cuda.max_memory_reserved(device)

    return {
        "baseline_allocated_mib": mib(baseline_allocated),
        "baseline_reserved_mib": mib(baseline_reserved),
        "peak_allocated_mib": mib(peak_allocated),
        "peak_reserved_mib": mib(peak_reserved),
        "forward_extra_allocated_mib": mib(
            max(0, peak_allocated - baseline_allocated)
        ),
    }


@torch.inference_mode()
def benchmark_latency(
    model: nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    amp: bool,
    repeats: int,
) -> Dict[str, Optional[float]]:
    if device.type != "cuda":
        # CPU fallback using perf_counter.
        import time

        timings: List[float] = []
        for _ in range(repeats):
            start = time.perf_counter()
            with autocast_context(device, amp):
                _ = model(sample)
            timings.append((time.perf_counter() - start) * 1000.0)
    else:
        timings = []

        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            with autocast_context(device, amp):
                _ = model(sample)
            end.record()

            synchronize(device)
            timings.append(float(start.elapsed_time(end)))

    mean_ms = statistics.mean(timings)
    median_ms = statistics.median(timings)
    std_ms = statistics.stdev(timings) if len(timings) > 1 else 0.0
    p95_ms = percentile(timings, 0.95)

    batch_size = int(sample.shape[0])
    throughput = (
        batch_size * 1000.0 / mean_ms
        if mean_ms > 0
        else float("nan")
    )

    return {
        "latency_mean_ms": mean_ms,
        "latency_median_ms": median_ms,
        "latency_std_ms": std_ms,
        "latency_p95_ms": p95_ms,
        "throughput_images_per_s": throughput,
    }


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def discover_checkpoints(
    explicit: Sequence[Path],
    runs: Optional[Path],
    checkpoint_name: str,
    include_patterns: Sequence[str],
    exclude_patterns: Sequence[str],
) -> List[Path]:
    found: List[Path] = []

    for path in explicit:
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        found.append(path)

    if runs is not None:
        if not runs.is_dir():
            raise NotADirectoryError(f"Runs directory not found: {runs}")

        for path in sorted(runs.rglob(checkpoint_name)):
            run_name = path.parent.name

            if include_patterns and not any(
                fnmatch.fnmatch(run_name, pattern)
                for pattern in include_patterns
            ):
                continue

            if exclude_patterns and any(
                fnmatch.fnmatch(run_name, pattern)
                for pattern in exclude_patterns
            ):
                continue

            found.append(path)

    unique: List[Path] = []
    seen = set()
    for path in found:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)

    return unique


# ---------------------------------------------------------------------------
# One-model profile
# ---------------------------------------------------------------------------

def profile_one(
    checkpoint_path: Path,
    *,
    img_size_override: Optional[int],
    batch_size: int,
    device: torch.device,
    amp: bool,
    warmup_iterations: int,
    repeat_iterations: int,
    memory_iterations: int,
    skip_flops: bool,
    layer_top: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    checkpoint = load_checkpoint(checkpoint_path)

    # Reuse the repository's validation reconstruction path. Build first on CPU
    # so fvcore profiling does not unnecessarily consume GPU memory.
    model, resolved_config = build_model_from_checkpoint(
        checkpoint,
        torch.device("cpu"),
    )
    model.eval()

    img_size = int(
        img_size_override
        if img_size_override is not None
        else resolved_config.get("img_size", 416)
    )

    if img_size <= 0:
        raise ValueError(f"Invalid img_size={img_size}")

    # The current repository requires spatial dimensions compatible with the
    # detector/backbone; the trained experiments use 416x416.
    sample_cpu = torch.randn(
        batch_size,
        3,
        img_size,
        img_size,
        dtype=torch.float32,
    )

    param_stats = exact_parameter_stats(model)
    top_groups = top_level_parameter_breakdown(model)
    leaf_rows = leaf_parameter_breakdown(model)

    fvcore_stats: Dict[str, Any]
    if skip_flops:
        fvcore_stats = {
            "flops": None,
            "gflops": None,
            "flop_unsupported_ops": {},
            "flops_by_module": {},
            "activations": None,
            "activations_m": None,
            "activation_unsupported_ops": {},
            "activations_by_module": {},
        }
    else:
        fvcore_stats = profile_fvcore(model, sample_cpu)

    # Move the exact same instantiated model to the requested runtime device for
    # VRAM and latency measurements.
    model = model.to(device)
    sample = sample_cpu.to(device)

    warm_up(
        model,
        sample,
        device,
        amp,
        warmup_iterations,
    )

    memory_stats = profile_cuda_memory(
        model,
        sample,
        device,
        amp,
        memory_iterations,
    )

    latency_stats = benchmark_latency(
        model,
        sample,
        device,
        amp,
        repeat_iterations,
    )

    run_name = checkpoint_path.parent.name
    epoch = checkpoint_epoch_display(checkpoint)

    summary: Dict[str, Any] = {
        "run": run_name,
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_epoch": epoch,
        "backbone": resolved_config.get("backbone", "presnet18"),
        "use_p2": config_bool(resolved_config, "use_p2"),
        "use_spd": config_bool(resolved_config, "use_spd"),
        "use_s2_fusion": config_bool(resolved_config, "use_s2_fusion"),
        "use_spd_detr": config_bool(resolved_config, "use_spd_detr"),
        "num_classes": int(resolved_config.get("num_classes", 0)),
        "num_queries": int(resolved_config.get("num_queries", 0)),
        "hidden_dim": int(resolved_config.get("hidden_dim", 0)),
        "decoder_layers": int(resolved_config.get("decoder_layers", 0)),
        "img_size": img_size,
        "batch_size": batch_size,
        "device": str(device),
        "amp": bool(amp),
        "torch_version": torch.__version__,
        "cuda_device_name": (
            torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None
        ),
        "total_params": param_stats["total_params"],
        "total_params_m": param_stats["total_params"] / 1e6,
        "trainable_params": param_stats["trainable_params"],
        "trainable_params_m": param_stats["trainable_params"] / 1e6,
        "non_trainable_params": param_stats["non_trainable_params"],
        "parameter_memory_mib": mib(param_stats["parameter_bytes"]),
        "top_level_parameters": top_groups,
        **fvcore_stats,
        **memory_stats,
        **latency_stats,
    }

    # Add context to layer rows for the combined CSV.
    contextual_layers: List[Dict[str, Any]] = []
    for row in leaf_rows[:layer_top] if layer_top > 0 else leaf_rows:
        contextual_layers.append(
            {
                "run": run_name,
                "checkpoint": str(checkpoint_path.resolve()),
                **row,
            }
        )

    return summary, contextual_layers


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return f"{value:.{digits}f}"
    return str(value)


def print_summary(result: Mapping[str, Any]) -> None:
    print()
    print("=" * 88)
    print(result["run"])
    print("=" * 88)
    print(f"Checkpoint          : {result['checkpoint']}")
    print(f"Epoch               : {result['checkpoint_epoch']}")
    print(f"Backbone            : {result['backbone']}")
    print(
        "Features            : "
        f"P2={result['use_p2']} "
        f"SPD={result['use_spd']} "
        f"S2Fusion={result['use_s2_fusion']} "
        f"SPD-DETR={result['use_spd_detr']}"
    )
    print(
        f"Input               : "
        f"{result['batch_size']} x 3 x "
        f"{result['img_size']} x {result['img_size']}"
    )
    print(f"Device              : {result['device']}")
    if result.get("cuda_device_name"):
        print(f"CUDA device         : {result['cuda_device_name']}")
    print(f"AMP benchmark       : {result['amp']}")

    print()
    print("PARAMETERS")
    print(f"Total               : {result['total_params']:,} "
          f"({result['total_params_m']:.3f} M)")
    print(f"Trainable           : {result['trainable_params']:,} "
          f"({result['trainable_params_m']:.3f} M)")
    print(f"Parameter memory    : {result['parameter_memory_mib']:.2f} MiB")

    for name, count in result["top_level_parameters"].items():
        pct = (
            100.0 * count / result["total_params"]
            if result["total_params"]
            else 0.0
        )
        print(
            f"  {name:<18s}: {count:>12,d} "
            f"({count / 1e6:>7.3f} M, {pct:>6.2f}%)"
        )

    print()
    print("COMPUTATION")
    if result["gflops"] is None:
        print("GFLOPs              : skipped")
        print("Activations         : skipped")
    else:
        print(f"GFLOPs              : {result['gflops']:.3f}")
        print(f"Activations         : {result['activations_m']:.3f} M")

        if result["flop_unsupported_ops"]:
            print(
                "WARNING unsupported FLOP ops: "
                + json.dumps(result["flop_unsupported_ops"], sort_keys=True)
            )
        else:
            print("Unsupported FLOP ops: none")

    print()
    print("MEMORY")
    if result["peak_allocated_mib"] is None:
        print("CUDA VRAM           : unavailable (CPU profile)")
    else:
        print(
            f"Baseline allocated  : "
            f"{result['baseline_allocated_mib']:.2f} MiB"
        )
        print(
            f"Peak allocated      : "
            f"{result['peak_allocated_mib']:.2f} MiB"
        )
        print(
            f"Forward extra       : "
            f"{result['forward_extra_allocated_mib']:.2f} MiB"
        )
        print(
            f"Peak reserved       : "
            f"{result['peak_reserved_mib']:.2f} MiB"
        )

    print()
    print("PURE FORWARD LATENCY")
    print(f"Mean                : {result['latency_mean_ms']:.3f} ms")
    print(f"Median              : {result['latency_median_ms']:.3f} ms")
    print(f"Std                 : {result['latency_std_ms']:.3f} ms")
    print(f"P95                 : {result['latency_p95_ms']:.3f} ms")
    print(
        f"Throughput          : "
        f"{result['throughput_images_per_s']:.2f} images/s"
    )


def print_comparison_table(results: Sequence[Mapping[str, Any]]) -> None:
    if not results:
        return

    headers = [
        "Run",
        "Params M",
        "GFLOPs",
        "Act M",
        "Peak MiB",
        "Extra MiB",
        "Mean ms",
        "P95 ms",
        "img/s",
    ]

    rows: List[List[str]] = []
    for result in results:
        rows.append(
            [
                str(result["run"]),
                fmt(result["total_params_m"]),
                fmt(result["gflops"]),
                fmt(result["activations_m"]),
                fmt(result["peak_allocated_mib"], 2),
                fmt(result["forward_extra_allocated_mib"], 2),
                fmt(result["latency_mean_ms"]),
                fmt(result["latency_p95_ms"]),
                fmt(result["throughput_images_per_s"], 2),
            ]
        )

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows))
        for i in range(len(headers))
    ]

    def emit(row: Sequence[str]) -> None:
        print(
            " | ".join(
                value.ljust(widths[i])
                for i, value in enumerate(row)
            )
        )

    print()
    print("=" * 88)
    print("MODEL COMPARISON")
    print("=" * 88)
    emit(headers)
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        emit(row)


def write_outputs(
    results: Sequence[Mapping[str, Any]],
    layer_rows: Sequence[Mapping[str, Any]],
    output_dir: Path,
) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "model_profile_report.json"
    csv_path = output_dir / "model_profile_report.csv"
    layers_path = output_dir / "model_profile_layers.csv"

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(list(results), handle, indent=2)

    summary_fields = [
        "run",
        "checkpoint",
        "checkpoint_epoch",
        "backbone",
        "use_p2",
        "use_spd",
        "use_s2_fusion",
        "use_spd_detr",
        "num_classes",
        "num_queries",
        "hidden_dim",
        "decoder_layers",
        "img_size",
        "batch_size",
        "device",
        "amp",
        "torch_version",
        "cuda_device_name",
        "total_params",
        "total_params_m",
        "trainable_params",
        "trainable_params_m",
        "non_trainable_params",
        "parameter_memory_mib",
        "flops",
        "gflops",
        "activations",
        "activations_m",
        "baseline_allocated_mib",
        "baseline_reserved_mib",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "forward_extra_allocated_mib",
        "latency_mean_ms",
        "latency_median_ms",
        "latency_std_ms",
        "latency_p95_ms",
        "throughput_images_per_s",
    ]

    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=summary_fields)
        writer.writeheader()
        for result in results:
            writer.writerow(
                {field: result.get(field) for field in summary_fields}
            )

    layer_fields = [
        "run",
        "checkpoint",
        "module",
        "type",
        "parameters",
        "trainable_parameters",
    ]

    with layers_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=layer_fields)
        writer.writeheader()
        for row in layer_rows:
            writer.writerow(
                {field: row.get(field) for field in layer_fields}
            )

    return csv_path, json_path, layers_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Profile RT-DETR checkpoints for parameters, GFLOPs, activations, "
            "CUDA VRAM and pure-forward latency."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        action="append",
        default=[],
        help="Checkpoint to profile. May be supplied more than once.",
    )
    parser.add_argument(
        "--runs",
        type=Path,
        default=None,
        help="Scan this directory recursively for checkpoint files.",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="best.pt",
        help="Checkpoint filename used with --runs (default: best.pt).",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help=(
            "Glob pattern for run directory names when scanning --runs. "
            "May be repeated, e.g. --include 'rtdetr_spatial_unet*'."
        ),
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Glob pattern to exclude when scanning --runs. May be repeated.",
    )

    parser.add_argument(
        "--img-size",
        type=int,
        default=None,
        help="Override checkpoint img_size. Default: use checkpoint config.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Profiling batch size (default: 1).",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Profiling device, e.g. cuda, cuda:0, cpu (default: cuda).",
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help="Use FP16 autocast for CUDA latency/VRAM profiling.",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="Number of warm-up forward passes (default: 30).",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=200,
        help="Number of timed forward passes (default: 200).",
    )
    parser.add_argument(
        "--memory-repeats",
        type=int,
        default=20,
        help="Forward passes used for peak memory profiling (default: 20).",
    )
    parser.add_argument(
        "--skip-flops",
        action="store_true",
        help="Skip fvcore FLOP and activation analysis.",
    )
    parser.add_argument(
        "--layer-top",
        type=int,
        default=0,
        help=(
            "Store only the N largest leaf modules in model_profile_layers.csv. "
            "0 stores all parameter-owning modules (default: 0)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for CSV/JSON reports (default: current directory).",
    )

    args = parser.parse_args()

    if not args.checkpoint and args.runs is None:
        parser.error("Supply --checkpoint and/or --runs.")

    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    if args.warmup < 0:
        parser.error("--warmup must be >= 0")
    if args.repeats < 1:
        parser.error("--repeats must be >= 1")
    if args.memory_repeats < 1:
        parser.error("--memory-repeats must be >= 1")
    if args.layer_top < 0:
        parser.error("--layer-top must be >= 0")

    return args


def main() -> int:
    args = parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit(
            "CUDA was requested but torch.cuda.is_available() is False."
        )

    checkpoints = discover_checkpoints(
        explicit=args.checkpoint,
        runs=args.runs,
        checkpoint_name=args.checkpoint_name,
        include_patterns=args.include,
        exclude_patterns=args.exclude,
    )

    if not checkpoints:
        raise SystemExit("No matching checkpoints found.")

    print(f"Found {len(checkpoints)} checkpoint(s).")
    print(f"PyTorch: {torch.__version__}")

    if device.type == "cuda":
        print(f"CUDA: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    results: List[Dict[str, Any]] = []
    all_layers: List[Dict[str, Any]] = []
    failures: List[Tuple[str, str]] = []

    for index, checkpoint_path in enumerate(checkpoints, start=1):
        print()
        print(
            f"[{index}/{len(checkpoints)}] "
            f"Profiling {checkpoint_path}"
        )

        try:
            result, layer_rows = profile_one(
                checkpoint_path,
                img_size_override=args.img_size,
                batch_size=args.batch_size,
                device=device,
                amp=args.amp,
                warmup_iterations=args.warmup,
                repeat_iterations=args.repeats,
                memory_iterations=args.memory_repeats,
                skip_flops=args.skip_flops,
                layer_top=args.layer_top,
            )

            results.append(result)
            all_layers.extend(layer_rows)
            print_summary(result)

        except Exception as exc:
            failures.append((str(checkpoint_path), str(exc)))
            print(f"FAILED: {exc}", file=sys.stderr)

        finally:
            if device.type == "cuda":
                torch.cuda.empty_cache()

    if results:
        print_comparison_table(results)

        csv_path, json_path, layers_path = write_outputs(
            results,
            all_layers,
            args.output_dir,
        )

        print()
        print(f"CSV summary : {csv_path.resolve()}")
        print(f"JSON report : {json_path.resolve()}")
        print(f"Layer CSV   : {layers_path.resolve()}")

    if failures:
        print()
        print("FAILURES")
        for path, error in failures:
            print(f"- {path}: {error}")

    # A partial batch remains useful. Error only when every model failed.
    return 0 if results else 1


if __name__ == "__main__":
    raise SystemExit(main())
