"""Checkpoint selection policy, shared by training and regression tests."""

from __future__ import annotations

import math
from typing import Any, Mapping


CHECKPOINT_METRIC = "map50_95"
# This is the repository's custom AP evaluator, not official COCOeval.
AP_EVALUATION = {
    "protocol": "repository_ap_v1",
    "ap_conf_thres": 0.001,
    "max_detections": 300,
}


def is_ap_improvement(score: float, best: float | None) -> bool:
    """Accept the first finite score (including zero); keep earlier ties."""
    if not math.isfinite(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"Validation mAP50:95 must be finite and in [0, 1], got {score}")
    return best is None or score > best


def restore_ap_selection(checkpoint: Mapping[str, Any]) -> tuple[float | None, int | None]:
    """Legacy loss-selected checkpoints need a fresh AP baseline."""
    if "checkpoint_metric" not in checkpoint:
        return None, None
    if checkpoint["checkpoint_metric"] != CHECKPOINT_METRIC:
        raise ValueError("Resume checkpoint uses a different selection metric")
    if checkpoint.get("ap_evaluation") != AP_EVALUATION:
        raise ValueError("Resume checkpoint uses a different AP evaluation protocol")
    score = float(checkpoint["best_validation_ap"])
    is_ap_improvement(score, None)  # Reject corrupted/non-finite stored scores.
    epoch = int(checkpoint["best_ap_epoch"])
    if epoch < 0 or epoch > int(checkpoint["epoch"]):
        raise ValueError("Resume checkpoint has an invalid best AP epoch")
    return score, epoch
