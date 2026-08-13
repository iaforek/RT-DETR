"""Validate RT-DETR-R18 or annotate one image.

This script mirrors the command style and metric output used by the author's
YOLO repositories. RT-DETR inference is NMS-free: each decoder query contributes
at most one class prediction and no Non-Maximum Suppression is applied.

Full validation example
-----------------------
python validate.py \
    --checkpoint runs/rtdetr_voc/best.pt \
    --root /mnt/scratch2/users/40464858/VOC_dataset/voc_yolo \
    --split val \
    --batch-size 32

Single-image example
--------------------
python validate.py \
    --checkpoint runs/rtdetr_voc/best.pt \
    --image /mnt/scratch2/users/40464858/coco128/images/train2017/000000000113.jpg \
    --conf-thres 0.05 \
    --out-path pred_vis_rtdetr.jpg
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch import Tensor
from torch.utils.data import DataLoader, Subset

from model import RTDETR, box_cxcywh_to_xyxy
from train import ROOT, YoloTxtDataset, choose_device, collate_fn, worker_init_fn


VOC_CLASS_NAMES = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor",
]

IOU_THRESHOLDS = np.round(np.arange(0.50, 0.96, 0.05), 2)

# COCO-style object-size buckets, measured as bounding-box area in the
# ORIGINAL image coordinates (pixels^2).
COCO_SIZE_RANGES = {
    "small": (0.0, float(32 ** 2)),
    "medium": (float(32 ** 2), float(96 ** 2)),
    "large": (float(96 ** 2), math.inf),
}


# -----------------------------------------------------------------------------
# Checkpoint and model helpers
# -----------------------------------------------------------------------------


def load_checkpoint_file(path: Path, device: torch.device) -> Mapping[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device)
    if not isinstance(checkpoint, Mapping):
        raise TypeError(
            f"Expected checkpoint mapping, received {type(checkpoint).__name__}"
        )
    return checkpoint


def checkpoint_config(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    config = checkpoint.get("config", {})
    return dict(config) if isinstance(config, Mapping) else {}


def resolve_class_names(num_classes: int, names_file: str | None) -> List[str]:
    if names_file:
        path = Path(names_file)
        if not path.is_file():
            raise FileNotFoundError(f"Class-names file does not exist: {path}")
        names = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(names) != num_classes:
            raise ValueError(
                f"Class-names file contains {len(names)} names, "
                f"but num_classes={num_classes}"
            )
        return names
    if num_classes == len(VOC_CLASS_NAMES):
        return VOC_CLASS_NAMES.copy()
    return [f"class_{index}" for index in range(num_classes)]


def build_model_from_checkpoint(
    checkpoint: Mapping[str, Any],
    device: torch.device,
    *,
    num_classes_override: int | None = None,
    num_queries_override: int | None = None,
    hidden_dim_override: int | None = None,
    decoder_layers_override: int | None = None,
) -> Tuple[RTDETR, Dict[str, Any]]:
    config = checkpoint_config(checkpoint)
    num_classes = int(
        num_classes_override
        if num_classes_override is not None
        else config.get("num_classes", 20)
    )
    num_queries = int(
        num_queries_override
        if num_queries_override is not None
        else config.get("num_queries", 300)
    )
    hidden_dim = int(
        hidden_dim_override
        if hidden_dim_override is not None
        else config.get("hidden_dim", 256)
    )
    decoder_layers = int(
        decoder_layers_override
        if decoder_layers_override is not None
        else config.get("decoder_layers", 3)
    )

    # Denoising queries are training-only and are not needed during validation.
    model = RTDETR(
        input_channels=3,
        num_classes=num_classes,
        num_queries=num_queries,
        hidden_dim=hidden_dim,
        num_decoder_layers=decoder_layers,
        num_denoising=int(config.get("num_denoising", 100)),
    ).to(device)
    state_dict = checkpoint.get("model", checkpoint)
    if not isinstance(state_dict, Mapping):
        raise TypeError("Checkpoint does not contain a model state dictionary")
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    resolved = {
        **config,
        "num_classes": num_classes,
        "num_queries": num_queries,
        "hidden_dim": hidden_dim,
        "decoder_layers": decoder_layers,
    }
    return model, resolved


# -----------------------------------------------------------------------------
# Geometry, matching and AP
# -----------------------------------------------------------------------------


def box_iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float64)
    intersection_x1 = np.maximum(box[0], boxes[:, 0])
    intersection_y1 = np.maximum(box[1], boxes[:, 1])
    intersection_x2 = np.minimum(box[2], boxes[:, 2])
    intersection_y2 = np.minimum(box[3], boxes[:, 3])
    intersection_width = np.maximum(0.0, intersection_x2 - intersection_x1)
    intersection_height = np.maximum(0.0, intersection_y2 - intersection_y1)
    intersection = intersection_width * intersection_height
    box_area = max(0.0, float(box[2] - box[0])) * max(
        0.0, float(box[3] - box[1])
    )
    boxes_area = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(
        0.0, boxes[:, 3] - boxes[:, 1]
    )
    union = box_area + boxes_area - intersection
    return intersection / np.maximum(union, 1e-12)


def average_precision_101_point(recall: np.ndarray, precision: np.ndarray) -> float:
    if recall.size == 0 or precision.size == 0:
        return 0.0
    values = []
    for recall_level in np.linspace(0.0, 1.0, 101):
        valid = recall >= recall_level
        values.append(float(precision[valid].max()) if valid.any() else 0.0)
    return float(np.mean(values))


def empty_prediction_data() -> Dict[str, np.ndarray]:
    return {
        "image_ids": np.zeros((0,), dtype=np.int64),
        "scores": np.zeros((0,), dtype=np.float64),
        "boxes": np.zeros((0, 4), dtype=np.float64),
    }


def box_areas_in_original_pixels(
    boxes: np.ndarray,
    original_size: Tuple[int, int],
) -> np.ndarray:
    """Return bbox areas in original-image pixel coordinates.

    boxes are normalized xyxy coordinates. original_size is (height, width).
    """
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if boxes.size == 0:
        return np.zeros((0,), dtype=np.float64)
    original_height, original_width = original_size
    widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * original_width
    heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1]) * original_height
    return widths * heights


def area_in_range(
    areas: np.ndarray,
    area_range: Tuple[float, float],
) -> np.ndarray:
    minimum_area, maximum_area = area_range
    if math.isinf(maximum_area):
        return areas >= minimum_area
    return (areas >= minimum_area) & (areas < maximum_area)


def match_class_predictions(
    prediction_data: Mapping[str, np.ndarray],
    ground_truth_by_image: Mapping[int, np.ndarray],
    iou_threshold: float,
    minimum_confidence: float,
) -> Dict[str, Any]:
    total_ground_truth = int(
        sum(boxes.shape[0] for boxes in ground_truth_by_image.values())
    )
    image_ids = np.asarray(prediction_data["image_ids"], dtype=np.int64)
    scores = np.asarray(prediction_data["scores"], dtype=np.float64)
    boxes = np.asarray(prediction_data["boxes"], dtype=np.float64).reshape(-1, 4)

    keep = scores >= minimum_confidence
    image_ids = image_ids[keep]
    scores = scores[keep]
    boxes = boxes[keep]
    if scores.size:
        order = np.argsort(-scores, kind="stable")
        image_ids = image_ids[order]
        scores = scores[order]
        boxes = boxes[order]

    true_positive = np.zeros(scores.shape[0], dtype=np.float64)
    false_positive = np.zeros(scores.shape[0], dtype=np.float64)
    matched_ious = np.zeros(scores.shape[0], dtype=np.float64)
    matched_ground_truth = {
        image_id: np.zeros(gt_boxes.shape[0], dtype=bool)
        for image_id, gt_boxes in ground_truth_by_image.items()
    }

    for prediction_index, (image_id, box) in enumerate(zip(image_ids, boxes)):
        image_id = int(image_id)
        gt_boxes = ground_truth_by_image.get(image_id)
        if gt_boxes is None or gt_boxes.shape[0] == 0:
            false_positive[prediction_index] = 1.0
            continue
        available = ~matched_ground_truth[image_id]
        if not available.any():
            false_positive[prediction_index] = 1.0
            continue
        ious = np.where(available, box_iou_one_to_many(box, gt_boxes), -1.0)
        best_gt = int(np.argmax(ious))
        best_iou = float(ious[best_gt])
        if best_iou >= iou_threshold:
            true_positive[prediction_index] = 1.0
            matched_ious[prediction_index] = best_iou
            matched_ground_truth[image_id][best_gt] = True
        else:
            false_positive[prediction_index] = 1.0

    cumulative_tp = np.cumsum(true_positive)
    cumulative_fp = np.cumsum(false_positive)
    recall_curve = (
        cumulative_tp / total_ground_truth
        if total_ground_truth > 0
        else np.zeros_like(cumulative_tp)
    )
    precision_curve = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp, 1e-12
    )
    ap = (
        average_precision_101_point(recall_curve, precision_curve)
        if total_ground_truth > 0
        else math.nan
    )
    tp = int(true_positive.sum())
    fp = int(false_positive.sum())
    fn = max(total_ground_truth - tp, 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(total_ground_truth, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    positive_ious = matched_ious[true_positive.astype(bool)]
    return {
        "ap": float(ap),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ground_truth": total_ground_truth,
        "predictions": int(scores.shape[0]),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "matched_ious": positive_ious,
        "mean_matched_iou": (
            float(positive_ious.mean()) if positive_ious.size else math.nan
        ),
    }


def match_class_predictions_by_size(
    prediction_data: Mapping[str, np.ndarray],
    ground_truth_by_image: Mapping[int, np.ndarray],
    image_sizes_by_id: Mapping[int, Tuple[int, int]],
    iou_threshold: float,
    minimum_confidence: float,
    area_range: Tuple[float, float],
) -> Dict[str, Any]:
    """Match one class while evaluating only one object-size bucket.

    Ground-truth boxes outside the requested size range are treated as ignored.
    A prediction that matches an ignored GT is ignored rather than counted as FP.
    An unmatched prediction is counted as FP only when the prediction itself is
    inside the requested size range.
    """
    gt_active_by_image: Dict[int, np.ndarray] = {}
    matched_ground_truth: Dict[int, np.ndarray] = {}
    total_ground_truth = 0

    for image_id, gt_boxes in ground_truth_by_image.items():
        original_size = image_sizes_by_id[image_id]
        gt_areas = box_areas_in_original_pixels(gt_boxes, original_size)
        active = area_in_range(gt_areas, area_range)
        gt_active_by_image[image_id] = active
        matched_ground_truth[image_id] = np.zeros(gt_boxes.shape[0], dtype=bool)
        total_ground_truth += int(active.sum())

    image_ids = np.asarray(prediction_data["image_ids"], dtype=np.int64)
    scores = np.asarray(prediction_data["scores"], dtype=np.float64)
    boxes = np.asarray(prediction_data["boxes"], dtype=np.float64).reshape(-1, 4)

    keep = scores >= minimum_confidence
    image_ids = image_ids[keep]
    scores = scores[keep]
    boxes = boxes[keep]

    if scores.size:
        order = np.argsort(-scores, kind="stable")
        image_ids = image_ids[order]
        scores = scores[order]
        boxes = boxes[order]

    true_positive = np.zeros(scores.shape[0], dtype=np.float64)
    false_positive = np.zeros(scores.shape[0], dtype=np.float64)
    ignored_prediction = np.zeros(scores.shape[0], dtype=bool)
    matched_ious = np.zeros(scores.shape[0], dtype=np.float64)

    for prediction_index, (image_id_value, box) in enumerate(zip(image_ids, boxes)):
        image_id = int(image_id_value)
        original_size = image_sizes_by_id[image_id]
        prediction_area = box_areas_in_original_pixels(
            box.reshape(1, 4), original_size
        )[0]
        prediction_in_range = bool(
            area_in_range(np.asarray([prediction_area]), area_range)[0]
        )

        gt_boxes = ground_truth_by_image.get(image_id)
        if gt_boxes is None or gt_boxes.shape[0] == 0:
            if prediction_in_range:
                false_positive[prediction_index] = 1.0
            else:
                ignored_prediction[prediction_index] = True
            continue

        gt_active = gt_active_by_image[image_id]
        matched = matched_ground_truth[image_id]
        ious = box_iou_one_to_many(box, gt_boxes)

        # Prefer an unmatched GT that belongs to the requested size bucket.
        active_available = gt_active & ~matched
        if active_available.any():
            active_ious = np.where(active_available, ious, -1.0)
            best_active = int(np.argmax(active_ious))
            best_active_iou = float(active_ious[best_active])
            if best_active_iou >= iou_threshold:
                true_positive[prediction_index] = 1.0
                matched_ious[prediction_index] = best_active_iou
                matched[best_active] = True
                continue

        # A prediction matching a GT outside this bucket is ignored, not FP.
        ignored_available = (~gt_active) & ~matched
        if ignored_available.any():
            ignored_ious = np.where(ignored_available, ious, -1.0)
            best_ignored = int(np.argmax(ignored_ious))
            best_ignored_iou = float(ignored_ious[best_ignored])
            if best_ignored_iou >= iou_threshold:
                ignored_prediction[prediction_index] = True
                matched[best_ignored] = True
                continue

        if prediction_in_range:
            false_positive[prediction_index] = 1.0
        else:
            ignored_prediction[prediction_index] = True

    considered = ~ignored_prediction
    tp_eval = true_positive[considered]
    fp_eval = false_positive[considered]

    cumulative_tp = np.cumsum(tp_eval)
    cumulative_fp = np.cumsum(fp_eval)
    recall_curve = (
        cumulative_tp / total_ground_truth
        if total_ground_truth > 0
        else np.zeros_like(cumulative_tp)
    )
    precision_curve = cumulative_tp / np.maximum(
        cumulative_tp + cumulative_fp, 1e-12
    )
    ap = (
        average_precision_101_point(recall_curve, precision_curve)
        if total_ground_truth > 0
        else math.nan
    )

    tp = int(true_positive.sum())
    fp = int(false_positive.sum())
    fn = max(total_ground_truth - tp, 0)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(total_ground_truth, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-12)
    positive_ious = matched_ious[true_positive.astype(bool)]

    return {
        "ap": float(ap),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "ground_truth": total_ground_truth,
        "predictions": int(considered.sum()),
        "ignored_predictions": int(ignored_prediction.sum()),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "matched_ious": positive_ious,
        "mean_matched_iou": (
            float(positive_ious.mean()) if positive_ious.size else math.nan
        ),
    }


def safe_mean(values: Sequence[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(np.mean(finite)) if finite else math.nan


def format_metric(value: float) -> str:
    return f"{value:.4f}" if math.isfinite(value) else "nan"


# -----------------------------------------------------------------------------
# Dataset inference
# -----------------------------------------------------------------------------


def concatenate_prediction_chunks(
    chunks: Sequence[Dict[str, List[np.ndarray]]],
) -> List[Dict[str, np.ndarray]]:
    combined: List[Dict[str, np.ndarray]] = []
    for class_chunks in chunks:
        if class_chunks["scores"]:
            combined.append(
                {
                    "image_ids": np.concatenate(class_chunks["image_ids"]),
                    "scores": np.concatenate(class_chunks["scores"]),
                    "boxes": np.concatenate(class_chunks["boxes"], axis=0),
                }
            )
        else:
            combined.append(empty_prediction_data())
    return combined


@torch.inference_mode()
def collect_predictions(
    model: RTDETR,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    ap_confidence_threshold: float,
    max_detections: int,
    amp_enabled: bool,
    log_interval: int,
) -> Tuple[
    List[Dict[str, np.ndarray]],
    List[Dict[int, np.ndarray]],
    Dict[int, Tuple[int, int]],
    int,
]:
    prediction_chunks: List[Dict[str, List[np.ndarray]]] = [
        {"image_ids": [], "scores": [], "boxes": []}
        for _ in range(num_classes)
    ]
    ground_truth_by_class: List[Dict[int, np.ndarray]] = [
        {} for _ in range(num_classes)
    ]
    image_sizes_by_id: Dict[int, Tuple[int, int]] = {}
    model.eval()
    image_offset = 0
    total_images = 0
    start_time = time.perf_counter()

    for batch_index, (images, targets) in enumerate(loader, start=1):
        images = images.to(device, non_blocking=True)
        with torch.amp.autocast(
            device_type=device.type,
            dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
            enabled=amp_enabled,
        ):
            output = model(images)
        logits = output["pred_logits"]
        boxes = output["pred_boxes"]
        if not isinstance(logits, Tensor) or not isinstance(boxes, Tensor):
            raise TypeError("RT-DETR output tensors are missing")
        scores, classes = logits.float().sigmoid().max(dim=-1)
        boxes_xyxy = box_cxcywh_to_xyxy(boxes.float()).clamp(0.0, 1.0)

        for local_index, target in enumerate(targets):
            global_image_id = image_offset + local_index

            if "orig_size" in target:
                original_height, original_width = (
                    int(value) for value in target["orig_size"].tolist()
                )
            else:
                # Backward-compatible fallback for datasets created before
                # orig_size was added to YoloTxtDataset.
                with Image.open(str(target["path"])) as source_image:
                    original_width, original_height = source_image.size

            image_sizes_by_id[global_image_id] = (
                original_height, original_width
            )

            target_labels = target["labels"].cpu().numpy().astype(np.int64)
            target_boxes = box_cxcywh_to_xyxy(target["boxes"]).cpu().numpy()
            for class_id in np.unique(target_labels):
                ground_truth_by_class[int(class_id)][global_image_id] = target_boxes[
                    target_labels == class_id
                ].astype(np.float64)

            image_scores = scores[local_index]
            image_classes = classes[local_index]
            image_boxes = boxes_xyxy[local_index]
            order = image_scores.argsort(descending=True)[:max_detections]
            image_scores = image_scores[order].cpu().numpy()
            image_classes = image_classes[order].cpu().numpy().astype(np.int64)
            image_boxes = image_boxes[order].cpu().numpy()
            valid = (
                np.isfinite(image_boxes).all(axis=1)
                & np.isfinite(image_scores)
                & (image_scores >= ap_confidence_threshold)
                & (image_boxes[:, 2] > image_boxes[:, 0])
                & (image_boxes[:, 3] > image_boxes[:, 1])
                & (image_classes >= 0)
                & (image_classes < num_classes)
            )
            image_scores = image_scores[valid]
            image_classes = image_classes[valid]
            image_boxes = image_boxes[valid]
            for class_id in np.unique(image_classes):
                selected = image_classes == class_id
                count = int(selected.sum())
                prediction_chunks[int(class_id)]["image_ids"].append(
                    np.full(count, global_image_id, dtype=np.int64)
                )
                prediction_chunks[int(class_id)]["scores"].append(
                    image_scores[selected].astype(np.float64)
                )
                prediction_chunks[int(class_id)]["boxes"].append(
                    image_boxes[selected].astype(np.float64)
                )

        image_offset += images.shape[0]
        total_images += images.shape[0]
        if log_interval > 0 and (
            batch_index % log_interval == 0 or batch_index == len(loader)
        ):
            elapsed = time.perf_counter() - start_time
            print(
                f"  batch {batch_index:4d}/{len(loader)} "
                f"images={total_images:5d} elapsed={elapsed:.1f}s"
            )

    return (
        concatenate_prediction_chunks(prediction_chunks),
        ground_truth_by_class,
        image_sizes_by_id,
        total_images,
    )


# -----------------------------------------------------------------------------
# Metrics and result files
# -----------------------------------------------------------------------------


def evaluate_metrics(
    predictions_by_class: Sequence[Mapping[str, np.ndarray]],
    ground_truth_by_class: Sequence[Mapping[int, np.ndarray]],
    class_names: Sequence[str],
    fixed_confidence_threshold: float,
    fixed_iou_threshold: float,
    ap_confidence_threshold: float,
) -> Dict[str, Any]:
    per_class: List[Dict[str, Any]] = []
    all_fixed_matched_ious: List[np.ndarray] = []
    micro_tp = micro_fp = micro_fn = 0

    print("\nCalculating AP across IoU thresholds 0.50:0.95...")
    for class_id, class_name in enumerate(class_names):
        fixed = match_class_predictions(
            predictions_by_class[class_id],
            ground_truth_by_class[class_id],
            iou_threshold=fixed_iou_threshold,
            minimum_confidence=fixed_confidence_threshold,
        )
        ap_values = [
            float(
                match_class_predictions(
                    predictions_by_class[class_id],
                    ground_truth_by_class[class_id],
                    iou_threshold=float(iou_threshold),
                    minimum_confidence=ap_confidence_threshold,
                )["ap"]
            )
            for iou_threshold in IOU_THRESHOLDS
        ]
        ap50 = ap_values[0]
        ap50_95 = safe_mean(ap_values)
        micro_tp += int(fixed["tp"])
        micro_fp += int(fixed["fp"])
        micro_fn += int(fixed["fn"])
        if fixed["matched_ious"].size:
            all_fixed_matched_ious.append(fixed["matched_ious"])
        row = {
            "class_id": class_id,
            "class_name": class_name,
            "ground_truth": int(fixed["ground_truth"]),
            "predictions_at_conf": int(fixed["predictions"]),
            "tp": int(fixed["tp"]),
            "fp": int(fixed["fp"]),
            "fn": int(fixed["fn"]),
            "precision": float(fixed["precision"]),
            "recall": float(fixed["recall"]),
            "f1": float(fixed["f1"]),
            "mean_matched_iou": float(fixed["mean_matched_iou"]),
            "ap50": float(ap50),
            "ap50_95": float(ap50_95),
            "ap_by_iou": {
                f"{threshold:.2f}": float(value)
                for threshold, value in zip(IOU_THRESHOLDS, ap_values)
            },
        }
        per_class.append(row)
        print(
            f"  {class_id:2d} {class_name:<14} "
            f"GT={fixed['ground_truth']:5d} "
            f"P={format_metric(float(fixed['precision']))} "
            f"R={format_metric(float(fixed['recall']))} "
            f"AP50={format_metric(ap50)} "
            f"AP50-95={format_metric(ap50_95)}"
        )

    included = [row for row in per_class if row["ground_truth"] > 0]
    precision_micro = micro_tp / max(micro_tp + micro_fp, 1)
    recall_micro = micro_tp / max(micro_tp + micro_fn, 1)
    f1_micro = 2.0 * precision_micro * recall_micro / max(
        precision_micro + recall_micro, 1e-12
    )
    mean_iou = (
        float(np.concatenate(all_fixed_matched_ious).mean())
        if all_fixed_matched_ious
        else math.nan
    )
    return {
        "summary": {
            "fixed_threshold": {
                "confidence": fixed_confidence_threshold,
                "iou": fixed_iou_threshold,
                "tp": micro_tp,
                "fp": micro_fp,
                "fn": micro_fn,
                "precision_micro": float(precision_micro),
                "recall_micro": float(recall_micro),
                "f1_micro": float(f1_micro),
                "precision_macro": safe_mean([row["precision"] for row in included]),
                "recall_macro": safe_mean([row["recall"] for row in included]),
                "f1_macro": safe_mean([row["f1"] for row in included]),
                "mean_matched_iou": mean_iou,
            },
            "ap": {
                "minimum_confidence": ap_confidence_threshold,
                "iou_thresholds": [float(value) for value in IOU_THRESHOLDS],
                "map50": safe_mean([row["ap50"] for row in included]),
                "map50_95": safe_mean([row["ap50_95"] for row in included]),
                "classes_with_ground_truth": len(included),
            },
        },
        "per_class": per_class,
    }


def evaluate_size_metrics(
    predictions_by_class: Sequence[Mapping[str, np.ndarray]],
    ground_truth_by_class: Sequence[Mapping[int, np.ndarray]],
    image_sizes_by_id: Mapping[int, Tuple[int, int]],
    class_names: Sequence[str],
    fixed_confidence_threshold: float,
    fixed_iou_threshold: float,
    ap_confidence_threshold: float,
) -> Dict[str, Any]:
    summaries: Dict[str, Dict[str, Any]] = {}
    per_class_rows: List[Dict[str, Any]] = []

    print("\nCalculating size-stratified metrics...")

    for size_name, area_range in COCO_SIZE_RANGES.items():
        micro_tp = micro_fp = micro_fn = 0
        matched_iou_chunks: List[np.ndarray] = []
        bucket_rows: List[Dict[str, Any]] = []

        for class_id, class_name in enumerate(class_names):
            fixed = match_class_predictions_by_size(
                predictions_by_class[class_id],
                ground_truth_by_class[class_id],
                image_sizes_by_id,
                iou_threshold=fixed_iou_threshold,
                minimum_confidence=fixed_confidence_threshold,
                area_range=area_range,
            )

            ap_values = [
                float(
                    match_class_predictions_by_size(
                        predictions_by_class[class_id],
                        ground_truth_by_class[class_id],
                        image_sizes_by_id,
                        iou_threshold=float(iou_threshold),
                        minimum_confidence=ap_confidence_threshold,
                        area_range=area_range,
                    )["ap"]
                )
                for iou_threshold in IOU_THRESHOLDS
            ]

            ap50 = ap_values[0]
            ap50_95 = safe_mean(ap_values)
            micro_tp += int(fixed["tp"])
            micro_fp += int(fixed["fp"])
            micro_fn += int(fixed["fn"])

            if fixed["matched_ious"].size:
                matched_iou_chunks.append(fixed["matched_ious"])

            row = {
                "size": size_name,
                "class_id": class_id,
                "class_name": class_name,
                "ground_truth": int(fixed["ground_truth"]),
                "predictions_at_conf": int(fixed["predictions"]),
                "ignored_predictions_at_conf": int(
                    fixed["ignored_predictions"]
                ),
                "tp": int(fixed["tp"]),
                "fp": int(fixed["fp"]),
                "fn": int(fixed["fn"]),
                "precision": float(fixed["precision"]),
                "recall": float(fixed["recall"]),
                "f1": float(fixed["f1"]),
                "mean_matched_iou": float(fixed["mean_matched_iou"]),
                "ap50": float(ap50),
                "ap50_95": float(ap50_95),
                "ap_by_iou": {
                    f"{threshold:.2f}": float(value)
                    for threshold, value in zip(IOU_THRESHOLDS, ap_values)
                },
            }
            bucket_rows.append(row)
            per_class_rows.append(row)

        included = [row for row in bucket_rows if row["ground_truth"] > 0]
        precision_micro = micro_tp / max(micro_tp + micro_fp, 1)
        recall_micro = micro_tp / max(micro_tp + micro_fn, 1)
        f1_micro = 2.0 * precision_micro * recall_micro / max(
            precision_micro + recall_micro, 1e-12
        )
        mean_iou = (
            float(np.concatenate(matched_iou_chunks).mean())
            if matched_iou_chunks
            else math.nan
        )

        summaries[size_name] = {
            "area_min_px2": float(area_range[0]),
            "area_max_px2": (
                None if math.isinf(area_range[1]) else float(area_range[1])
            ),
            "ground_truth": int(sum(row["ground_truth"] for row in bucket_rows)),
            "fixed_threshold": {
                "confidence": fixed_confidence_threshold,
                "iou": fixed_iou_threshold,
                "tp": micro_tp,
                "fp": micro_fp,
                "fn": micro_fn,
                "precision_micro": float(precision_micro),
                "recall_micro": float(recall_micro),
                "f1_micro": float(f1_micro),
                "precision_macro": safe_mean(
                    [row["precision"] for row in included]
                ),
                "recall_macro": safe_mean(
                    [row["recall"] for row in included]
                ),
                "f1_macro": safe_mean([row["f1"] for row in included]),
                "mean_matched_iou": mean_iou,
            },
            "ap": {
                "minimum_confidence": ap_confidence_threshold,
                "map50": safe_mean([row["ap50"] for row in included]),
                "map50_95": safe_mean([row["ap50_95"] for row in included]),
                "classes_with_ground_truth": len(included),
            },
        }

    return {
        "summary": summaries,
        "per_class": per_class_rows,
    }


WITHIN_CLASS_METRIC_FIELDS = (
    ("precision", "P"),
    ("recall", "R"),
    ("f1", "F1"),
    ("mean_matched_iou", "IoU"),
    ("ap50", "AP50"),
    ("ap50_95", "AP50-95"),
)


def exact_two_sided_sign_test(positive_count: int, negative_count: int) -> float:
    """Exact two-sided sign-test p-value, excluding ties."""
    trials = int(positive_count + negative_count)
    if trials <= 0:
        return math.nan
    tail = min(int(positive_count), int(negative_count))
    probability = sum(math.comb(trials, index) for index in range(tail + 1))
    probability /= float(2 ** trials)
    return min(1.0, 2.0 * probability)


def evaluate_within_class_scale(
    size_results: Mapping[str, Any],
    minimum_ground_truth_per_bucket: int,
) -> Dict[str, Any]:
    """Compare the same semantic class across small, medium and large buckets.

    A class is included only when every size bucket contains at least
    ``minimum_ground_truth_per_bucket`` ground-truth instances. This avoids
    interpreting very sparse class/size combinations as stable scale effects.
    """
    rows_by_class: Dict[int, Dict[str, Mapping[str, Any]]] = {}
    for row in size_results["per_class"]:
        class_id = int(row["class_id"])
        rows_by_class.setdefault(class_id, {})[str(row["size"])] = row

    per_class: List[Dict[str, Any]] = []
    for class_id in sorted(rows_by_class):
        buckets = rows_by_class[class_id]
        if not all(size_name in buckets for size_name in ("small", "medium", "large")):
            continue

        small = buckets["small"]
        medium = buckets["medium"]
        large = buckets["large"]
        ground_truth_counts = {
            "small": int(small["ground_truth"]),
            "medium": int(medium["ground_truth"]),
            "large": int(large["ground_truth"]),
        }
        if min(ground_truth_counts.values()) < minimum_ground_truth_per_bucket:
            continue

        class_row: Dict[str, Any] = {
            "class_id": class_id,
            "class_name": str(small["class_name"]),
            "gt_small": ground_truth_counts["small"],
            "gt_medium": ground_truth_counts["medium"],
            "gt_large": ground_truth_counts["large"],
        }

        for metric_name, _ in WITHIN_CLASS_METRIC_FIELDS:
            small_value = float(small[metric_name])
            medium_value = float(medium[metric_name])
            large_value = float(large[metric_name])
            class_row[f"{metric_name}_small"] = small_value
            class_row[f"{metric_name}_medium"] = medium_value
            class_row[f"{metric_name}_large"] = large_value

            finite = all(
                math.isfinite(value)
                for value in (small_value, medium_value, large_value)
            )
            if finite:
                class_row[f"{metric_name}_delta_medium_small"] = (
                    medium_value - small_value
                )
                class_row[f"{metric_name}_delta_large_medium"] = (
                    large_value - medium_value
                )
                class_row[f"{metric_name}_delta_large_small"] = (
                    large_value - small_value
                )
                class_row[f"{metric_name}_monotonic"] = bool(
                    small_value <= medium_value <= large_value
                )
            else:
                class_row[f"{metric_name}_delta_medium_small"] = math.nan
                class_row[f"{metric_name}_delta_large_medium"] = math.nan
                class_row[f"{metric_name}_delta_large_small"] = math.nan
                class_row[f"{metric_name}_monotonic"] = False

        per_class.append(class_row)

    summary_rows: List[Dict[str, Any]] = []
    for metric_name, metric_label in WITHIN_CLASS_METRIC_FIELDS:
        finite_rows = [
            row
            for row in per_class
            if all(
                math.isfinite(float(row[f"{metric_name}_{size_name}"]))
                for size_name in ("small", "medium", "large")
            )
        ]
        small_values = np.asarray(
            [row[f"{metric_name}_small"] for row in finite_rows], dtype=np.float64
        )
        medium_values = np.asarray(
            [row[f"{metric_name}_medium"] for row in finite_rows], dtype=np.float64
        )
        large_values = np.asarray(
            [row[f"{metric_name}_large"] for row in finite_rows], dtype=np.float64
        )

        delta_medium_small = medium_values - small_values
        delta_large_medium = large_values - medium_values
        delta_large_small = large_values - small_values
        positive_count = int((delta_large_small > 0.0).sum())
        negative_count = int((delta_large_small < 0.0).sum())
        tie_count = int((delta_large_small == 0.0).sum())
        monotonic_count = int(
            ((small_values <= medium_values) & (medium_values <= large_values)).sum()
        )
        n = int(len(finite_rows))

        summary_rows.append(
            {
                "metric": metric_name,
                "metric_label": metric_label,
                "classes": n,
                "small_mean": float(small_values.mean()) if n else math.nan,
                "medium_mean": float(medium_values.mean()) if n else math.nan,
                "large_mean": float(large_values.mean()) if n else math.nan,
                "small_median": float(np.median(small_values)) if n else math.nan,
                "medium_median": float(np.median(medium_values)) if n else math.nan,
                "large_median": float(np.median(large_values)) if n else math.nan,
                "mean_delta_medium_small": (
                    float(delta_medium_small.mean()) if n else math.nan
                ),
                "mean_delta_large_medium": (
                    float(delta_large_medium.mean()) if n else math.nan
                ),
                "mean_delta_large_small": (
                    float(delta_large_small.mean()) if n else math.nan
                ),
                "median_delta_large_small": (
                    float(np.median(delta_large_small)) if n else math.nan
                ),
                "large_better_count": positive_count,
                "large_worse_count": negative_count,
                "large_equal_count": tie_count,
                "large_better_fraction": (
                    positive_count / max(positive_count + negative_count, 1)
                ),
                "monotonic_count": monotonic_count,
                "monotonic_fraction": monotonic_count / max(n, 1),
                "sign_test_p_two_sided": exact_two_sided_sign_test(
                    positive_count, negative_count
                ),
            }
        )

    return {
        "minimum_ground_truth_per_bucket": int(minimum_ground_truth_per_bucket),
        "eligible_classes": len(per_class),
        "summary": summary_rows,
        "per_class": per_class,
    }


def print_within_class_scale_summary(results: Mapping[str, Any]) -> None:
    print("\n" + "=" * 118)
    print(
        "WITHIN-CLASS SCALE ANALYSIS "
        f"(minimum GT per size bucket = {results['minimum_ground_truth_per_bucket']})"
    )
    print(
        "Same semantic classes are paired across small, medium and large object sizes."
    )
    print("=" * 118)
    print(
        f"{'Metric':<10} {'N':>4} {'Small':>9} {'Medium':>9} {'Large':>9} "
        f"{'Delta L-S':>11} {'L>S':>9} {'Monotonic':>11} {'Sign p':>10}"
    )
    print("-" * 118)
    for row in results["summary"]:
        print(
            f"{row['metric_label']:<10} {row['classes']:4d} "
            f"{format_metric(row['small_mean']):>9} "
            f"{format_metric(row['medium_mean']):>9} "
            f"{format_metric(row['large_mean']):>9} "
            f"{format_metric(row['mean_delta_large_small']):>11} "
            f"{row['large_better_count']:3d}/{row['classes']:<4d} "
            f"{row['monotonic_count']:3d}/{row['classes']:<6d} "
            f"{format_metric(row['sign_test_p_two_sided']):>10}"
        )
    print("=" * 118)
    print(f"Eligible classes: {results['eligible_classes']}")


def print_size_summary(size_results: Mapping[str, Any]) -> None:
    print("\n" + "=" * 104)
    print("OBJECT SIZE SUMMARY (COCO bbox-area thresholds in original image pixels)")
    print("=" * 104)
    print(
        f"{'Size':<8} {'GT':>7} {'TP':>7} {'FP':>7} {'FN':>7} "
        f"{'P':>9} {'R':>9} {'F1':>9} {'IoU':>9} {'mAP50':>9} {'mAP50-95':>11}"
    )
    print("-" * 104)

    for size_name in ("small", "medium", "large"):
        row = size_results["summary"][size_name]
        fixed = row["fixed_threshold"]
        ap = row["ap"]
        print(
            f"{size_name:<8} {row['ground_truth']:7d} "
            f"{fixed['tp']:7d} {fixed['fp']:7d} {fixed['fn']:7d} "
            f"{format_metric(fixed['precision_micro']):>9} "
            f"{format_metric(fixed['recall_micro']):>9} "
            f"{format_metric(fixed['f1_micro']):>9} "
            f"{format_metric(fixed['mean_matched_iou']):>9} "
            f"{format_metric(ap['map50']):>9} "
            f"{format_metric(ap['map50_95']):>11}"
        )
    print("=" * 104)


def print_summary(results: Mapping[str, Any]) -> None:
    fixed = results["summary"]["fixed_threshold"]
    ap = results["summary"]["ap"]
    print("\n" + "=" * 72)
    print("VALIDATION SUMMARY")
    print("=" * 72)
    print(
        f"Fixed threshold: confidence >= {fixed['confidence']:.3f}, "
        f"IoU >= {fixed['iou']:.2f}"
    )
    print(f"TP / FP / FN        : {fixed['tp']} / {fixed['fp']} / {fixed['fn']}")
    print(f"Precision (micro)   : {format_metric(fixed['precision_micro'])}")
    print(f"Recall (micro)      : {format_metric(fixed['recall_micro'])}")
    print(f"F1 (micro)          : {format_metric(fixed['f1_micro'])}")
    print(f"Precision (macro)   : {format_metric(fixed['precision_macro'])}")
    print(f"Recall (macro)      : {format_metric(fixed['recall_macro'])}")
    print(f"F1 (macro)          : {format_metric(fixed['f1_macro'])}")
    print(f"Mean matched IoU    : {format_metric(fixed['mean_matched_iou'])}")
    print(f"mAP@0.50            : {format_metric(ap['map50'])}")
    print(f"mAP@0.50:0.95       : {format_metric(ap['map50_95'])}")
    print(
        f"Classes included    : {ap['classes_with_ground_truth']} "
        "(classes with ground truth)"
    )
    print("=" * 72)


def make_json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): make_json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return make_json_safe(value.tolist())
    if isinstance(value, np.generic):
        return make_json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def save_results(
    results: Mapping[str, Any],
    output_dir: Path,
    run_metadata: Mapping[str, Any],
    size_results: Mapping[str, Any] | None = None,
) -> Tuple[Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "metrics.json"
    csv_path = output_dir / "per_class_metrics.csv"
    json_payload: Dict[str, Any] = {
        "run": dict(run_metadata),
        "summary": results["summary"],
        "per_class": results["per_class"],
    }
    if size_results is not None:
        json_payload["size_metrics"] = size_results

    json_path.write_text(
        json.dumps(
            make_json_safe(json_payload),
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    fieldnames = [
        "class_id",
        "class_name",
        "ground_truth",
        "predictions_at_conf",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
        "mean_matched_iou",
        "ap50",
        "ap50_95",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in results["per_class"]:
            writer.writerow({field: row[field] for field in fieldnames})
    print(f"Saved JSON metrics : {json_path}")
    print(f"Saved class table  : {csv_path}")

    if size_results is not None:
        size_csv_path = output_dir / "size_metrics.csv"
        size_fieldnames = [
            "size",
            "area_min_px2",
            "area_max_px2",
            "ground_truth",
            "tp",
            "fp",
            "fn",
            "precision_micro",
            "recall_micro",
            "f1_micro",
            "precision_macro",
            "recall_macro",
            "f1_macro",
            "mean_matched_iou",
            "map50",
            "map50_95",
            "classes_with_ground_truth",
        ]
        with size_csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=size_fieldnames)
            writer.writeheader()
            for size_name in ("small", "medium", "large"):
                summary = size_results["summary"][size_name]
                fixed = summary["fixed_threshold"]
                ap = summary["ap"]
                writer.writerow(
                    {
                        "size": size_name,
                        "area_min_px2": summary["area_min_px2"],
                        "area_max_px2": summary["area_max_px2"],
                        "ground_truth": summary["ground_truth"],
                        "tp": fixed["tp"],
                        "fp": fixed["fp"],
                        "fn": fixed["fn"],
                        "precision_micro": fixed["precision_micro"],
                        "recall_micro": fixed["recall_micro"],
                        "f1_micro": fixed["f1_micro"],
                        "precision_macro": fixed["precision_macro"],
                        "recall_macro": fixed["recall_macro"],
                        "f1_macro": fixed["f1_macro"],
                        "mean_matched_iou": fixed["mean_matched_iou"],
                        "map50": ap["map50"],
                        "map50_95": ap["map50_95"],
                        "classes_with_ground_truth": ap[
                            "classes_with_ground_truth"
                        ],
                    }
                )

        per_class_size_csv = output_dir / "per_class_size_metrics.csv"
        per_class_size_fields = [
            "size",
            "class_id",
            "class_name",
            "ground_truth",
            "predictions_at_conf",
            "ignored_predictions_at_conf",
            "tp",
            "fp",
            "fn",
            "precision",
            "recall",
            "f1",
            "mean_matched_iou",
            "ap50",
            "ap50_95",
        ]
        with per_class_size_csv.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=per_class_size_fields)
            writer.writeheader()
            for row in size_results["per_class"]:
                writer.writerow(
                    {field: row[field] for field in per_class_size_fields}
                )

        print(f"Saved size table   : {size_csv_path}")
        print(f"Saved class/size   : {per_class_size_csv}")

        within_class = size_results.get("within_class_analysis")
        if within_class is not None:
            within_summary_csv = output_dir / "within_class_scale_summary.csv"
            within_summary_fields = [
                "metric",
                "metric_label",
                "classes",
                "small_mean",
                "medium_mean",
                "large_mean",
                "small_median",
                "medium_median",
                "large_median",
                "mean_delta_medium_small",
                "mean_delta_large_medium",
                "mean_delta_large_small",
                "median_delta_large_small",
                "large_better_count",
                "large_worse_count",
                "large_equal_count",
                "large_better_fraction",
                "monotonic_count",
                "monotonic_fraction",
                "sign_test_p_two_sided",
            ]
            with within_summary_csv.open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(file, fieldnames=within_summary_fields)
                writer.writeheader()
                for row in within_class["summary"]:
                    writer.writerow(
                        {field: row[field] for field in within_summary_fields}
                    )

            within_class_csv = output_dir / "within_class_scale_metrics.csv"
            base_fields = [
                "class_id",
                "class_name",
                "gt_small",
                "gt_medium",
                "gt_large",
            ]
            metric_fields: List[str] = []
            for metric_name, _ in WITHIN_CLASS_METRIC_FIELDS:
                metric_fields.extend(
                    [
                        f"{metric_name}_small",
                        f"{metric_name}_medium",
                        f"{metric_name}_large",
                        f"{metric_name}_delta_medium_small",
                        f"{metric_name}_delta_large_medium",
                        f"{metric_name}_delta_large_small",
                        f"{metric_name}_monotonic",
                    ]
                )
            within_class_fields = base_fields + metric_fields
            with within_class_csv.open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(file, fieldnames=within_class_fields)
                writer.writeheader()
                for row in within_class["per_class"]:
                    writer.writerow(
                        {field: row[field] for field in within_class_fields}
                    )

            print(f"Saved within summary: {within_summary_csv}")
            print(f"Saved within classes: {within_class_csv}")

    return json_path, csv_path


# -----------------------------------------------------------------------------
# Single-image inference and annotation
# -----------------------------------------------------------------------------


def class_colour(class_id: int) -> Tuple[int, int, int]:
    palette = (
        (230, 57, 70),
        (29, 53, 87),
        (69, 123, 157),
        (42, 157, 143),
        (233, 196, 106),
        (244, 162, 97),
        (231, 111, 81),
        (131, 56, 236),
        (0, 150, 136),
        (255, 111, 0),
    )
    return palette[class_id % len(palette)]


def draw_detections(
    image: Image.Image,
    detections: np.ndarray,
    class_names: Sequence[str],
) -> Image.Image:
    annotated = image.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated)
    font = ImageFont.load_default()
    line_width = max(2, round(min(annotated.size) / 250))
    for x1, y1, x2, y2, confidence, class_value in detections:
        class_id = int(class_value)
        colour = class_colour(class_id)
        class_name = (
            class_names[class_id]
            if 0 <= class_id < len(class_names)
            else f"class_{class_id}"
        )
        label = f"{class_name} {confidence:.3f}"
        box = tuple(int(round(value)) for value in (x1, y1, x2, y2))
        draw.rectangle(box, outline=colour, width=line_width)
        text_box = draw.textbbox((0, 0), label, font=font)
        text_width = text_box[2] - text_box[0]
        text_height = text_box[3] - text_box[1]
        label_x = max(0, box[0])
        label_y = max(0, box[1] - text_height - 6)
        draw.rectangle(
            (label_x, label_y, label_x + text_width + 6, label_y + text_height + 6),
            fill=colour,
        )
        draw.text((label_x + 3, label_y + 3), label, fill="white", font=font)
    return annotated


@torch.inference_mode()
def infer_one(
    image_path: str | Path,
    model: RTDETR,
    class_names: Sequence[str],
    device: torch.device,
    *,
    img_size: int = 416,
    conf_thres: float = 0.05,
    max_detections: int = 300,
    out_path: str | Path = "pred_vis_rtdetr.jpg",
    amp_enabled: bool = True,
) -> np.ndarray:
    image_path = Path(image_path)
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")
    original = Image.open(image_path).convert("RGB")
    original_width, original_height = original.size
    resized = original.resize((img_size, img_size), Image.Resampling.BILINEAR)
    array = np.asarray(resized, dtype=np.float32).copy()
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0) / 255.0
    tensor = tensor.to(device)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.amp.autocast(
        device_type=device.type,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        enabled=amp_enabled,
    ):
        output = model(tensor)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    logits = output["pred_logits"]
    boxes = output["pred_boxes"]
    if not isinstance(logits, Tensor) or not isinstance(boxes, Tensor):
        raise TypeError("RT-DETR output tensors are missing")
    scores, classes = logits.float().sigmoid().max(dim=-1)
    boxes_xyxy = box_cxcywh_to_xyxy(boxes.float()).clamp(0.0, 1.0)
    order = scores[0].argsort(descending=True)[:max_detections]
    scores = scores[0, order]
    classes = classes[0, order]
    boxes_xyxy = boxes_xyxy[0, order]
    keep = scores >= conf_thres
    scores = scores[keep].cpu().numpy()
    classes = classes[keep].cpu().numpy().astype(np.float64)
    boxes_xyxy = boxes_xyxy[keep].cpu().numpy()
    boxes_xyxy[:, [0, 2]] *= original_width
    boxes_xyxy[:, [1, 3]] *= original_height
    detections = (
        np.concatenate(
            (boxes_xyxy, scores[:, None], classes[:, None]),
            axis=1,
        )
        if scores.size
        else np.zeros((0, 6), dtype=np.float64)
    )
    annotated = draw_detections(original, detections, class_names)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(out_path)

    print(f"Image              : {image_path}")
    print(f"Original size      : {original_width} x {original_height}")
    print(f"Model input        : {img_size} x {img_size} (stretched)")
    print(f"Confidence threshold: {conf_thres:.3f}")
    print("Inference path     : RT-DETR NMS-free")
    print(f"Detections retained: {detections.shape[0]}")
    print(f"Inference time     : {elapsed_ms:.2f} ms")
    print(f"Annotated image    : {out_path}")
    return detections


# -----------------------------------------------------------------------------
# Entry points
# -----------------------------------------------------------------------------


def validate_dataset(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint_file(checkpoint_path, device)
    model, config = build_model_from_checkpoint(
        checkpoint,
        device,
        num_classes_override=args.num_classes,
        num_queries_override=args.num_queries,
        hidden_dim_override=args.hidden_dim,
        decoder_layers_override=args.decoder_layers,
    )
    root = args.root or str(config.get("root", ROOT))
    img_size = args.img_size or int(config.get("img_size", 416))
    num_classes = int(config["num_classes"])
    class_names = resolve_class_names(num_classes, args.class_names)
    dataset: Any = YoloTxtDataset(
        root=root,
        split=args.split,
        img_size=img_size,
        num_classes=num_classes,
        augment=False,
    )
    if args.limit is not None:
        dataset = Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_fn,
        worker_init_fn=worker_init_fn,
        persistent_workers=args.workers > 0,
    )

    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {int(checkpoint['epoch']) + 1}")
    else:
        print("Checkpoint epoch: not stored")
    print(f"Dataset root: {root}")
    print(f"Split: {args.split}")
    print(f"Images: {len(dataset)}")
    print(f"Image size: {img_size}")
    print(f"Classes: {num_classes}")
    print("Model: RT-DETR-R18")
    print(f"Object queries: {config['num_queries']}")
    print(f"Decoder layers: {config['decoder_layers']}")
    print(f"Maximum detections per image: {args.max_detections}")
    print(f"AMP inference: {amp_enabled}")

    start = time.perf_counter()
    predictions, ground_truth, image_sizes, image_count = collect_predictions(
        model,
        loader,
        device,
        num_classes,
        args.ap_conf_thres,
        args.max_detections,
        amp_enabled,
        args.log_interval,
    )
    inference_seconds = time.perf_counter() - start
    results = evaluate_metrics(
        predictions,
        ground_truth,
        class_names,
        args.conf_thres,
        args.match_iou,
        args.ap_conf_thres,
    )
    print_summary(results)

    size_results = None
    if args.size_metrics:
        size_results = evaluate_size_metrics(
            predictions,
            ground_truth,
            image_sizes,
            class_names,
            args.conf_thres,
            args.match_iou,
            args.ap_conf_thres,
        )
        print_size_summary(size_results)

        if args.within_class_analysis:
            within_class_results = evaluate_within_class_scale(
                size_results,
                args.within_class_min_gt,
            )
            size_results["within_class_analysis"] = within_class_results
            print_within_class_scale_summary(within_class_results)

    milliseconds_per_image = 1000.0 * inference_seconds / max(image_count, 1)
    save_results(
        results,
        Path(args.output_dir),
        {
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": (
                int(checkpoint["epoch"]) + 1 if "epoch" in checkpoint else None
            ),
            "dataset_root": root,
            "split": args.split,
            "images": image_count,
            "img_size": img_size,
            "num_classes": num_classes,
            "model": "RT-DETR-R18",
            "num_queries": config["num_queries"],
            "decoder_layers": config["decoder_layers"],
            "max_detections": args.max_detections,
            "inference_seconds": inference_seconds,
            "milliseconds_per_image": milliseconds_per_image,
            "device": str(device),
            "amp": amp_enabled,
            "size_metrics": bool(args.size_metrics),
            "within_class_analysis": bool(args.within_class_analysis),
            "within_class_min_gt": int(args.within_class_min_gt),
        },
        size_results=size_results,
    )
    print(
        f"Inference time      : {inference_seconds:.2f}s "
        f"({milliseconds_per_image:.2f} ms/image including loading)"
    )


def annotate_image(args: argparse.Namespace) -> None:
    device = choose_device(args.device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    checkpoint_path = Path(args.checkpoint)
    checkpoint = load_checkpoint_file(checkpoint_path, device)
    model, config = build_model_from_checkpoint(
        checkpoint,
        device,
        num_classes_override=args.num_classes,
        num_queries_override=args.num_queries,
        hidden_dim_override=args.hidden_dim,
        decoder_layers_override=args.decoder_layers,
    )
    img_size = args.img_size or int(config.get("img_size", 416))
    class_names = resolve_class_names(int(config["num_classes"]), args.class_names)
    print(f"PyTorch: {torch.__version__}")
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    if "epoch" in checkpoint:
        print(f"Checkpoint epoch: {int(checkpoint['epoch']) + 1}")
    else:
        print("Checkpoint epoch: not stored")
    infer_one(
        args.image,
        model,
        class_names,
        device,
        img_size=img_size,
        conf_thres=args.conf_thres,
        max_detections=args.max_detections,
        out_path=args.out_path,
        amp_enabled=amp_enabled,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate RT-DETR-R18 or annotate one image"
    )
    parser.add_argument("--checkpoint", default="runs/rtdetr_voc/best.pt")
    parser.add_argument("--root", default=None)
    parser.add_argument("--split", default="val")
    parser.add_argument("--image", default=None)
    parser.add_argument("--out-path", default="pred_vis_rtdetr.jpg")
    parser.add_argument("--img-size", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--num-queries", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--decoder-layers", type=int, default=None)
    parser.add_argument("--class-names", default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--conf-thres", type=float, default=0.25)
    parser.add_argument("--match-iou", type=float, default=0.50)
    parser.add_argument("--ap-conf-thres", type=float, default=0.001)
    parser.add_argument("--max-detections", type=int, default=300)
    parser.add_argument(
        "--size-metrics",
        action="store_true",
        help=(
            "Report COCO-style small/medium/large bbox metrics using "
            "original-image pixel areas"
        ),
    )
    parser.add_argument(
        "--within-class-analysis",
        action="store_true",
        help=(
            "Compare the same classes across small/medium/large buckets and "
            "save paired scale-effect metrics; requires --size-metrics"
        ),
    )
    parser.add_argument(
        "--within-class-min-gt",
        type=int,
        default=20,
        help=(
            "Minimum ground-truth instances required in EACH size bucket for "
            "a class to enter within-class analysis (default: 20)"
        ),
    )
    parser.add_argument("--output-dir", default="runs/rtdetr_voc/validation")
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    for name in ("conf_thres", "match_iou", "ap_conf_thres"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if args.img_size is not None and (args.img_size <= 0 or args.img_size % 32):
        parser.error("--img-size must be positive and divisible by 32")
    if args.batch_size <= 0 or args.workers < 0:
        parser.error("--batch-size must be positive and --workers cannot be negative")
    if args.max_detections <= 0:
        parser.error("--max-detections must be positive")
    if args.within_class_min_gt <= 0:
        parser.error("--within-class-min-gt must be positive")
    if args.within_class_analysis and not args.size_metrics:
        parser.error("--within-class-analysis requires --size-metrics")
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be positive")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.image:
        annotate_image(arguments)
    else:
        validate_dataset(arguments)
