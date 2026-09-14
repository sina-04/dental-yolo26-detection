from __future__ import annotations

import math
from typing import Any


def tile_windows(width: int, height: int, count: int = 3, overlap: float = 0.25) -> list[tuple[int, int, int, int]]:
    if width <= 0 or height <= 0 or count < 1:
        raise ValueError("Image dimensions and tile count must be positive")
    if not 0.0 <= overlap < 1.0:
        raise ValueError("Tile overlap must be in [0, 1)")
    if count == 1:
        return [(0, 0, width, height)]
    tile_width = min(width, math.ceil(width / (count - (count - 1) * overlap)))
    maximum_start = max(0, width - tile_width)
    starts = [round(index * maximum_start / (count - 1)) for index in range(count)]
    return [(start, 0, min(width, start + tile_width), height) for start in starts]


def box_iou(left: list[float], right: list[float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-12)


def merge_detections(detections: list[dict[str, Any]], iou_threshold: float = 0.55) -> list[dict[str, Any]]:
    """Class-aware weighted box fusion for full-frame and tile predictions."""
    remaining = sorted(detections, key=lambda item: float(item["confidence"]), reverse=True)
    merged: list[dict[str, Any]] = []
    while remaining:
        anchor = remaining.pop(0)
        cluster = [anchor]
        keep: list[dict[str, Any]] = []
        for candidate in remaining:
            if (
                int(candidate["class_id"]) == int(anchor["class_id"])
                and box_iou(list(anchor["xyxy"]), list(candidate["xyxy"])) >= iou_threshold
            ):
                cluster.append(candidate)
            else:
                keep.append(candidate)
        remaining = keep
        weights = [max(float(item["confidence"]), 1e-6) for item in cluster]
        total = sum(weights)
        coordinates = [
            sum(float(item["xyxy"][index]) * weight for item, weight in zip(cluster, weights, strict=True)) / total
            for index in range(4)
        ]
        fused = dict(anchor)
        fused["xyxy"] = coordinates
        fused["confidence"] = max(weights)
        fused["sources"] = sorted({str(item.get("source_view", "full")) for item in cluster})
        merged.append(fused)
    return merged


def apply_class_thresholds(
    detections: list[dict[str, Any]], thresholds: dict[int, float] | float
) -> list[dict[str, Any]]:
    return [
        detection for detection in detections
        if float(detection["confidence"])
        >= (float(thresholds.get(int(detection["class_id"]), 1.0)) if isinstance(thresholds, dict) else thresholds)
    ]
