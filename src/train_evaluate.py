from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import random
import shutil
import subprocess
import sys
import time
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO, __version__ as ultralytics_version
import yaml

from src.common import IMAGE_EXTENSIONS, load_names, sha256_file, stable_fingerprint, write_json
from src.inference_utils import merge_detections, tile_windows
from src.pathology import PRIMARY_SOURCE_CLASSES, require_clinician_review, require_support_gate


ROOT = Path(__file__).resolve().parents[1]
DATA_YAML = ROOT / "dataset" / "data.yaml"
RUNS = ROOT / "runs"
REPORTS = ROOT / "reports"
ARTIFACTS = ROOT / "artifacts"


@dataclass
class GroundTruth:
    class_id: int
    xyxy: tuple[float, float, float, float]


@dataclass
class Prediction:
    class_id: int
    confidence: float
    xyxy: tuple[float, float, float, float]


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    return "0" if torch.cuda.is_available() else "cpu"


def training_images(data_yaml: Path) -> list[Path]:
    configuration = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(configuration.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    train = Path(configuration["train"])
    if not train.is_absolute():
        train = root / train
    if train.suffix.lower() == ".txt":
        return [Path(line.strip()) for line in train.read_text(encoding="utf-8").splitlines() if line.strip()]
    return sorted(path for path in train.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)


def create_runtime_data_yaml(data_yaml: Path, max_train_images: int | None, seed: int, tag: str) -> tuple[Path, int]:
    images = training_images(data_yaml)
    if not max_train_images or max_train_images >= len(images):
        return data_yaml, len(images)
    configuration = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(configuration.get("path", data_yaml.parent))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    labels_by_image: dict[Path, set[int]] = {}
    class_candidates: dict[int, list[Path]] = defaultdict(list)
    for image in images:
        label = root / "labels" / "train" / f"{image.stem}.txt"
        classes = {int(line.split()[0]) for line in label.read_text(encoding="utf-8").splitlines() if line.strip()}
        labels_by_image[image] = classes
        for class_id in classes:
            class_candidates[class_id].append(image)
    rng = random.Random(seed)
    selected: set[Path] = set()
    for class_id in sorted(class_candidates, key=lambda item: len(class_candidates[item])):
        slots = max_train_images - len(selected)
        if slots <= 0:
            break
        candidates = class_candidates[class_id].copy()
        rng.shuffle(candidates)
        selected.update(candidates[: min(5, len(candidates), slots)])
    remaining = [image for image in images if image not in selected]
    rng.shuffle(remaining)
    selected.update(remaining[: max(0, max_train_images - len(selected))])
    chosen = sorted(selected)
    runtime_dir = DATA_YAML.parent / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    train_list = runtime_dir / f"{tag}_subset_train.txt"
    train_list.write_text("\n".join(path.resolve().as_posix() for path in chosen) + "\n", encoding="utf-8")
    base = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    base["path"] = root.resolve().as_posix()
    base["train"] = train_list.resolve().as_posix()
    runtime_yaml = runtime_dir / f"{tag}_subset.yaml"
    runtime_yaml.write_text(yaml.safe_dump(base, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return runtime_yaml, len(chosen)


def train_experiment(
    name: str,
    model_checkpoint: str,
    epochs: int,
    imgsz: int,
    batch: int,
    device: str,
    seed: int,
    augmented: bool,
    data_yaml: Path,
    train_images: int,
    workers: int,
    cache: bool | str,
    amp: bool,
    patience: int,
    save_period: int,
    resume: bool,
    data_variant: str,
    dataset_fingerprint: str,
    code_commit: str,
    training_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    run_dir = RUNS / name
    existing_last = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"
    completed_before = sum(1 for _ in results_csv.open(encoding="utf-8")) - 1 if results_csv.exists() else 0
    completion_path = run_dir / "run_complete.json"
    already_complete = completion_path.exists() and (run_dir / "weights" / "best.pt").exists()
    parameters: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": workers,
        "patience": patience,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "cos_lr": True,
        "weight_decay": 0.0005,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.0,
        "degrees": 0.0,
        "translate": 0.0,
        "scale": 0.0,
        "multi_scale": 0.0,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.0,
        "seed": seed,
        "deterministic": True,
        "plots": True,
        "cache": cache,
        "amp": amp,
        "save_period": save_period,
        "project": str(RUNS),
        "name": name,
        "exist_ok": True,
        "verbose": True,
    }
    protected = {"data", "project", "name", "exist_ok", "device"}
    forbidden = protected.intersection(training_overrides or {})
    if forbidden:
        raise ValueError(f"Training overrides cannot replace protected keys: {sorted(forbidden)}")
    parameters.update(training_overrides or {})
    manifest = {
        "schema_version": 1,
        "experiment": name,
        "data_variant": data_variant,
        "dataset_fingerprint": dataset_fingerprint,
        "code_commit": code_commit,
        "model_checkpoint": model_checkpoint,
        "training_images": train_images,
        "configuration": {
            key: value
            for key, value in parameters.items()
            if key not in {"data", "project", "name", "exist_ok", "verbose"}
        },
    }
    manifest["fingerprint"] = stable_fingerprint(manifest)
    manifest_path = run_dir / "run_manifest.json"
    if manifest_path.exists():
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            existing_inputs = {key: value for key, value in existing_manifest.items() if key not in {"code_commit", "fingerprint"}}
            current_inputs = {key: value for key, value in manifest.items() if key not in {"code_commit", "fingerprint"}}
            same_inputs = existing_inputs == current_inputs
            has_training_state = existing_last.exists() or results_csv.exists() or (run_dir / "weights" / "best.pt").exists()
            if same_inputs and already_complete:
                # Preserve the commit that actually trained this completed run.
                manifest = existing_manifest
            elif same_inputs and not has_training_state:
                # A setup-time failure produced no checkpoint or metrics. It is
                # safe to retry that run with the repaired implementation.
                write_json(manifest_path, manifest)
            else:
                raise RuntimeError(f"Refusing incompatible resume for {name}; use a new run prefix or results root")
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError(f"Existing run {name} has no compatibility manifest")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        write_json(manifest_path, manifest)

    checkpoint = str(existing_last) if resume and existing_last.exists() and not already_complete else model_checkpoint
    model = YOLO(checkpoint)
    trainer_class = None
    if data_variant == "balanced":
        from src.balanced_trainer import PatientBalancedDetectionTrainer, configure_patient_groups

        configure_patient_groups(REPORTS / "dataset_manifest.csv", seed)
        trainer_class = PatientBalancedDetectionTrainer
    started = time.monotonic()
    if already_complete:
        print(f"Skipping completed experiment {name}: {completed_before}/{epochs} epochs already present.", flush=True)
        model = YOLO(str(run_dir / "weights" / "best.pt"))
    elif resume and existing_last.exists():
        print(f"Resuming {name} from {existing_last}", flush=True)
        model.train(resume=True, epochs=epochs, **({"trainer": trainer_class} if trainer_class else {}))
    else:
        model.train(**parameters, **({"trainer": trainer_class} if trainer_class else {}))
    duration = time.monotonic() - started
    save_dir = run_dir if already_complete else Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    last = save_dir / "weights" / "last.pt"
    validation_model = YOLO(str(best))
    metrics = validation_model.val(
        data=str(data_yaml),
        split="val",
        imgsz=imgsz,
        batch=batch,
        device=device,
        workers=workers,
        plots=True,
        project=str(RUNS),
        name=f"{name}_val",
        exist_ok=True,
    )
    values = {key: float(value) for key, value in metrics.results_dict.items()}
    results_csv = save_dir / "results.csv"
    completed = sum(1 for _ in results_csv.open(encoding="utf-8")) - 1 if results_csv.exists() else epochs
    write_json(
        completion_path,
        {"status": "complete", "epochs_completed": completed, "stopped_early": completed < epochs},
    )
    return {
        "name": name,
        "model": model_checkpoint,
        "epochs_requested": epochs,
        "epochs_completed": completed,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "augmentation": "offline class-aware medical augmentation" if augmented else "none",
        "data_variant": data_variant,
        "dataset_fingerprint": dataset_fingerprint,
        "run_manifest": manifest,
        "duration_seconds": duration,
        "train_images": train_images,
        "best": str(best),
        "last": str(last),
        "validation": values,
        "parameters": parameters,
    }


def load_ground_truth(image_path: Path) -> list[GroundTruth]:
    split = image_path.parent.name
    label = DATA_YAML.parent / "labels" / split / f"{image_path.stem}.txt"
    truths: list[GroundTruth] = []
    if not label.exists():
        return truths
    for line in label.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        class_id = int(parts[0])
        x, y, width, height = map(float, parts[1:])
        truths.append(GroundTruth(class_id, (x - width / 2, y - height / 2, x + width / 2, y + height / 2)))
    return truths


def iou(left: tuple[float, float, float, float], right: tuple[float, float, float, float]) -> float:
    x1, y1 = max(left[0], right[0]), max(left[1], right[1])
    x2, y2 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return intersection / max(left_area + right_area - intersection, 1e-12)


def cache_predictions(
    model: YOLO,
    split: str,
    imgsz: int,
    device: str,
    tile_config: dict[str, Any] | None = None,
) -> dict[str, list[Prediction]]:
    image_dir = DATA_YAML.parent / "images" / split
    predictions: dict[str, list[Prediction]] = {}
    # The operating-point search starts at 0.05, so retaining lower-confidence
    # detections only increases memory and matching cost without affecting it.
    if tile_config and tile_config.get("enabled"):
        for image_path in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                continue
            height, width = image.shape[:2]
            raw: list[dict[str, Any]] = []

            def add_result(result: Any, x_offset: int, source_view: str) -> None:
                if result.boxes is None:
                    return
                for box in result.boxes:
                    coordinates = [float(value) for value in box.xyxy[0].tolist()]
                    coordinates[0] += x_offset
                    coordinates[2] += x_offset
                    raw.append({
                        "class_id": int(box.cls.item()), "confidence": float(box.conf.item()),
                        "xyxy": coordinates, "source_view": source_view,
                    })

            add_result(model.predict(image, conf=0.05, iou=0.7, imgsz=imgsz, device=device, verbose=False)[0], 0, "full")
            if width / max(height, 1) > float(tile_config.get("aspect_ratio_trigger", 1.4)):
                for index, (x1, y1, x2, y2) in enumerate(tile_windows(
                    width, height, int(tile_config.get("tiles", 3)), float(tile_config.get("overlap", 0.25))
                ), start=1):
                    add_result(
                        model.predict(image[y1:y2, x1:x2], conf=0.05, iou=0.7, imgsz=imgsz, device=device, verbose=False)[0],
                        x1, f"tile_{index}",
                    )
            entries = []
            for item in merge_detections(raw, float(tile_config.get("merge_iou", 0.55))):
                x1, y1, x2, y2 = item["xyxy"]
                entries.append(Prediction(
                    int(item["class_id"]), float(item["confidence"]),
                    (x1 / width, y1 / height, x2 / width, y2 / height),
                ))
            predictions[image_path.stem] = entries
        return predictions

    results = model.predict(source=str(image_dir), conf=0.05, iou=0.7, imgsz=imgsz, device=device, stream=True, verbose=False)
    for result in results:
        entries: list[Prediction] = []
        if result.boxes is not None:
            coordinates = result.boxes.xyxyn.detach().cpu().numpy()
            classes = result.boxes.cls.detach().cpu().numpy()
            confidences = result.boxes.conf.detach().cpu().numpy()
            for coordinate, class_id, confidence in zip(coordinates, classes, confidences, strict=True):
                entries.append(Prediction(int(class_id), float(confidence), tuple(map(float, coordinate))))
        predictions[Path(result.path).stem] = entries
    return predictions


def _prediction_threshold(threshold: float | dict[int, float], class_id: int) -> float:
    return float(threshold.get(class_id, 1.0)) if isinstance(threshold, dict) else float(threshold)


def score_image(
    truths: list[GroundTruth], predictions: list[Prediction], threshold: float | dict[int, float]
) -> dict[str, Any]:
    active = [
        prediction for prediction in predictions
        if prediction.confidence >= _prediction_threshold(threshold, prediction.class_id)
    ]
    matched_truths: set[int] = set()
    matched_predictions: set[int] = set()
    tp = misclassified = poor_localization = 0
    per_class: dict[int, Counter[str]] = defaultdict(Counter)

    candidates: list[tuple[float, int, int]] = []
    for prediction_index, prediction in enumerate(active):
        for truth_index, truth in enumerate(truths):
            candidates.append((iou(prediction.xyxy, truth.xyxy), prediction_index, truth_index))
    for overlap, prediction_index, truth_index in sorted(candidates, reverse=True):
        if prediction_index in matched_predictions or truth_index in matched_truths or overlap < 0.10:
            continue
        prediction, truth = active[prediction_index], truths[truth_index]
        # A low-overlap, wrong-class pair is not a legitimate match; leave both
        # unmatched so they are counted as one FP and one FN below.
        if overlap < 0.50 and prediction.class_id != truth.class_id:
            continue
        matched_predictions.add(prediction_index)
        matched_truths.add(truth_index)
        if overlap >= 0.50 and prediction.class_id == truth.class_id:
            tp += 1
            per_class[truth.class_id]["tp"] += 1
        elif overlap >= 0.50:
            misclassified += 1
            per_class[truth.class_id]["fn"] += 1
            per_class[prediction.class_id]["fp"] += 1
        elif prediction.class_id == truth.class_id:
            poor_localization += 1
            per_class[truth.class_id]["fn"] += 1
            per_class[prediction.class_id]["fp"] += 1

    fp = len(active) - len(matched_predictions)
    fn = len(truths) - len(matched_truths)
    for index, prediction in enumerate(active):
        if index not in matched_predictions:
            per_class[prediction.class_id]["fp"] += 1
    for index, truth in enumerate(truths):
        if index not in matched_truths:
            per_class[truth.class_id]["fn"] += 1
    return {
        "tp": tp,
        "fp": fp + misclassified + poor_localization,
        "fn": fn + misclassified + poor_localization,
        "misclassification": misclassified,
        "poor_localization": poor_localization,
        "per_class": per_class,
    }


def threshold_analysis(predictions: dict[str, list[Prediction]], split: str, thresholds: list[float]) -> tuple[list[dict[str, float]], float]:
    image_dir = DATA_YAML.parent / "images" / split
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    truth_cache = {image.stem: load_ground_truth(image) for image in images}
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        totals = Counter()
        for image in images:
            totals.update({key: value for key, value in score_image(truth_cache[image.stem], predictions.get(image.stem, []), threshold).items() if isinstance(value, int)})
        precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
        recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append({"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, "tp": totals["tp"], "fp": totals["fp"], "fn": totals["fn"]})
    recall_candidates = [row for row in rows if row["recall"] >= 0.80]
    selected = max(recall_candidates or rows, key=lambda row: (row["f1"], row["recall"], -row["threshold"]))
    return rows, float(selected["threshold"])


def per_class_threshold_analysis(
    predictions: dict[str, list[Prediction]],
    split: str,
    thresholds: list[float],
    class_count: int,
    minimum_precision: float = 0.60,
) -> tuple[list[dict[str, float]], dict[int, float]]:
    """Select recall-oriented F2 thresholds using validation data only."""
    image_dir = DATA_YAML.parent / "images" / split
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    truth_cache = {image.stem: load_ground_truth(image) for image in images}
    rows: list[dict[str, float]] = []
    selected: dict[int, float] = {}
    for class_id in range(class_count):
        candidates: list[dict[str, float]] = []
        for threshold in thresholds:
            totals = Counter()
            for image in images:
                truths = [truth for truth in truth_cache[image.stem] if truth.class_id == class_id]
                class_predictions = [item for item in predictions.get(image.stem, []) if item.class_id == class_id]
                scored = score_image(truths, class_predictions, threshold)
                totals.update({key: value for key, value in scored.items() if isinstance(value, int)})
            precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
            recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
            f2 = 5 * precision * recall / max(4 * precision + recall, 1e-12)
            row = {
                "class_id": class_id, "threshold": threshold, "precision": precision,
                "recall": recall, "f2": f2, "tp": totals["tp"], "fp": totals["fp"], "fn": totals["fn"],
            }
            rows.append(row)
            candidates.append(row)
        eligible = [row for row in candidates if row["precision"] >= minimum_precision]
        winner = max(eligible or candidates, key=lambda row: (row["f2"], row["recall"], -row["threshold"]))
        selected[class_id] = float(winner["threshold"])
    return rows, selected


def aggregate_custom_metrics(predictions: dict[str, list[Prediction]], split: str, threshold: float | dict[int, float], class_count: int) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    totals = Counter()
    per_class: dict[int, Counter[str]] = defaultdict(Counter)
    per_image: list[dict[str, Any]] = []
    image_dir = DATA_YAML.parent / "images" / split
    for image in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
        scored = score_image(load_ground_truth(image), predictions.get(image.stem, []), threshold)
        totals.update({key: value for key, value in scored.items() if isinstance(value, int)})
        for class_id, counts in scored["per_class"].items():
            per_class[class_id].update(counts)
        row: dict[str, Any] = {"image_id": image.stem, **{key: value for key, value in scored.items() if isinstance(value, int)}}
        for class_id in range(class_count):
            counts = scored["per_class"][class_id]
            for key in ("tp", "fp", "fn"):
                row[f"class_{class_id}_{key}"] = counts[key]
        per_image.append(row)
    precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
    recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
    summary = {"threshold": threshold, "precision": precision, "recall": recall, "f1": 2 * precision * recall / max(precision + recall, 1e-12), **totals}
    class_rows = []
    for class_id in range(class_count):
        values = per_class[class_id]
        p = values["tp"] / max(values["tp"] + values["fp"], 1)
        r = values["tp"] / max(values["tp"] + values["fn"], 1)
        class_rows.append({"class_id": class_id, "support": values["tp"] + values["fn"], "tp": values["tp"], "fp": values["fp"], "fn": values["fn"], "precision": p, "recall": r, "f1": 2 * p * r / max(p + r, 1e-12)})
    return summary, class_rows, per_image


def bootstrap_operating_metrics(
    per_image: list[dict[str, Any]],
    manifest_path: Path,
    class_count: int,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Patient-level bootstrap intervals for macro operating-point metrics."""
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        group_by_image = {row["image_id"]: row["patient_group_hash"] for row in csv.DictReader(handle)}
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_image:
        grouped[group_by_image.get(str(row["image_id"]), str(row["image_id"]))].append(row)
    groups = sorted(grouped)
    if not groups or samples <= 0:
        return {}
    rng = random.Random(seed)
    distributions: dict[str, list[float]] = defaultdict(list)
    for _ in range(samples):
        drawn = [rng.choice(groups) for _ in groups]
        class_metrics: dict[str, list[float]] = defaultdict(list)
        for class_id in range(class_count):
            totals = Counter()
            for group in drawn:
                for row in grouped[group]:
                    for key in ("tp", "fp", "fn"):
                        totals[key] += int(row.get(f"class_{class_id}_{key}", 0))
            precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
            recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
            if totals["tp"] + totals["fn"]:
                class_metrics["precision"].append(precision)
                class_metrics["recall"].append(recall)
                class_metrics["f1"].append(2 * precision * recall / max(precision + recall, 1e-12))
        for key, values in class_metrics.items():
            distributions[f"macro_{key}"].append(float(np.mean(values)) if values else 0.0)
    intervals: dict[str, dict[str, float]] = {}
    for name, values in distributions.items():
        intervals[name] = {
            "lower_95": float(np.percentile(values, 2.5)),
            "median": float(np.percentile(values, 50.0)),
            "upper_95": float(np.percentile(values, 97.5)),
        }
    return intervals


def macro_operating_metrics(class_rows: list[dict[str, Any]]) -> dict[str, float]:
    supported = [row for row in class_rows if int(row.get("support", 0)) > 0]
    if not supported:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    return {
        key: float(np.mean([float(row[key]) for row in supported]))
        for key in ("precision", "recall", "f1")
    }


def paired_bootstrap_delta(
    candidate: list[dict[str, Any]],
    baseline: list[dict[str, Any]],
    manifest_path: Path,
    class_count: int,
    samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Paired patient-level bootstrap of candidate-minus-baseline macro metrics."""
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        group_by_image = {row["image_id"]: row["patient_group_hash"] for row in csv.DictReader(handle)}
    candidate_by_id = {str(row["image_id"]): row for row in candidate}
    baseline_by_id = {str(row["image_id"]): row for row in baseline}
    groups: dict[str, list[str]] = defaultdict(list)
    for image_id in sorted(set(candidate_by_id) & set(baseline_by_id)):
        groups[group_by_image.get(image_id, image_id)].append(image_id)
    group_ids = sorted(groups)
    if not group_ids or samples <= 0:
        return {}
    rng = random.Random(seed)
    distributions: dict[str, list[float]] = defaultdict(list)

    def sampled_macro(rows: dict[str, dict[str, Any]], drawn: list[str]) -> dict[str, float]:
        metrics: dict[str, list[float]] = defaultdict(list)
        for class_id in range(class_count):
            totals = Counter()
            for group in drawn:
                for image_id in groups[group]:
                    row = rows[image_id]
                    for key in ("tp", "fp", "fn"):
                        totals[key] += int(row.get(f"class_{class_id}_{key}", 0))
            if totals["tp"] + totals["fn"] == 0:
                continue
            precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
            recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
            metrics["precision"].append(precision)
            metrics["recall"].append(recall)
            metrics["f1"].append(2 * precision * recall / max(precision + recall, 1e-12))
        return {key: float(np.mean(values)) if values else 0.0 for key, values in metrics.items()}

    for _ in range(samples):
        drawn = [rng.choice(group_ids) for _ in group_ids]
        candidate_metrics = sampled_macro(candidate_by_id, drawn)
        baseline_metrics = sampled_macro(baseline_by_id, drawn)
        for key in ("precision", "recall", "f1"):
            distributions[f"macro_{key}"].append(candidate_metrics.get(key, 0.0) - baseline_metrics.get(key, 0.0))
    return {
        name: {
            "lower_95": float(np.percentile(values, 2.5)),
            "median": float(np.percentile(values, 50.0)),
            "upper_95": float(np.percentile(values, 97.5)),
        }
        for name, values in distributions.items()
    }


def render_examples(model_predictions: dict[str, list[Prediction]], per_image: list[dict[str, Any]], threshold: float | dict[int, float], names: list[str], count: int = 10) -> list[dict[str, Any]]:
    image_dir = DATA_YAML.parent / "images" / "test"
    ranked = sorted(per_image, key=lambda row: (-(row["fn"] + row["fp"] + row["misclassification"] + row["poor_localization"]), row["image_id"]))
    selectors = [
        ("correct_detection", lambda row: row["tp"] > 0 and row["fp"] == 0 and row["fn"] == 0),
        ("false_negative", lambda row: row["fn"] > 0),
        ("false_positive", lambda row: row["fp"] > 0),
        ("misclassification", lambda row: row["misclassification"] > 0),
        ("poor_localization", lambda row: row["poor_localization"] > 0),
    ]
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    for _, predicate in selectors:
        added = 0
        for row in (item for item in ranked if predicate(item) and item["image_id"] not in selected_ids):
            selected.append(row)
            selected_ids.add(row["image_id"])
            added += 1
            if added >= 2 or len(selected) >= count:
                break
    selected.extend(row for row in ranked if row["image_id"] not in selected_ids and len(selected) < count)
    selected = selected[:count]
    output_dir = REPORTS / "inference_examples"
    output_dir.mkdir(parents=True, exist_ok=True)
    for old_example in output_dir.glob("example_*.jpg"):
        old_example.unlink()
    analysis_rows: list[dict[str, Any]] = []
    for index, row in enumerate(selected, start=1):
        image_path = next(image_dir.glob(f"{row['image_id']}.*"))
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        height, width = image.shape[:2]
        truths = load_ground_truth(image_path)
        active_predictions = [
            prediction for prediction in model_predictions.get(row["image_id"], [])
            if prediction.confidence >= _prediction_threshold(threshold, prediction.class_id)
        ]
        for truth in truths:
            x1, y1, x2, y2 = truth.xyxy
            cv2.rectangle(image, (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height)), (40, 220, 40), 2)
            cv2.putText(image, f"GT {names[truth.class_id]}", (int(x1 * width), max(15, int(y1 * height) - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (40, 220, 40), 1, cv2.LINE_AA)
        for prediction in active_predictions:
            x1, y1, x2, y2 = prediction.xyxy
            cv2.rectangle(image, (int(x1 * width), int(y1 * height)), (int(x2 * width), int(y2 * height)), (40, 40, 240), 2)
            cv2.putText(image, f"P {names[prediction.class_id]} {prediction.confidence:.2f}", (int(x1 * width), min(height - 5, int(y2 * height) + 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (40, 40, 240), 1, cv2.LINE_AA)
        output = output_dir / f"example_{index:02d}_{row['image_id']}.jpg"
        cv2.imwrite(str(output), image, [cv2.IMWRITE_JPEG_QUALITY, 93])
        if row["misclassification"]:
            category = "misclassification"
        elif row["poor_localization"]:
            category = "poor_localization"
        elif row["fn"]:
            category = "false_negative"
        elif row["fp"]:
            category = "false_positive"
        else:
            category = "correct_detection"
        cause = {
            "misclassification": "Overlapping visual features, class imbalance, or annotation inconsistency.",
            "poor_localization": "Diffuse boundary, small target, or polygon-to-box approximation.",
            "false_negative": "Low contrast, small lesion, limited training, or conservative threshold.",
            "false_positive": "Anatomical structure resembles a labeled finding or source-domain shift.",
            "correct_detection": "Representative correctly localized prediction.",
        }[category]
        expected_counts = Counter(names[truth.class_id] for truth in truths)
        predicted_counts = Counter(names[prediction.class_id] for prediction in active_predictions)
        expected_result = "; ".join(f"{name} x{count_value}" for name, count_value in expected_counts.most_common()) or "No labeled object"
        model_prediction = "; ".join(f"{name} x{count_value}" for name, count_value in predicted_counts.most_common()) or "No prediction at selected threshold"
        outcomes = []
        if row["tp"]:
            outcomes.append("correct detection")
        if row["fp"]:
            outcomes.append("false positive")
        if row["fn"]:
            outcomes.append("false negative")
        if row["misclassification"]:
            outcomes.append("misclassification")
        if row["poor_localization"]:
            outcomes.append("poor localization")
        analysis_rows.append({
            "example": output.name,
            "image_id": row["image_id"],
            "category": category,
            "observed_outcomes": "; ".join(outcomes) or "no thresholded error",
            "expected_result": expected_result,
            "model_prediction": model_prediction,
            "tp": row["tp"],
            "fp": row["fp"],
            "fn": row["fn"],
            "probable_cause": cause,
            "proposed_improvement": "Clinical relabel review, more patient-diverse data, and class-aware sampling.",
        })
    return analysis_rows


def plot_thresholds(rows: list[dict[str, float]]) -> None:
    REPORTS.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 5))
    for key in ("precision", "recall", "f1"):
        plt.plot([row["threshold"] for row in rows], [row[key] for row in rows], marker="o", label=key)
    plt.xlabel("Confidence threshold")
    plt.ylabel("Score at IoU 0.50")
    plt.ylim(0, 1)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(REPORTS / "threshold_analysis.png", dpi=180)
    plt.close()


def environment_report(device: str) -> dict[str, Any]:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "ultralytics": ultralytics_version,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "torch_cuda": torch.version.cuda,
        "gpu": gpu,
        "selected_device": device,
    }


def support_tier(support: int) -> str:
    if support == 0:
        return "N/E"
    if support < 20:
        return "very_low"
    if support < 50:
        return "limited"
    return "supported"


def current_git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    )
    return completed.stdout.strip() or "unknown"


def dataset_fingerprint() -> str:
    candidates = [DATA_YAML.parent / "fingerprint.json", *[parent / "fingerprint.json" for parent in DATA_YAML.parents]]
    path = next((candidate for candidate in candidates if candidate.exists()), None)
    if path is None:
        raise FileNotFoundError("Run src.prepare_dataset to create dataset/fingerprint.json")
    payload = json.loads(path.read_text(encoding="utf-8"))
    return str(payload["fingerprint"])


def require_audit_approval(path: Path, fingerprint: str) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(
            "Manual annotation audit approval is missing. Review all 31 contact sheets and run src.approve_audit."
        )
    approval = json.loads(path.read_text(encoding="utf-8"))
    if approval.get("status") != "approved" or approval.get("dataset_fingerprint") != fingerprint:
        raise RuntimeError("Manual annotation audit approval does not match the current dataset fingerprint")
    return approval


def apply_profile(args: argparse.Namespace) -> None:
    if not args.profile:
        return
    profile = yaml.safe_load(args.profile.read_text(encoding="utf-8"))
    allowed = {
        "model", "baseline_epochs", "tuned_epochs", "imgsz", "batch", "workers", "patience",
        "cache", "seed", "run_prefix", "save_period", "amp", "data_view", "experiments",
        "require_primary_support", "allow_insufficient_support", "bootstrap_samples",
        "minimum_precision", "tile_inference", "tuning", "require_clinician_review",
    }
    unknown = set(profile) - allowed
    if unknown:
        raise ValueError(f"Unknown profile keys: {sorted(unknown)}")
    for key, value in profile.items():
        setattr(args, key, value)


def train_phase(args: argparse.Namespace, names: list[str], fingerprint: str, code_commit: str, device: str) -> dict[str, Any]:
    runtime = DATA_YAML.parent / "runtime"
    base_source = runtime / "base.yaml"
    augmented_source = runtime / "augmented.yaml"
    if not base_source.exists() or not augmented_source.exists():
        raise FileNotFoundError("Prepared base/augmented training views are missing; rebuild the dataset")
    base_yaml, base_images = create_runtime_data_yaml(base_source, args.max_train_images, args.seed, "base")
    augmented_yaml, augmented_images = create_runtime_data_yaml(
        augmented_source, args.max_train_images, args.seed, "augmented"
    )
    cache: bool | str = False if args.cache == "false" else args.cache
    model_slug = Path(args.model).stem.replace(".", "_")
    prefix = f"{args.run_prefix.strip('_')}_" if args.run_prefix.strip("_") else ""
    common = {
        "device": device,
        "seed": args.seed,
        "workers": args.workers,
        "cache": cache,
        "amp": args.amp,
        "patience": args.patience,
        "save_period": args.save_period,
        "resume": args.resume,
        "dataset_fingerprint": fingerprint,
        "code_commit": code_commit,
    }
    if args.experiments:
        experiments = []
        for specification in args.experiments:
            variant = str(specification.get("data_variant", "base"))
            source = runtime / f"{variant}.yaml"
            if not source.exists():
                raise FileNotFoundError(f"Training data variant is missing: {source}")
            runtime_yaml, image_count = create_runtime_data_yaml(
                source, args.max_train_images, args.seed, f"{variant}_{specification['name']}"
            )
            checkpoint = str(specification.get("model", args.model))
            experiments.append(train_experiment(
                f"{prefix}{specification['name']}",
                model_checkpoint=checkpoint,
                epochs=int(specification.get("epochs", args.tuned_epochs)),
                imgsz=int(specification.get("imgsz", args.imgsz)),
                batch=int(specification.get("batch", args.batch)),
                augmented=variant != "base",
                data_yaml=runtime_yaml,
                train_images=image_count,
                data_variant=variant,
                training_overrides=dict(specification.get("training", {})),
                **common,
            ))
    else:
        experiments = [
            train_experiment(
                f"{prefix}baseline_{model_slug}", model_checkpoint=args.model,
                epochs=args.baseline_epochs, imgsz=args.imgsz, batch=args.batch, augmented=False,
                data_yaml=base_yaml, train_images=base_images, data_variant="base", **common,
            ),
            train_experiment(
                f"{prefix}medical_aug_{model_slug}", model_checkpoint=args.model,
                epochs=args.tuned_epochs, imgsz=args.imgsz, batch=args.batch, augmented=True,
                data_yaml=augmented_yaml, train_images=augmented_images, data_variant="augmented", **common,
            ),
    ]
    map_key = "metrics/mAP50-95(B)"
    selection_pool = experiments
    advancement: dict[str, dict[str, Any]] = {}
    if len(names) == len(PRIMARY_SOURCE_CLASSES) and experiments:
        baseline = experiments[0]
        baseline_map = baseline["validation"].get(map_key, 0.0)
        baseline_recall = baseline["validation"].get("metrics/recall(B)", 0.0)
        selection_pool = [baseline]
        advancement[baseline["name"]] = {"advanced": True, "reason": "acceptance baseline"}
        for candidate in experiments[1:]:
            candidate_map = candidate["validation"].get(map_key, 0.0)
            candidate_recall = candidate["validation"].get("metrics/recall(B)", 0.0)
            map_gate = candidate_map >= baseline_map * 1.02
            recall_gate = candidate_recall >= baseline_recall - 0.02
            advanced = map_gate and recall_gate
            advancement[candidate["name"]] = {
                "advanced": advanced, "map50_95_relative_to_baseline": candidate_map / max(baseline_map, 1e-12) - 1,
                "recall_delta": candidate_recall - baseline_recall,
            }
            if advanced:
                selection_pool.append(candidate)
    selected = max(
        selection_pool,
        key=lambda experiment: (
            experiment["validation"].get(map_key, float("-inf")),
            experiment["validation"].get("metrics/recall(B)", float("-inf")),
        ),
    )
    shutil.copy2(selected["best"], ARTIFACTS / "best.pt")
    shutil.copy2(selected["last"], ARTIFACTS / "last.pt")
    experiment_rows = []
    for experiment in experiments:
        values = experiment["validation"]
        precision = values.get("metrics/precision(B)", 0.0)
        recall = values.get("metrics/recall(B)", 0.0)
        experiment_rows.append({
            "experiment": experiment["name"], "model": experiment["model"],
            "main_change": experiment["augmentation"], "data_variant": experiment["data_variant"],
            "epochs": experiment["epochs_completed"], "imgsz": experiment["imgsz"],
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "map50": values.get("metrics/mAP50(B)", 0.0),
            "map50_95": values.get(map_key, 0.0),
            "duration_seconds": experiment["duration_seconds"], "selected": experiment is selected,
        })
    write_csv(REPORTS / "experiments.csv", experiment_rows, list(experiment_rows[0]))
    selection = {
        "selected_experiment": selected["name"],
        "selection_metric": "validation mAP50-95; validation recall tie-breaker",
        "advancement_gates": advancement,
        "dataset_fingerprint": fingerprint,
        "code_commit": code_commit,
        "selected_checkpoint_sha256": sha256_file(ARTIFACTS / "best.pt"),
        "environment": environment_report(device),
        "experiments": experiments,
    }
    write_json(REPORTS / "selection.json", selection)
    write_json(ARTIFACTS / "training_configuration.json", selection)
    freeze = subprocess.run(
        [os.fspath(Path(sys.executable)), "-m", "pip", "freeze"], capture_output=True, text=True, check=True
    ).stdout
    (ARTIFACTS / "environment_freeze.txt").write_text(freeze, encoding="utf-8")
    print(json.dumps(selection, indent=2))
    return selection


def test_phase(args: argparse.Namespace, names: list[str], fingerprint: str, code_commit: str, device: str) -> dict[str, Any]:
    selection_path = REPORTS / "selection.json"
    if not selection_path.exists() or not (ARTIFACTS / "best.pt").exists():
        raise FileNotFoundError("Complete the training/selection phase before test evaluation")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if selection.get("dataset_fingerprint") != fingerprint or selection.get("code_commit") != code_commit:
        raise RuntimeError("Selection manifest is incompatible with the current dataset or code commit")
    selected_experiment = next(
        experiment for experiment in selection["experiments"]
        if experiment["name"] == selection["selected_experiment"]
    )
    evaluation_imgsz = int(selected_experiment["imgsz"])
    checkpoint_hash = sha256_file(ARTIFACTS / "best.pt")
    final_path = REPORTS / "final_metrics.json"
    if final_path.exists() and not args.force_test:
        existing = json.loads(final_path.read_text(encoding="utf-8"))
        if existing.get("dataset_fingerprint") == fingerprint and existing.get("checkpoint_sha256") == checkpoint_hash:
            print("Reusing the matching held-out test evaluation. Use --force-test to rerun explicitly.")
            return existing
        raise RuntimeError("Existing test results do not match the selected checkpoint or dataset")

    base_yaml = DATA_YAML.parent / "runtime" / "base.yaml"
    final_model = YOLO(str(ARTIFACTS / "best.pt"))
    val_predictions = cache_predictions(final_model, "val", evaluation_imgsz, device, args.tile_inference)
    thresholds = [round(value, 2) for value in np.arange(0.05, 0.80, 0.05)]
    threshold_rows, selected_threshold = threshold_analysis(val_predictions, "val", thresholds)
    write_csv(REPORTS / "threshold_analysis.csv", threshold_rows, list(threshold_rows[0]))
    plot_thresholds(threshold_rows)
    per_class_threshold_rows, selected_thresholds = per_class_threshold_analysis(
        val_predictions, "val", thresholds, len(names), args.minimum_precision
    )
    write_csv(
        REPORTS / "per_class_threshold_analysis.csv",
        per_class_threshold_rows,
        list(per_class_threshold_rows[0]),
    )
    test_metrics = final_model.val(
        data=str(base_yaml), split="test", imgsz=evaluation_imgsz, batch=args.batch, device=device,
        workers=args.workers, plots=True, project=str(RUNS), name="final_test", exist_ok=True,
    )
    test_standard = {key: float(value) for key, value in test_metrics.results_dict.items()}
    test_predictions = cache_predictions(final_model, "test", evaluation_imgsz, device, args.tile_inference)
    test_custom, class_rows, per_image = aggregate_custom_metrics(
        test_predictions, "test", selected_thresholds, len(names)
    )
    bootstrap_intervals = bootstrap_operating_metrics(
        per_image, REPORTS / "dataset_manifest.csv", len(names), args.bootstrap_samples, args.seed
    )
    maps = list(map(float, getattr(test_metrics.box, "maps", [0.0] * len(names))))
    for row, name, class_map in zip(class_rows, names, maps, strict=False):
        row["class_name"] = name
        row["map50_95"] = class_map
        row["evidence_tier"] = support_tier(int(row["support"]))
    write_csv(
        REPORTS / "per_class_test_metrics.csv", class_rows,
        ["class_id", "class_name", "support", "evidence_tier", "tp", "fp", "fn", "precision", "recall", "f1", "map50_95"],
    )
    write_csv(REPORTS / "per_image_test_errors.csv", per_image, list(per_image[0]))
    error_rows = render_examples(test_predictions, per_image, selected_thresholds, names, 10)
    write_csv(REPORTS / "error_analysis.csv", error_rows, list(error_rows[0]) if error_rows else ["example"])
    acceptance: dict[str, Any] | None = None
    if len(names) == len(PRIMARY_SOURCE_CLASSES):
        baseline_experiment = next(
            (item for item in selection["experiments"] if "e0_yolo26s_640" in item["name"]),
            selection["experiments"][0],
        )
        baseline_model = YOLO(str(baseline_experiment["best"]))
        baseline_imgsz = int(baseline_experiment["imgsz"])
        baseline_val_predictions = cache_predictions(baseline_model, "val", baseline_imgsz, device)
        _, baseline_thresholds = per_class_threshold_analysis(
            baseline_val_predictions, "val", thresholds, len(names), args.minimum_precision
        )
        baseline_ultralytics = baseline_model.val(
            data=str(base_yaml), split="test", imgsz=baseline_imgsz,
            batch=int(baseline_experiment["batch"]), device=device, workers=args.workers,
            plots=False, verbose=False,
        )
        baseline_predictions = cache_predictions(baseline_model, "test", baseline_imgsz, device)
        _, baseline_class_rows, baseline_per_image = aggregate_custom_metrics(
            baseline_predictions, "test", baseline_thresholds, len(names)
        )
        baseline_map = float(baseline_ultralytics.results_dict.get("metrics/mAP50-95(B)", 0.0))
        candidate_map = float(test_standard.get("metrics/mAP50-95(B)", 0.0))
        baseline_macro = macro_operating_metrics(baseline_class_rows)
        candidate_macro = macro_operating_metrics(class_rows)
        paired_intervals = paired_bootstrap_delta(
            per_image, baseline_per_image, REPORTS / "dataset_manifest.csv",
            len(names), args.bootstrap_samples, args.seed,
        )
        recall_regressions = {
            names[index]: float(class_rows[index]["recall"]) - float(baseline_class_rows[index]["recall"])
            for index in range(len(names))
        }
        support_payload = json.loads((REPORTS / "pathology_support.json").read_text(encoding="utf-8"))
        gates = {
            "support_gate": bool(support_payload.get("support_gate_passed")),
            "map50_95_relative_gain_at_least_25pct": (
                (candidate_map - baseline_map) / max(baseline_map, 1e-12) >= 0.25
            ),
            "macro_recall_gain_at_least_15_points": candidate_macro["recall"] - baseline_macro["recall"] >= 0.15,
            "no_class_recall_regression_over_5_points": min(recall_regressions.values(), default=0.0) >= -0.05,
            "bootstrap_macro_recall_excludes_zero": paired_intervals.get("macro_recall", {}).get("lower_95", -1.0) > 0.0,
        }
        acceptance = {
            "accepted": all(gates.values()), "gates": gates,
            "baseline_experiment": baseline_experiment["name"],
            "baseline_map50_95": baseline_map, "candidate_map50_95": candidate_map,
            "map50_95_relative_gain": (candidate_map - baseline_map) / max(baseline_map, 1e-12),
            "baseline_macro": baseline_macro, "candidate_macro": candidate_macro,
            "macro_recall_absolute_gain": candidate_macro["recall"] - baseline_macro["recall"],
            "per_class_recall_delta": recall_regressions,
            "paired_patient_bootstrap_delta_95": paired_intervals,
        }
        write_json(REPORTS / "acceptance_report.json", acceptance)
    payload = {
        **selection,
        "validation_selected_threshold": selected_threshold,
        "validation_selected_thresholds": {
            str(class_id): {"class_name": names[class_id], "threshold": threshold}
            for class_id, threshold in selected_thresholds.items()
        },
        "test_ultralytics_metrics": test_standard,
        "test_threshold_metrics_iou50": test_custom,
        "patient_bootstrap_95": bootstrap_intervals,
        "tile_inference": args.tile_inference or {"enabled": False},
        "v2_acceptance": acceptance,
        "checkpoint_sha256": checkpoint_hash,
        "evaluation_imgsz": evaluation_imgsz,
        "test_evaluation_id": stable_fingerprint({
            "dataset_fingerprint": fingerprint,
            "checkpoint_sha256": checkpoint_hash,
            "threshold": selected_threshold,
        }),
        "test_forced_rerun": bool(args.force_test),
    }
    write_json(final_path, payload)
    write_json(ARTIFACTS / "training_configuration.json", payload)
    print(json.dumps(payload, indent=2))
    return payload


def main() -> None:
    global DATA_YAML, RUNS, REPORTS, ARTIFACTS
    parser = argparse.ArgumentParser(description="Train/select and separately test YOLO26 experiments.")
    parser.add_argument("--phase", choices=("train", "test", "all"), default="all")
    parser.add_argument("--profile", type=Path)
    parser.add_argument("--baseline-epochs", type=int, default=100)
    parser.add_argument("--tuned-epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-images", type=int, default=None)
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--data-yaml", type=Path, default=DATA_YAML)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--run-prefix", default="panoramic31_yolo26s_t4_v1")
    parser.add_argument("--workers", type=int, default=0 if platform.system() == "Windows" else 2)
    parser.add_argument("--cache", choices=("false", "ram", "disk"), default="disk")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force-test", action="store_true")
    parser.add_argument("--data-view")
    parser.add_argument("--experiments", default=None)
    parser.add_argument("--require-primary-support", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--require-clinician-review", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--allow-insufficient-support", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--minimum-precision", type=float, default=0.60)
    parser.add_argument("--tile-inference", default=None)
    parser.add_argument("--tuning", default=None)
    parser.add_argument("--audit-approval", type=Path, default=ROOT / "reports" / "manual_audit_approval.json")
    args = parser.parse_args()
    apply_profile(args)
    if args.data_view and args.data_yaml.resolve() == (ROOT / "dataset" / "data.yaml").resolve():
        args.data_yaml = ROOT / "dataset" / str(args.data_view) / "data.yaml"
    DATA_YAML = args.data_yaml.resolve()
    output_root = args.output_root.resolve()
    RUNS, REPORTS, ARTIFACTS = output_root / "runs", output_root / "reports", output_root / "artifacts"
    if not DATA_YAML.exists():
        raise FileNotFoundError("Run src.prepare_dataset before training")
    REPORTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    names = load_names(DATA_YAML)
    fingerprint = dataset_fingerprint()
    require_audit_approval(args.audit_approval.resolve(), fingerprint)
    if args.require_primary_support:
        require_support_gate(args.audit_approval.resolve().parent / "pathology_support.json", args.allow_insufficient_support)
    if args.require_clinician_review:
        require_clinician_review(
            args.audit_approval.resolve().parent / "pathology_clinician_review.json", fingerprint
        )
    commit = current_git_commit()
    device = choose_device(args.device)
    if args.phase in {"train", "all"}:
        train_phase(args, names, fingerprint, commit, device)
    if args.phase in {"test", "all"}:
        test_phase(args, names, fingerprint, commit, device)


if __name__ == "__main__":
    main()
