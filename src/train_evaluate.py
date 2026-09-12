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

from src.common import IMAGE_EXTENSIONS, load_names, write_json


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


def create_runtime_data_yaml(max_train_images: int | None, seed: int) -> tuple[Path, int]:
    image_dir = DATA_YAML.parent / "images" / "train"
    images = sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS)
    if not max_train_images or max_train_images >= len(images):
        return DATA_YAML, len(images)
    labels_by_image: dict[Path, set[int]] = {}
    class_candidates: dict[int, list[Path]] = defaultdict(list)
    for image in images:
        label = DATA_YAML.parent / "labels" / "train" / f"{image.stem}.txt"
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
    train_list = runtime_dir / "train.txt"
    train_list.write_text("\n".join(path.resolve().as_posix() for path in chosen) + "\n", encoding="utf-8")
    base = yaml.safe_load(DATA_YAML.read_text(encoding="utf-8"))
    # Runtime YAML lives in dataset/runtime, so anchor its relative val/test
    # entries to the actual generated dataset root.
    base["path"] = DATA_YAML.parent.resolve().as_posix()
    base["train"] = train_list.resolve().as_posix()
    runtime_yaml = runtime_dir / "data_runtime.yaml"
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
) -> dict[str, Any]:
    run_dir = RUNS / name
    existing_last = run_dir / "weights" / "last.pt"
    results_csv = run_dir / "results.csv"
    completed_before = sum(1 for _ in results_csv.open(encoding="utf-8")) - 1 if results_csv.exists() else 0
    already_complete = completed_before >= epochs and (run_dir / "weights" / "best.pt").exists()
    checkpoint = str(existing_last) if resume and existing_last.exists() and not already_complete else model_checkpoint
    model = YOLO(checkpoint)
    parameters: dict[str, Any] = {
        "data": str(data_yaml),
        "epochs": epochs,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "workers": workers,
        "patience": patience,
        "optimizer": "AdamW" if augmented else "auto",
        "lr0": 0.001 if augmented else 0.01,
        "lrf": 0.01,
        "cos_lr": augmented,
        "weight_decay": 0.0005,
        "mosaic": 0.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "hsv_h": 0.0,
        "hsv_s": 0.0,
        "hsv_v": 0.12 if augmented else 0.0,
        "degrees": 7.0 if augmented else 0.0,
        "translate": 0.03 if augmented else 0.0,
        "scale": 0.10 if augmented else 0.0,
        "multi_scale": 0.10 if augmented else 0.0,
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
    started = time.monotonic()
    if already_complete:
        print(f"Skipping completed experiment {name}: {completed_before}/{epochs} epochs already present.", flush=True)
        model = YOLO(str(run_dir / "weights" / "best.pt"))
    elif resume and existing_last.exists():
        print(f"Resuming {name} from {existing_last}", flush=True)
        model.train(resume=True, epochs=epochs)
    else:
        model.train(**parameters)
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
    return {
        "name": name,
        "model": model_checkpoint,
        "epochs_requested": epochs,
        "epochs_completed": completed,
        "imgsz": imgsz,
        "batch": batch,
        "device": device,
        "augmentation": "class-aware medical augmentation + cosine AdamW" if augmented else "minimal",
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


def cache_predictions(model: YOLO, split: str, imgsz: int, device: str) -> dict[str, list[Prediction]]:
    image_dir = DATA_YAML.parent / "images" / split
    predictions: dict[str, list[Prediction]] = {}
    # The operating-point search starts at 0.05, so retaining lower-confidence
    # detections only increases memory and matching cost without affecting it.
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


def score_image(truths: list[GroundTruth], predictions: list[Prediction], threshold: float) -> dict[str, Any]:
    active = [prediction for prediction in predictions if prediction.confidence >= threshold]
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
    rows: list[dict[str, float]] = []
    for threshold in thresholds:
        totals = Counter()
        for image in images:
            totals.update({key: value for key, value in score_image(load_ground_truth(image), predictions.get(image.stem, []), threshold).items() if isinstance(value, int)})
        precision = totals["tp"] / max(totals["tp"] + totals["fp"], 1)
        recall = totals["tp"] / max(totals["tp"] + totals["fn"], 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        rows.append({"threshold": threshold, "precision": precision, "recall": recall, "f1": f1, "tp": totals["tp"], "fp": totals["fp"], "fn": totals["fn"]})
    recall_candidates = [row for row in rows if row["recall"] >= 0.80]
    selected = max(recall_candidates or rows, key=lambda row: (row["f1"], row["recall"], -row["threshold"]))
    return rows, float(selected["threshold"])


def aggregate_custom_metrics(predictions: dict[str, list[Prediction]], split: str, threshold: float, class_count: int) -> tuple[dict[str, float], list[dict[str, Any]], list[dict[str, Any]]]:
    totals = Counter()
    per_class: dict[int, Counter[str]] = defaultdict(Counter)
    per_image: list[dict[str, Any]] = []
    image_dir = DATA_YAML.parent / "images" / split
    for image in sorted(path for path in image_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
        scored = score_image(load_ground_truth(image), predictions.get(image.stem, []), threshold)
        totals.update({key: value for key, value in scored.items() if isinstance(value, int)})
        for class_id, counts in scored["per_class"].items():
            per_class[class_id].update(counts)
        per_image.append({"image_id": image.stem, **{key: value for key, value in scored.items() if isinstance(value, int)}})
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


def render_examples(model_predictions: dict[str, list[Prediction]], per_image: list[dict[str, Any]], threshold: float, names: list[str], count: int = 10) -> list[dict[str, Any]]:
    image_dir = DATA_YAML.parent / "images" / "test"
    ranked = sorted(per_image, key=lambda row: (-(row["fn"] + row["fp"] + row["misclassification"] + row["poor_localization"]), row["image_id"]))
    selected = ranked[:count]
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
        active_predictions = [prediction for prediction in model_predictions.get(row["image_id"], []) if prediction.confidence >= threshold]
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


def main() -> None:
    global DATA_YAML, RUNS, REPORTS, ARTIFACTS
    parser = argparse.ArgumentParser(description="Train, select, test, infer, and analyze YOLO26 experiments.")
    parser.add_argument("--baseline-epochs", type=int, default=20)
    parser.add_argument("--tuned-epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=512)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-images", type=int, default=None, help="Deterministic class-covered compute profile; validation/test stay complete.")
    parser.add_argument("--model", default="yolo26n.pt", help="Ultralytics model checkpoint, for example yolo26s.pt on Colab.")
    parser.add_argument("--data-yaml", type=Path, default=DATA_YAML)
    parser.add_argument("--output-root", type=Path, default=ROOT, help="Root containing runs/, reports/, and artifacts/.")
    parser.add_argument("--run-prefix", default="", help="Prefix added to experiment names, such as colab_t4.")
    parser.add_argument("--workers", type=int, default=0 if platform.system() == "Windows" else 4)
    parser.add_argument("--cache", choices=("false", "ram", "disk"), default="false")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--save-period", type=int, default=1)
    parser.add_argument("--resume", action="store_true", help="Resume incomplete experiments from their last checkpoint.")
    args = parser.parse_args()
    DATA_YAML = args.data_yaml.resolve()
    output_root = args.output_root.resolve()
    RUNS = output_root / "runs"
    REPORTS = output_root / "reports"
    ARTIFACTS = output_root / "artifacts"
    if not DATA_YAML.exists():
        raise FileNotFoundError("Run src.prepare_dataset before training")
    names = load_names(DATA_YAML)
    device = choose_device(args.device)
    runtime_data_yaml, train_images = create_runtime_data_yaml(args.max_train_images, args.seed)
    REPORTS.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    cache: bool | str = False if args.cache == "false" else args.cache
    model_slug = Path(args.model).stem.replace(".", "_")
    prefix = f"{args.run_prefix.strip('_')}_" if args.run_prefix.strip("_") else ""
    common = {
        "model_checkpoint": args.model,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": device,
        "seed": args.seed,
        "data_yaml": runtime_data_yaml,
        "train_images": train_images,
        "workers": args.workers,
        "cache": cache,
        "amp": args.amp,
        "patience": args.patience,
        "save_period": args.save_period,
        "resume": args.resume,
    }
    experiments = [
        train_experiment(f"{prefix}baseline_{model_slug}", epochs=args.baseline_epochs, augmented=False, **common),
        train_experiment(f"{prefix}tuned_{model_slug}_medical_aug", epochs=args.tuned_epochs, augmented=True, **common),
    ]
    map_key = "metrics/mAP50-95(B)"
    selected = max(experiments, key=lambda experiment: experiment["validation"].get(map_key, float("-inf")))
    shutil.copy2(selected["best"], ARTIFACTS / "best.pt")
    shutil.copy2(selected["last"], ARTIFACTS / "last.pt")

    final_model = YOLO(str(ARTIFACTS / "best.pt"))
    val_predictions = cache_predictions(final_model, "val", args.imgsz, device)
    thresholds = [round(value, 2) for value in np.arange(0.05, 0.80, 0.05)]
    threshold_rows, selected_threshold = threshold_analysis(val_predictions, "val", thresholds)
    write_csv(REPORTS / "threshold_analysis.csv", threshold_rows, list(threshold_rows[0]))
    plot_thresholds(threshold_rows)

    test_metrics = final_model.val(data=str(runtime_data_yaml), split="test", imgsz=args.imgsz, batch=args.batch, device=device, workers=args.workers, plots=True, project=str(RUNS), name="final_test", exist_ok=True)
    test_standard = {key: float(value) for key, value in test_metrics.results_dict.items()}
    test_predictions = cache_predictions(final_model, "test", args.imgsz, device)
    test_custom, class_rows, per_image = aggregate_custom_metrics(test_predictions, "test", selected_threshold, len(names))
    maps = list(map(float, getattr(test_metrics.box, "maps", [0.0] * len(names))))
    for row, name, class_map in zip(class_rows, names, maps, strict=False):
        row["class_name"] = name
        row["map50_95"] = class_map
    write_csv(REPORTS / "per_class_test_metrics.csv", class_rows, ["class_id", "class_name", "support", "tp", "fp", "fn", "precision", "recall", "f1", "map50_95"])
    write_csv(REPORTS / "per_image_test_errors.csv", per_image, list(per_image[0]))
    error_rows = render_examples(test_predictions, per_image, selected_threshold, names, 10)
    write_csv(REPORTS / "error_analysis.csv", error_rows, list(error_rows[0]) if error_rows else ["example"])

    experiment_rows = []
    for experiment in experiments:
        values = experiment["validation"]
        precision = values.get("metrics/precision(B)", 0.0)
        recall = values.get("metrics/recall(B)", 0.0)
        experiment_rows.append({
            "experiment": experiment["name"],
            "model": experiment["model"],
            "main_change": experiment["augmentation"],
            "epochs": experiment["epochs_completed"],
            "imgsz": experiment["imgsz"],
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12),
            "map50": values.get("metrics/mAP50(B)", 0.0),
            "map50_95": values.get(map_key, 0.0),
            "duration_seconds": experiment["duration_seconds"],
            "selected": experiment is selected,
        })
    write_csv(REPORTS / "experiments.csv", experiment_rows, list(experiment_rows[0]))

    payload = {
        "selected_experiment": selected["name"],
        "selection_metric": "validation mAP50-95",
        "validation_selected_threshold": selected_threshold,
        "test_ultralytics_metrics": test_standard,
        "test_threshold_metrics_iou50": test_custom,
        "environment": environment_report(device),
        "experiments": experiments,
    }
    write_json(REPORTS / "final_metrics.json", payload)
    write_json(ARTIFACTS / "training_configuration.json", payload)
    freeze = subprocess.run([os.fspath(Path(sys.executable)), "-m", "pip", "freeze"], capture_output=True, text=True, check=True).stdout
    (ARTIFACTS / "environment_freeze.txt").write_text(freeze, encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
