from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import random
import shutil
import subprocess
from typing import Any

from ultralytics import YOLO
import yaml

from src.balanced_trainer import PatientBalancedDetectionTrainer, configure_patient_groups
from src.common import load_names, sha256_file, stable_fingerprint, write_json
from src.pathology import require_clinician_review, require_support_gate


ROOT = Path(__file__).resolve().parents[1]


def sample_search_space(space: dict[str, list[Any]], rng: random.Random) -> dict[str, float]:
    sampled: dict[str, float] = {}
    for name, specification in space.items():
        low, high = float(specification[0]), float(specification[1])
        if len(specification) > 2 and specification[2] == "log":
            sampled[name] = math.exp(rng.uniform(math.log(low), math.log(high)))
        else:
            sampled[name] = rng.uniform(low, high)
    return sampled


def train_candidate(
    *,
    checkpoint: str,
    data_yaml: Path,
    project: Path,
    name: str,
    epochs: int,
    imgsz: int,
    batch: int,
    workers: int,
    patience: int,
    seed: int,
    device: str,
    overrides: dict[str, float],
) -> dict[str, Any]:
    run = project / name
    result_path = run / "candidate.json"
    if result_path.exists() and (run / "weights" / "best.pt").exists():
        return json.loads(result_path.read_text(encoding="utf-8"))
    last = run / "weights" / "last.pt"
    configure_patient_groups(project.parent / "reports" / "dataset_manifest.csv", seed)
    model = YOLO(str(last) if last.exists() else checkpoint)
    if last.exists():
        model.train(resume=True, epochs=epochs, trainer=PatientBalancedDetectionTrainer)
    else:
        parameters: dict[str, Any] = {
            "data": str(data_yaml), "epochs": epochs, "imgsz": imgsz, "batch": batch,
            "workers": workers, "patience": patience, "optimizer": "AdamW", "lr0": 0.001,
            "lrf": 0.01, "cos_lr": True, "weight_decay": 0.0005, "mosaic": 0.0,
            "mixup": 0.0, "cutmix": 0.0, "copy_paste": 0.0, "hsv_h": 0.0,
            "hsv_s": 0.0, "hsv_v": 0.0, "flipud": 0.0, "fliplr": 0.0,
            "deterministic": True, "seed": seed, "amp": True, "cache": "disk", "plots": True,
            "device": device, "project": str(project), "name": name, "exist_ok": True,
        }
        parameters.update(overrides)
        model.train(**parameters, trainer=PatientBalancedDetectionTrainer)
    best = run / "weights" / "best.pt"
    validation = YOLO(str(best)).val(
        data=str(data_yaml), split="val", imgsz=imgsz, batch=batch, workers=workers,
        device=device, plots=False, verbose=False,
    )
    metrics = {key: float(value) for key, value in validation.results_dict.items()}
    record: dict[str, Any] = {
        "name": name, "model": checkpoint, "epochs_requested": epochs, "imgsz": imgsz,
        "batch": batch, "seed": seed, "data_variant": "balanced", "augmentation": "patient-balanced pathology view",
        "training_overrides": overrides, "validation": metrics,
        "best": str(best), "last": str(last),
    }
    record["fingerprint"] = stable_fingerprint({key: value for key, value in record.items() if key not in {"best", "last"}})
    write_json(result_path, record)
    return record


def write_trials(path: Path, records: list[dict[str, Any]]) -> None:
    rows = []
    for record in records:
        metrics = record["validation"]
        rows.append({
            "name": record["name"], "model": record["model"], "imgsz": record["imgsz"],
            "seed": record["seed"], "map50_95": metrics.get("metrics/mAP50-95(B)", 0.0),
            "map50": metrics.get("metrics/mAP50(B)", 0.0),
            "precision": metrics.get("metrics/precision(B)", 0.0),
            "recall": metrics.get("metrics/recall(B)", 0.0),
            "overrides": json.dumps(record["training_overrides"], sort_keys=True),
        })
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only pathology hyperparameter search and finalists.")
    parser.add_argument("--profile", type=Path, default=ROOT / "configs" / "pathology_a100.yaml")
    parser.add_argument("--data-yaml", type=Path, default=ROOT / "dataset" / "views" / "pathology" / "data.yaml")
    parser.add_argument("--output-root", type=Path, default=ROOT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--screen-only", action="store_true")
    args = parser.parse_args()
    profile = yaml.safe_load(args.profile.read_text(encoding="utf-8"))
    tuning = profile.get("tuning")
    if not tuning:
        raise ValueError("The selected profile has no tuning section")
    support_report = ROOT / "reports" / "pathology_support.json"
    require_support_gate(support_report, bool(profile.get("allow_insufficient_support", False)))
    output = args.output_root.resolve()
    reports, artifacts = output / "reports", output / "artifacts"
    selection_path = reports / "selection.json"
    if not selection_path.exists():
        raise FileNotFoundError("Run the architecture experiment stage before tuning")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    if profile.get("require_clinician_review"):
        require_clinician_review(
            ROOT / "reports" / "pathology_clinician_review.json", selection["dataset_fingerprint"]
        )
    architecture = next(
        item for item in selection["experiments"] if item["name"] == selection["selected_experiment"]
    )
    data_yaml = args.data_yaml.resolve().parent / "runtime" / "balanced.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Balanced pathology view is missing: {data_yaml}")
    checkpoint = str(architecture["model"])
    imgsz, batch = int(architecture["imgsz"]), int(architecture["batch"])
    rng = random.Random(int(profile.get("seed", 42)))
    screen: list[dict[str, Any]] = []
    for index in range(int(tuning["trials"])):
        overrides = sample_search_space(tuning["search_space"], rng)
        screen.append(train_candidate(
            checkpoint=checkpoint, data_yaml=data_yaml, project=output / "tuning_runs",
            name=f"screen_{index + 1:02d}", epochs=int(tuning["epochs"]), imgsz=imgsz,
            batch=batch, workers=int(profile.get("workers", 4)), patience=int(tuning["epochs"]),
            seed=int(profile.get("seed", 42)), device=args.device, overrides=overrides,
        ))
        write_json(reports / "tuning_trials.json", screen)
        write_trials(reports / "tuning_trials.csv", screen)
    ranked = sorted(
        screen,
        key=lambda item: (
            item["validation"].get("metrics/mAP50-95(B)", 0.0),
            item["validation"].get("metrics/recall(B)", 0.0),
        ),
        reverse=True,
    )
    architecture_map = float(architecture["validation"].get("metrics/mAP50-95(B)", 0.0))
    architecture_recall = float(architecture["validation"].get("metrics/recall(B)", 0.0))
    advanced = [
        item for item in ranked
        if item["validation"].get("metrics/mAP50-95(B)", 0.0) >= architecture_map * 1.02
        and item["validation"].get("metrics/recall(B)", 0.0) >= architecture_recall - 0.02
    ]
    finalists_pool = advanced or ranked
    if args.screen_only:
        write_json(reports / "tuning_selection.json", {
            "status": "screened", "advancement_gate_passed_by": len(advanced),
            "top": finalists_pool[: int(tuning["finalists"])],
        })
        return

    finalists: list[dict[str, Any]] = []
    for index, candidate in enumerate(finalists_pool[: int(tuning["finalists"])], start=1):
        finalists.append(train_candidate(
            checkpoint=checkpoint, data_yaml=data_yaml, project=output / "tuning_runs",
            name=f"finalist_{index}_seed42", epochs=int(profile.get("tuned_epochs", 120)), imgsz=imgsz,
            batch=batch, workers=int(profile.get("workers", 4)), patience=int(profile.get("patience", 30)),
            seed=42, device=args.device, overrides=candidate["training_overrides"],
        ))
    winner = max(
        finalists,
        key=lambda item: (
            item["validation"].get("metrics/mAP50-95(B)", 0.0),
            item["validation"].get("metrics/recall(B)", 0.0),
        ),
    )
    confirmations = [winner]
    for seed in [value for value in tuning.get("seeds", [42, 43, 44]) if int(value) != 42]:
        confirmations.append(train_candidate(
            checkpoint=checkpoint, data_yaml=data_yaml, project=output / "tuning_runs",
            name=f"winner_seed{seed}", epochs=int(profile.get("tuned_epochs", 120)), imgsz=imgsz,
            batch=batch, workers=int(profile.get("workers", 4)), patience=int(profile.get("patience", 30)),
            seed=int(seed), device=args.device, overrides=winner["training_overrides"],
        ))
    artifacts.mkdir(parents=True, exist_ok=True)
    shutil.copy2(winner["best"], artifacts / "best.pt")
    shutil.copy2(winner["last"], artifacts / "last.pt")
    winner["dataset_fingerprint"] = selection["dataset_fingerprint"]
    selection["experiments"].extend(finalists)
    selection["selected_experiment"] = winner["name"]
    selection["selection_metric"] = "validation mAP50-95; recall tie-breaker after randomized tuning"
    selection["selected_checkpoint_sha256"] = sha256_file(artifacts / "best.pt")
    selection["tuning_confirmations"] = confirmations
    selection["code_commit"] = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False
    ).stdout.strip() or "unknown"
    write_json(selection_path, selection)
    write_json(artifacts / "training_configuration.json", selection)
    write_json(reports / "tuning_selection.json", {
        "status": "complete", "advancement_gate_passed_by": len(advanced),
        "screen_top": finalists_pool[: int(tuning["finalists"])],
        "finalists": finalists, "winner": winner, "seed_confirmations": confirmations,
    })
    print(json.dumps({"winner": winner["name"], "confirmations": len(confirmations)}, indent=2))


if __name__ == "__main__":
    main()
