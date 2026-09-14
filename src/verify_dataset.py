from __future__ import annotations

import argparse
import csv
from collections import defaultdict
import json
from pathlib import Path

import yaml

from src.common import IMAGE_EXTENSIONS, write_json
from src.pathology import PRIMARY_SOURCE_CLASSES, support_failures


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
REPORTS = ROOT / "reports"


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a prepared dental YOLO dataset.")
    parser.add_argument("--dataset", type=Path, default=DATASET)
    parser.add_argument("--reports-root", type=Path, default=REPORTS)
    args = parser.parse_args()
    dataset = args.dataset.resolve()
    reports = args.reports_root.resolve()
    configuration = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))
    class_count = len(configuration["names"])
    failures: list[str] = []
    if class_count != 31:
        failures.append(f"expected 31 classes, found {class_count}")
    fingerprint_path = dataset / "fingerprint.json"
    dataset_fingerprint = ""
    if not fingerprint_path.exists():
        failures.append("dataset/fingerprint.json is missing")
    else:
        try:
            dataset_fingerprint = str(json.loads(fingerprint_path.read_text(encoding="utf-8"))["fingerprint"])
        except (json.JSONDecodeError, KeyError, OSError):
            failures.append("dataset/fingerprint.json is invalid")
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        image_stems = {path.stem for path in (dataset / "images" / split).iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
        label_paths = list((dataset / "labels" / split).glob("*.txt"))
        label_stems = {path.stem for path in label_paths}
        if image_stems != label_stems:
            failures.append(f"{split}: image/label stem mismatch")
        instances = 0
        for label in label_paths:
            for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
                parts = line.split()
                if len(parts) != 5:
                    failures.append(f"{label}:{line_number}: expected five fields")
                    continue
                class_id = int(parts[0])
                coordinates = list(map(float, parts[1:]))
                if not 0 <= class_id < class_count:
                    failures.append(f"{label}:{line_number}: class out of range")
                if not all(0 <= value <= 1 for value in coordinates) or coordinates[2] <= 0 or coordinates[3] <= 0:
                    failures.append(f"{label}:{line_number}: invalid normalized box")
                instances += 1
        counts[split] = {"images": len(image_stems), "labels": len(label_stems), "instances": instances}

    runtime = dataset / "runtime"
    view_stems: dict[str, set[str]] = {}
    for view in ("base", "augmented"):
        path = runtime / f"{view}_train.txt"
        if not path.exists():
            failures.append(f"runtime/{view}_train.txt is missing")
            view_stems[view] = set()
            continue
        listed = [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        missing = [str(path) for path in listed if not path.exists()]
        if missing:
            failures.append(f"runtime/{view}_train.txt lists {len(missing)} missing images")
        view_stems[view] = {path.stem for path in listed}
    if any("_aug" in stem for stem in view_stems.get("base", set())):
        failures.append("base training view contains augmented images")
    if not view_stems.get("base", set()).issubset(view_stems.get("augmented", set())):
        failures.append("augmented training view is not a superset of the base view")
    generated_train_stems = {
        path.stem for path in (dataset / "images" / "train").iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
    }
    if view_stems.get("augmented", set()) != generated_train_stems:
        failures.append("augmented training view does not exactly match generated training images")
    for split in ("val", "test"):
        if any("_aug" in path.stem for path in (dataset / "images" / split).iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS):
            failures.append(f"{split}: augmented images are forbidden")

    manifest = list(csv.DictReader((reports / "dataset_manifest.csv").open(encoding="utf-8")))
    by_patient: dict[str, set[str]] = defaultdict(set)
    by_hash: dict[str, set[str]] = defaultdict(set)
    split_by_id: dict[str, str] = {}
    for row in manifest:
        by_patient[row["patient_group_hash"]].add(row["split"])
        by_hash[row["sha256"]].add(row["split"])
        split_by_id[row["image_id"]] = row["split"]
    patient_leakage = sum(len(splits) > 1 for splits in by_patient.values())
    exact_hash_leakage = sum(bool(digest) and len(splits) > 1 for digest, splits in by_hash.items())
    near_pairs = list(csv.DictReader((reports / "near_duplicate_candidates.csv").open(encoding="utf-8")))
    near_leakage = sum(split_by_id[row["left"]] != split_by_id[row["right"]] for row in near_pairs)
    if patient_leakage:
        failures.append(f"{patient_leakage} patient groups cross splits")
    if exact_hash_leakage:
        failures.append(f"{exact_hash_leakage} exact hashes cross splits")
    if near_leakage:
        failures.append(f"{near_leakage} listed near-duplicate pairs cross splits")
    audit = json.loads((reports / "dataset_audit.json").read_text(encoding="utf-8"))
    if dataset_fingerprint and audit.get("dataset_fingerprint") != dataset_fingerprint:
        failures.append("dataset fingerprint does not match dataset audit")

    pathology = dataset / "views" / "pathology"
    support_path = reports / "pathology_support.json"
    if not pathology.exists() or not support_path.exists():
        failures.append("primary pathology dataset view or support report is missing")
        pathology_counts: dict[str, dict[str, int]] = {}
        support_gate_passed = False
    else:
        pathology_config = yaml.safe_load((pathology / "data.yaml").read_text(encoding="utf-8"))
        pathology_class_count = len(pathology_config.get("names", {}))
        if pathology_class_count != len(PRIMARY_SOURCE_CLASSES):
            failures.append(
                f"pathology view: expected {len(PRIMARY_SOURCE_CLASSES)} classes, found {pathology_class_count}"
            )
        pathology_counts = {}
        for split in ("train", "val", "test"):
            image_stems = {
                path.stem for path in (pathology / "images" / split).iterdir()
                if path.suffix.lower() in IMAGE_EXTENSIONS
            }
            labels = list((pathology / "labels" / split).glob("*.txt"))
            if image_stems != {path.stem for path in labels}:
                failures.append(f"pathology/{split}: image/label stem mismatch")
            instances = 0
            for label in labels:
                for line_number, line in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
                    parts = line.split()
                    if len(parts) != 5 or not 0 <= int(float(parts[0])) < pathology_class_count:
                        failures.append(f"{label}:{line_number}: invalid remapped pathology label")
                    instances += 1
            pathology_counts[split] = {"images": len(image_stems), "labels": len(labels), "instances": instances}
        support = json.loads(support_path.read_text(encoding="utf-8"))
        recalculated_failures = support_failures(
            support.get("patient_group_support", {}), support.get("minimum_patient_groups", {})
        )
        if recalculated_failures != support.get("support_failures", []):
            failures.append("pathology support report is internally inconsistent")
        support_gate_passed = not recalculated_failures
    payload = {
        "status": "pass" if not failures else "fail",
        "counts": counts,
        "class_count": class_count,
        "dataset_fingerprint": dataset_fingerprint,
        "training_view_counts": {name: len(stems) for name, stems in view_stems.items()},
        "patient_group_cross_split": patient_leakage,
        "exact_hash_cross_split": exact_hash_leakage,
        "listed_near_duplicate_cross_split": near_leakage,
        "pathology_counts": pathology_counts,
        "pathology_support_gate_passed": support_gate_passed,
        "failures": failures,
    }
    write_json(reports / "dataset_verification.json", payload)
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
