from __future__ import annotations

import csv
from collections import defaultdict
import json
from pathlib import Path

import yaml

from src.common import IMAGE_EXTENSIONS, write_json


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "dataset"
REPORTS = ROOT / "reports"


def main() -> None:
    configuration = yaml.safe_load((DATASET / "data.yaml").read_text(encoding="utf-8"))
    class_count = len(configuration["names"])
    failures: list[str] = []
    counts: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        image_stems = {path.stem for path in (DATASET / "images" / split).iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS}
        label_paths = list((DATASET / "labels" / split).glob("*.txt"))
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

    manifest = list(csv.DictReader((REPORTS / "dataset_manifest.csv").open(encoding="utf-8")))
    by_patient: dict[str, set[str]] = defaultdict(set)
    by_hash: dict[str, set[str]] = defaultdict(set)
    split_by_id: dict[str, str] = {}
    for row in manifest:
        by_patient[row["patient_group_hash"]].add(row["split"])
        by_hash[row["sha256"]].add(row["split"])
        split_by_id[row["image_id"]] = row["split"]
    patient_leakage = sum(len(splits) > 1 for splits in by_patient.values())
    exact_hash_leakage = sum(bool(digest) and len(splits) > 1 for digest, splits in by_hash.items())
    near_pairs = list(csv.DictReader((REPORTS / "near_duplicate_candidates.csv").open(encoding="utf-8")))
    near_leakage = sum(split_by_id[row["left"]] != split_by_id[row["right"]] for row in near_pairs)
    if patient_leakage:
        failures.append(f"{patient_leakage} patient groups cross splits")
    if exact_hash_leakage:
        failures.append(f"{exact_hash_leakage} exact hashes cross splits")
    if near_leakage:
        failures.append(f"{near_leakage} listed near-duplicate pairs cross splits")
    payload = {
        "status": "pass" if not failures else "fail",
        "counts": counts,
        "class_count": class_count,
        "patient_group_cross_split": patient_leakage,
        "exact_hash_cross_split": exact_hash_leakage,
        "listed_near_duplicate_cross_split": near_leakage,
        "failures": failures,
    }
    write_json(REPORTS / "dataset_verification.json", payload)
    print(json.dumps(payload, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
