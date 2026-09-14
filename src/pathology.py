from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
from typing import Iterable

import yaml

from src.common import IMAGE_EXTENSIONS, list_images, stable_fingerprint, write_json


# Source IDs are intentionally retained here. The generated pathology view remaps
# them to contiguous YOLO IDs in this order and records the mapping in its YAML.
PRIMARY_SOURCE_CLASSES: dict[int, str] = {
    0: "Caries",
    7: "Periapical lesion",
    8: "Retained root",
    10: "Root Piece",
    11: "impacted tooth",
    13: "Bone Loss",
}

DEFAULT_SUPPORT_MINIMUMS = {"train": 200, "val": 50, "test": 50}


def remap_boxes(lines: Iterable[str], source_ids: Iterable[int]) -> list[str]:
    """Keep selected YOLO boxes and remap source IDs to contiguous IDs."""
    mapping = {source_id: target_id for target_id, source_id in enumerate(source_ids)}
    remapped: list[str] = []
    for raw in lines:
        parts = raw.split()
        if len(parts) != 5:
            continue
        source_id = int(float(parts[0]))
        if source_id in mapping:
            remapped.append(" ".join([str(mapping[source_id]), *parts[1:]]))
    return remapped


def support_failures(
    support: dict[str, dict[str, int]],
    minimums: dict[str, int] | None = None,
) -> list[str]:
    minimums = minimums or DEFAULT_SUPPORT_MINIMUMS
    failures: list[str] = []
    for class_name, split_counts in support.items():
        for split, minimum in minimums.items():
            actual = int(split_counts.get(split, 0))
            if actual < minimum:
                failures.append(f"{class_name}/{split}: {actual} patient groups; requires {minimum}")
    return failures


def _link_or_copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        try:
            destination.symlink_to(source.resolve())
        except OSError:
            shutil.copy2(source, destination)


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def build_primary_view(
    dataset_root: Path,
    reports_root: Path,
    source_classes: dict[int, str] | None = None,
    support_minimums: dict[str, int] | None = None,
    seed: int = 42,
) -> dict[str, object]:
    """Create a disk-efficient six-class dataset view and support audit.

    Every original image is retained so images without a primary finding become
    legitimate negatives. Images are hard-linked where possible; labels are
    remapped into a dedicated view so the 31-class dataset remains unchanged.
    """
    source_classes = source_classes or PRIMARY_SOURCE_CLASSES
    support_minimums = support_minimums or DEFAULT_SUPPORT_MINIMUMS
    source_ids = list(source_classes)
    view_root = dataset_root / "views" / "pathology"
    if view_root.exists():
        shutil.rmtree(view_root)

    for split in ("train", "val", "test"):
        for image in list_images(dataset_root / "images" / split):
            _link_or_copy(image, view_root / "images" / split / image.name)
            source_label = dataset_root / "labels" / split / f"{image.stem}.txt"
            lines = source_label.read_text(encoding="utf-8").splitlines() if source_label.exists() else []
            target_label = view_root / "labels" / split / f"{image.stem}.txt"
            target_label.parent.mkdir(parents=True, exist_ok=True)
            selected = remap_boxes(lines, source_ids)
            target_label.write_text("\n".join(selected) + ("\n" if selected else ""), encoding="utf-8")

    manifest = _read_manifest(reports_root / "dataset_manifest.csv")
    by_id = {row["image_id"]: row for row in manifest}
    support_sets: dict[str, dict[str, set[str]]] = {
        name: {split: set() for split in ("train", "val", "test")} for name in source_classes.values()
    }
    for split in ("train", "val", "test"):
        for label in (dataset_root / "labels" / split).glob("*.txt"):
            if "_aug" in label.stem or label.stem not in by_id:
                continue
            row = by_id[label.stem]
            group = row["patient_group_hash"]
            classes = {
                int(line.split()[0])
                for line in label.read_text(encoding="utf-8").splitlines()
                if len(line.split()) == 5 and int(float(line.split()[0])) in source_classes
            }
            for source_id in classes:
                support_sets[source_classes[source_id]][split].add(group)

    support = {
        name: {split: len(groups) for split, groups in split_groups.items()}
        for name, split_groups in support_sets.items()
    }
    failures = support_failures(support, support_minimums)

    runtime = view_root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    all_train = list_images(view_root / "images" / "train")
    base_train = [image for image in all_train if "_aug" not in image.stem]
    # The balanced trainer consumes the full original-image list and performs
    # patient-group weighting at batch-sampling time.
    views = {"base": base_train, "augmented": all_train, "balanced": base_train}
    names = {index: name for index, name in enumerate(source_classes.values())}
    source_mapping = {index: source_id for index, source_id in enumerate(source_ids)}
    for name, images in views.items():
        train_list = runtime / f"{name}_train.txt"
        train_list.write_text("\n".join(path.resolve().as_posix() for path in images) + "\n", encoding="utf-8")
        data = {
            "path": view_root.resolve().as_posix(),
            "train": train_list.resolve().as_posix(),
            "val": "images/val",
            "test": "images/test",
            "names": names,
            "source_class_ids": source_mapping,
        }
        (runtime / f"{name}.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    (view_root / "data.yaml").write_text(
        yaml.safe_dump({"train": "images/train", "val": "images/val", "test": "images/test", "names": names, "source_class_ids": source_mapping}, sort_keys=False),
        encoding="utf-8",
    )

    payload: dict[str, object] = {
        "schema_version": 1,
        "view": "pathology",
        "seed": seed,
        "source_dataset_fingerprint": json.loads(
            (dataset_root / "fingerprint.json").read_text(encoding="utf-8")
        ).get("fingerprint"),
        "source_to_primary": {str(source_id): index for index, source_id in enumerate(source_ids)},
        "primary_names": names,
        "patient_group_support": support,
        "minimum_patient_groups": support_minimums,
        "support_gate_passed": not failures,
        "support_failures": failures,
        "training_view_counts": {name: len(images) for name, images in views.items()},
    }
    payload["fingerprint"] = stable_fingerprint(payload)
    write_json(reports_root / "pathology_support.json", payload)
    write_json(view_root / "view_manifest.json", payload)
    return payload


def require_support_gate(report_path: Path, allow_insufficient: bool = False) -> dict[str, object]:
    if not report_path.exists():
        raise FileNotFoundError(f"Pathology support report is missing: {report_path}")
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not payload.get("support_gate_passed") and not allow_insufficient:
        details = "; ".join(payload.get("support_failures", []))
        raise RuntimeError(
            "Primary pathology patient-support gate failed. Collect and review independent examinations before "
            f"a final run, or use an explicitly exploratory profile. {details}"
        )
    return payload


def require_clinician_review(path: Path, dataset_fingerprint: str) -> dict[str, object]:
    if not path.exists():
        raise FileNotFoundError(
            "Clinician pathology-review approval is missing. Complete the review and run src.approve_pathology_review."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "approved" or payload.get("dataset_fingerprint") != dataset_fingerprint:
        raise RuntimeError("Clinician pathology-review approval does not match the current dataset fingerprint")
    return payload
