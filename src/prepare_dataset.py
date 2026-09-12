from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
import concurrent.futures
from dataclasses import dataclass, field
import json
import inspect
import math
from pathlib import Path
import random
import re
import shutil

import albumentations as A
import cv2
import imagehash
from PIL import Image, ImageDraw
import yaml

from src.common import IMAGE_EXTENSIONS, finite_unit, list_images, load_names, opaque_id, sha256_file, stable_fingerprint, write_json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = PROJECT_ROOT / "dataset"
REPORT_ROOT = PROJECT_ROOT / "reports"
SEED = 42
DATASET_HANDLE = "lokisilvres/dental-disease-panoramic-detection-dataset/versions/6"
GROUPING_VERSION = "panoramic-filename-phash-v2"
SPLIT_VERSION = "greedy-multilabel-75-15-10-v2"
SPLIT_TARGETS = {"train": 0.75, "val": 0.15, "test": 0.10}


def dataset_class_count(path: Path) -> int | None:
    """Return the number of classes in a YOLO data file, or None if invalid."""
    try:
        return len(load_names(path / "data.yaml"))
    except (FileNotFoundError, KeyError, TypeError, ValueError, yaml.YAMLError):
        return None


def resolve_dataset_root(explicit: Path | None, candidates: list[Path], expected_classes: int, label: str) -> Path:
    """Locate a YOLO dataset without relying on a machine-specific extraction path."""
    search_roots = [explicit] if explicit else candidates
    matches: list[Path] = []
    for search_root in search_roots:
        if search_root is None or not search_root.exists():
            continue
        direct = search_root.resolve()
        if dataset_class_count(direct) == expected_classes:
            matches.append(direct)
        for data_yaml in search_root.rglob("data.yaml"):
            root = data_yaml.parent.resolve()
            if dataset_class_count(root) == expected_classes:
                matches.append(root)
    unique = sorted(set(matches), key=lambda path: (len(path.parts), str(path)))
    if not unique:
        searched = ", ".join(str(path) for path in search_roots if path is not None)
        raise FileNotFoundError(f"Could not locate the {label} YOLO dataset ({expected_classes} classes) under: {searched}")
    return unique[0]


@dataclass
class Box:
    class_id: int
    x: float
    y: float
    w: float
    h: float

    def line(self) -> str:
        return f"{self.class_id} {self.x:.6f} {self.y:.6f} {self.w:.6f} {self.h:.6f}"


@dataclass
class Record:
    source: str
    source_split: str
    image: Path
    label: Path | None
    patient_group: str
    boxes: list[Box] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)
    sha256: str = ""
    phash: str = ""
    width: int = 0
    height: int = 0
    output_id: str = ""
    split: str = ""


def patient_key(stem: str) -> str:
    base = re.split(r"\.rf\.", stem, maxsplit=1, flags=re.IGNORECASE)[0]
    cleaned = re.sub(r"^[0-9a-f]{8}-", "", base, flags=re.IGNORECASE)
    tokens = re.split(r"[_-]+", cleaned)
    patient_tokens: list[str] = []
    for token in tokens:
        if re.fullmatch(r"(?:19|20)\d{2}", token) or re.search(r"\d{6,}", token):
            break
        if token.lower() in {"jpg", "jpeg", "png", "xray", "opg"}:
            continue
        if not any(character.isdigit() for character in token):
            patient_tokens.append(token.lower())
    identity = "_".join(patient_tokens) or base
    return f"panoramic_{opaque_id(identity, 20)}"


def parse_label(path: Path | None, class_count: int) -> tuple[list[Box], list[str], int]:
    boxes: list[Box] = []
    issues: list[str] = []
    polygon_count = 0
    if path is None or not path.exists():
        return boxes, ["missing_label"], polygon_count
    for line_number, raw in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        raw = raw.strip()
        if not raw:
            continue
        parts = raw.split()
        try:
            local_class = int(float(parts[0]))
            values = [float(item) for item in parts[1:]]
        except (ValueError, IndexError):
            issues.append(f"malformed_line:{line_number}")
            continue
        if not 0 <= local_class < class_count:
            issues.append(f"invalid_class:{line_number}:{local_class}")
            continue
        if len(values) == 4:
            x, y, width, height = values
        elif len(values) >= 6 and len(values) % 2 == 0:
            xs, ys = values[0::2], values[1::2]
            x_min, x_max = min(xs), max(xs)
            y_min, y_max = min(ys), max(ys)
            x, y = (x_min + x_max) / 2, (y_min + y_max) / 2
            width, height = x_max - x_min, y_max - y_min
            polygon_count += 1
        else:
            issues.append(f"invalid_coordinate_count:{line_number}:{len(values)}")
            continue
        if not finite_unit((x, y, width, height)):
            issues.append(f"out_of_bounds:{line_number}")
            continue
        x_min, x_max = x - width / 2, x + width / 2
        y_min, y_max = y - height / 2, y + height / 2
        if width <= 0 or height <= 0:
            issues.append(f"nonpositive_box:{line_number}")
            continue
        if x_min < 0 or y_min < 0 or x_max > 1 or y_max > 1:
            x_min, x_max = max(0.0, x_min), min(1.0, x_max)
            y_min, y_max = max(0.0, y_min), min(1.0, y_max)
            x, y = (x_min + x_max) / 2, (y_min + y_max) / 2
            width, height = x_max - x_min, y_max - y_min
            issues.append(f"clipped_box:{line_number}")
        if width * height < 1e-5:
            issues.append(f"tiny_box:{line_number}")
        boxes.append(Box(local_class, x, y, width, height))
    if not boxes:
        issues.append("empty_annotation")
    return boxes, issues, polygon_count


def locate_label(label_dir: Path, image: Path) -> Path | None:
    candidate = label_dir / f"{image.stem}.txt"
    return candidate if candidate.exists() else None


def collect_source(root: Path, class_count: int) -> tuple[list[Record], int]:
    inputs: list[tuple[str, Path, Path]] = []
    for split_name in ("train", "valid", "val", "test"):
        split_root = root / split_name
        if not split_root.exists():
            continue
        image_dir, label_dir = split_root / "images", split_root / "labels"
        for image_path in list_images(image_dir):
            inputs.append((split_name, image_path, label_dir))

    def inspect(item: tuple[str, Path, Path]) -> tuple[Record, int]:
        split_name, image_path, label_dir = item
        label_path = locate_label(label_dir, image_path)
        boxes, issues, polygon_count = parse_label(label_path, class_count)
        record = Record(
                source="panoramic",
                source_split="val" if split_name == "valid" else split_name,
                image=image_path,
                label=label_path,
                patient_group=patient_key(image_path.stem),
                boxes=boxes,
                issues=issues,
            )
        try:
            with Image.open(image_path) as opened:
                opened.verify()
            with Image.open(image_path) as opened:
                record.width, record.height = opened.size
                record.phash = str(imagehash.phash(opened.convert("L")))
            record.sha256 = sha256_file(image_path)
        except Exception as error:  # corrupted inputs are recorded and excluded
            record.issues.append(f"corrupt_image:{type(error).__name__}")
        return record, polygon_count

    records: list[Record] = []
    polygon_total = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, (len(inputs) or 1))) as executor:
        for index, (record, polygon_count) in enumerate(executor.map(inspect, inputs), start=1):
            records.append(record)
            polygon_total += polygon_count
            if index % 1000 == 0:
                print(f"panoramic: inspected {index}/{len(inputs)} images", flush=True)
    return records, polygon_total


def remove_exact_duplicates(records: list[Record]) -> tuple[list[Record], list[dict[str, str]]]:
    groups: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        if record.sha256:
            groups[record.sha256].append(record)
    removed: list[dict[str, str]] = []
    keep_ids: set[int] = set()
    for digest, duplicates in groups.items():
        duplicates.sort(key=lambda record: (-len(record.boxes), str(record.image)))
        keeper = duplicates[0]
        keep_ids.add(id(keeper))
        for duplicate in duplicates[1:]:
            removed.append({"source": duplicate.source, "sha256": digest, "kept": opaque_id(str(keeper.image)), "removed": opaque_id(str(duplicate.image))})
    unique = [record for record in records if not record.sha256 or id(record) in keep_ids]
    return unique, removed


def merge_identical_perceptual_groups(records: list[Record]) -> int:
    groups: dict[tuple[str, str], list[Record]] = defaultdict(list)
    for record in records:
        if record.phash:
            groups[(record.source, record.phash)].append(record)
    changed = 0
    for duplicates in groups.values():
        patient_groups = {record.patient_group for record in duplicates}
        if len(patient_groups) <= 1:
            continue
        merged = f"{duplicates[0].source}_{opaque_id('|'.join(sorted(patient_groups)), 20)}"
        for record in duplicates:
            if record.patient_group != merged:
                record.patient_group = merged
                changed += 1
    return changed


def find_near_duplicate_pairs(records: list[Record], max_distance: int = 3) -> list[tuple[Record, Record, int]]:
    """Find pHash-near images without an O(n²) full scan.

    A 64-bit hash within distance three must share at least one of four
    16-bit segments. Segment buckets therefore produce a complete candidate
    set for the configured distance while keeping the full dataset practical.
    """
    buckets: dict[tuple[int, int], list[int]] = defaultdict(list)
    hashed = [(record, int(record.phash, 16)) for record in records if record.phash]
    pairs: list[tuple[Record, Record, int]] = []
    seen: set[tuple[int, int]] = set()
    for right_index, (right, right_hash) in enumerate(hashed):
        candidates: set[int] = set()
        for segment in range(4):
            value = (right_hash >> (segment * 16)) & 0xFFFF
            candidates.update(buckets[(segment, value)])
        for left_index in candidates:
            key = (left_index, right_index)
            if key in seen:
                continue
            seen.add(key)
            left, left_hash = hashed[left_index]
            distance = (left_hash ^ right_hash).bit_count()
            if distance <= max_distance and left.sha256 != right.sha256:
                pairs.append((left, right, distance))
        for segment in range(4):
            value = (right_hash >> (segment * 16)) & 0xFFFF
            buckets[(segment, value)].append(right_index)
    return pairs


def merge_near_duplicate_groups(records: list[Record], pairs: list[tuple[Record, Record, int]]) -> int:
    """Transitively place perceptually near images in one split group."""
    index_by_record = {id(record): index for index, record in enumerate(records)}
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for left, right, _ in pairs:
        union(index_by_record[id(left)], index_by_record[id(right)])
    clusters: dict[int, list[Record]] = defaultdict(list)
    for index, record in enumerate(records):
        clusters[find(index)].append(record)
    changed = 0
    for cluster in clusters.values():
        groups = sorted({record.patient_group for record in cluster})
        if len(groups) <= 1:
            continue
        merged = f"panoramic_{opaque_id('|'.join(groups), 20)}"
        for record in cluster:
            if record.patient_group != merged:
                record.patient_group = merged
                changed += 1
    return changed


def near_duplicate_rows(
    pairs: list[tuple[Record, Record, int]], limit: int | None = None
) -> list[dict[str, object]]:
    selected = pairs if limit is None else pairs[:limit]
    return [
        {
            "left": left.output_id,
            "right": right.output_id,
            "distance": distance,
            "same_patient_group": left.patient_group == right.patient_group,
        }
        for left, right, distance in selected
    ]


def assign_splits(records: list[Record], class_count: int) -> dict[str, object]:
    targets = SPLIT_TARGETS
    grouped: dict[str, list[Record]] = defaultdict(list)
    for record in records:
        grouped[record.patient_group].append(record)
    total_images = len(records)
    total_classes = Counter(box.class_id for record in records for box in record.boxes)
    state = {split: {"images": 0, "classes": Counter()} for split in targets}

    def rarity(group_records: list[Record]) -> float:
        present = {box.class_id for record in group_records for box in record.boxes}
        return sum(1 / max(total_classes[class_id], 1) for class_id in present)

    class_groups: dict[int, set[str]] = {class_id: set() for class_id in range(class_count)}
    for group, members in grouped.items():
        for class_id in {box.class_id for record in members for box in record.boxes}:
            class_groups[class_id].add(group)
    forced_train_classes = [class_id for class_id, groups in class_groups.items() if 0 < len(groups) < 3]
    forced_train_groups = set().union(*(class_groups[class_id] for class_id in forced_train_classes)) if forced_train_classes else set()
    for group in sorted(forced_train_groups):
        members = grouped[group]
        for record in members:
            record.split = "train"
        state["train"]["images"] += len(members)
        state["train"]["classes"].update(box.class_id for record in members for box in record.boxes)

    groups = sorted(
        ((group, members) for group, members in grouped.items() if group not in forced_train_groups),
        key=lambda item: (-rarity(item[1]), -len(item[1]), item[0]),
    )
    for group_name, group_records in groups:
        group_classes = Counter(box.class_id for record in group_records for box in record.boxes)
        best_split = "train"
        best_score = float("inf")
        for candidate, _ in targets.items():
            image_score = 0.0
            class_score = 0.0
            for split, fraction in targets.items():
                image_after = state[split]["images"] + (len(group_records) if split == candidate else 0)
                image_target = total_images * fraction
                image_score += ((image_after - image_target) / max(total_images, 1)) ** 2
                for class_id, total in total_classes.items():
                    class_after = state[split]["classes"][class_id] + (group_classes[class_id] if split == candidate else 0)
                    class_target = total * fraction
                    class_score += ((class_after - class_target) / max(total, 1)) ** 2
            score = image_score + class_score / max(class_count, 1)
            if score < best_score:
                best_score, best_split = score, candidate
        for record in group_records:
            record.split = best_split
        state[best_split]["images"] += len(group_records)
        state[best_split]["classes"].update(group_classes)

    # Ensure every class represented by at least three independent groups appears
    # in all three splits. Move the smallest safe group and keep the global ratio
    # close to target; classes with fewer groups are inherently unevaluable in all splits.
    group_classes = {group: {box.class_id for record in members for box in record.boxes} for group, members in grouped.items()}
    coverage: dict[int, dict[str, set[str]]] = {
        class_id: {split: set() for split in targets} for class_id in range(class_count)
    }
    for group, members in grouped.items():
        split = members[0].split
        for class_id in group_classes[group]:
            coverage[class_id][split].add(group)
    for class_id in sorted(range(class_count), key=lambda item: sum(len(groups) for groups in coverage[item].values())):
        if sum(len(groups) for groups in coverage[class_id].values()) < 3:
            continue
        for missing_split in [split for split in targets if not coverage[class_id][split]]:
            candidates: list[tuple[int, int, str, str]] = []
            for donor in targets:
                if donor == missing_split or len(coverage[class_id][donor]) <= 1:
                    continue
                for group in coverage[class_id][donor]:
                    if group in forced_train_groups:
                        continue
                    lost_singletons = sum(len(coverage[other][donor]) == 1 for other in group_classes[group])
                    if lost_singletons == 0:
                        candidates.append((lost_singletons, len(grouped[group]), group, donor))
            if not candidates:
                continue
            _, _, selected_group, donor = min(candidates)
            for record in grouped[selected_group]:
                record.split = missing_split
            for other in group_classes[selected_group]:
                coverage[other][donor].discard(selected_group)
                coverage[other][missing_split].add(selected_group)
    return {
        "class_group_counts": {str(class_id): len(groups) for class_id, groups in class_groups.items()},
        "forced_train_classes": forced_train_classes,
        "forced_train_groups": len(forced_train_groups),
    }


def save_box_labels(path: Path, boxes: list[Box]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(box.line() for box in boxes) + ("\n" if boxes else ""), encoding="utf-8")


def materialize(records: list[Record], output_root: Path) -> None:
    for split in ("train", "val", "test"):
        (output_root / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_root / "labels" / split).mkdir(parents=True, exist_ok=True)
    counters = Counter()
    for record in sorted(records, key=lambda item: (item.source, str(item.image))):
        counters[record.source] += 1
        record.output_id = f"{record.source}_{counters[record.source]:06d}_{opaque_id(str(record.image), 8)}"
        extension = record.image.suffix.lower() if record.image.suffix.lower() in IMAGE_EXTENSIONS else ".jpg"
        shutil.copy2(record.image, output_root / "images" / record.split / f"{record.output_id}{extension}")
        save_box_labels(output_root / "labels" / record.split / f"{record.output_id}.txt", record.boxes)


def build_augmentation(seed: int) -> A.Compose:
    """Create the same conservative policy for Albumentations and AlbumentationsX."""
    if "angle_range" in inspect.signature(A.Rotate).parameters:
        rotate = A.Rotate(angle_range=(-7, 7), border_mode=cv2.BORDER_CONSTANT, p=0.70)
        brightness = A.RandomBrightnessContrast(
            brightness_range=(-0.10, 0.10), contrast_range=(-0.10, 0.10), p=0.65
        )
        blur = A.GaussianBlur(blur_range=(3, 3), p=1.0)
        clahe = A.CLAHE(clip_range=(1.0, 2.0), p=1.0)
        bbox_params = A.BboxParams(coord_format="yolo", label_fields=["class_labels"], min_visibility=0.70)
    else:
        rotate = A.Rotate(limit=(-7, 7), border_mode=cv2.BORDER_CONSTANT, p=0.70)
        brightness = A.RandomBrightnessContrast(
            brightness_limit=(-0.10, 0.10), contrast_limit=(-0.10, 0.10), p=0.65
        )
        blur = A.GaussianBlur(blur_limit=(3, 3), p=1.0)
        clahe = A.CLAHE(clip_limit=(1.0, 2.0), p=1.0)
        bbox_params = A.BboxParams(format="yolo", label_fields=["class_labels"], min_visibility=0.70)
    return A.Compose([rotate, brightness, A.OneOf([blur, clahe], p=0.25)], bbox_params=bbox_params, seed=seed)


def augmentation_copy_plan(
    records: list[Record],
    fraction: float,
    minority_target_instances: int,
    max_augmentations_per_image: int,
    seed: int,
) -> dict[str, int]:
    """Plan deterministic train-only augmentation with extra emphasis on rare classes.

    The cap prevents a single patient image from dominating training even when a
    class has only one example. Augmentation improves invariance, but it does not
    pretend that transformed copies are independent clinical observations.
    """
    candidates = [record for record in records if record.split == "train" and record.boxes]
    shuffled = list(candidates)
    random.Random(seed).shuffle(shuffled)
    copies = {record.output_id: 0 for record in candidates}
    uniform_count = max(0, round(len(shuffled) * max(0.0, fraction)))
    for record in shuffled[:uniform_count]:
        copies[record.output_id] = 1

    if minority_target_instances <= 0 or max_augmentations_per_image <= 0:
        return copies
    class_counts = Counter(box.class_id for record in candidates for box in record.boxes)
    for record in candidates:
        required = 0
        for class_id in {box.class_id for box in record.boxes}:
            count = class_counts[class_id]
            if count < minority_target_instances:
                required = max(required, math.ceil(minority_target_instances / max(1, count)) - 1)
        copies[record.output_id] = max(copies[record.output_id], min(max_augmentations_per_image, required))
    return copies


def augment_training(
    records: list[Record],
    output_root: Path,
    fraction: float,
    minority_target_instances: int,
    max_augmentations_per_image: int,
    seed: int,
) -> int:
    if fraction <= 0 and minority_target_instances <= 0:
        return 0
    transform = build_augmentation(seed)
    candidates = [record for record in records if record.split == "train" and record.boxes]
    copy_plan = augmentation_copy_plan(
        records,
        fraction,
        minority_target_instances,
        max_augmentations_per_image,
        seed,
    )
    created = 0
    for record in candidates:
        source = next((path for path in (output_root / "images" / "train").glob(f"{record.output_id}.*")), None)
        if source is None:
            continue
        image = cv2.imread(str(source), cv2.IMREAD_COLOR)
        if image is None:
            continue
        bboxes = [(box.x, box.y, box.w, box.h) for box in record.boxes]
        classes = [box.class_id for box in record.boxes]
        for copy_index in range(1, copy_plan.get(record.output_id, 0) + 1):
            augmented = transform(image=image, bboxes=bboxes, class_labels=classes)
            if len(augmented["bboxes"]) == 0:
                continue
            output_id = f"{record.output_id}_aug{copy_index}"
            image_path = output_root / "images" / "train" / f"{output_id}.jpg"
            cv2.imwrite(str(image_path), augmented["image"], [cv2.IMWRITE_JPEG_QUALITY, 95])
            boxes = [Box(int(class_id), *map(float, bbox)) for bbox, class_id in zip(augmented["bboxes"], augmented["class_labels"], strict=True)]
            save_box_labels(output_root / "labels" / "train" / f"{output_id}.txt", boxes)
            created += 1
    return created


def draw_contact_sheet(records: list[Record], names: list[str], class_id: int, output: Path, seed: int) -> int:
    candidates = [record for record in records if any(box.class_id == class_id for box in record.boxes)]
    random.Random(seed + class_id).shuffle(candidates)
    selected = candidates[:5]
    if not selected:
        return 0
    panels: list[Image.Image] = []
    for record in selected:
        with Image.open(record.image).convert("RGB") as opened:
            image = opened.copy()
        draw = ImageDraw.Draw(image)
        for box in record.boxes:
            x1 = (box.x - box.w / 2) * image.width
            y1 = (box.y - box.h / 2) * image.height
            x2 = (box.x + box.w / 2) * image.width
            y2 = (box.y + box.h / 2) * image.height
            color = (255, 60, 60) if box.class_id == class_id else (80, 190, 255)
            draw.rectangle((x1, y1, x2, y2), outline=color, width=max(2, image.width // 500))
        image.thumbnail((720, 360))
        panel = Image.new("RGB", (740, 390), "black")
        panel.paste(image, ((740 - image.width) // 2, 25))
        ImageDraw.Draw(panel).text((8, 6), f"{record.output_id} | target: {names[class_id]}", fill="white")
        panels.append(panel)
    sheet = Image.new("RGB", (740, 390 * len(panels)), "black")
    for index, panel in enumerate(panels):
        sheet.paste(panel, (0, index * 390))
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output, quality=92)
    return len(selected)


def write_training_views(output_root: Path, names: list[str]) -> dict[str, int]:
    runtime = output_root / "runtime"
    runtime.mkdir(parents=True, exist_ok=True)
    train_images = sorted(list_images(output_root / "images" / "train"))
    base_images = [path for path in train_images if "_aug" not in path.stem]
    views = {"base": base_images, "augmented": train_images}
    counts: dict[str, int] = {}
    for name, images in views.items():
        train_list = runtime / f"{name}_train.txt"
        train_list.write_text("\n".join(path.resolve().as_posix() for path in images) + "\n", encoding="utf-8")
        configuration = {
            "path": output_root.resolve().as_posix(),
            "train": train_list.resolve().as_posix(),
            "val": "images/val",
            "test": "images/test",
            "names": {index: class_name for index, class_name in enumerate(names)},
        }
        (runtime / f"{name}.yaml").write_text(
            yaml.safe_dump(configuration, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        counts[name] = len(images)
    return counts


def native_split_overlap(records: list[Record]) -> dict[str, int]:
    groups: dict[str, set[str]] = defaultdict(set)
    for record in records:
        groups[record.source_split].add(record.patient_group)
    return {
        "train_val": len(groups["train"] & groups["val"]),
        "train_test": len(groups["train"] & groups["test"]),
        "val_test": len(groups["val"] & groups["test"]),
    }


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit, split, and prepare the Dental X-Ray Panoramic Dataset.")
    parser.add_argument("--output", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--reports-root", type=Path, default=REPORT_ROOT)
    parser.add_argument("--source-root", type=Path, help="Extracted panoramic YOLO dataset or a parent directory containing it.")
    parser.add_argument("--augment-fraction", type=float, default=0.15)
    parser.add_argument("--minority-target-instances", type=int, default=128)
    parser.add_argument("--max-augmentations-per-image", type=int, default=8)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--resume", action="store_true", help="Reuse/overwrite a previously generated output without deleting it.")
    parser.add_argument("--rebuild", action="store_true", help="Delete and regenerate only the project dataset output.")
    args = parser.parse_args()
    if args.rebuild and args.output.exists():
        if args.output.resolve() != OUTPUT_ROOT.resolve():
            raise ValueError("--rebuild is restricted to the generated project dataset directory")
        shutil.rmtree(args.output)
    if args.output.exists() and any(args.output.iterdir()) and not args.resume:
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {args.output}")
    if args.resume:
        for generated_directory in ("images", "labels", "runtime"):
            path = args.output / generated_directory
            if path.exists():
                shutil.rmtree(path)
        annotation_audit = args.reports_root / "annotation_audit"
        if annotation_audit.exists():
            shutil.rmtree(annotation_audit)

    source_root = resolve_dataset_root(
        args.source_root,
        [PROJECT_ROOT / "Dental X-Ray Panoramic Dataset", PROJECT_ROOT / "raw" / "dental-xray-panoramic"],
        expected_classes=31,
        label="Dental X-Ray Panoramic Dataset",
    )
    names = [name.strip() for name in load_names(source_root / "data.yaml")]
    contract = {
        "dataset_handle": DATASET_HANDLE,
        "source_data_yaml_sha256": sha256_file(source_root / "data.yaml"),
        "class_names": names,
        "grouping_version": GROUPING_VERSION,
        "split_version": SPLIT_VERSION,
        "split_targets": SPLIT_TARGETS,
        "seed": args.seed,
        "augmentation": {
            "fraction": args.augment_fraction,
            "minority_target_instances": args.minority_target_instances,
            "max_augmentations_per_image": args.max_augmentations_per_image,
        },
    }
    panoramic_records, polygon_annotations = collect_source(source_root, len(names))
    publisher_split_overlap = native_split_overlap(panoramic_records)
    all_records = [record for record in panoramic_records if not any(issue.startswith("corrupt_image") for issue in record.issues)]
    all_records, exact_duplicates = remove_exact_duplicates(all_records)
    contract["source_content_digest"] = stable_fingerprint([
        {
            "sha256": record.sha256,
            "phash": record.phash,
            "patient_group_hash": record.patient_group,
            "publisher_split": record.source_split,
            "boxes": [box.line() for box in record.boxes],
        }
        for record in sorted(all_records, key=lambda item: (item.sha256, item.phash, item.patient_group))
    ])
    dataset_fingerprint = stable_fingerprint(contract)
    perceptual_group_merges = merge_identical_perceptual_groups(all_records)
    near_pairs = find_near_duplicate_pairs(all_records)
    near_group_merges = merge_near_duplicate_groups(all_records, near_pairs)
    split_metadata = assign_splits(all_records, len(names))
    materialize(all_records, args.output)
    augmented_count = augment_training(
        all_records,
        args.output,
        args.augment_fraction,
        args.minority_target_instances,
        args.max_augmentations_per_image,
        args.seed,
    )
    training_view_counts = write_training_views(args.output, names)

    data_yaml = {
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {index: name for index, name in enumerate(names)},
    }
    (args.output / "data.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False, allow_unicode=True), encoding="utf-8")
    write_json(args.output / "fingerprint.json", {"fingerprint": dataset_fingerprint, "contract": contract})

    audit_sample_counts: dict[str, int] = {}
    for class_id in range(len(names)):
        count = draw_contact_sheet(
            all_records,
            names,
            class_id,
            args.reports_root / "annotation_audit" / f"class_{class_id:02d}.jpg",
            args.seed,
        )
        audit_sample_counts[str(class_id)] = count

    class_rows: list[dict[str, object]] = []
    for class_id, name in enumerate(names):
        for split in ("all", "train", "val", "test"):
            subset = all_records if split == "all" else [record for record in all_records if record.split == split]
            class_rows.append({
                "class_id": class_id,
                "class_name": name,
                "split": split,
                "images": sum(any(box.class_id == class_id for box in record.boxes) for record in subset),
                "instances": sum(box.class_id == class_id for record in subset for box in record.boxes),
            })
    write_csv(args.reports_root / "class_distribution.csv", class_rows, ["class_id", "class_name", "split", "images", "instances"])

    manifest_rows = [{
        "image_id": record.output_id,
        "source": record.source,
        "original_split": record.source_split,
        "patient_group_hash": record.patient_group,
        "split": record.split,
        "width": record.width,
        "height": record.height,
        "instances": len(record.boxes),
        "issues": "|".join(record.issues),
        "sha256": record.sha256,
        "phash": record.phash,
    } for record in all_records]
    write_csv(args.reports_root / "dataset_manifest.csv", manifest_rows, list(manifest_rows[0]))

    near_rows = near_duplicate_rows(near_pairs)
    write_csv(args.reports_root / "near_duplicate_candidates.csv", near_rows, ["left", "right", "distance", "same_patient_group"])
    write_json(args.reports_root / "exact_duplicates_removed.json", exact_duplicates)

    issue_counts = Counter(issue.split(":", 1)[0] for record in all_records for issue in record.issues)
    split_counts = Counter(record.split for record in all_records)
    source_counts = Counter(record.source for record in all_records)
    patient_splits: dict[str, set[str]] = defaultdict(set)
    for record in all_records:
        patient_splits[record.patient_group].add(record.split)
    patient_leakage = sum(len(splits) > 1 for splits in patient_splits.values())
    audit = {
        "seed": args.seed,
        "dataset": "Dental X-Ray Panoramic Dataset",
        "dataset_handle": DATASET_HANDLE,
        "dataset_version": 6,
        "license": "Apache 2.0",
        "dataset_fingerprint": dataset_fingerprint,
        "dataset_contract": contract,
        "source_counts": dict(source_counts),
        "publisher_split_group_overlap": publisher_split_overlap,
        "split_counts": dict(split_counts),
        "class_count": len(names),
        "patient_groups": len(patient_splits),
        "patient_group_leakage": patient_leakage,
        "exact_duplicates_removed": len(exact_duplicates),
        "near_duplicate_pairs": len(near_pairs),
        "near_duplicate_pairs_reported": len(near_rows),
        "records_regrouped_by_identical_perceptual_hash": perceptual_group_merges,
        "records_regrouped_by_near_duplicate_hash": near_group_merges,
        "polygon_annotations_converted": polygon_annotations,
        "augmented_training_images": augmented_count,
        "training_view_counts": training_view_counts,
        "split_metadata": split_metadata,
        "minority_target_instances": args.minority_target_instances,
        "max_augmentations_per_image": args.max_augmentations_per_image,
        "issue_counts": dict(issue_counts),
        "images_after_exact_deduplication": len(all_records),
        "instances": sum(len(record.boxes) for record in all_records),
        "raw_filename_privacy": "Raw panoramic filenames may contain identifying text. Processed filenames and manifests use opaque IDs only.",
        "manual_audit_status": "pending",
        "manual_audit_samples_per_class": audit_sample_counts,
    }
    write_json(args.reports_root / "dataset_audit.json", audit)
    write_json(
        args.reports_root / "manual_audit_status.json",
        {
            "status": "pending",
            "dataset_fingerprint": dataset_fingerprint,
            "instructions": "Review every reports/annotation_audit/class_XX.jpg sheet before approving training.",
            "samples_per_class": audit_sample_counts,
        },
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
