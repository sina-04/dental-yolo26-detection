from __future__ import annotations

from collections import Counter, defaultdict
import csv
import math
from pathlib import Path
import random
import re
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler
from ultralytics.data.build import InfiniteDataLoader, seed_worker
from ultralytics.models.yolo.detect.train import DetectionTrainer
from ultralytics.utils.torch_utils import torch_distributed_zero_first


_GROUP_BY_STEM: dict[str, str] = {}
_SEED = 42


def configure_patient_groups(manifest_path: Path, seed: int) -> None:
    global _GROUP_BY_STEM, _SEED
    with manifest_path.open(encoding="utf-8", newline="") as handle:
        _GROUP_BY_STEM = {
            row["image_id"]: row["patient_group_hash"] for row in csv.DictReader(handle)
        }
    _SEED = seed


def original_stem(stem: str) -> str:
    return re.sub(r"_aug\d+$", "", stem)


class GroupAwareBatchSampler(Sampler[list[int]]):
    """Oversample rare-class patient groups without repeating a group in a batch."""

    def __init__(
        self,
        groups: list[str],
        classes: list[set[int]],
        batch_size: int,
        seed: int = 42,
        max_repeat: int = 4,
    ) -> None:
        self.batch_size = batch_size
        self.seed = seed
        self.epoch = 0
        self.indices: dict[str, list[int]] = defaultdict(list)
        self.group_classes: dict[str, set[int]] = defaultdict(set)
        for index, group in enumerate(groups):
            self.indices[group].append(index)
            self.group_classes[group].update(classes[index])
        class_groups = Counter()
        for values in self.group_classes.values():
            class_groups.update(values)
        largest = max(class_groups.values(), default=1)
        self.repeats = {
            group: max(
                [min(max_repeat, max(1, math.ceil(math.sqrt(largest / max(class_groups[class_id], 1))))) for class_id in values]
                or [1]
            )
            for group, values in self.group_classes.items()
        }
        self.sample_count = sum(self.repeats.values())

    def __len__(self) -> int:
        return math.ceil(self.sample_count / self.batch_size)

    def __iter__(self) -> Iterator[list[int]]:
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        remaining = dict(self.repeats)
        cursors = Counter()
        while remaining:
            candidates = list(remaining)
            rng.shuffle(candidates)
            candidates.sort(key=lambda group: remaining[group], reverse=True)
            selected = candidates[: self.batch_size]
            batch: list[int] = []
            for group in selected:
                options = self.indices[group]
                batch.append(options[cursors[group] % len(options)])
                cursors[group] += 1
                remaining[group] -= 1
                if remaining[group] == 0:
                    del remaining[group]
            yield batch


class GroupAwareIndexSampler(Sampler[int]):
    """Flatten patient-unique batches for a standard DataLoader batch size.

    Passing ``batch_sampler`` directly makes PyTorch expose ``batch_size=None``.
    Ultralytics 8.4.x expects a numeric ``train_loader.batch_size`` while it
    builds the training pipeline, so let DataLoader recreate the same batch
    boundaries from this flattened index stream instead.
    """

    def __init__(self, batch_sampler: GroupAwareBatchSampler) -> None:
        self.batch_sampler = batch_sampler

    def __len__(self) -> int:
        return self.batch_sampler.sample_count

    def __iter__(self) -> Iterator[int]:
        for batch in self.batch_sampler:
            yield from batch


class PatientBalancedDetectionTrainer(DetectionTrainer):
    """Single-GPU Ultralytics trainer with patient-unique weighted batches."""

    def get_dataloader(self, dataset_path: str, batch_size: int = 16, rank: int = 0, mode: str = "train"):
        if mode != "train":
            return super().get_dataloader(dataset_path, batch_size, rank, mode)
        if rank not in {-1, 0}:
            raise RuntimeError("Patient-balanced training currently supports one GPU process")
        with torch_distributed_zero_first(rank):
            dataset = self.build_dataset(dataset_path, mode, batch_size)
        groups = [
            _GROUP_BY_STEM.get(original_stem(Path(path).stem), original_stem(Path(path).stem))
            for path in dataset.im_files
        ]
        classes = [
            {int(value) for value in np.asarray(label.get("cls", [])).reshape(-1).tolist()}
            for label in dataset.labels
        ]
        batch_sampler = GroupAwareBatchSampler(groups, classes, batch_size, seed=_SEED, max_repeat=4)
        sampler = GroupAwareIndexSampler(batch_sampler)
        workers = min(self.args.workers, max(0, len(batch_sampler) - 1))
        generator = torch.Generator()
        generator.manual_seed(_SEED)
        return InfiniteDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            sampler=sampler,
            drop_last=False,
            num_workers=workers,
            collate_fn=getattr(dataset, "collate_fn", None),
            worker_init_fn=seed_worker,
            generator=generator,
            pin_memory=torch.cuda.is_available(),
            prefetch_factor=4 if workers else None,
        )
