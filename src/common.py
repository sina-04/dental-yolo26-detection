from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Iterable

import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def slugify(value: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "_", value.strip().lower())
    return value.strip("_")


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def opaque_id(value: str, length: int = 16) -> str:
    return hashlib.sha256(value.encode("utf-8", errors="ignore")).hexdigest()[:length]


def load_names(yaml_path: Path) -> list[str]:
    payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
    names = payload["names"]
    if isinstance(names, dict):
        return [str(names[index]) for index in sorted(names, key=lambda item: int(item))]
    return [str(name) for name in names]


def list_images(path: Path) -> list[Path]:
    return sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() in IMAGE_EXTENSIONS)


def finite_unit(values: Iterable[float]) -> bool:
    return all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in values)


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def stable_fingerprint(payload: object) -> str:
    """Return a deterministic SHA-256 fingerprint for JSON-compatible data."""
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
