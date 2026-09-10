"""Download a large HTTP resource with resumable byte-range segments."""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time

import requests


def _resolve(url: str) -> tuple[str, int]:
    with requests.get(url, allow_redirects=False, stream=True, timeout=60) as response:
        response.raise_for_status()
        target = response.headers.get("Location", url)
    head = requests.head(target, timeout=60)
    head.raise_for_status()
    size = int(head.headers["Content-Length"])
    if head.headers.get("Accept-Ranges", "").lower() != "bytes":
        raise RuntimeError("Server does not advertise byte-range downloads")
    return target, size


def download(url: str, output: Path, workers: int) -> None:
    target, size = _resolve(url)
    part_dir = output.with_suffix(output.suffix + f".parts-{workers}")
    part_dir.mkdir(parents=True, exist_ok=True)
    chunk_size = math.ceil(size / workers)
    lock = threading.Lock()
    started = time.monotonic()

    def fetch(index: int) -> Path:
        start = index * chunk_size
        end = min(size - 1, start + chunk_size - 1)
        part = part_dir / f"{index:03d}.part"
        expected = end - start + 1
        existing = part.stat().st_size if part.exists() else 0
        if existing == expected:
            return part
        if existing > expected:
            part.unlink()
            existing = 0
        resume = part.with_suffix(".resume")
        if resume.exists():
            with part.open("ab") as destination, resume.open("rb") as source:
                shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
            resume.unlink()
            existing = part.stat().st_size
        command = [
                "curl.exe",
                "--fail",
                "--location",
                "--retry",
                "5",
                "--retry-delay",
                "3",
                "--silent",
                "--show-error",
                "--range",
                f"{start + existing}-{end}",
                "--output",
                str(resume),
                target,
            ]
        completed = subprocess.run(command, check=False)
        if resume.exists():
            with part.open("ab") as destination, resume.open("rb") as source:
                shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
            resume.unlink()
        if completed.returncode:
            raise subprocess.CalledProcessError(completed.returncode, command)
        actual = part.stat().st_size
        if actual != expected:
            raise RuntimeError(f"Segment {index} is {actual} bytes; expected {expected}")
        with lock:
            downloaded = sum(p.stat().st_size for p in part_dir.glob("*.part"))
            elapsed = max(time.monotonic() - started, 0.001)
            print(
                f"segment {index + 1}/{workers} complete; "
                f"{downloaded / size:.1%}; {downloaded / elapsed / 1024 / 1024:.2f} MiB/s",
                flush=True,
            )
        return part

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(fetch, range(workers)))

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".assembling")
    with temporary.open("wb") as destination:
        for part in parts:
            with part.open("rb") as source:
                shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
    if temporary.stat().st_size != size:
        raise RuntimeError("Assembled file size does not match the remote resource")
    os.replace(temporary, output)
    for part in parts:
        part.unlink()
    part_dir.rmdir()
    print(f"saved {output} ({size} bytes)", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("url")
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    download(args.url, args.output, args.workers)


if __name__ == "__main__":
    main()
