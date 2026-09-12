from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_HANDLE = "lokisilvres/dental-disease-panoramic-detection-dataset/versions/6"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def download_dataset(handle: str, destination: Path) -> Path:
    marker = destination / ".codex_complete"
    if marker.is_file():
        print(f"Reusing completed Kaggle download: {destination}", flush=True)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and any(destination.iterdir()):
        has_manifest = any(destination.rglob("data.yaml"))
        has_image = any(path.suffix.lower() in {".jpg", ".jpeg", ".png"} for path in destination.rglob("*"))
        if has_manifest and has_image:
            marker.write_text(handle + "\n", encoding="utf-8")
            print(f"Reusing extracted Kaggle download: {destination}", flush=True)
            return destination
        raise FileExistsError(f"Incomplete dataset directory exists: {destination}")
    import kagglehub

    resolved = Path(kagglehub.dataset_download(handle, output_dir=str(destination)))
    marker.write_text(handle + "\n", encoding="utf-8")
    print(f"Downloaded {handle} to {resolved}", flush=True)
    return resolved


def prepared_dataset_is_valid(dataset: Path, reports: Path, expected_marker: dict[str, object] | None = None) -> bool:
    verification = reports / "dataset_verification.json"
    marker = dataset / ".colab_prepared.json"
    if not marker.exists() or not verification.exists() or not (dataset / "images" / "train").exists():
        return False
    try:
        marker_data = json.loads(marker.read_text(encoding="utf-8"))
        verification_passed = json.loads(verification.read_text(encoding="utf-8")).get("status") == "pass"
        marker_matches = expected_marker is None or all(marker_data.get(key) == value for key, value in expected_marker.items())
        return verification_passed and marker_matches
    except (json.JSONDecodeError, OSError):
        return False


def prepare(
    data_root: Path,
    augment_fraction: float,
    minority_target_instances: int,
    max_augmentations_per_image: int,
    seed: int,
    rebuild: bool,
) -> tuple[Path, Path]:
    dataset = PROJECT_ROOT / "dataset"
    reports = PROJECT_ROOT / "reports"
    marker_payload = {
        "dataset_handle": DATASET_HANDLE,
        "augment_fraction": augment_fraction,
        "minority_target_instances": minority_target_instances,
        "max_augmentations_per_image": max_augmentations_per_image,
        "seed": seed,
    }
    if prepared_dataset_is_valid(dataset, reports, marker_payload) and not rebuild:
        print("Prepared dataset already passed verification; skipping rebuild.", flush=True)
        return dataset, reports

    source = download_dataset(DATASET_HANDLE, data_root / "panoramic")
    command = [
        sys.executable,
        "-m",
        "src.prepare_dataset",
        "--source-root",
        str(source),
        "--output",
        str(dataset),
        "--reports-root",
        str(reports),
        "--augment-fraction",
        str(augment_fraction),
        "--minority-target-instances",
        str(minority_target_instances),
        "--max-augmentations-per-image",
        str(max_augmentations_per_image),
        "--seed",
        str(seed),
    ]
    command.append("--rebuild" if rebuild else "--resume")
    run(command)
    run(
        [
            sys.executable,
            "-m",
            "src.verify_dataset",
            "--dataset",
            str(dataset),
            "--reports-root",
            str(reports),
        ]
    )
    (dataset / ".colab_prepared.json").write_text(
        json.dumps(marker_payload, indent=2) + "\n",
        encoding="utf-8",
    )
    return dataset, reports


def copy_dataset_reports(source: Path, results_root: Path) -> None:
    destination = results_root / "reports"
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "class_distribution.csv",
        "dataset_audit.json",
        "dataset_manifest.csv",
        "dataset_verification.json",
        "exact_duplicates_removed.json",
        "near_duplicate_candidates.csv",
    ):
        path = source / name
        if path.exists():
            shutil.copy2(path, destination / name)


def train(args: argparse.Namespace, dataset: Path, dataset_reports: Path) -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is unavailable. Run the Colab dependency cell first.") from error
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. In Colab choose Runtime > Change runtime type > GPU.")

    gpu = torch.cuda.get_device_name(0)
    memory_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Using {gpu} with {memory_gib:.1f} GiB VRAM", flush=True)
    args.results_root.mkdir(parents=True, exist_ok=True)
    copy_dataset_reports(dataset_reports, args.results_root)
    run(
        [
            sys.executable,
            "-m",
            "src.train_evaluate",
            "--model",
            args.model,
            "--baseline-epochs",
            str(args.baseline_epochs),
            "--tuned-epochs",
            str(args.tuned_epochs),
            "--imgsz",
            str(args.imgsz),
            "--batch",
            str(args.batch),
            "--workers",
            str(args.workers),
            "--device",
            "0",
            "--seed",
            str(args.seed),
            "--data-yaml",
            str(dataset / "data.yaml"),
            "--output-root",
            str(args.results_root),
            "--run-prefix",
            args.run_prefix,
            "--cache",
            args.cache,
            "--patience",
            str(args.patience),
            "--save-period",
            "1",
            "--resume",
        ]
    )
    run(
        [
            sys.executable,
            "-m",
            "src.generate_report",
            "--results-root",
            str(args.results_root),
            "--dataset-reports-root",
            str(dataset_reports),
        ]
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Download, prepare, verify, and train the dental detector on Google Colab.")
    parser.add_argument("--stage", choices=("prepare", "train", "all"), default="all")
    parser.add_argument("--data-root", type=Path, default=Path("/content/dental_yolo26_data"))
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("/content/drive/MyDrive/dental-yolo26-detection/colab-results"),
    )
    parser.add_argument("--model", default="yolo26s.pt")
    parser.add_argument("--baseline-epochs", type=int, default=15)
    parser.add_argument("--tuned-epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--cache", choices=("false", "disk", "ram"), default="false")
    parser.add_argument("--augment-fraction", type=float, default=0.15)
    parser.add_argument("--minority-target-instances", type=int, default=128)
    parser.add_argument("--max-augmentations-per-image", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-prefix", default="colab_full")
    parser.add_argument("--rebuild-data", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.data_root = args.data_root.resolve()
    args.results_root = args.results_root.resolve()
    dataset = PROJECT_ROOT / "dataset"
    dataset_reports = PROJECT_ROOT / "reports"
    if args.stage in {"prepare", "all"}:
        dataset, dataset_reports = prepare(
            args.data_root,
            args.augment_fraction,
            args.minority_target_instances,
            args.max_augmentations_per_image,
            args.seed,
            args.rebuild_data,
        )
    if args.stage in {"train", "all"}:
        if not prepared_dataset_is_valid(dataset, dataset_reports):
            raise RuntimeError("The dataset is not prepared and verified. Run with --stage prepare or --stage all.")
        train(args, dataset, dataset_reports)


if __name__ == "__main__":
    main()
