from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_HANDLE = "lokisilvres/dental-disease-panoramic-detection-dataset/versions/6"
DEFAULT_RESULTS_ROOT = Path("/content/drive/MyDrive/dental-yolo26-detection/panoramic31-yolo26s-t4-v1")
DEFAULT_PROFILE = PROJECT_ROOT / "configs" / "colab_t4.yaml"


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


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
    fingerprint = dataset / "fingerprint.json"
    required = (marker, verification, fingerprint, dataset / "images" / "train", dataset / "runtime" / "base.yaml")
    if not all(path.exists() for path in required):
        return False
    try:
        marker_data = read_json(marker)
        fingerprint_value = read_json(fingerprint).get("fingerprint")
        verification_passed = read_json(verification).get("status") == "pass"
        marker_matches = expected_marker is None or all(marker_data.get(key) == value for key, value in expected_marker.items())
        return bool(
            verification_passed
            and fingerprint_value
            and marker_data.get("dataset_fingerprint") == fingerprint_value
            and marker_matches
        )
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
    marker_payload: dict[str, object] = {
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
        sys.executable, "-m", "src.prepare_dataset", "--source-root", str(source),
        "--output", str(dataset), "--reports-root", str(reports),
        "--augment-fraction", str(augment_fraction),
        "--minority-target-instances", str(minority_target_instances),
        "--max-augmentations-per-image", str(max_augmentations_per_image),
        "--seed", str(seed), "--rebuild" if rebuild else "--resume",
    ]
    run(command)
    run([sys.executable, "-m", "src.verify_dataset", "--dataset", str(dataset), "--reports-root", str(reports)])
    marker_payload["dataset_fingerprint"] = read_json(dataset / "fingerprint.json")["fingerprint"]
    (dataset / ".colab_prepared.json").write_text(json.dumps(marker_payload, indent=2) + "\n", encoding="utf-8")
    return dataset, reports


def copy_dataset_reports(source: Path, results_root: Path) -> None:
    destination = results_root / "reports"
    destination.mkdir(parents=True, exist_ok=True)
    for name in (
        "class_distribution.csv", "dataset_audit.json", "dataset_manifest.csv",
        "dataset_verification.json", "exact_duplicates_removed.json",
        "near_duplicate_candidates.csv", "manual_audit_status.json",
        "manual_audit_approval.json",
    ):
        path = source / name
        if path.exists():
            shutil.copy2(path, destination / name)
    sheets = source / "annotation_audit"
    if sheets.exists():
        shutil.copytree(sheets, destination / "annotation_audit", dirs_exist_ok=True)


def audit_is_approved(dataset: Path, reports: Path, results_root: Path) -> bool:
    fingerprint = read_json(dataset / "fingerprint.json").get("fingerprint")
    local = reports / "manual_audit_approval.json"
    persisted = results_root / "reports" / "manual_audit_approval.json"
    approval_path = local if local.exists() else persisted
    if not approval_path.exists():
        return False
    approval = read_json(approval_path)
    valid = approval.get("status") == "approved" and approval.get("dataset_fingerprint") == fingerprint
    if valid and approval_path == persisted:
        reports.mkdir(parents=True, exist_ok=True)
        shutil.copy2(persisted, local)
    return valid


def require_gpu() -> None:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("PyTorch is unavailable. Run the Colab dependency cell first.") from error
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required. In Colab choose Runtime > Change runtime type > GPU.")
    memory_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
    print(f"Using {torch.cuda.get_device_name(0)} with {memory_gib:.1f} GiB VRAM", flush=True)


def run_model_phase(args: argparse.Namespace, dataset: Path, reports: Path, phase: str) -> None:
    require_gpu()
    args.results_root.mkdir(parents=True, exist_ok=True)
    copy_dataset_reports(reports, args.results_root)
    if not audit_is_approved(dataset, reports, args.results_root):
        raise RuntimeError(
            "Manual annotation audit is not approved for this dataset fingerprint. Review all 31 contact sheets "
            "and run python -m src.approve_audit before training or testing."
        )
    command = [
        sys.executable, "-m", "src.train_evaluate", "--phase", phase,
        "--profile", str(args.profile), "--data-yaml", str(dataset / "data.yaml"),
        "--output-root", str(args.results_root), "--device", "0", "--resume",
    ]
    if phase == "test" and args.force_test:
        command.append("--force-test")
    run(command)


def generate_report(args: argparse.Namespace, reports: Path) -> None:
    if not (args.results_root / "reports" / "final_metrics.json").exists():
        raise FileNotFoundError("Held-out test metrics are missing. Run --stage test before generating the final report.")
    run([
        sys.executable, "-m", "src.generate_report", "--results-root", str(args.results_root),
        "--dataset-reports-root", str(reports),
    ])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Staged, resumable panoramic dental YOLO26 workflow for Google Colab.")
    parser.add_argument("--stage", choices=("prepare", "train", "test", "report", "all"), default="prepare")
    parser.add_argument("--data-root", type=Path, default=Path("/content/dental_yolo26_data"))
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--augment-fraction", type=float, default=0.15)
    parser.add_argument("--minority-target-instances", type=int, default=128)
    parser.add_argument("--max-augmentations-per-image", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--force-test", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.data_root = args.data_root.resolve()
    args.results_root = args.results_root.resolve()
    args.profile = args.profile.resolve()
    if not args.profile.exists():
        raise FileNotFoundError(f"Training profile not found: {args.profile}")
    dataset = PROJECT_ROOT / "dataset"
    reports = PROJECT_ROOT / "reports"
    if args.stage in {"prepare", "all"}:
        dataset, reports = prepare(
            args.data_root, args.augment_fraction, args.minority_target_instances,
            args.max_augmentations_per_image, args.seed, args.rebuild_data,
        )
    if args.stage in {"train", "test", "all"} and not prepared_dataset_is_valid(dataset, reports):
        raise RuntimeError("The dataset is not prepared and verified. Run with --stage prepare first.")
    if args.stage in {"train", "all"}:
        run_model_phase(args, dataset, reports, "train")
    if args.stage in {"test", "all"}:
        run_model_phase(args, dataset, reports, "test")
    if args.stage in {"report", "all"}:
        audit_reports = reports if (reports / "dataset_audit.json").exists() else args.results_root / "reports"
        generate_report(args, audit_reports)


if __name__ == "__main__":
    main()
