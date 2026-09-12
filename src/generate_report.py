from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def percentage(value: float) -> str:
    return f"{100 * value:.2f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate human-readable reports from a completed training run.")
    parser.add_argument("--results-root", type=Path, default=ROOT)
    parser.add_argument("--dataset-reports-root", type=Path, default=REPORTS)
    args = parser.parse_args()
    results_root = args.results_root.resolve()
    reports = results_root / "reports"
    audit_reports = args.dataset_reports_root.resolve()
    audit = read_json(audit_reports / "dataset_audit.json")
    final = read_json(reports / "final_metrics.json")
    experiments = read_csv(reports / "experiments.csv")
    class_rows = [row for row in read_csv(audit_reports / "class_distribution.csv") if row["split"] == "all"]
    per_class = read_csv(reports / "per_class_test_metrics.csv")
    errors = read_csv(reports / "error_analysis.csv")
    approval_path = audit_reports / "manual_audit_approval.json"
    approval = read_json(approval_path) if approval_path.exists() else {"status": "missing", "reviewer": "N/A"}
    environment = final["environment"]
    base_available = audit["split_counts"].get("train", 0)
    augmented_available = audit.get("training_view_counts", {}).get(
        "augmented", base_available + audit.get("augmented_training_images", 0)
    )
    base_experiment = next((item for item in final["experiments"] if item.get("data_variant") == "base"), final["experiments"][0])
    augmented_experiment = next(
        (item for item in final["experiments"] if item.get("data_variant") == "augmented"), final["experiments"][-1]
    )
    base_executed = base_experiment.get("train_images", base_available)
    augmented_executed = augmented_experiment.get("train_images", augmented_available)
    model_checkpoint = str(final["experiments"][0].get("model", "YOLO26"))
    compute_profile = "full-data" if base_executed >= base_available and augmented_executed >= augmented_available else "compute-limited"
    test = final["test_ultralytics_metrics"]
    custom = final["test_threshold_metrics_iou50"]

    experiment_table = "\n".join(
        f"| {row['experiment']} | {row['model']} | {row['main_change']} | {int(float(row['epochs']))} | {percentage(float(row['precision']))} | {percentage(float(row['recall']))} | {float(row['map50']):.4f} | {float(row['map50_95']):.4f} | {row['selected']} |"
        for row in experiments
    )
    class_table = "\n".join(
        f"| {row['class_id']} | {row['class_name']} | {row['images']} | {row['instances']} |"
        for row in class_rows
    )
    metric_lines = []
    for row in per_class:
        if int(float(row["support"])) == 0:
            metric_lines.append(f"| {row['class_id']} | {row['class_name']} | 0 | N/E | N/E | N/E | N/E | N/E |")
        else:
            metric_lines.append(
                f"| {row['class_id']} | {row['class_name']} | {row['support']} | {row.get('evidence_tier', 'N/E')} | {float(row['precision']):.3f} | {float(row['recall']):.3f} | {float(row['f1']):.3f} | {float(row['map50_95']):.3f} |"
            )
    metric_table = "\n".join(metric_lines)
    error_table = "\n".join(
        f"| {row.get('example', '')} | {row.get('observed_outcomes', '')} | {row.get('expected_result', '')} | {row.get('model_prediction', '')} | {row.get('probable_cause', '')} | {row.get('proposed_improvement', '')} |"
        for row in errors
    )

    report = f"""# Dental YOLO26 Detection Project — Final Report

## Executive summary

This project uses only the 31-class Dental X-Ray Panoramic Dataset to build a YOLO26 object detector. Segmentation polygons are converted to axis-aligned detection boxes when present. Patient/exam groups inferred from filenames are hashed before being written to processed manifests, and no inferred patient group crosses train, validation, and test splits.

The selected model is **{final['selected_experiment']}**, chosen only by validation mAP50-95. Its held-out test mAP50-95 is **{test.get('metrics/mAP50-95(B)', 0.0):.4f}**, mAP50 is **{test.get('metrics/mAP50(B)', 0.0):.4f}**, precision is **{test.get('metrics/precision(B)', 0.0):.4f}**, and recall is **{test.get('metrics/recall(B)', 0.0):.4f}**. At the validation-selected confidence threshold of **{final['validation_selected_threshold']:.2f}**, the custom IoU=0.50 test F1 is **{custom['f1']:.4f}**.

The executed **{compute_profile}** profile used **{base_executed:,}/{base_available:,} baseline-view images** and **{augmented_executed:,}/{augmented_available:,} augmented-view images**. Validation and test sets remained complete. Hardware and software details are recorded below so local and Colab runs can be compared without conflating their compute budgets.

The result is **not clinically usable**: at the validation-selected threshold it missed **{custom['fn']:,} of {custom['tp'] + custom['fn']:,}** labeled test objects (recall {custom['recall']:.4f}). Aggregate metrics must not be read in isolation, especially for classes with very limited test support.

This is an educational research result, not a clinically validated diagnostic device.

## Data provenance and licensing

- [Dental Disease Panoramic Detection Dataset](https://www.kaggle.com/datasets/lokisilvres/dental-disease-panoramic-detection-dataset), Kaggle version 6, Apache 2.0.

The panoramic export contains 31 classes spanning diseases, treatments, devices, teeth, and anatomy. The source images are not redistributed by this repository.

## Quality control and preparation

- Usable images after corruption checks and exact-deduplication: **{sum(audit['split_counts'].values()) if 'split_counts' in audit else 'see dataset_manifest.csv'}**.
- Source counts: `{audit.get('source_counts', {})}`.
- Final split counts: `{audit.get('split_counts', {})}`.
- Exact duplicate files removed: **{audit.get('exact_duplicates_removed', 0)}**.
- Segmentation polygons converted to boxes: **{audit.get('polygon_annotations_converted', 0)}**.
- Patient-group leakage count: **{audit.get('patient_group_leakage', 0)}**.
- Offline medically mild AlbumentationsX training images: **{audit.get('augmented_training_images', 0)}**.
- Dataset fingerprint: **`{audit.get('dataset_fingerprint', 'missing')}`**.
- Manual contact-sheet audit: **{approval.get('status', 'missing')}**, reviewer **{approval.get('reviewer', 'N/A')}**.
- Coordinate/annotation issues: `{audit.get('issue_counts', {})}`.

Automatic QC checked image decodability, missing/malformed labels, class ranges, coordinate ranges, non-positive and tiny boxes, exact hashes, perceptual-hash near-duplicate candidates, source pairing, and split leakage. One annotated sample per observed class is generated locally in `reports/annotation_audit/`. Those clinical-image derivatives are excluded from the public Git repository pending explicit privacy and redistribution review. The montages support local review but do not substitute for a dentist/radiologist annotation audit.

## Class distribution

| ID | Class | Images | Instances |
|---:|---|---:|---:|
{class_table}

## Split strategy

The pipeline reconstructs splits from inferred patient/exam groups instead of trusting the publisher-provided Roboflow split. Tokens likely to identify a patient are normalized, immediately hashed, and never written in clear text to the processed manifest. A deterministic greedy multilabel group allocation targets 75% train, 15% validation, and 10% test while reducing class-distribution drift. Exact visual duplicates are removed before splitting; identical perceptual hashes are grouped and near-duplicate candidates are listed for review.

Because patient identifiers were inferred from filenames rather than verified against a clinical master index, patient separation is best-effort and remains a limitation.

## Model and training configuration

- Ultralytics: **{environment['ultralytics']}**
- Model/checkpoint: **`{model_checkpoint}`**, COCO-pretrained transfer learning
- Python: **{environment['python']}**
- PyTorch: **{environment['torch']}**
- Device: **{environment['selected_device']}** ({environment.get('gpu') or 'CPU'})
- Mosaic, MixUp, CutMix, Copy-Paste: **disabled**
- Hue/saturation, vertical flip, horizontal flip: **disabled**
- Early stopping, weight decay 0.0005, deterministic seed 42: **enabled**
- Offline AlbumentationsX: rotation ±7°, mild brightness/contrast, low-probability 3×3 Gaussian blur or CLAHE; bounding boxes transformed together with images

The checkpoint and image size were selected explicitly for the recorded hardware profile. The Colab workflow defaults to a YOLO26s model and full-resolution training; local runs can use YOLO26n when compute is limited.

## Validation experiment comparison

| Experiment | Model | Main change | Epochs | Precision | Recall | mAP50 | mAP50-95 | Selected |
|---|---|---|---:|---:|---:|---:|---:|---|
{experiment_table}

## Final held-out test evaluation

The held-out test set was evaluated once after experiment selection. Ultralytics mAP uses confidence-swept precision-recall curves; the operating threshold was selected on validation data, never on test data.

- Precision: **{test.get('metrics/precision(B)', 0.0):.4f}**
- Recall/sensitivity: **{test.get('metrics/recall(B)', 0.0):.4f}**
- mAP50: **{test.get('metrics/mAP50(B)', 0.0):.4f}**
- mAP50-95: **{test.get('metrics/mAP50-95(B)', 0.0):.4f}**
- Validation-selected threshold: **{final['validation_selected_threshold']:.2f}**
- Thresholded test F1 at IoU 0.50: **{custom['f1']:.4f}**
- Thresholded TP / FP / FN: **{custom['tp']} / {custom['fp']} / {custom['fn']}**

### Per-class test metrics

| ID | Class | Test support | Evidence tier | Precision | Recall | F1 | mAP50-95 |
|---:|---|---:|---|---:|---:|---:|---:|
{metric_table}

`N/E` means not estimable because the held-out test split contains no ground-truth instance of that ultra-rare class; it is not a zero-performance claim. `very_low` (<20 objects) and `limited` (<50 objects) results are exploratory rather than reliable class-level estimates.

The confusion matrix and training/validation loss plots are retained under `runs/`; `reports/threshold_analysis.png` shows the validation precision-recall-F1 operating-point tradeoff.

## Error analysis and unseen inference examples

Ground truth is green and model prediction is red in the ten locally saved test examples. The images are excluded from the public Git repository pending explicit privacy and redistribution review; their privacy-hashed IDs and structured results remain in `reports/error_analysis.csv`.

| Example | Observed outcomes | Expected result | Model prediction | Probable cause | Proposed improvement |
|---|---|---|---|---|---|
{error_table}

Recurring risks include small lesions, low contrast, overlapping anatomy, severe class imbalance, publisher annotation inconsistency, and polygon-to-box loss of shape information. The next scientifically useful step is blinded review of false negatives and annotation candidates by qualified dental clinicians, followed by class-aware additional collection rather than indiscriminate synthetic augmentation.

## Overfitting prevention actually used

- Hashed patient/exam group splitting before materialization
- Exact duplicate removal and near-duplicate audit
- COCO-pretrained transfer learning
- Two validation-controlled experiments
- Early stopping
- Weight decay
- Medically constrained augmentation
- Small model selected for the data/compute regime
- Test isolation until final selection

Dropout was not claimed or used as a detector regularizer.

## Legacy reference comparison

The upstream Dental-Disease-Detection application demonstrates that the same 31-label domain can support immediate multiclass inference with separately distributed weights. Safe checkpoint metadata inspection identifies that model as an Ultralytics **YOLOv8x-seg** segmentation model (Ultralytics 8.3.0), not YOLO26. Its embedded validation summary reports box mAP50 **0.2971**, box mAP50-95 **0.1567**, and mask mAP50-95 **0.1160**. These figures are contextual only: the repository does not contain its training code or split construction, and the publisher's native split has substantial inferred patient/exam overlap. It is therefore not treated as a reproducible benchmark or directly compared for model selection.

## Limitations and medical-AI warning

The source is a public secondary dataset with uncertain clinical sampling, demographics, device distributions, labeling protocol, and patient metadata. Its taxonomy mixes anatomy, disease, treatments, and devices. Bounding boxes derived from segmentation discard lesion shape. Some classes may be too rare to estimate stable performance, and high aggregate mAP can hide clinically important minority-class failure. Patient grouping is inferred, not externally verified. There is no external-site, prospective, reader-study, calibration, robustness, fairness, security, regulatory, or clinical-utility validation.

False negatives could miss disease; false positives could trigger unnecessary concern or follow-up. Predictions must not be interpreted as diagnoses or used for patient care.

## Reproducibility artifacts

- `src/prepare_dataset.py`: validation, conversion, grouping, splitting, augmentation, and audit
- `src/train_evaluate.py`: transfer learning, experiment selection, final test evaluation, threshold analysis, inference, and error analysis
- `src/infer.py`: validation-thresholded prediction CLI with annotated images and JSON detections
- `dataset/data.yaml`: portable generated-dataset configuration and 31-class ontology
- `dataset/fingerprint.json`: immutable data/preparation contract identifier
- `reports/dataset_manifest.csv`: privacy-hashed split manifest
- `reports/dataset_audit.json`: QC summary
- `reports/experiments.csv`: experiment ledger
- `reports/final_metrics.json`: complete selected/test metrics
- `artifacts/best.pt` and `artifacts/last.pt`: selected YOLO26 checkpoints
- `artifacts/training_configuration.json` and `environment_freeze.txt`: exact run details
"""
    results_root.mkdir(parents=True, exist_ok=True)
    (results_root / "FINAL_REPORT.md").write_text(report, encoding="utf-8")

    progress = """# Deliverable Progress

| Deliverable | Progress | Completed | Problems encountered | Knowledge / evidence gained | Next action |
|---|---:|---|---|---|---|
| Training source code | 100% | End-to-end preparation, training, validation, testing, inference, and error-analysis CLIs | Local compute constrained model scale and epoch budget | Reproducible YOLO26 workflow | Re-run on a larger GPU for longer experiments |
| Final model weights | 100% | Selected `best.pt` and retained `last.pt` | Model capacity depends on the assigned accelerator | Validation-only selection preserves test isolation | Compare larger models only on validation data |
| Dataset and configuration | 100% | Single panoramic source, polygons boxified, group split, data.yaml generated | Patient groups use inferred IDs only | Single-domain training removes the former cross-domain merge | Clinician review and verified patient index |
| Evaluation report | 100% | Full test metrics, per-class table, losses, confusion matrix, PR and threshold analysis | Rare classes yield unstable estimates | Per-class reporting exposes imbalance | Add confidence intervals with larger external test data |
| Ten unseen examples | 100% | Representative held-out test visualizations | Some error types may not occur in a small test set | Model behavior is inspectable | Blinded expert review |
| Error analysis | 100% | Per-image structured categories and candidate causes | Candidate causes are hypotheses, not clinical adjudication | False-negative audit is highest priority | Clinician adjudication |
| Overfitting report | 100% | Only techniques actually used are documented | Compute limits model-size comparison | Leakage control matters more than cosmetic tuning | External validation |
| Training configuration | 100% | Versions, seed, device, model, augmentation, hyperparameters, duration | Low-memory GPU | Exact environment is frozen | Reproduce on CUDA GPU |
| Experiment comparison | 100% | Baseline vs medically augmented experiment | Same model size due VRAM | Validation-only selection preserved test integrity | Add yolo26s/m experiments remotely |
| Reproducibility package | 100% | Code, configs, reports, weights, README, lock-style freeze | Raw Kaggle data must remain license-compliant | Entire run is scriptable | Archive with checksums under approved license terms |

## Mentor discussion

- Is the 31-class mixed clinical ontology appropriate, or should pathology and treatment/device findings be modeled separately?
- Can a verified patient identifier replace filename-based grouping?
- Which false-negative classes are clinically highest priority?
- Is external-site data available for a genuine generalization test?
"""
    (results_root / "PROGRESS.md").write_text(progress, encoding="utf-8")


if __name__ == "__main__":
    main()
