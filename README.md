# Dental YOLO26 Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sina-04/dental-yolo26-detection/blob/main/notebooks/dental_yolo26_colab.ipynb)

A reproducible Ultralytics YOLO26 object-detection project for panoramic dental X-rays. It uses only the **Dental X-Ray Panoramic Dataset** and preserves all 31 source classes. The previous intraoral dataset and mixed 38-class ontology are no longer part of this project.

> Educational research only. This is not a clinically validated diagnostic system and must not be used for patient care.

## Current status

The repository contains the complete preparation, training, validation-selection, held-out testing, reporting, and inference workflow. It does not claim new YOLO26 results until the staged Colab run has produced them. Generated data, clinical-image derivatives, model weights, and run outputs are intentionally excluded from Git.

## Dataset

The sole source is [Dental Disease Panoramic Detection Dataset, version 6](https://www.kaggle.com/datasets/lokisilvres/dental-disease-panoramic-detection-dataset), referred to in this project as **Dental X-Ray Panoramic Dataset**.

- Source-reported license: Apache 2.0
- Native YOLO export: 9,481 train, 2,871 validation, 1,580 test images
- Total native images: 13,932
- Detection classes: 31
- Labels: YOLO boxes and polygons; polygons are converted to axis-aligned boxes

The native split is not used. Filename-derived exam groups overlap substantially across its partitions, so preparation removes exact duplicates, groups visually similar variants, and deterministically reconstructs a 75/15/10 train/validation/test split. Grouping is best-effort because a verified clinical patient index is unavailable.

The full 31-label ontology is retained. Very rare classes remain in the model but their class-level metrics are explicitly marked `N/E`, `very_low`, or `limited` according to held-out support; they are not presented as reliable estimates.

## Experiment design

Two validation-controlled YOLO26s experiments use identical optimizer, schedule, image size, and regularization settings:

1. `baseline`: original training images only.
2. `medical_aug`: original plus capped, class-aware, medically constrained augmentations.

Both run for up to 100 epochs with early stopping. The selected checkpoint is determined by validation mAP50-95, with validation recall as the tie-breaker. The test split is evaluated only in the separate test phase, after selection. Its operating confidence threshold is chosen on validation data.

The default Google Colab T4 profile is in `configs/colab_t4.yaml`: YOLO26s, 640 px, batch 16, AdamW, cosine learning-rate decay, 20-epoch patience, disk cache, AMP, seed 42, and checkpoint saving every epoch.

## Data and audit safeguards

Preparation performs:

- image decoding and annotation validation;
- polygon-to-box conversion;
- exact SHA-256 deduplication;
- perceptual-hash grouping of identical and near-duplicate images;
- privacy-hashed filename-derived exam grouping;
- deterministic multilabel group splitting;
- train-only augmentation with no synthetic validation/test images;
- isolated `base` and `augmented` training manifests;
- a dataset-contract fingerprint; and
- five-image annotation contact sheets for every class.

Training is blocked until a reviewer inspects all 31 contact sheets and records approval for the current dataset fingerprint. Changing the source manifest, class order, split/grouping version, seed, or augmentation settings changes the fingerprint and invalidates stale approvals and run resumptions.

## Project structure

```text
.
├── configs/
│   └── colab_t4.yaml             # canonical free-T4 training profile
├── dataset/
│   └── data.yaml                 # public 31-class ontology template
├── notebooks/
│   └── dental_yolo26_colab.ipynb # staged prepare/audit/train/test/report run
├── src/
│   ├── approve_audit.py          # fingerprint-bound annotation-audit approval
│   ├── colab_workflow.py         # resumable Colab stage orchestration
│   ├── common.py                 # hashing, paths, JSON, and YAML helpers
│   ├── generate_report.py        # final report and deliverable progress
│   ├── infer.py                  # annotated-image and JSON inference CLI
│   ├── prepare_dataset.py        # QC, dedupe, grouping, split, augmentation
│   ├── train_evaluate.py         # train/select and isolated test phases
│   └── verify_dataset.py         # leakage and training-view integrity checks
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── requirements-colab.txt
```

Generated paths such as `dataset/images/`, `dataset/labels/`, `dataset/runtime/`, `dataset/fingerprint.json`, `reports/`, `runs/`, `artifacts/`, and `inference_output/` are ignored by Git.

## Recommended Google Colab workflow

Open the notebook using the badge and select a T4 GPU. It installs `requirements-colab.txt`, downloads only Kaggle dataset version 6 to Colab's ephemeral SSD, and writes resumable training outputs to:

```text
MyDrive/dental-yolo26-detection/panoramic31-yolo26s-t4-v1/
```

The notebook intentionally uses separate cells:

1. Prepare and verify the dataset.
2. Display and review all 31 annotation contact sheets.
3. Record fingerprint-bound audit approval.
4. Train both experiments and select using validation data.
5. Evaluate the held-out test once.
6. Generate the final report and progress table.

Equivalent commands inside the cloned repository are:

```bash
python -m src.colab_workflow --stage prepare --rebuild-data
python -m src.approve_audit --reviewer "Reviewer name" --notes "Concise findings after reviewing all 31 sheets"
python -m src.colab_workflow --stage train --profile configs/colab_t4.yaml
python -m src.colab_workflow --stage test --profile configs/colab_t4.yaml
python -m src.colab_workflow --stage report
```

Do not use `--force-test` for iterative tuning. It exists only for an intentional rerun of the same final checkpoint and dataset.

## Local preparation

Python 3.11 or newer is recommended. A CUDA GPU is needed for the canonical training profile.

```powershell
git clone https://github.com/sina-04/dental-yolo26-detection.git
cd dental-yolo26-detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place the extracted dataset under `Dental X-Ray Panoramic Dataset/`, or pass a parent directory containing its 31-class YOLO export:

```powershell
python -m src.prepare_dataset --source-root "D:\datasets\Dental X-Ray Panoramic Dataset" --rebuild
python -m src.verify_dataset
```

Review `reports/annotation_audit/class_00.jpg` through `class_30.jpg`, then approve and train:

```powershell
python -m src.approve_audit --reviewer "Reviewer name" --notes "Review findings"
python -m src.train_evaluate --phase train --profile configs/colab_t4.yaml --device 0 --resume
python -m src.train_evaluate --phase test --profile configs/colab_t4.yaml --device 0
python -m src.generate_report
```

## Inference

After the test phase, inference defaults to the confidence threshold selected on validation data:

```powershell
python -m src.infer "path\to\panoramic-xray.jpg"
```

For Drive-backed Colab results or another checkpoint:

```bash
python -m src.infer /content/xrays \
  --weights /content/drive/MyDrive/dental-yolo26-detection/panoramic31-yolo26s-t4-v1/artifacts/best.pt \
  --metrics /content/drive/MyDrive/dental-yolo26-detection/panoramic31-yolo26s-t4-v1/reports/final_metrics.json \
  --output /content/predictions
```

Each image produces an annotated JPEG and a JSON file containing class ID, class name, confidence, and pixel `xyxy` coordinates.

## Relationship to the reference application

[Loki-Silvres/Dental-Disease-Detection](https://github.com/Loki-Silvres/Dental-Disease-Detection) is a useful functional inference reference for this domain. Its separately distributed checkpoint is a trained Ultralytics YOLOv8x-seg model, while this project targets YOLO26 object detection and adds the missing reproducible preparation, training, evaluation, experiment comparison, leakage controls, error analysis, and Colab workflow.

Metadata safely inspected from that legacy checkpoint reports box mAP50 0.2971, box mAP50-95 0.1567, and mask mAP50-95 0.1160. These are contextual figures only: the repository lacks its training code and verifiable split construction, and its results must not be treated as a directly comparable benchmark.

## Outputs after a completed run

- `artifacts/best.pt` and `last.pt`
- `artifacts/training_configuration.json` and `environment_freeze.txt`
- `reports/selection.json`, `experiments.csv`, and `final_metrics.json`
- per-class test metrics with evidence tiers
- validation threshold analysis
- confusion matrices and training curves under `runs/`
- ten structured held-out error examples
- `FINAL_REPORT.md` and `PROGRESS.md`

## Tests

```powershell
python -m unittest discover -s tests -v
```

Tests cover label conversion, privacy-preserving grouping, near-duplicate grouping, rare-class split behavior, isolated training views, dataset fingerprints, audit approval matching, inference threshold resolution, and resumable Colab markers.

## Limitations

The public secondary dataset has uncertain demographics, acquisition hardware, sampling, annotation protocol, and patient metadata. Its taxonomy mixes pathology, treatments, devices, anatomy, and tooth state. Polygon conversion loses shape. Several labels have too little support for stable estimates. There is no external-site, prospective, calibration, fairness, robustness, reader-study, regulatory, or clinical-utility validation.

False negatives can miss disease and false positives can create unnecessary concern. All labels and failure cases require qualified dental-clinician review before any further applied research.

## License

The upstream dataset's terms apply independently to its files. No separate open-source license is granted by this repository unless a `LICENSE` file is added later.
