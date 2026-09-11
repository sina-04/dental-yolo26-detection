# Dental YOLO26 Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sina-04/dental-yolo26-detection/blob/main/notebooks/dental_yolo26_colab.ipynb)

An end-to-end, reproducible dental object-detection project built with Ultralytics YOLO26. It combines an intraoral tooth-anatomy dataset with a panoramic dental-disease dataset, converts incompatible annotations into one 38-class detection ontology, prevents group leakage, trains two experiments, selects the final checkpoint using validation results, and evaluates it once on a held-out test set.

> [!WARNING]
> This repository is an educational medical-AI experiment, not a clinically validated diagnostic device. The current checkpoint has very low recall and must not be used for diagnosis, treatment, triage, or patient care.

## Repository description

End-to-end YOLO26 dental object detection pipeline combining anatomy and panoramic disease datasets, with leakage-safe patient grouping, annotation conversion, medical augmentation, reproducible training, evaluation, error analysis, and model weight.

The paragraph above is exactly 250 characters and is used as the GitHub repository description.

## Highlights

- Uses both requested public Kaggle datasets.
- Reconciles incompatible source IDs into 7 `anatomy_*` and 31 `disease_*` classes.
- Converts 136,891 YOLO segmentation polygons into axis-aligned detection boxes.
- Removes exact duplicates and groups patient/exam and perceptual near-duplicates before splitting.
- Hashes inferred patient identifiers before writing processed manifests.
- Applies only medically conservative augmentation.
- Compares a baseline against a medically augmented YOLO26n experiment.
- Selects the model using validation mAP50-95, without using test performance for selection.
- Includes final weights, metrics, plots, threshold analysis, and 10 held-out inference examples.
- Includes a resumable Google Colab workflow that downloads exact Kaggle versions, uses the full training split, and persists every epoch checkpoint to Google Drive.

## Datasets

1. [Dental Anatomy Dataset — YOLOv8](https://www.kaggle.com/datasets/saisiddartha69/dental-anatomy-dataset-yolov8)
   - Kaggle version used: 1
   - Seven tooth-anatomy classes
   - Primarily intraoral photographs

2. [Dental Disease Panoramic Detection Dataset](https://www.kaggle.com/datasets/lokisilvres/dental-disease-panoramic-detection-dataset)
   - Kaggle version used: 6
   - Thirty-one disease, treatment, device, tooth, and anatomical classes
   - Panoramic dental radiographs

The raw datasets are intentionally excluded from Git because of their size, privacy sensitivity, and upstream license terms. Download them directly from Kaggle and review each dataset page before redistribution. The anatomy dataset also contains inconsistent license declarations between Kaggle metadata and its embedded export; resolve that discrepancy before republishing any raw data.

## Dataset preparation summary

| Item | Result |
|---|---:|
| Usable original images | 14,599 |
| Anatomy images | 724 |
| Panoramic disease images | 13,875 |
| Combined classes | 38 |
| Exact duplicate images removed | 57 |
| Polygon annotations converted | 136,891 |
| Original train images | 10,935 |
| Offline augmented train images | 546 |
| Final train images | 11,481 |
| Validation images | 2,192 |
| Test images | 1,472 |
| Validation instances | 16,850 |
| Test instances | 8,758 |
| Detected patient-group leakage | 0 |
| Detected exact-hash cross-split leakage | 0 |
| Reviewed near-duplicate cross-split leakage | 0 |

The split targets approximately 75% train, 15% validation, and 10% test while keeping inferred patient/exam groups together and improving minority-class coverage when independent groups permit it. Patient grouping is inferred from filenames rather than verified against a clinical master patient index, so it remains a documented limitation.

## Annotation and augmentation policy

Disease segmentation polygons are converted to normalized YOLO detection boxes:

```text
class_id x_center y_center width height
```

The training pipeline uses medically conservative transformations:

- Rotation up to ±7 degrees
- Mild brightness and contrast variation
- Low-probability blur
- Low-probability CLAHE
- Bounding-box-aware geometric transformation

The following are disabled: Mosaic, MixUp, CutMix, Copy-Paste, hue/saturation shifts, horizontal flip, and vertical flip.

## Model and executed experiments

The metrics below document the already executed 2 GB local baseline. They are retained for provenance and should not be confused with the stronger Colab profile described in the next section. Colab results are written to Drive and can replace these published metrics only after that run finishes and is reviewed.

- Framework: [Ultralytics YOLO26](https://docs.ultralytics.com/models/yolo26/)
- Model: YOLO26n
- Pretrained checkpoint: `yolo26n.pt`
- Image size: 320
- Batch size: 8
- Seed: 42
- GPU: NVIDIA GeForce MX330, 2 GB VRAM
- Training profile: deterministic 1,200-image class-covered subset
- Validation and test sets: complete

| Experiment | Epochs | Main change | Val precision | Val recall | Val mAP50 | Val mAP50-95 |
|---|---:|---|---:|---:|---:|---:|
| `baseline_yolo26n` | 1 | Minimal augmentation | 0.0182 | 0.0287 | 0.0130 | 0.0105 |
| `tuned_yolo26n_medical_aug` | 2 | Medical augmentation | 0.8590 | 0.0197 | 0.0392 | 0.0260 |

The medically augmented experiment was selected only by validation mAP50-95.

## Recommended: full-data Google Colab training

Open [`notebooks/dental_yolo26_colab.ipynb`](notebooks/dental_yolo26_colab.ipynb) with the badge at the top of this README and choose a GPU runtime. The notebook mounts Google Drive, clones this repository, installs Colab-safe dependencies without replacing Colab's PyTorch/CUDA build, downloads the exact public Kaggle dataset versions to the runtime SSD, rebuilds and verifies the combined dataset, and starts resumable training.

The default Colab profile is deliberately stronger than the historical local run:

| Setting | Local baseline | Colab full-data profile |
|---|---:|---:|
| Model | YOLO26n | YOLO26s |
| Image size | 320 | 640 |
| Training images | 1,200 class-covered subset | Complete prepared training split |
| Baseline epochs | 1 | 15 |
| Tuned epochs | 2 | 40 |
| Batch size | 8 | 32 |
| Checkpoint persistence | Local disk | Google Drive after every epoch |

The Colab command used by the notebook is:

```bash
python -m src.colab_workflow \
  --stage train \
  --results-root /content/drive/MyDrive/dental-yolo26-detection/colab-results \
  --model yolo26s.pt \
  --baseline-epochs 15 \
  --tuned-epochs 40 \
  --imgsz 640 \
  --batch 32 \
  --workers 4 \
  --patience 12 \
  --seed 42
```

Rerun the training cell after a Colab interruption; incomplete experiments resume from their Drive-backed `last.pt`. Batch size 32 has been validated on a 14.6 GiB Tesla T4. If the assigned GPU runs out of memory, reduce `--batch 32` to `--batch 16` (or 8 on a smaller accelerator). Raw datasets remain on Colab's temporary SSD rather than being copied into Git or Drive. Final weights, metrics, plots, environment details, and reports are stored under `MyDrive/dental-yolo26-detection/colab-results/`.

## Held-out test results

| Metric | Result |
|---|---:|
| Precision | 0.8733 |
| Recall | 0.0197 |
| mAP50 | 0.0409 |
| mAP50-95 | 0.0280 |
| Validation-selected confidence threshold | 0.05 |
| Thresholded precision at IoU 0.50 | 0.5747 |
| Thresholded recall at IoU 0.50 | 0.0057 |
| Thresholded F1 at IoU 0.50 | 0.0113 |
| TP / FP / FN | 50 / 37 / 8,708 |

The high precision must not be interpreted by itself. At the selected threshold, the model missed 8,708 of 8,758 labeled test objects. This severe recall limitation is consistent with the short, compute-constrained training profile and extreme class imbalance. The checkpoint is useful as a reproducible baseline only.

## Project structure

```text
.
├── artifacts/
│   ├── best.pt
│   ├── last.pt
│   ├── training_configuration.json
│   └── environment_freeze.txt
├── dataset/
│   └── data.yaml                 # images/ and labels/ are generated locally
├── reports/
│   ├── annotation_audit/          # generated locally; images excluded from Git
│   ├── inference_examples/        # generated locally; images excluded from Git
│   ├── dataset_audit.json
│   ├── dataset_verification.json
│   ├── experiments.csv
│   ├── final_metrics.json
│   ├── per_class_test_metrics.csv
│   ├── error_analysis.csv
│   └── threshold_analysis.png
├── runs/                         # privacy-safe Ultralytics plots and results
├── src/
│   ├── colab_workflow.py
│   ├── common.py
│   ├── prepare_dataset.py
│   ├── verify_dataset.py
│   ├── train_evaluate.py
│   └── generate_report.py
├── tests/
│   └── test_pipeline.py
├── notebooks/
│   └── dental_yolo26_colab.ipynb
├── tools/
│   └── segmented_download.py
├── FINAL_REPORT.md
├── PROGRESS.md
├── requirements.txt
├── requirements-colab.txt
└── README.md
```

## Requirements

- Windows, Linux, or macOS
- Python 3.12 recommended
- NVIDIA CUDA GPU recommended for training
- Sufficient disk space for both Kaggle datasets and the generated dataset

The pinned requirements use PyTorch CUDA 11.8. If your GPU requires a different CUDA build, install the appropriate PyTorch build from the [official PyTorch selector](https://pytorch.org/get-started/locally/) before installing the remaining dependencies.

On Colab, use `requirements-colab.txt`. It intentionally does not pin PyTorch or CUDA, preserving the accelerator-compatible build supplied by Colab.

## Installation

From PowerShell:

```powershell
git clone https://github.com/sina-04/dental-yolo26-detection.git
cd dental-yolo26-detection

py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Linux/macOS activation equivalents can use `.venv/bin/python` instead.

## Download and place the datasets

Download both archives from the Kaggle links above and extract them so the repository contains:

```text
Dental Dataset/
└── Dental Dataset/
    ├── data.yaml
    ├── train/
    ├── valid/
    └── test/

YOLO & COCO/
└── YOLO/
    └── YOLO/
        ├── data.yaml
        ├── train/
        ├── valid/
        └── test/
```

The preparation script detects these locations automatically. Raw and generated images are excluded by `.gitignore`.

## Build the processed dataset

```powershell
.\.venv\Scripts\python.exe -m src.prepare_dataset --augment-fraction 0.05 --seed 42
```

If a generated `dataset/` already exists and you deliberately want to replace it:

```powershell
.\.venv\Scripts\python.exe -m src.prepare_dataset --augment-fraction 0.05 --seed 42 --rebuild
```

`--rebuild` is restricted in code to the generated project `dataset/` directory.

## Verify the dataset

```powershell
.\.venv\Scripts\python.exe -m src.verify_dataset
```

A valid build returns `"status": "pass"` and zero cross-split leakage failures.

## Reproduce the historical low-memory training run

```powershell
.\.venv\Scripts\python.exe -m src.train_evaluate `
  --baseline-epochs 1 `
  --tuned-epochs 2 `
  --imgsz 320 `
  --batch 8 `
  --max-train-images 1200 `
  --device auto `
  --seed 42

.\.venv\Scripts\python.exe -m src.generate_report
```

The complete validation and test sets remain in use when `--max-train-images` limits the training set.

## Run a longer full-data experiment outside Colab

Use appropriate hardware and adjust the following example to your GPU capacity:

```powershell
.\.venv\Scripts\python.exe -m src.train_evaluate `
  --model yolo26s.pt `
  --baseline-epochs 20 `
  --tuned-epochs 30 `
  --imgsz 512 `
  --batch 4 `
  --device auto `
  --seed 42
```

Do not compare a new model against the published test result repeatedly. Choose experiments using validation data and reserve test evaluation for the final selected configuration.

## Run inference

Single image:

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; model=YOLO('artifacts/best.pt'); model.predict(source=r'C:\path\to\image.jpg', conf=0.05, imgsz=320, device=0, save=True)"
```

Folder of images:

```powershell
.\.venv\Scripts\python.exe -c "from ultralytics import YOLO; model=YOLO('artifacts/best.pt'); model.predict(source=r'C:\path\to\images', conf=0.05, imgsz=320, device=0, save=True)"
```

Ultralytics saves annotated outputs under `runs/detect/predict*`.

For CPU inference, replace `device=0` with `device='cpu'`.

## Run tests

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

The unit tests cover detection-label preservation, polygon-to-box conversion, and privacy-preserving patient grouping. Dataset integrity checks are performed separately by `src.verify_dataset`.

## Reports and artifacts

- [`FINAL_REPORT.md`](FINAL_REPORT.md): methodology, metrics, limitations, per-class results, and error analysis
- [`PROGRESS.md`](PROGRESS.md): deliverable-by-deliverable completion report
- [`reports/final_metrics.json`](reports/final_metrics.json): machine-readable final metrics
- [`reports/experiments.csv`](reports/experiments.csv): experiment comparison ledger
- [`reports/error_analysis.csv`](reports/error_analysis.csv): expected and predicted classes for qualitative examples
- `reports/inference_examples/`: five anatomy and five panoramic held-out visualizations, generated locally and excluded from the public repository pending privacy and redistribution review
- [`artifacts/best.pt`](artifacts/best.pt): validation-selected YOLO26 checkpoint
- [`artifacts/last.pt`](artifacts/last.pt): final epoch checkpoint

Green boxes indicate ground truth and red boxes indicate predictions in the saved qualitative examples.

## Known limitations

- Training used 1,200 of 11,481 available training images and only three experiment epochs in total.
- The datasets represent very different image domains: intraoral photography and panoramic radiography.
- The combined ontology mixes anatomy, diseases, treatments, and devices.
- Several classes contain too few independent examples for stable evaluation.
- Polygon-to-box conversion discards lesion shape.
- Patient grouping is inferred from filenames rather than verified identifiers.
- No external-site, prospective, reader-study, calibration, fairness, robustness, regulatory, or clinical-utility validation was performed.
- Dataset labels and example images have not been adjudicated by a qualified dental radiologist as part of this project.

## Responsible use

Do not use this model to make or support clinical decisions. False negatives can miss disease and false positives can cause unnecessary concern. Any future medical use would require verified patient-level data governance, expert annotation review, external validation, subgroup analysis, calibration, human-factors evaluation, security assessment, and applicable regulatory approval.

## Reproducibility

The repository retains:

- Exact random seed and executed hyperparameters
- Selected and last checkpoints
- Pinned direct requirements and a full environment freeze
- Dataset audit and privacy-hashed manifest
- Validation experiment ledger
- Per-class and per-image test metrics
- Training, validation, PR, threshold, and confusion-matrix plots
- Source code and unit tests

The source datasets themselves are not redistributed. Download the exact upstream versions from Kaggle.

## License and attribution

No license is granted for the raw datasets by this repository. Their upstream terms apply independently. Review the Kaggle dataset pages and embedded export metadata before use or redistribution. No separate open-source license has been assigned to this repository unless a `LICENSE` file is added later.

## Author

[Sina Rezaei (`sina-04`)](https://github.com/sina-04)
