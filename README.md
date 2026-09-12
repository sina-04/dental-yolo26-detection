# Dental YOLO26 Detection

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sina-04/dental-yolo26-detection/blob/main/notebooks/dental_yolo26_colab.ipynb)

An end-to-end Ultralytics YOLO26 pipeline for detecting dental findings in panoramic X-rays. The project now uses only the **Dental X-Ray Panoramic Dataset** and its 31-class YOLO export; the former intraoral anatomy dataset and mixed 38-class ontology have been removed.

> This is an educational research project, not a clinically validated diagnostic system. Do not use its predictions for patient care.

## What the pipeline does

- Locates the dataset from either the local project folder or a downloaded Kaggle directory.
- Validates images and YOLO annotations, including polygon-to-box conversion when needed.
- Removes exact duplicates and groups Roboflow variants by a privacy-preserving patient/exam hash.
- Rebuilds deterministic 75/15/10 train, validation, and test splits to reduce leakage.
- Applies capped, train-only class-aware augmentation for minority classes.
- Compares a baseline and a medically constrained augmented YOLO26 experiment.
- Selects the checkpoint using validation mAP50-95, then evaluates the test split once.
- Produces per-class metrics, threshold analysis, error analysis, and reproducibility metadata.

## Dataset

The only data source is the [Dental Disease Panoramic Detection Dataset](https://www.kaggle.com/datasets/lokisilvres/dental-disease-panoramic-detection-dataset), used as the local **Dental X-Ray Panoramic Dataset** folder.

- Kaggle version: 6
- License reported by the source: Apache 2.0
- Native export formats: YOLO and COCO
- Native YOLO split: 9,481 train, 2,871 validation, and 1,580 test images
- Detection classes: 31

The raw dataset is excluded from Git because it is large and source filenames may contain identifying text. Review the upstream license and privacy implications before redistributing any images. The preparation pipeline replaces raw filenames with opaque IDs in generated data and manifests.

### Classes

`Caries`, `Crown`, `Filling`, `Implant`, `Malaligned`, `Mandibular Canal`, `Missing teeth`, `Periapical lesion`, `Retained root`, `Root Canal Treatment`, `Root Piece`, `impacted tooth`, `maxillary sinus`, `Bone Loss`, `Fracture teeth`, `Permanent Teeth`, `Supra Eruption`, `TAD`, `abutment`, `attrition`, `bone defect`, `gingival former`, `metal band`, `orthodontic brackets`, `permanent retainer`, `post - core`, `plating`, `wire`, `Cyst`, `Root resorption`, and `Primary teeth`.

## Project structure

```text
.
├── dataset/
│   └── data.yaml                 # portable 31-class generated-dataset config
├── notebooks/
│   └── dental_yolo26_colab.ipynb # end-to-end Colab workflow
├── src/
│   ├── colab_workflow.py         # Kaggle download, prepare, verify, train
│   ├── common.py                 # shared file, hashing, and YAML helpers
│   ├── generate_report.py        # final report and progress generator
│   ├── prepare_dataset.py        # QC, grouping, splitting, augmentation
│   ├── train_evaluate.py         # training, selection, test, error analysis
│   └── verify_dataset.py         # processed-dataset integrity checks
├── tests/
│   └── test_pipeline.py
├── requirements.txt
└── requirements-colab.txt
```

These local/generated paths are intentionally not committed:

```text
Dental X-Ray Panoramic Dataset/  # raw YOLO/COCO source and source checkpoints
dataset/images/                  # prepared images
dataset/labels/                  # prepared labels
artifacts/                       # selected weights and environment snapshot
reports/                         # audit, metrics, plots, and error analysis
runs/                            # Ultralytics run outputs
```

## Local setup

Requirements:

- Python 3.11 or newer
- A CUDA-capable GPU is recommended for training
- Enough disk space for the raw and rebuilt datasets

```powershell
git clone https://github.com/sina-04/dental-yolo26-detection.git
cd dental-yolo26-detection
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Place the extracted dataset at the default location:

```text
Dental X-Ray Panoramic Dataset/
└── YOLO/
    └── YOLO/
        ├── data.yaml
        ├── train/{images,labels}/
        ├── valid/{images,labels}/
        └── test/{images,labels}/
```

You can instead pass any parent directory containing the 31-class YOLO export with `--source-root`.

## Prepare and verify

Rebuild the processed dataset from the local default path:

```powershell
python -m src.prepare_dataset --rebuild
python -m src.verify_dataset
```

For a custom extraction location:

```powershell
python -m src.prepare_dataset --source-root "D:\datasets\Dental X-Ray Panoramic Dataset" --rebuild
```

Preparation ignores the publisher's split assignment and reconstructs groups so variants inferred to belong to the same patient/exam cannot cross splits. Because the grouping key is inferred from filenames rather than verified against a clinical patient index, leakage prevention remains best-effort.

Useful preparation options:

```text
--augment-fraction 0.15
--minority-target-instances 128
--max-augmentations-per-image 8
--seed 42
```

## Train and evaluate

Run the default local profile:

```powershell
python -m src.train_evaluate --model yolo26n.pt --device auto
python -m src.generate_report
```

For a constrained smoke run, cap only the training images; validation and test remain complete:

```powershell
python -m src.train_evaluate --model yolo26n.pt --max-train-images 1200 --baseline-epochs 1 --tuned-epochs 2
```

Training outputs are written to `runs/`, `reports/`, and `artifacts/`. These outputs are ignored so a checkpoint trained on the retired mixed ontology cannot be mistaken for a compatible 31-class model. Publish new weights only after retraining and documenting their metrics.

## Google Colab

Open the notebook with the badge above and select a GPU runtime. The workflow:

1. mounts Google Drive;
2. clones this repository;
3. installs Colab-safe dependencies;
4. downloads only Kaggle version 6 of the panoramic dataset;
5. prepares and verifies the 31-class dataset on Colab's local SSD;
6. resumes training from Drive-backed checkpoints when available; and
7. saves metrics, reports, and weights under `MyDrive/dental-yolo26-detection/colab-results/`.

The equivalent command is:

```bash
python -m src.colab_workflow \
  --stage all \
  --data-root /content/dental_yolo26_data \
  --results-root /content/drive/MyDrive/dental-yolo26-detection/colab-results \
  --model yolo26s.pt \
  --imgsz 640 \
  --batch 32 \
  --baseline-epochs 15 \
  --tuned-epochs 100
```

If a Colab GPU runs out of memory, reduce `--batch` to 16 or 8. Use `--rebuild-data` when changing dataset-preparation options or migrating from the former two-dataset pipeline.

## Tests

```powershell
python -m unittest discover -s tests -v
```

The tests cover label parsing, polygon-to-box conversion, privacy-preserving patient grouping, nested dataset discovery, augmentation caps, and resumable Colab markers. `src.verify_dataset` performs the full generated-dataset integrity check.

## Medical-AI limitations

The source is a public secondary dataset with uncertain demographics, acquisition devices, clinical sampling, annotation protocol, and patient metadata. The labels mix pathology, treatments, devices, teeth, and anatomy. Some classes are rare, publisher annotations may be inconsistent, and bounding boxes derived from polygons discard lesion shape. There is no external-site, prospective, calibration, robustness, fairness, reader-study, regulatory, or clinical-utility validation.

False negatives can miss disease and false positives can cause unnecessary concern. A qualified dental clinician must review the labels, failure cases, and any intended use before further research or deployment.

## License

No license is granted for the raw dataset by this repository; the upstream dataset terms apply independently. No separate open-source license has been assigned to this repository unless a `LICENSE` file is added later.
