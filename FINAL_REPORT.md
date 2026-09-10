# Dental YOLO26 Detection Project — Final Report

## Executive summary

This project uses both requested Kaggle datasets to build a single YOLO26 object detector. The class ontologies are intentionally namespaced (`anatomy_*` and `disease_*`) because the first dataset labels tooth anatomy while the second labels diseases, treatments, and anatomical structures. Segmentation polygons in the disease dataset were converted to axis-aligned detection boxes. Patient/exam groups inferred from filenames were hashed before being written to processed manifests, and no patient group crosses train, validation, and test splits.

The selected model is **tuned_yolo26n_medical_aug**, chosen only by validation mAP50-95. Its held-out test mAP50-95 is **0.0280**, mAP50 is **0.0409**, precision is **0.8733**, and recall is **0.0197**. At the validation-selected confidence threshold of **0.05**, the custom IoU=0.50 test F1 is **0.0113**.

The executed local training profile used **1,200 of 11,481 available training images** with deterministic rare-class coverage because the available MX330 has only 2 GB VRAM. Validation and test sets remained complete. This result is a compute-constrained baseline, not the final accuracy ceiling of the full dataset.

The result is **not clinically usable**: at the validation-selected 0.05 threshold it missed **8,708 of 8,758** labeled test objects (recall 0.0057). The high aggregate precision must not be read in isolation; the model is severely recall-limited after this short local training profile.

This is an educational research result, not a clinically validated diagnostic device.

## Data provenance and licensing

- [Dental Anatomy Dataset — YOLOv8](https://www.kaggle.com/datasets/saisiddartha69/dental-anatomy-dataset-yolov8), Kaggle version 1, Kaggle metadata license CC BY-SA 4.0. The embedded Roboflow export declares CC BY 4.0; downstream users should resolve this license inconsistency before redistribution.
- [Dental Disease Panoramic Detection Dataset](https://www.kaggle.com/datasets/lokisilvres/dental-disease-panoramic-detection-dataset), Kaggle version 6, Apache 2.0.

The anatomy export contains seven tooth-type classes. The panoramic export contains 31 classes, including diseases, treatments, devices, teeth, and anatomy. These are not semantically interchangeable, so merging by numeric class ID would have been invalid.

## Quality control and preparation

- Usable images after corruption checks and exact-deduplication: **14599**.
- Source counts: `{'anatomy': 724, 'disease': 13875}`.
- Final split counts: `{'train': 10935, 'val': 2192, 'test': 1472}`.
- Exact duplicate files removed: **57**.
- Disease segmentation polygons converted to boxes: **136891**.
- Patient-group leakage count: **0**.
- Offline medically mild AlbumentationsX training images: **546**.
- Coordinate/annotation issues: `{'clipped_box': 51, 'empty_annotation': 45, 'tiny_box': 5, 'out_of_bounds': 3}`.

Automatic QC checked image decodability, missing/malformed labels, class ranges, coordinate ranges, non-positive and tiny boxes, exact hashes, perceptual-hash near-duplicate candidates, source pairing, and split leakage. One annotated sample per observed class is generated locally in `reports/annotation_audit/`. Those clinical-image derivatives are excluded from the public Git repository pending explicit privacy and redistribution review. The montages support local review but do not substitute for a dentist/radiologist annotation audit.

## Class distribution

| ID | Class | Images | Instances |
|---:|---|---:|---:|
| 0 | anatomy_1st_molar | 679 | 2137 |
| 1 | anatomy_1st_premolar | 721 | 2543 |
| 2 | anatomy_2nd_molar | 459 | 1043 |
| 3 | anatomy_2nd_premolar | 707 | 2405 |
| 4 | anatomy_canine | 722 | 2593 |
| 5 | anatomy_central_incisor | 724 | 2609 |
| 6 | anatomy_lateral_incisor | 724 | 2605 |
| 7 | disease_caries | 3046 | 10671 |
| 8 | disease_crown | 4065 | 11189 |
| 9 | disease_filling | 9905 | 48846 |
| 10 | disease_implant | 610 | 1782 |
| 11 | disease_malaligned | 12 | 18 |
| 12 | disease_mandibular_canal | 320 | 619 |
| 13 | disease_missing_teeth | 1644 | 3467 |
| 14 | disease_periapical_lesion | 2359 | 5250 |
| 15 | disease_retained_root | 72 | 174 |
| 16 | disease_root_canal_treatment | 5462 | 19020 |
| 17 | disease_root_piece | 954 | 2603 |
| 18 | disease_impacted_tooth | 11369 | 27876 |
| 19 | disease_maxillary_sinus | 233 | 462 |
| 20 | disease_bone_loss | 1489 | 3130 |
| 21 | disease_fracture_teeth | 10 | 11 |
| 22 | disease_permanent_teeth | 5 | 12 |
| 23 | disease_supra_eruption | 39 | 48 |
| 24 | disease_tad | 3 | 4 |
| 25 | disease_abutment | 22 | 33 |
| 26 | disease_attrition | 15 | 44 |
| 27 | disease_bone_defect | 1 | 1 |
| 28 | disease_gingival_former | 7 | 14 |
| 29 | disease_metal_band | 36 | 65 |
| 30 | disease_orthodontic_brackets | 75 | 136 |
| 31 | disease_permanent_retainer | 5 | 8 |
| 32 | disease_post_core | 172 | 315 |
| 33 | disease_plating | 2 | 8 |
| 34 | disease_wire | 136 | 231 |
| 35 | disease_cyst | 5 | 5 |
| 36 | disease_root_resorption | 2 | 3 |
| 37 | disease_primary_teeth | 47 | 228 |

## Split strategy

The pipeline reconstructs splits from patient/exam groups instead of trusting the publisher-provided Roboflow split. For the anatomy set, the original figure identifier is the conservative group. For the disease set, tokens likely to identify a patient are normalized, immediately hashed, and never written in clear text to the processed manifest. A deterministic greedy multilabel group allocation targets 75% train, 15% validation, and 10% test while reducing class-distribution drift. Exact visual duplicates are removed before splitting; perceptual near-duplicates remain grouped and are listed for review.

Because patient identifiers were inferred from filenames rather than verified against a clinical master index, patient separation is best-effort and remains a limitation.

## Model and training configuration

- Ultralytics: **8.4.145**
- Model/checkpoint: **YOLO26n / `yolo26n.pt`**, COCO-pretrained transfer learning
- Python: **3.12.10**
- PyTorch: **2.7.1+cu118**
- Device: **0** (NVIDIA GeForce MX330)
- Mosaic, MixUp, CutMix, Copy-Paste: **disabled**
- Hue/saturation, vertical flip, horizontal flip: **disabled**
- Early stopping, weight decay 0.0005, deterministic seed 42: **enabled**
- Offline AlbumentationsX: rotation ±7°, mild brightness/contrast, low-probability 3×3 Gaussian blur or CLAHE; bounding boxes transformed together with images

YOLO26n was selected instead of YOLO26s because the available MX330 has only 2 GB VRAM. This compute-driven choice is documented rather than presented as an accuracy-optimal model-size comparison.

## Validation experiment comparison

| Experiment | Model | Main change | Epochs | Precision | Recall | mAP50 | mAP50-95 | Selected |
|---|---|---|---:|---:|---:|---:|---:|---|
| baseline_yolo26n | yolo26n.pt | minimal | 1 | 1.82% | 2.87% | 0.0130 | 0.0105 | False |
| tuned_yolo26n_medical_aug | yolo26n.pt | medical_mild | 2 | 85.90% | 1.97% | 0.0392 | 0.0260 | True |

## Final held-out test evaluation

The held-out test set was evaluated once after experiment selection. Ultralytics mAP uses confidence-swept precision-recall curves; the operating threshold was selected on validation data, never on test data.

- Precision: **0.8733**
- Recall/sensitivity: **0.0197**
- mAP50: **0.0409**
- mAP50-95: **0.0280**
- Validation-selected threshold: **0.05**
- Thresholded test F1 at IoU 0.50: **0.0113**
- Thresholded TP / FP / FN: **50 / 37 / 8708**

### Per-class test metrics

| ID | Class | Test support | Precision | Recall | F1 | mAP50-95 |
|---:|---|---:|---:|---:|---:|---:|
| 0 | anatomy_1st_molar | 102 | 0.000 | 0.000 | 0.000 | 0.002 |
| 1 | anatomy_1st_premolar | 112 | 0.000 | 0.000 | 0.000 | 0.000 |
| 2 | anatomy_2nd_molar | 39 | 0.000 | 0.000 | 0.000 | 0.000 |
| 3 | anatomy_2nd_premolar | 111 | 0.000 | 0.000 | 0.000 | 0.001 |
| 4 | anatomy_canine | 112 | 0.444 | 0.036 | 0.066 | 0.204 |
| 5 | anatomy_central_incisor | 112 | 0.738 | 0.402 | 0.520 | 0.459 |
| 6 | anatomy_lateral_incisor | 112 | 0.167 | 0.009 | 0.017 | 0.304 |
| 7 | disease_caries | 184 | 0.000 | 0.000 | 0.000 | 0.000 |
| 8 | disease_crown | 23 | 0.000 | 0.000 | 0.000 | 0.000 |
| 9 | disease_filling | 4866 | 0.000 | 0.000 | 0.000 | 0.000 |
| 10 | disease_implant | 9 | 0.000 | 0.000 | 0.000 | 0.000 |
| 11 | disease_malaligned | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| 12 | disease_mandibular_canal | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| 13 | disease_missing_teeth | 10 | 0.000 | 0.000 | 0.000 | 0.000 |
| 14 | disease_periapical_lesion | 10 | 0.000 | 0.000 | 0.000 | 0.000 |
| 15 | disease_retained_root | 6 | 0.000 | 0.000 | 0.000 | 0.000 |
| 16 | disease_root_canal_treatment | 51 | 0.000 | 0.000 | 0.000 | 0.000 |
| 17 | disease_root_piece | 11 | 0.000 | 0.000 | 0.000 | 0.000 |
| 18 | disease_impacted_tooth | 2833 | 0.000 | 0.000 | 0.000 | 0.004 |
| 19 | disease_maxillary_sinus | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| 20 | disease_bone_loss | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| 21 | disease_fracture_teeth | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| 22 | disease_permanent_teeth | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| 23 | disease_supra_eruption | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| 24 | disease_tad | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| 25 | disease_abutment | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| 26 | disease_attrition | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| 27 | disease_bone_defect | 0 | N/E | N/E | N/E | N/E |
| 28 | disease_gingival_former | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| 29 | disease_metal_band | 3 | 0.000 | 0.000 | 0.000 | 0.000 |
| 30 | disease_orthodontic_brackets | 6 | 0.000 | 0.000 | 0.000 | 0.006 |
| 31 | disease_permanent_retainer | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| 32 | disease_post_core | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| 33 | disease_plating | 0 | N/E | N/E | N/E | N/E |
| 34 | disease_wire | 2 | 0.000 | 0.000 | 0.000 | 0.000 |
| 35 | disease_cyst | 1 | 0.000 | 0.000 | 0.000 | 0.000 |
| 36 | disease_root_resorption | 0 | N/E | N/E | N/E | N/E |
| 37 | disease_primary_teeth | 19 | 0.000 | 0.000 | 0.000 | 0.000 |

`N/E` means not estimable because the held-out test split contains no ground-truth instance of that ultra-rare class; it is not a zero-performance claim.

The confusion matrix and training/validation loss plots are retained under `runs/`; `reports/threshold_analysis.png` shows the validation precision-recall-F1 operating-point tradeoff.

## Error analysis and unseen inference examples

Ground truth is green and model prediction is red in the ten locally saved test examples. The images are excluded from the public Git repository pending explicit privacy and redistribution review; their privacy-hashed IDs and structured results remain in `reports/error_analysis.csv`.

| Example | Observed outcomes | Expected result | Model prediction | Probable cause | Proposed improvement |
|---|---|---|---|---|---|
| example_01_anatomy_000375_b2f220ed.jpg | false positive; false negative; misclassification | anatomy_central_incisor x4; anatomy_lateral_incisor x4; anatomy_canine x4; anatomy_1st_premolar x4; anatomy_2nd_premolar x4; anatomy_1st_molar x4; anatomy_2nd_molar x3 | anatomy_central_incisor x3; anatomy_lateral_incisor x1 | Overlapping visual features, class imbalance, or annotation inconsistency. | Clinical relabel review, more patient-diverse data, and class-aware sampling. |
| example_02_anatomy_000044_03e0264a.jpg | correct detection; false positive; false negative; misclassification | anatomy_central_incisor x4; anatomy_lateral_incisor x4; anatomy_canine x4; anatomy_1st_premolar x4; anatomy_2nd_molar x4; anatomy_1st_molar x4; anatomy_2nd_premolar x3 | anatomy_central_incisor x4; anatomy_lateral_incisor x2 | Overlapping visual features, class imbalance, or annotation inconsistency. | Clinical relabel review, more patient-diverse data, and class-aware sampling. |
| example_03_anatomy_000376_a6f2bb96.jpg | correct detection; false positive; false negative; misclassification | anatomy_central_incisor x4; anatomy_lateral_incisor x4; anatomy_canine x4; anatomy_1st_premolar x4; anatomy_2nd_premolar x4; anatomy_1st_molar x4 | anatomy_central_incisor x3; anatomy_canine x2; anatomy_lateral_incisor x1 | Overlapping visual features, class imbalance, or annotation inconsistency. | Clinical relabel review, more patient-diverse data, and class-aware sampling. |
| example_04_anatomy_000028_58098eba.jpg | correct detection; false positive; false negative; misclassification | anatomy_central_incisor x4; anatomy_lateral_incisor x4; anatomy_canine x4; anatomy_1st_premolar x4; anatomy_2nd_premolar x4; anatomy_1st_molar x4; anatomy_2nd_molar x2 | anatomy_central_incisor x2 | Overlapping visual features, class imbalance, or annotation inconsistency. | Clinical relabel review, more patient-diverse data, and class-aware sampling. |
| example_05_anatomy_000047_df2d411a.jpg | correct detection; false positive; false negative; misclassification | anatomy_central_incisor x4; anatomy_lateral_incisor x4; anatomy_canine x4; anatomy_1st_premolar x4; anatomy_2nd_premolar x4; anatomy_1st_molar x4; anatomy_2nd_molar x3 | anatomy_central_incisor x2; anatomy_canine x1 | Overlapping visual features, class imbalance, or annotation inconsistency. | Clinical relabel review, more patient-diverse data, and class-aware sampling. |
| example_06_disease_013717_353df961.jpg | false negative | disease_crown x8; disease_root_piece x8; disease_missing_teeth x4; disease_maxillary_sinus x2; disease_mandibular_canal x2; disease_filling x1; disease_root_canal_treatment x1 | No prediction at selected threshold | Low contrast, small lesion, limited training, or conservative threshold. | Clinical relabel review, more patient-diverse data, and class-aware sampling. |
| example_07_disease_009871_726a6ccc.jpg | false negative | disease_caries x21 | No prediction at selected threshold | Low contrast, small lesion, limited training, or conservative threshold. | Clinical relabel review, more patient-diverse data, and class-aware sampling. |
| example_08_disease_013554_0a1c64b9.jpg | false negative | disease_primary_teeth x17; disease_caries x4 | No prediction at selected threshold | Low contrast, small lesion, limited training, or conservative threshold. | Clinical relabel review, more patient-diverse data, and class-aware sampling. |
| example_09_disease_013658_750513b9.jpg | false negative | disease_caries x21 | No prediction at selected threshold | Low contrast, small lesion, limited training, or conservative threshold. | Clinical relabel review, more patient-diverse data, and class-aware sampling. |
| example_10_disease_000942_3bc87ae3.jpg | false negative | disease_caries x19 | No prediction at selected threshold | Low contrast, small lesion, limited training, or conservative threshold. | Clinical relabel review, more patient-diverse data, and class-aware sampling. |

Recurring risks include small lesions, low contrast, overlapping anatomy, severe class imbalance, publisher annotation inconsistency, polygon-to-box loss of shape information, and domain differences between anatomy imagery and panoramic radiographs. The next scientifically useful step is blinded review of false negatives and annotation candidates by qualified dental clinicians, followed by class-aware additional collection rather than indiscriminate synthetic augmentation.

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

## Limitations and medical-AI warning

The datasets are public secondary datasets with uncertain clinical sampling, demographics, device distributions, labeling protocols, and patient metadata. The combined taxonomy mixes anatomy, disease, treatments, and devices. Bounding boxes derived from segmentation discard lesion shape. Some classes may be too rare to estimate stable performance, and high aggregate mAP can hide clinically important minority-class failure. Patient grouping is inferred, not externally verified. There is no external-site, prospective, reader-study, calibration, robustness, fairness, security, regulatory, or clinical-utility validation.

False negatives could miss disease; false positives could trigger unnecessary concern or follow-up. Predictions must not be interpreted as diagnoses or used for patient care.

## Reproducibility artifacts

- `src/prepare_dataset.py`: acquisition extraction, validation, conversion, grouping, splitting, augmentation, and audit
- `src/train_evaluate.py`: transfer learning, experiment selection, final test evaluation, threshold analysis, inference, and error analysis
- `dataset/data.yaml`: absolute local dataset configuration and 38-class ontology
- `reports/dataset_manifest.csv`: privacy-hashed split manifest
- `reports/dataset_audit.json`: QC summary
- `reports/experiments.csv`: experiment ledger
- `reports/final_metrics.json`: complete selected/test metrics
- `artifacts/best.pt` and `artifacts/last.pt`: selected YOLO26 checkpoints
- `artifacts/training_configuration.json` and `environment_freeze.txt`: exact run details
