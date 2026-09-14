# Metric Improvement Plan

## Primary objective

The v2 primary detector uses six source labels: Caries, Periapical lesion, Retained root, Root Piece, impacted tooth, and Bone Loss. The unchanged 31-class workflow remains a secondary benchmark.

The acceptance baseline is YOLO26s at 640 px, reproduced on the same v2 dataset fingerprint and split. The final candidate must achieve:

- at least 25% relative improvement in macro mAP50-95;
- at least 15 percentage points improvement in macro recall;
- no primary-class recall regression greater than five percentage points; and
- patient-bootstrap intervals that exclude zero improvement for operating-point macro metrics.

Ordinary accuracy and precision without recall are not acceptance metrics.

## Data gate

Preparation remaps the six labels into a contiguous pathology view while keeping every image, including negatives. Final training requires at least 200/50/50 independent train/validation/test patient groups for every primary class. The support report is written to `reports/pathology_support.json`.

The strict profile stops when this gate fails. The exploratory profile allows method development on current public data but its results cannot satisfy acceptance. All validation and test annotations, plus model-mined false positives and false negatives, require clinician review before final training.

## Experiment sequence

1. Reproduce YOLO26s/640.
2. Compare YOLO26s/1024 and YOLO26m/1024.
3. Compare the patient-balanced training view.
4. Screen 24 medically constrained hyperparameter configurations for 25 epochs on validation data only.
5. Train the best three configurations for up to 120 epochs with patience 30.
6. Confirm the winner with seeds 42, 43, and 44.
7. Evaluate full-image plus three-tile inference, merging class-matched boxes at IoU 0.55.
8. Freeze the winner and run the test stage once.

The search permits mild rotation, translation, scale, brightness/contrast, and CLAHE. Vertical flips, hue shifts, MixUp, CutMix, and aggressive geometry remain disabled.

## Evaluation contract

Model and threshold selection use validation data. Each class receives a validation-selected F2 threshold subject to precision of at least 0.60. The final report includes mAP50-95, mAP50, precision, recall, F1/F2, per-class support, patient-level bootstrap intervals, inference latency, dataset fingerprint, checkpoint hash, and all seeds.

The test set is not used for tuning. Existing v1 metrics remain historical because the v2 split and ontology view have a new fingerprint.
