# Deliverable Progress

| Deliverable | Progress | Completed | Problems encountered | Knowledge / evidence gained | Next action |
|---|---:|---|---|---|---|
| Training source code | 100% | End-to-end preparation, training, validation, testing, inference, and error-analysis CLIs | Local compute constrained model scale and epoch budget | Reproducible YOLO26 workflow | Re-run on a larger GPU for longer experiments |
| Final model weights | 100% | Selected `best.pt` and retained `last.pt` | YOLO26s was infeasible on 2 GB VRAM | Nano is the viable local baseline | Benchmark s/m on cloud GPU without opening test set |
| Dataset and configuration | 100% | Both sources combined, polygons boxified, group split, data.yaml generated | Source ontologies differ; inferred IDs only | Namespacing avoids false semantic merges | Clinician review and verified patient index |
| Evaluation report | 100% | Full test metrics, per-class table, losses, confusion matrix, PR and threshold analysis | Rare classes yield unstable estimates | Per-class reporting exposes imbalance | Add confidence intervals with larger external test data |
| Ten unseen examples | 100% | Representative held-out test visualizations | Some error types may not occur in a small test set | Model behavior is inspectable | Blinded expert review |
| Error analysis | 100% | Per-image structured categories and candidate causes | Candidate causes are hypotheses, not clinical adjudication | False-negative audit is highest priority | Clinician adjudication |
| Overfitting report | 100% | Only techniques actually used are documented | Compute limits model-size comparison | Leakage control matters more than cosmetic tuning | External validation |
| Training configuration | 100% | Versions, seed, device, model, augmentation, hyperparameters, duration | Low-memory GPU | Exact environment is frozen | Reproduce on CUDA GPU |
| Experiment comparison | 100% | Baseline vs medically augmented experiment | Same model size due VRAM | Validation-only selection preserved test integrity | Add yolo26s/m experiments remotely |
| Reproducibility package | 100% | Code, configs, reports, weights, README, lock-style freeze | Raw Kaggle data must remain license-compliant | Entire run is scriptable | Archive with checksums under approved license terms |

## Mentor discussion

- Is the 38-class namespaced ontology clinically appropriate, or should anatomy and pathology be separate models?
- Can a verified patient identifier replace filename-based grouping?
- Which false-negative classes are clinically highest priority?
- Is external-site data available for a genuine generalization test?
