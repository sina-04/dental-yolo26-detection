# Manual Annotation Spot Audit

Date: 2026-09-09  
Reviewer: Codex visual inspection; **not a qualified dental clinician**

## Scope

The pipeline generated one labeled visualization for every observed class. Ten representative sheets were visually inspected in detail: anatomy first molar, caries, periapical lesion, bone defect, retained root, fractured teeth, TAD, cyst, root resorption, and primary teeth. The complete 38-class sheet set remains in `reports/annotation_audit/` for clinician review.

## Findings

- Image/label pairing is visually aligned in the inspected samples; boxes generally overlap the intended tooth or jaw region rather than appearing globally displaced.
- The anatomy dataset contains visible publication/watermark text and is an intraoral-photo domain, unlike the panoramic radiographs. This creates a strong source/domain cue that the model may learn.
- The disease dataset has many dense overlapping objects per radiograph. Converted boxes around fillings, crowns, root-canal treatment, and impacted teeth often overlap substantially.
- Several source segmentation regions become very large, coarse axis-aligned boxes after conversion, especially bone loss, orthodontic brackets, primary teeth, and permanent teeth. These boxes include unrelated anatomy and weaken localization precision.
- Small targets such as caries, TAD, fractured teeth, and periapical lesions occupy very few pixels at reduced training resolutions and are vulnerable to false negatives.
- The class distribution is extreme: `disease_filling` has 48,846 instances while bone defect has 1, plating has 8, root resorption has 3, and several other classes have fewer than 20 instances. Metrics for these rare classes cannot be stable.
- Some inspected labels are clinically ambiguous to a non-specialist. No claim is made that disease identity is correct merely because the geometry is plausible.

## Decision

Proceed with a clearly labeled educational baseline while retaining all automated QC flags. Do not treat the source annotations as clinically validated. Before any clinical interpretation, a dentist or oral/maxillofacial radiologist should adjudicate at least all rare classes, all false negatives, the 51 clipped boxes, the 45 empty annotations, and a stratified sample of high-frequency classes.

## Recommended annotation actions

1. Separate anatomical-region, treatment/device, tooth-state, and pathology taxonomies or train task-specific heads.
2. Re-annotate diffuse findings with clinically agreed box rules; consider retaining segmentation for bone loss and large anatomical regions.
3. Remove or mask watermarks only if licensing and study design permit, then test whether source-domain shortcuts remain.
4. Obtain more independent patient groups for classes with fewer than three groups; augmentation cannot replace independent clinical cases.
5. Conduct blinded double-review with disagreement resolution and report inter-rater agreement.
