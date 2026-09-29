# Model Card: AlzhiNet baseline

Fill this in after each frozen baseline run. Leave nothing as "TBD" when you
report a number publicly.

## Summary
- **Task:** 5-class CDR severity staging (Normal, MCI, Mild, Moderate, Severe) from MRI slices
- **Architecture:** AlzhiNet, 3 conv blocks (Conv-BatchNorm-ReLU-MaxPool) + global average pooling + 2 FC layers, trained from scratch (~640K params total)
- **Status:** baseline run pending on the patient-level split

## Run details
| Field | Value |
|---|---|
| Date / git commit | |
| Gaussian sigma (and why) | see `results/sigma_sweep.png` |
| Random seed | 42 |
| Unique patients (train / val / test) | |
| Images per class (train / val / test) | |
| Multi-visit patients | stratified by most severe recorded label |

## Test results (patient-level split)
| Metric | Value |
|---|---|
| Accuracy | |
| Quadratic-weighted Cohen's kappa | |
| Per-class recall (Normal / MCI / Mild / Moderate / Severe) | |

Confusion matrix: attach `results/test_report.json`.

## Intended use
Research and education on MRI-based severity staging. Not for diagnosis,
screening, or any clinical decision.

## Limitations
- Trained and evaluated on one data source (NACC); no external validation
- Rare classes (typically Severe) have few held-out patients, so per-class
  metrics for them are high-variance
- 2D slices only; no volumetric context
- JPEG conversion is lossy compared with the original scan format
- Add anything else you observe (scanner mix, demographics, class imbalance)

## Data
NACC data under its data use agreement; not redistributed here.
