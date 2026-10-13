# Model Card: AlzhiNet baseline

Fill this in after each frozen baseline run. Leave nothing as "TBD" when you
report a number publicly.

## Summary
- **Task:** 5-class CDR severity staging (Normal, MCI, Mild, Moderate, Severe) from MRI slices
- **Architecture:** AlzhiNet, 3 conv blocks (Conv-BatchNorm-ReLU-MaxPool) + global average pooling + 2 FC layers, trained from scratch (~640K params total)
- **Status:** baseline run pending on the patient-level split (the sigma ensemble, below, is optional and also pending)

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

## Sigma ensemble (optional)
Four AlzhiNet models, one per Gaussian sigma, trained on the same patient-level
split and combined by averaging softmax probabilities (`ensemble.py`). Fill in
from `ensemble/ensemble_results.json`; report it even if it does not beat the
best single model.

| Model | Accuracy | Quadratic-weighted kappa |
|---|---|---|
| sigma 1.0 | | |
| sigma 1.25 | | |
| sigma 1.5 | | |
| sigma 2.0 | | |
| **Ensemble (mean of 4)** | | |

## Intended use
Research and education on MRI-based severity staging. Not for diagnosis,
screening, or any clinical decision.

## Limitations
- Trained and evaluated on one data source (NACC); no external validation
- Rare classes (typically Severe) have few held-out patients, so per-class
  metrics for them are high-variance
- 2D slices only; no volumetric context
- JPEG conversion is lossy compared with the original scan format
- Dataset mean/std for the single-model flow is computed over all smoothed
  images (including val/test); the ensemble uses train-only stats
- The sigma ensemble varies only one preprocessing parameter on a single
  architecture, so its diversity is narrow and a gain over the best single
  sigma is not guaranteed
- Numbers from `scripts/generate_sample_data.py` are synthetic smoke-test
  output and must never be reported as results
- Add anything else you observe (scanner mix, demographics, class imbalance)

## Data
NACC data under its data use agreement; not redistributed here.
