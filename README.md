# AlzhiNet — CNN-Based MRI Classifier for Alzheimer's Severity Staging

A convolutional neural network pipeline that harmonizes NACC clinical
records with MRI scans and classifies each patient's Alzheimer's severity
on the standard **Clinical Dementia Rating (CDR)** scale: Normal, MCI
(very mild), Mild, Moderate, or Severe. The pipeline includes patient-level
data harmonization, Gaussian noise reduction, model training with
class-imbalance handling, ordinal-aware evaluation, and Grad-CAM
explainability.

> This research was supported through the SURI initiative under the
> mentorship of Dr. Sriram Srinivasan and Dr. Ruth Agada. Team members
> contributing on this project: Lawrence Miggins, Chibueze Oburuoh, Kevin
> Elias Mejia, Darryl Lomax Jr, Lauren Buriss.

## Severity scale used

This project follows the standard CDR scale used by NACC, not an ad-hoc
labeling scheme:

| CDR score | Label | Meaning |
|---|---|---|
| 0 | Normal | No cognitive impairment |
| 0.5 | MCI | Mild Cognitive Impairment / "very mild" |
| 1 | Mild | Mild dementia |
| 2 | Moderate | Moderate dementia |
| 3 | Severe | Severe dementia |

This mapping is defined once, in `config.py`, and every other script
(`data_pipeline.py`, `train.py`, `gradcam_cli.py`) imports it from there —
nothing hardcodes its own copy of the label list anymore.

## Pipeline

```
data_pipeline.py     Harmonize NACC CSV + MRI images, organize by class,
                      split into train/val/test at the PATIENT level
        │
        ▼
preprocessing.py      Gaussian smoothing (sigma 1.0-2.0) to reduce MRI
                      scanner noise; also computes dataset-specific
                      normalization stats
        │
        ▼
train.py              Train AlzhiNet with class-weighted loss, early
                      stopping, and ordinal-aware evaluation (quadratic-
                      weighted Cohen's Kappa + confusion matrix)
        │
        ▼
gradcam_cli.py         Classify a single image and visualize which regions
                      of the MRI the model attended to (explainability)
```

## Repo structure

```
├── config.py           # single source of truth: paths, CDR label mapping, hyperparameters
├── AppModel.py          # AlzhiNet architecture (imported by train.py and gradcam_cli.py)
├── data_pipeline.py     # harmonize CSV+images, patient-level stratified split
├── preprocessing.py     # Gaussian smoothing, sigma sweep, dataset stats
├── train.py             # training loop + evaluation (kappa, confusion matrix)
├── gradcam_cli.py        # single-image classification + Grad-CAM visualization
├── requirements.txt
└── FINE_TUNING_GUIDE.md # how to tune, standardize, and cross-validate the baseline
```

`CNN.py` and the original `gradcamCLI.py` are superseded by the files
above — see "What changed" below for why.

## Setup

```bash
git clone https://github.com/aphan37/CNNModel.git
cd CNNModel
pip install -r requirements.txt
```

Place your NACC clinical CSV and labeled MRI `.jpg` folder in the repo
root, matching the paths at the top of `config.py`.

## How to run

```bash
# 1. Harmonize CSV + images, split by PATIENT (not by file) into train/val/test
python data_pipeline.py

# 2. Pick a Gaussian sigma (visually compare 1.0-2.0), then apply it
python preprocessing.py sweep      # writes results/sigma_sweep.png
python preprocessing.py apply      # applies config.GAUSSIAN_SIGMA to all images

# 3. Compute this dataset's own normalization stats (saved automatically,
#    train.py and gradcam_cli.py load it — no manual copy-paste)
python preprocessing.py stats

# 4. Train + evaluate
python train.py

# 5. Classify a single image with Grad-CAM explainability
python gradcam_cli.py --image test_images/sample.jpg --gradcam
```

See [`FINE_TUNING_GUIDE.md`](FINE_TUNING_GUIDE.md) for the full tuning
process, what counts as the frozen "baseline," and how to run
cross-validation before reporting a final number.

## What changed from the original pipeline, and why

- **Standard CDR labels everywhere.** The original label mapping had 0.5
  and 1 swapped relative to the real CDR scale, and labeled CDR=1
  "CognitivelyIntact" (CDR=1 is mild *dementia*, not intact cognition).
  Now defined once in `config.py` and imported everywhere else.
- **Patient-level split, not image-level.** NACC patients have multiple
  scans/visits; splitting by individual image file let the same patient's
  scans land in both train and test, which the model can exploit —
  likely the real explanation behind previously reported 98–99% accuracy.
  `data_pipeline.py` now splits by NACCID, stratified by severity.
- **Gaussian smoothing added.** `preprocessing.py` applies Gaussian
  smoothing (sigma 1.0–2.0, tunable) between harmonization and model
  input to suppress scanner noise while preserving anatomical edges —
  previously referenced in the README but not actually implemented.
- **Dataset-specific normalization**, not ImageNet stats — this model is
  trained from scratch, not fine-tuned from an ImageNet backbone.
- **Grad-CAM fixed, not removed.** The original `gradcamCLI.py` imported
  a module (`cnn_app_model`) that doesn't exist — Grad-CAM has been
  non-functional since whatever rename left that import stale. It's a
  meaningful piece of the project (explainability matters for
  clinical-adjacent ML), so it's fixed rather than dropped: correct
  import, correct label order, a dynamic conv-layer lookup instead of a
  hardcoded index, consistent normalization, and a non-deprecated hook API.
- **Ordinal-aware evaluation.** Severity is ordinal (Normal < MCI < Mild <
  Moderate < Severe); plain accuracy treats a Normal-vs-Severe mistake the
  same as a Mild-vs-Moderate mistake. `train.py` now reports
  quadratic-weighted Cohen's Kappa alongside accuracy, plus a full
  confusion matrix (the original only printed a scalar accuracy despite
  the README promising a confusion matrix).
- **Class-weighted loss, early stopping, LR scheduling, reproducibility
  seeding** — none of which the original training loop had.

## Tests

```bash
python -m pytest tests/ -v
```

12 tests cover the properties this pipeline depends on: standard CDR label
mapping, clinical (not alphabetical) class ordering, empty-class handling,
patient-level splitting with no leakage across train/val/test, Gaussian
smoothing, and Grad-CAM output. They run automatically in CI on every push.

## Requirements

- Python 3.9+
- PyTorch, torchvision
- pandas, scikit-learn
- OpenCV (`opencv-python`)
- matplotlib (for the sigma sweep and Grad-CAM display)
- tqdm
