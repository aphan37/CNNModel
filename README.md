# AlzhiNet: MRI-Based Alzheimer's Severity Staging

[![Tests](https://github.com/aphan37/CNNModel/actions/workflows/tests.yml/badge.svg)](https://github.com/aphan37/CNNModel/actions/workflows/tests.yml)

A PyTorch pipeline that harmonizes NACC clinical records with MRI scans and
classifies each patient's severity on the standard **Clinical Dementia
Rating (CDR)** scale. It covers patient-level data splitting, Gaussian
noise reduction, class-imbalance-aware training, ordinal evaluation,
Grad-CAM explainability, and a Streamlit demo.

> **Research project, not a medical device.** Not validated for clinical use.

> Supported through the SURI initiative under the mentorship of Dr. Sriram
> Srinivasan and Dr. Ruth Agada. Team: Lawrence Miggins, Chibueze Oburuoh,
> Kevin Elias Mejia, Darryl Lomax Jr, Lauren Buriss.

## Severity scale (standard CDR)

| CDR | Label | Meaning |
|---|---|---|
| 0 | Normal | No cognitive impairment |
| 0.5 | MCI | Mild cognitive impairment / very mild |
| 1 | Mild | Mild dementia |
| 2 | Moderate | Moderate dementia |
| 3 | Severe | Severe dementia |

Defined once in `config.py` and imported by every script.

## Pipeline

```
data_pipeline.py   Harmonize NACC CSV + MRI .jpg files, split by PATIENT
      |
preprocessing.py   Gaussian smoothing (sigma 1.0-2.0), dataset mean/std
      |
train.py           AlzhiNet, class-weighted loss, early stopping,
      |            quadratic-weighted kappa + confusion matrix
      |
gradcam_cli.py / app.py   Single-image prediction + Grad-CAM (CLI / Streamlit)
```

## Quick start

```bash
git clone https://github.com/aphan37/CNNModel.git
cd CNNModel
pip install -r requirements.txt
python -m pytest tests/ -v        # 15 tests, no data required
```

To train, place the NACC clinical CSV and labeled `.jpg` folder in the repo
root (paths at the top of `config.py`), then:

```bash
python data_pipeline.py                 # organize + patient-level split
python preprocessing.py sweep           # compare sigma 1.0-2.0 -> results/sigma_sweep.png
python preprocessing.py apply           # apply config.GAUSSIAN_SIGMA
python preprocessing.py stats           # dataset mean/std -> results/dataset_stats.json
python train.py                         # train + evaluate -> models/, results/

python gradcam_cli.py --image path/to/scan.jpg --gradcam
streamlit run app.py
```

See [`FINE_TUNING_GUIDE.md`](FINE_TUNING_GUIDE.md) for the tuning process and
baseline protocol, and [`MODEL_CARD.md`](MODEL_CARD.md) for results and limits.

## Design decisions

- **Patient-level splitting.** NACC patients have multiple visits/scans, so
  images are grouped by NACCID and stratified by severity; no patient appears
  in more than one of train/val/test. Splitting by file would leak patient
  anatomy across splits and inflate accuracy.
- **Gaussian smoothing (sigma 1.0-2.0).** Suppresses high-frequency scanner
  noise while keeping anatomical edges; label-independent, so it adds no leakage.
- **Ordinal-aware evaluation.** Severity is ordered, so a Normal-vs-Severe
  miss should cost more than Mild-vs-Moderate. Reports quadratic-weighted
  Cohen's kappa alongside accuracy, and class indices follow clinical order
  (not alphabetical folder order).
- **Class-weighted loss** for NACC's skewed severity distribution.
- **Dataset-specific normalization**, since the model trains from scratch.
- **Explainability.** Grad-CAM shows which regions drove each prediction.

## Results

Results are reported in [`MODEL_CARD.md`](MODEL_CARD.md) once the leakage-free
patient-level run is complete. Earlier image-level-split runs reported ~99%
accuracy; those numbers are likely inflated by patient leakage and are
intentionally not reported here.

## Data

MRI images and clinical records come from the NACC (National Alzheimer's
Coordinating Center) data center and are governed by its data use agreement.
**No patient data is included in this repository**, and `.gitignore` excludes
CSVs, images, and checkpoints so none is committed by accident.

## Repo layout

```
config.py            paths, CDR mapping, hyperparameters (single source of truth)
AppModel.py          AlzhiNet architecture (shared by train / Grad-CAM / app)
data_pipeline.py     harmonization + patient-level stratified split
preprocessing.py     Gaussian smoothing, sigma sweep, dataset stats
train.py             training + evaluation
gradcam_cli.py       CLI classifier + Grad-CAM
app.py               Streamlit demo
tests/               pytest suite (labels, ordering, leakage, preprocessing, Grad-CAM, app)
```
