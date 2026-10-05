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

ensemble.py           (optional) 4 models, one per Gaussian sigma, combined
                       by averaging softmax probabilities -- see below
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
python data_pipeline.py organize        # harmonize CSV + images by class
python preprocessing.py sweep           # compare sigma 1.0-2.0 -> results/sigma_sweep.png
python preprocessing.py apply           # apply config.GAUSSIAN_SIGMA
python preprocessing.py stats           # dataset mean/std -> results/dataset_stats.json
python data_pipeline.py split           # patient-level stratified train/val/test split
python train.py                         # train + evaluate -> models/, results/

python gradcam_cli.py --image path/to/scan.jpg --gradcam
streamlit run app.py
```

Order matters: `organize` must run before `preprocessing.py apply` (which
smooths the organized images), and `split` must run after (it splits from
the smoothed output). Running them out of order fails fast with a clear
error rather than silently doing the wrong thing.

### Try it without real data

No NACC access yet, or just want to confirm the pipeline itself works?

```bash
python scripts/generate_sample_data.py     # synthetic patients + fake MRI-like images
python data_pipeline.py organize
python preprocessing.py apply && python preprocessing.py stats
python data_pipeline.py split
python train.py --epochs 3 --patience 3    # fast smoke run, not a real baseline
python gradcam_cli.py --image dataset/test/Severe/<any_file>.jpg --gradcam
```

This runs the exact same code path as a real run, just on synthetic
"brain slices" with a severity-correlated visual signal. It proves the
pipeline works end to end; it proves nothing about real diagnostic
accuracy. The same flow runs automatically in CI on every push (see
`tests/test_integration.py`).

See [`FINE_TUNING_GUIDE.md`](FINE_TUNING_GUIDE.md) for the tuning process and
baseline protocol, and [`MODEL_CARD.md`](MODEL_CARD.md) for results and limits.

## Sigma ensemble (optional, for extra accuracy)

Instead of picking one Gaussian sigma from the sweep, `ensemble.py` trains
4 models -- one per sigma in `config.ENSEMBLE_SIGMAS` (1.0, 1.25, 1.5, 2.0)
-- on the **same patient-level split**, then combines their predictions by
averaging softmax probabilities (soft voting). This is cheap to build
(reuses the existing preprocessing/training code) and gives the 4 models
some real diversity, since noise sensitivity differs by sigma, without
needing extra data or a second architecture.

```bash
python ensemble.py prepare     # assigns ONE shared patient split, smooths x4, stats x4
python ensemble.py train       # trains all 4 models on the real baseline schedule
python ensemble.py evaluate    # combines predictions, reports ensemble vs. each model

python ensemble.py all --epochs 3 --patience 3   # fast smoke run, all 3 steps
```

All 4 models are guaranteed to share the exact same train/val/test
patients -- the split is computed once (`data_pipeline.py assign`) and
reused across all 4 sigma variants, so their test accuracy numbers are
directly comparable and the combined evaluation is valid. This is checked
automatically by `tests/test_ensemble.py`.

**Honest expectation-setting:** averaging 4 variants of the *same*
architecture, differing only in one preprocessing parameter, gives real but
modest diversity. It's a reasonable cheap first ensemble to try, not a
guaranteed accuracy win -- `ensemble.py evaluate` reports when the ensemble
does and doesn't beat the best individual model, and both outcomes are
worth recording in `MODEL_CARD.md`. A heterogeneous-architecture ensemble
(AlzhiNet + a pretrained ResNet18/DenseNet121) would likely help more, at
roughly 4x the compute cost -- a reasonable next step if this one doesn't
move the needle.

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
- **Global average pooling before the FC head.** The original architecture
  flattened the full 256x28x28 feature map straight into a 1024-unit FC
  layer — about 205 million parameters in that one layer alone, the large
  majority of the entire model. Global average pooling (standard in
  ResNet, DenseNet, and most modern CNNs) collapses each channel to one
  value first, cutting that layer to ~263K parameters with no loss of the
  spatial features the conv blocks extract, and makes training realistic
  on a modest GPU or CPU.

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
scripts/generate_sample_data.py   synthetic data generator for smoke-testing without real data
tests/               unit tests, app tests, and an end-to-end integration test
```

### Tests

```bash
python -m pytest tests/ -v
```

17 tests: label mapping, clinical (not alphabetical) class ordering, empty-class
handling, patient-level split leakage, Gaussian smoothing, Grad-CAM, the
Streamlit app, a full end-to-end single-model pipeline run, and a full
end-to-end sigma-ensemble run (including verifying all 4 variants share
identical patients) -- all on synthetic data, ~60-90 seconds total. All run
in CI on every push.
