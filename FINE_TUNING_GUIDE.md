# Fine-Tuning & Standardization Guide

This defines the baseline process for this pipeline going forward, so future
runs (and anyone else reading the repo) know what "standard" means here and
what's safe to tune.

## Run order

```bash
# 1. Harmonize CSV + images by class
python data_pipeline.py organize

# 2. Denoise: compare sigma candidates visually, pick one, then apply it
python preprocessing.py sweep     # writes results/sigma_sweep.png
# edit config.GAUSSIAN_SIGMA based on what you see, then:
python preprocessing.py apply

# 3. Compute this dataset's own normalization stats (auto-saved and
#    auto-loaded by train.py / gradcam_cli.py -- no manual copy-paste)
python preprocessing.py stats

# 4. Split by PATIENT (not by file) into train/val/test, stratified by severity
python data_pipeline.py split

# 5. Train + evaluate
python train.py

# 6. Classify a single image with Grad-CAM explainability
python gradcam_cli.py --image test_images/sample.jpg --gradcam
```

The order is enforced, not just suggested: `organize` must run before
`preprocessing.py apply` (which smooths the organized images), and `split`
must run after (it splits from the smoothed output). Running a step before
its dependency is ready fails immediately with a clear error rather than
silently producing something wrong.

**No real data yet?** `python scripts/generate_sample_data.py` generates a
small synthetic dataset (fake patients, fake MRI-like images) that exercises
this exact same flow, so you can confirm the pipeline itself works before
ever touching NACC data. See the README's "Try it without real data" section.

## What counts as the "baseline" — freeze this before tuning anything

A baseline is one specific, reproducible configuration you compare every
future change against. Freeze:

- `GAUSSIAN_SIGMA` (pick from the sweep, document why in the model card below)
- `RANDOM_SEED = 42`
- The patient-level split itself — re-running `data_pipeline.py` with the
  same seed should reproduce the exact same train/val/test patient sets
- Architecture (`AlzhiNet` as defined in `train.py`)
- `LEARNING_RATE`, `BATCH_SIZE`, `WEIGHT_DECAY` as set in `config.py`

Record the baseline's test-set accuracy, quadratic-weighted kappa, and
confusion matrix (all auto-saved to `results/test_report.json`) as the
number every future experiment is measured against.

## What to tune, and in what order

Tune one axis at a time — changing two things at once means you can't
attribute the result to either one.

1. **Gaussian sigma** (1.0–2.0 sweep) — do this first, since it changes the
   input data itself. Re-run steps 2–4 for each candidate sigma and compare
   kappa, not just accuracy.
2. **Learning rate** — try `3e-4` and `3e-3` around the current `1e-3`.
   Watch `results/training_history.json`'s `val_loss` curve: if it's
   noisy/diverging, LR is too high; if it barely moves, too low.
3. **Batch size** — 16 vs. 32 vs. 64, constrained by your GPU memory. Larger
   batches generally need a proportionally larger LR.
4. **Class weighting strategy** — the current approach is inverse-frequency
   weighting (`compute_class_weights` in `train.py`). If the rarest class
   (usually Severe) is still consistently missed, consider oversampling
   that class's patients instead of/in addition to loss weighting.
5. **Data augmentation strength** — current setup uses only a horizontal
   flip + ±10° rotation, deliberately conservative for medical imaging.
   Only increase this if you see overfitting (train accuracy far exceeds
   val accuracy) despite dropout and weight decay.
6. **Architecture depth** — only after 1–5 are exhausted. Options in rough
   order of effort: add a 4th conv block; switch to a pretrained backbone
   (ResNet18/DenseNet121) with the first layer adapted for your input
   channels, which often helps a lot with a dataset this size but changes
   the normalization-stats story (you'd go back to ImageNet stats if using
   ImageNet-pretrained weights).

## Cross-validation (recommended before reporting a final number)

A single train/val/test split can get a lucky or unlucky patient
assignment, especially for the rarer severity classes. Once you're happy
with a configuration from the tuning process above, validate it with
**k-fold cross-validation at the patient level** (k=5 is standard) rather
than trusting one split's test accuracy as the final reported number.
`sklearn.model_selection.StratifiedGroupKFold` (grouping by NACCID,
stratifying by severity) is the direct sklearn tool for this — same idea as
`patient_level_stratified_split()` in `data_pipeline.py`, repeated 5 times
with different held-out folds, then averaging the kappa/accuracy across
folds.

## Model card — fill this in for every baseline you freeze

Keep a short model card (a markdown file per run, e.g.
`results/model_card_v1.md`) recording:

- Date, git commit hash
- `GAUSSIAN_SIGMA` used and why (link to the sweep image)
- Dataset mean/std used
- Number of unique patients / images per split, per class (from
  `data_pipeline.py`'s printed counts)
- How multi-visit patients with more than one severity label were handled
  (currently: most-severe-label-wins for stratification — note if you
  change this)
- Final test accuracy, quadratic-weighted kappa, confusion matrix
- Known limitations (e.g., "Severe class has only N test patients, treat
  per-class Severe metrics with caution")

This is what turns "I trained a model" into "I can tell you exactly how
this model was produced and how confident to be in its numbers" — which is
the difference that matters most when this project comes up in an
interview.
