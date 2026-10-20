<!--
Insert this section into README.md right after the "## Pipeline" code block
(the one ending at "python gradcam_cli.py / app.py"), and before "## Quick start".
It documents ensemble.py, which exists in the repo but isn't currently mentioned.
-->

## Optional: sigma ensemble

`ensemble.py` trains 4 AlzhiNet models — one per Gaussian smoothing sigma in
`config.ENSEMBLE_SIGMAS` — on the **same patient-level train/val/test split**,
then combines their predictions by averaging softmax probabilities (soft
voting). This is the standard, overfitting-safe way to combine a small
number of models when there isn't enough held-out data to safely train a
stacking meta-learner on top.

The split is computed once (`data_pipeline.compute_patient_split_assignment()`)
and reused for all 4 sigma variants, so every model sees exactly the same
patients in train/val/test — without this, each variant could get a
different random split and the per-model and ensemble accuracy numbers
would not be comparable.

```bash
python ensemble.py prepare                          # organize, assign split, smooth x4, stats x4
python ensemble.py train                            # train all 4 models (real baseline schedule)
python ensemble.py train --epochs 3 --patience 3     # fast smoke run
python ensemble.py evaluate                          # combine predictions, report ensemble vs. individual
python ensemble.py all --epochs 3 --patience 3       # prepare + train + evaluate, one command
```

`evaluate` reports quadratic-weighted kappa and a confusion matrix for the
ensemble alongside each individual sigma model, so you can see whether
averaging actually helps before relying on it.
