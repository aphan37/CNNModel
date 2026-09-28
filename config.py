"""
config.py

Single source of truth for paths and hyperparameters. Import this instead
of hardcoding values across preprocessing.py / data_pipeline.py / train.py —
this is what lets you fine-tune the pipeline without hunting through
multiple files, and lets you reproduce a specific run later.
"""

import os

# === Paths ===
COMMERCIAL_FILE_PATH = "commercial_nacc65a.csv"
LABELLED_IMAGE_FOLDER = "labeledNACCImages"
ORGANIZED_DIR = "organizedNACCImages"          # raw images, organized by class (untouched originals)
PREPROCESSED_DIR = "preprocessedNACCImages"    # Gaussian-smoothed versions, organized by class
SPLIT_DIR = "dataset"                          # final train/val/test split, built from PREPROCESSED_DIR
MODEL_OUT_DIR = "models"
RESULTS_DIR = "results"

os.makedirs(MODEL_OUT_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASET_STATS_PATH = os.path.join(RESULTS_DIR, "dataset_stats.json")
BEST_MODEL_PATH = os.path.join(MODEL_OUT_DIR, "best_model.pth")

# === CDR -> severity label mapping (standard Clinical Dementia Rating scale) ===
# 0 = Normal, 0.5 = Very Mild / MCI, 1 = Mild, 2 = Moderate, 3 = Severe.
# This ORDER matters — it's used for the quadratic-weighted kappa metric,
# which penalizes a Normal-vs-Severe mistake more than a Mild-vs-Moderate one.
ALZHEIMERS_CATEGORY = {
    0:   "Normal",
    0.5: "MCI",        # Mild Cognitive Impairment / "very mild"
    1:   "Mild",
    2:   "Moderate",
    3:   "Severe",
}
CLASS_ORDER = ["Normal", "MCI", "Mild", "Moderate", "Severe"]  # for ordinal metrics

# === Gaussian preprocessing ===
# Sigma controls smoothing strength. Per project spec, sweep within 1.0-2.0
# and pick the value that gives the best validation performance (see
# preprocessing.py's `sweep_sigma_and_save_grid` for a way to compare
# candidates visually before committing to one).
GAUSSIAN_SIGMA = 1.5
GAUSSIAN_SIGMA_SWEEP_CANDIDATES = [1.0, 1.25, 1.5, 1.75, 2.0]

# === Split ===
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42

# === Training ===
IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 50               # upper bound — early stopping will likely halt sooner
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EARLY_STOPPING_PATIENCE = 7   # epochs with no val-loss improvement before stopping
LR_SCHEDULER_PATIENCE = 3     # epochs with no val-loss improvement before reducing LR
LR_SCHEDULER_FACTOR = 0.5
