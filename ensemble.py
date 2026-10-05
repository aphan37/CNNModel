"""
ensemble.py

Trains 4 AlzhiNet models, each on the same patient-level split but a
different Gaussian sigma (config.ENSEMBLE_SIGMAS), then combines their
predictions by averaging softmax probabilities (soft voting) -- the
standard, overfitting-safe way to combine a small number of models when
you don't have enough held-out data to safely train a stacking
meta-learner on top.

Hard requirement this script enforces: all 4 models share the EXACT same
train/val/test patients. This is done by computing the patient->split
assignment ONCE (data_pipeline.compute_patient_split_assignment(), based
on the pre-smoothing organized images, since which patient goes where
doesn't depend on sigma), saving it to disk, then reusing it to build
each sigma variant's dataset folder. Without this, each variant could get
a different random split and the 4 models' test accuracy numbers --  and
the ensemble's combined evaluation -- would not be valid to compare.

Usage:
    python ensemble.py prepare                    # organize, assign split, smooth x4, stats x4
    python ensemble.py train                      # train all 4 models (real baseline schedule)
    python ensemble.py train --epochs 3 --patience 3   # fast smoke run
    python ensemble.py evaluate                    # combine predictions, report ensemble vs individual
    python ensemble.py all --epochs 3 --patience 3     # prepare + train + evaluate, one command
"""

import argparse
import json
import os

import numpy as np
from sklearn.metrics import classification_report, cohen_kappa_score, confusion_matrix

import config
import data_pipeline
import preprocessing
import train as train_module


def prepare():
    if not os.path.isdir(config.ORGANIZED_DIR):
        data_pipeline.harmonize_and_organize()
    else:
        print(f"{config.ORGANIZED_DIR} already exists — skipping organize (delete it to redo).")

    if not os.path.exists(config.PATIENT_SPLIT_ASSIGNMENT_PATH):
        data_pipeline.compute_patient_split_assignment()
    else:
        print(f"Patient split assignment already exists at {config.PATIENT_SPLIT_ASSIGNMENT_PATH} "
              f"— reusing so all sigma variants share identical patients. Delete it to force a new split.")

    for sigma in config.ENSEMBLE_SIGMAS:
        paths = config.ensemble_paths(sigma)
        print(f"\n--- sigma={sigma} ---")
        preprocessing.apply_to_directory(config.ORGANIZED_DIR, paths["preprocessed"], sigma=sigma)
        data_pipeline.build_dataset_from_assignment(paths["preprocessed"], paths["dataset"])
        # Stats computed from this sigma's TRAIN split only -- slightly more
        # correct than the single-model flow's whole-preprocessed-dir stats
        # (which includes val/test images), avoiding a small amount of
        # normalization-stat leakage.
        preprocessing.compute_dataset_mean_std(
            os.path.join(paths["dataset"], "train"), save_path=paths["stats_path"]
        )


def train_all(epochs=None, patience=None):
    for sigma in config.ENSEMBLE_SIGMAS:
        paths = config.ensemble_paths(sigma)
        print(f"\n=== Training sigma={sigma} model ===")
        if epochs is not None:
            config.NUM_EPOCHS = epochs
        if patience is not None:
            config.EARLY_STOPPING_PATIENCE = patience
        train_module.train(
            split_dir=paths["dataset"],
            model_dir=paths["model_dir"],
            stats_path=paths["stats_path"],
            results_dir=paths["results_dir"],
        )


def _load_predictions(sigma):
    paths = config.ensemble_paths(sigma)
    pred_path = os.path.join(paths["results_dir"], "test_predictions.json")
    if not os.path.exists(pred_path):
        raise RuntimeError(f"{pred_path} not found — run `python ensemble.py train` first.")
    with open(pred_path) as f:
        return json.load(f)


def evaluate():
    per_sigma = {sigma: _load_predictions(sigma) for sigma in config.ENSEMBLE_SIGMAS}

    reference = per_sigma[config.ENSEMBLE_SIGMAS[0]]
    class_order = reference["class_order"]
    n_classes = len(class_order)

    probs_by_sigma_file = {
        sigma: dict(zip(data["filenames"], data["probs"])) for sigma, data in per_sigma.items()
    }
    true_by_file = dict(zip(reference["filenames"], reference["true_labels"]))

    # Align by filename rather than assuming identical order across the 4
    # models' saved predictions, and only ensemble files every model has a
    # prediction for.
    common_files = set(reference["filenames"])
    for sigma in config.ENSEMBLE_SIGMAS:
        model_files = set(probs_by_sigma_file[sigma].keys())
        if model_files != common_files:
            print(f"  [WARN] sigma={sigma}'s test set differs from the others by "
                  f"{len(model_files ^ common_files)} file(s) — check all 4 used the "
                  f"same patient_split_assignment.json.")
        common_files &= model_files
    common_files = sorted(common_files)
    print(f"Combining predictions over {len(common_files)} test images common to all "
          f"{len(config.ENSEMBLE_SIGMAS)} models.")

    label_indices = list(range(n_classes))
    true_labels = np.array([true_by_file[f] for f in common_files])

    print("\n=== Individual model results ===")
    comparison = {}
    ensemble_probs = np.zeros((len(common_files), n_classes))
    for sigma in config.ENSEMBLE_SIGMAS:
        model_probs = np.array([probs_by_sigma_file[sigma][f] for f in common_files])
        ensemble_probs += model_probs

        model_preds = model_probs.argmax(axis=1)
        acc = 100 * np.mean(model_preds == true_labels)
        kappa = cohen_kappa_score(true_labels, model_preds, weights="quadratic")
        comparison[str(sigma)] = {"accuracy": acc, "quadratic_weighted_kappa": kappa}
        print(f"  sigma={sigma}: accuracy={acc:.2f}%  quadratic-weighted kappa={kappa:.4f}")

    ensemble_probs /= len(config.ENSEMBLE_SIGMAS)
    ensemble_preds = ensemble_probs.argmax(axis=1)

    ensemble_accuracy = 100 * np.mean(ensemble_preds == true_labels)
    ensemble_kappa = cohen_kappa_score(true_labels, ensemble_preds, weights="quadratic")
    ensemble_cm = confusion_matrix(true_labels, ensemble_preds, labels=label_indices)
    ensemble_report = classification_report(
        true_labels, ensemble_preds, labels=label_indices, target_names=class_order, zero_division=0
    )

    print(f"\n=== Ensemble (soft-voting average of {len(config.ENSEMBLE_SIGMAS)} models) ===")
    print(f"Accuracy: {ensemble_accuracy:.2f}%")
    print(f"Quadratic-weighted Kappa: {ensemble_kappa:.4f}")
    print(f"\nConfusion matrix (rows=true, cols=predicted):\n{class_order}\n{ensemble_cm}")
    print(f"\n{ensemble_report}")

    best_individual_kappa = max(v["quadratic_weighted_kappa"] for v in comparison.values())
    if ensemble_kappa > best_individual_kappa:
        print(f"Ensemble kappa ({ensemble_kappa:.4f}) beats every individual model "
              f"(best individual: {best_individual_kappa:.4f}) — ensembling helped on this run.")
    else:
        print(f"[NOTE] Ensemble kappa ({ensemble_kappa:.4f}) did not beat the best individual "
              f"model ({best_individual_kappa:.4f}) here. With 4 variants of the same architecture "
              f"differing only in preprocessing, a gain isn't guaranteed, especially on a small "
              f"test set — this is a real result worth reporting, not a failure of the code.")

    os.makedirs(config.ENSEMBLE_DIR, exist_ok=True)
    out_path = os.path.join(config.ENSEMBLE_DIR, "ensemble_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "sigmas": config.ENSEMBLE_SIGMAS,
            "n_test_files_used": len(common_files),
            "individual_models": comparison,
            "ensemble": {
                "accuracy": ensemble_accuracy,
                "quadratic_weighted_kappa": ensemble_kappa,
                "confusion_matrix": ensemble_cm.tolist(),
                "class_order": class_order,
            },
        }, f, indent=2)
    print(f"\nSaved comparison -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="4-model sigma ensemble for AlzhiNet")
    parser.add_argument("command", choices=["prepare", "train", "evaluate", "all"])
    parser.add_argument("--epochs", type=int, default=None, help="Override config.NUM_EPOCHS for all 4 models")
    parser.add_argument("--patience", type=int, default=None, help="Override config.EARLY_STOPPING_PATIENCE")
    args = parser.parse_args()

    if args.command == "prepare":
        prepare()
    elif args.command == "train":
        train_all(epochs=args.epochs, patience=args.patience)
    elif args.command == "evaluate":
        evaluate()
    elif args.command == "all":
        prepare()
        train_all(epochs=args.epochs, patience=args.patience)
        evaluate()
