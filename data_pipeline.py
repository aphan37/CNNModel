"""
data_pipeline.py

Step 1 of the pipeline: harmonize the NACC clinical CSV with the MRI images,
organize images into class folders, then split into train/val/test at the
PATIENT level (not the image level) to avoid leakage.

Key fixes vs. the original CNN.py:
  1. Correct CDR label mapping (see config.ALZHEIMERS_CATEGORY) — the
     original had 0.5 and 1 mislabeled relative to the standard CDR scale.
  2. Patient-level (NACCID-level) split with stratification, instead of
     splitting individual image files. Since NACC patients have multiple
     visits/scans, splitting by file lets the same patient appear in both
     train and test, which the model can exploit to "recognize" a patient's
     anatomy rather than learn disease signal — this is a common cause of
     inflated accuracy in medical imaging pipelines.
  3. Uses shutil.copy instead of shutil.move, and skips files that are
     already organized — the original script emptied its source folder on
     first run, making a second run silently do nothing.

Run this AFTER preprocessing.py has produced the Gaussian-smoothed images
in config.PREPROCESSED_DIR — or point ORGANIZE_SOURCE_DIR at
config.ORGANIZED_DIR if you want to split before smoothing (order doesn't
matter for correctness, since smoothing is deterministic and label-blind;
this repo's default order is: organize raw -> smooth -> split).
"""

import json
import os
import shutil
from collections import defaultdict

import pandas as pd
from sklearn.model_selection import train_test_split

import config


def harmonize_and_organize():
    """
    Reads the NACC clinical CSV, maps CDRGLOB to a severity label using the
    corrected mapping in config.py, and copies each matched image into
    config.ORGANIZED_DIR/<label>/. Non-destructive and idempotent — safe
    to re-run.
    """
    print("Loading classification data...")
    classification_df = pd.read_csv(config.COMMERCIAL_FILE_PATH)
    classification_df["Type"] = classification_df["CDRGLOB"].map(config.ALZHEIMERS_CATEGORY)

    unmapped = classification_df["Type"].isna().sum()
    if unmapped > 0:
        print(f"  [WARN] {unmapped} rows had a CDRGLOB value not in ALZHEIMERS_CATEGORY "
              f"and will be excluded — check config.py's mapping covers all values in your data.")

    os.makedirs(config.ORGANIZED_DIR, exist_ok=True)
    for label in config.CLASS_ORDER:
        os.makedirs(os.path.join(config.ORGANIZED_DIR, label), exist_ok=True)

    print("Organizing images (copy, not move — source folder stays intact)...")
    matched, unmatched, already_present = 0, 0, 0

    for file_name in os.listdir(config.LABELLED_IMAGE_FOLDER):
        if not file_name.lower().endswith(".jpg"):
            continue

        base_id = file_name.split("_")[0]
        match = classification_df.loc[classification_df["NACCID"].astype(str) == base_id]

        if match.empty or pd.isna(match["Type"].values[0]):
            unmatched += 1
            continue

        classification_type = match["Type"].values[0]
        target_path = os.path.join(config.ORGANIZED_DIR, classification_type, file_name)

        if os.path.exists(target_path):
            already_present += 1
            continue

        shutil.copy(os.path.join(config.LABELLED_IMAGE_FOLDER, file_name), target_path)
        matched += 1

    print(f"  Organized: {matched} new, {already_present} already cached, {unmatched} unmatched/excluded.")
    if unmatched > 0:
        print(f"  [WARN] {unmatched} images had no matching NACCID or an unmapped CDR value — "
              f"these are silently excluded from training. Investigate if this number is large.")


def _safe_stratified_split(ids, labels, test_size):
    """train_test_split with stratify= raises ValueError if any class has
    fewer than 2 members on either side of the split -- which WILL happen
    for rare classes (Moderate/Severe are typically the smallest in NACC
    severity data, and small synthetic/pilot datasets hit this constantly).
    Rather than crash the whole pipeline over one rare class, this falls
    back to an unstratified split when that happens, with a loud warning
    so you know per-class proportions may be uneven in that split."""
    try:
        return train_test_split(
            ids, labels, test_size=test_size, stratify=labels, random_state=config.RANDOM_SEED
        )
    except ValueError as e:
        print(f"  [WARN] stratified split failed ({e}) — falling back to a random "
              f"(non-stratified) split for this partition. Per-class proportions may "
              f"be uneven here; check the printed counts below before trusting "
              f"per-class val/test metrics for the affected class.")
        return train_test_split(ids, labels, test_size=test_size, random_state=config.RANDOM_SEED)


def _extract_naccid(file_name: str) -> str:
    """NACCID is assumed to be the underscore-delimited prefix of the filename,
    e.g. 'NACC123456_visit2_slice04.jpg' -> 'NACC123456'. If your naming
    convention differs, update this function — everything downstream
    depends on it being correct."""
    return file_name.split("_")[0]


def patient_level_stratified_split(source_dir: str = None):
    """
    Groups images by patient (NACCID), assigns each patient a single split
    (train/val/test) based on stratified sampling over their most severe
    recorded label, then copies every one of that patient's images into
    the corresponding split folder. This guarantees no patient's images
    cross the train/val/test boundary.
    """
    source_dir = source_dir or config.PREPROCESSED_DIR
    if not os.path.isdir(source_dir):
        raise RuntimeError(
            f"{source_dir} not found. Run preprocessing.py apply first, "
            f"or pass source_dir=config.ORGANIZED_DIR to split pre-smoothing."
        )

    print(f"Building patient -> images map from {source_dir}...")
    patient_files = defaultdict(list)   # naccid -> list of (label, file_path)
    patient_labels = defaultdict(set)   # naccid -> set of labels seen (usually 1, but be defensive)

    for label in config.CLASS_ORDER:
        label_dir = os.path.join(source_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for file_name in os.listdir(label_dir):
            if not file_name.lower().endswith(".jpg"):
                continue
            naccid = _extract_naccid(file_name)
            patient_files[naccid].append((label, os.path.join(label_dir, file_name)))
            patient_labels[naccid].add(label)

    n_patients = len(patient_files)
    print(f"  Found {n_patients} unique patients across {sum(len(v) for v in patient_files.values())} images.")

    multi_label_patients = [pid for pid, labels in patient_labels.items() if len(labels) > 1]
    if multi_label_patients:
        print(f"  [NOTE] {len(multi_label_patients)} patients have images spanning more than one "
              f"severity label (e.g., progression across visits). Using each patient's most "
              f"severe recorded label for stratification — this is a judgment call worth "
              f"documenting in your model card.")

    def most_severe_label(pid):
        labels_present = patient_labels[pid]
        return max(labels_present, key=lambda l: config.CLASS_ORDER.index(l))

    patient_ids = list(patient_files.keys())
    strat_labels = [most_severe_label(pid) for pid in patient_ids]

    # split patients (not images) into train/val/test, stratified by severity
    train_ids, temp_ids, train_strat, temp_strat = _safe_stratified_split(
        patient_ids, strat_labels, test_size=(1 - config.SPLIT_RATIOS["train"])
    )
    relative_test_size = config.SPLIT_RATIOS["test"] / (config.SPLIT_RATIOS["val"] + config.SPLIT_RATIOS["test"])
    val_ids, test_ids, _, _ = _safe_stratified_split(
        temp_ids, temp_strat, test_size=relative_test_size
    )

    split_map = {"train": set(train_ids), "val": set(val_ids), "test": set(test_ids)}
    print(f"  Patient split -> train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")

    for split in config.SPLIT_RATIOS:
        for label in config.CLASS_ORDER:
            os.makedirs(os.path.join(config.SPLIT_DIR, split, label), exist_ok=True)

    print("Copying images into split folders...")
    counts = defaultdict(int)
    for split_name, id_set in split_map.items():
        for pid in id_set:
            for label, src_path in patient_files[pid]:
                file_name = os.path.basename(src_path)
                dst_path = os.path.join(config.SPLIT_DIR, split_name, label, file_name)
                if not os.path.exists(dst_path):
                    shutil.copy(src_path, dst_path)
                    counts[(split_name, label)] += 1

    print("\nFinal image counts per split/class:")
    for split_name in config.SPLIT_RATIOS:
        row = [f"{label}={counts[(split_name, label)]}" for label in config.CLASS_ORDER]
        print(f"  {split_name}: {', '.join(row)}")

    print("\n[CHECK] If any class shows 0 or very few images in val/test, that class is too rare "
          "for a reliable held-out estimate — consider merging adjacent severity classes or "
          "collecting more data for that class before trusting per-class metrics.")


def compute_patient_split_assignment(source_dir: str = None):
    """
    Like patient_level_stratified_split(), but instead of copying images
    into split folders for ONE preprocessed version, this just determines
    and SAVES which split (train/val/test) each patient belongs to, as
    JSON. Used by the sigma ensemble so all 4 models train/validate/test
    on the exact same patients, just with different Gaussian smoothing —
    without this, each sigma variant could end up with a different random
    split, making the 4 models' test accuracy numbers not comparable to
    each other and the ensemble's "same test set" requirement violated.

    Reads from config.ORGANIZED_DIR (pre-smoothing) since the patient/label
    grouping doesn't depend on sigma at all.
    """
    source_dir = source_dir or config.ORGANIZED_DIR
    if not os.path.isdir(source_dir):
        raise RuntimeError(f"{source_dir} not found. Run 'python data_pipeline.py organize' first.")

    print(f"Building patient -> label map from {source_dir}...")
    patient_labels = defaultdict(set)
    for label in config.CLASS_ORDER:
        label_dir = os.path.join(source_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for file_name in os.listdir(label_dir):
            if file_name.lower().endswith(".jpg"):
                patient_labels[_extract_naccid(file_name)].add(label)

    def most_severe_label(pid):
        return max(patient_labels[pid], key=lambda l: config.CLASS_ORDER.index(l))

    patient_ids = list(patient_labels.keys())
    strat_labels = [most_severe_label(pid) for pid in patient_ids]

    train_ids, temp_ids, train_strat, temp_strat = _safe_stratified_split(
        patient_ids, strat_labels, test_size=(1 - config.SPLIT_RATIOS["train"])
    )
    relative_test_size = config.SPLIT_RATIOS["test"] / (config.SPLIT_RATIOS["val"] + config.SPLIT_RATIOS["test"])
    val_ids, test_ids, _, _ = _safe_stratified_split(temp_ids, temp_strat, test_size=relative_test_size)

    assignment = {pid: "train" for pid in train_ids}
    assignment.update({pid: "val" for pid in val_ids})
    assignment.update({pid: "test" for pid in test_ids})

    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    with open(config.PATIENT_SPLIT_ASSIGNMENT_PATH, "w") as f:
        json.dump(assignment, f, indent=2)

    print(f"  Patient split -> train: {len(train_ids)}, val: {len(val_ids)}, test: {len(test_ids)}")
    print(f"  Saved assignment -> {config.PATIENT_SPLIT_ASSIGNMENT_PATH}")
    return assignment


def build_dataset_from_assignment(preprocessed_dir: str, out_dir: str):
    """
    Copies images from a (sigma-smoothed) preprocessed directory into
    out_dir/train|val|test/<class>/, using the patient->split assignment
    saved by compute_patient_split_assignment() — so every sigma variant's
    dataset folder has the identical set of patients in each split.
    """
    if not os.path.exists(config.PATIENT_SPLIT_ASSIGNMENT_PATH):
        raise RuntimeError(
            f"{config.PATIENT_SPLIT_ASSIGNMENT_PATH} not found. "
            f"Run compute_patient_split_assignment() first (shared across all sigma variants)."
        )
    with open(config.PATIENT_SPLIT_ASSIGNMENT_PATH) as f:
        assignment = json.load(f)

    for split in config.SPLIT_RATIOS:
        for label in config.CLASS_ORDER:
            os.makedirs(os.path.join(out_dir, split, label), exist_ok=True)

    counts = defaultdict(int)
    unassigned = 0
    for label in config.CLASS_ORDER:
        label_dir = os.path.join(preprocessed_dir, label)
        if not os.path.isdir(label_dir):
            continue
        for file_name in os.listdir(label_dir):
            if not file_name.lower().endswith(".jpg"):
                continue
            pid = _extract_naccid(file_name)
            split = assignment.get(pid)
            if split is None:
                unassigned += 1
                continue
            dst_path = os.path.join(out_dir, split, label, file_name)
            if not os.path.exists(dst_path):
                shutil.copy(os.path.join(label_dir, file_name), dst_path)
                counts[(split, label)] += 1

    print(f"  Built dataset at {out_dir}")
    for split in config.SPLIT_RATIOS:
        row = [f"{label}={counts[(split, label)]}" for label in config.CLASS_ORDER]
        print(f"    {split}: {', '.join(row)}")
    if unassigned:
        print(f"  [WARN] {unassigned} images belonged to a patient not in the saved "
              f"assignment (e.g. new data added after assign ran) — excluded.")


if __name__ == "__main__":
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else "organize"

    if command == "organize":
        harmonize_and_organize()
    elif command == "split":
        patient_level_stratified_split()
    elif command == "assign":
        compute_patient_split_assignment()
    else:
        print("Usage: python data_pipeline.py [organize|split|assign]")
        print("  organize: harmonize CSV + images by class")
        print("  split:    single-model patient-level split (self-contained)")
        print("  assign:   save a patient->split assignment for the sigma ensemble "
              "(see ensemble.py)")
