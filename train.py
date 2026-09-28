"""
train.py

Step 3-4 of the pipeline: train and evaluate AlzhiNet on the patient-level
split produced by data_pipeline.py.

Key fixes vs. the original CNN.py:
  1. Reproducibility: seeds set for random/numpy/torch.
  2. Dataset-specific normalization stats instead of ImageNet's (this
     model is trained from scratch, not fine-tuned from an ImageNet
     backbone — see preprocessing.py's compute_dataset_mean_std()).
  3. Class-weighted loss to counter NACC's typical severity imbalance
     (usually skewed toward Normal/MCI) so the model can't win on
     accuracy by mostly predicting the majority class.
  4. Early stopping + ReduceLROnPlateau, instead of a fixed 10 epochs.
  5. Model definition moved to AppModel.py (single source of truth, also
     used by gradcam_cli.py) with BatchNorm added to each conv block.
  6. Fixed class-index ordering: torchvision's ImageFolder assigns class
     indices by ALPHABETICAL folder name (MCI, Mild, Moderate, Normal,
     Severe), not clinical severity order. That silently breaks the
     quadratic-weighted kappa metric below, which needs indices in true
     severity order (Normal < MCI < Mild < Moderate < Severe) to mean
     anything. OrderedImageFolder below fixes this.
  7. Confusion matrix + classification report + quadratic-weighted Cohen's
     Kappa on the test set. Kappa matters because this is an ORDINAL task
     (Normal < MCI < Mild < Moderate < Severe) — plain accuracy treats a
     Normal-vs-Severe mistake the same as a Mild-vs-Moderate mistake,
     which is not clinically equivalent.
"""

import json
import os
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config
from AppModel import AlzhiNet


def set_seed(seed: int = config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


_IMG_EXTENSIONS = (".jpg", ".jpeg", ".png", ".ppm", ".bmp", ".pgm", ".tif", ".tiff", ".webp")


class OrderedImageFolder(datasets.ImageFolder):
    """ImageFolder assigns class indices alphabetically by default, which
    breaks the ordinal severity assumption the kappa metric depends on.
    This forces indices to follow config.CLASS_ORDER instead
    (Normal=0, MCI=1, Mild=2, Moderate=3, Severe=4).

    Also excludes any class folder with zero image files — ImageFolder
    raises a hard error if a *listed* class has no valid files (which can
    easily happen for a rare class like Severe in a small split), so we
    drop empty ones here and warn instead of crashing. The index for a
    present class is still taken from the FULL config.CLASS_ORDER, so
    indices stay consistent across train/val/test even if one split is
    missing a class the others have."""

    def find_classes(self, directory):
        classes = []
        for c in config.CLASS_ORDER:
            class_dir = os.path.join(directory, c)
            if os.path.isdir(class_dir) and any(
                f.lower().endswith(_IMG_EXTENSIONS) for f in os.listdir(class_dir)
            ):
                classes.append(c)

        missing = [c for c in config.CLASS_ORDER if c not in classes]
        if missing:
            print(f"  [WARN] {directory}: no images found for class(es) {missing} — "
                  f"excluded from this split's loader. Model output layer still "
                  f"sized for all {len(config.CLASS_ORDER)} classes.")

        class_to_idx = {cls_name: config.CLASS_ORDER.index(cls_name) for cls_name in classes}
        return classes, class_to_idx


def build_transforms(mean, std):
    train_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        # Mild, anatomically-conservative augmentation. Horizontal flip is
        # included on the assumption of axial brain slices (roughly
        # left-right symmetric) — remove it if your images are a view
        # where laterality is diagnostically meaningful.
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    eval_transform = transforms.Compose([
        transforms.Resize((config.IMAGE_SIZE, config.IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return train_transform, eval_transform


def compute_class_weights(dataset: OrderedImageFolder) -> torch.Tensor:
    # Sized by the FULL class set (config.CLASS_ORDER), not just whichever
    # classes happen to be present in this particular split — otherwise a
    # split missing a rare class (e.g. no Severe patients this run) would
    # produce a weight vector shorter than the model's output layer.
    n_classes = len(config.CLASS_ORDER)
    counts = np.zeros(n_classes)
    for _, label_idx in dataset.samples:
        counts[label_idx] += 1
    counts_safe = np.maximum(counts, 1)  # avoid divide-by-zero for an empty class
    weights = counts_safe.sum() / (n_classes * counts_safe)
    print("Class counts (train):", dict(zip(config.CLASS_ORDER, counts.astype(int))))
    print("Class weights applied:", dict(zip(config.CLASS_ORDER, np.round(weights, 3))))
    if (counts == 0).any():
        zero_classes = [c for c, n in zip(config.CLASS_ORDER, counts) if n == 0]
        print(f"  [WARN] classes with ZERO training samples: {zero_classes} — "
              f"model cannot learn these classes at all from this split.")
    return torch.tensor(weights, dtype=torch.float32)


def load_dataset_stats():
    """Loads mean/std saved by `python preprocessing.py stats`. Falls back
    to a neutral placeholder with a loud warning if that step hasn't been
    run yet — training will still work, but normalization won't be tuned
    to your actual data."""
    if os.path.exists(config.DATASET_STATS_PATH):
        with open(config.DATASET_STATS_PATH) as f:
            stats = json.load(f)
        print(f"Loaded dataset stats from {config.DATASET_STATS_PATH}: "
              f"mean={stats['mean']}, std={stats['std']}")
        return stats["mean"], stats["std"]

    print(f"  [WARN] {config.DATASET_STATS_PATH} not found — run "
          f"`python preprocessing.py stats` first for accurate normalization. "
          f"Falling back to a neutral placeholder for now.")
    return [0.5, 0.5, 0.5], [0.25, 0.25, 0.25]


def train():
    set_seed()

    dataset_mean, dataset_std = load_dataset_stats()
    train_transform, eval_transform = build_transforms(dataset_mean, dataset_std)

    train_dataset = OrderedImageFolder(root=os.path.join(config.SPLIT_DIR, "train"), transform=train_transform)
    val_dataset = OrderedImageFolder(root=os.path.join(config.SPLIT_DIR, "val"), transform=eval_transform)
    test_dataset = OrderedImageFolder(root=os.path.join(config.SPLIT_DIR, "test"), transform=eval_transform)

    # Class indices now follow config.CLASS_ORDER (clinical severity order),
    # not alphabetical folder order — this is what makes the kappa metric
    # below meaningful. class_names is always the FULL 5-class list, even if
    # a given split happens to be missing a rare class, so the output layer
    # size and metric labels stay consistent across train/val/test.
    class_names = config.CLASS_ORDER

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = AlzhiNet(num_classes=len(class_names)).to(device)
    class_weights = compute_class_weights(train_dataset).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=config.LR_SCHEDULER_FACTOR, patience=config.LR_SCHEDULER_PATIENCE
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(config.NUM_EPOCHS):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                val_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        val_loss /= len(val_loader)
        val_acc = 100 * correct / total

        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Epoch {epoch+1:02d} | train_loss={train_loss:.4f} | "
              f"val_loss={val_loss:.4f} | val_acc={val_acc:.2f}% | lr={current_lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            torch.save(model.state_dict(), os.path.join(config.MODEL_OUT_DIR, "best_model.pth"))
            print("  Saved best model so far.")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.EARLY_STOPPING_PATIENCE:
                print(f"  No val_loss improvement in {config.EARLY_STOPPING_PATIENCE} epochs — stopping early.")
                break

    with open(os.path.join(config.RESULTS_DIR, "training_history.json"), "w") as f:
        json.dump(history, f, indent=2)

    evaluate_on_test(model, test_loader, class_names, device)


def evaluate_on_test(model, test_loader, class_names, device):
    model.load_state_dict(torch.load(os.path.join(config.MODEL_OUT_DIR, "best_model.pth")))
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    label_indices = list(range(len(class_names)))  # config.CLASS_ORDER indices, 0..4

    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    cm = confusion_matrix(all_labels, all_preds, labels=label_indices)
    report = classification_report(all_labels, all_preds, labels=label_indices,
                                    target_names=class_names, zero_division=0)

    print(f"\n=== Test Results ===")
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Quadratic-weighted Cohen's Kappa: {kappa:.4f}  "
          f"(this is the metric that matters more than accuracy here — it "
          f"penalizes far-apart severity misses more than adjacent-class misses)")
    print(f"\nConfusion matrix (rows=true, cols=predicted):\n{class_names}\n{cm}")
    print(f"\nClassification report:\n{report}")

    with open(os.path.join(config.RESULTS_DIR, "test_report.json"), "w") as f:
        json.dump({
            "accuracy": accuracy,
            "quadratic_weighted_kappa": kappa,
            "confusion_matrix": cm.tolist(),
            "class_order": class_names,
        }, f, indent=2)


if __name__ == "__main__":
    train()
