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
  5. BatchNorm added to each conv block — speeds convergence and improves
     training stability, especially important for a from-scratch model.
  6. Confusion matrix + classification report + quadratic-weighted Cohen's
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
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    cohen_kappa_score,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import config


def set_seed(seed: int = config.RANDOM_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class AlzhiNet(nn.Module):
    """Same conv-block structure as the original, with BatchNorm added
    after every conv layer for faster, more stable convergence."""

    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(256 * 28 * 28, 1024)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


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


def compute_class_weights(dataset: datasets.ImageFolder) -> torch.Tensor:
    counts = np.zeros(len(dataset.classes))
    for _, label_idx in dataset.samples:
        counts[label_idx] += 1
    counts = np.maximum(counts, 1)  # avoid divide-by-zero for an empty class
    weights = counts.sum() / (len(counts) * counts)
    print("Class counts (train):", dict(zip(dataset.classes, counts.astype(int))))
    print("Class weights applied:", dict(zip(dataset.classes, np.round(weights, 3))))
    return torch.tensor(weights, dtype=torch.float32)


def train():
    set_seed()

    # NOTE: replace these with the output of `python preprocessing.py stats`
    # run against your actual dataset — these are placeholders.
    dataset_mean = [0.5, 0.5, 0.5]
    dataset_std = [0.25, 0.25, 0.25]
    train_transform, eval_transform = build_transforms(dataset_mean, dataset_std)

    train_dataset = datasets.ImageFolder(root=os.path.join(config.SPLIT_DIR, "train"), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=os.path.join(config.SPLIT_DIR, "val"), transform=eval_transform)
    test_dataset = datasets.ImageFolder(root=os.path.join(config.SPLIT_DIR, "test"), transform=eval_transform)

    # sanity check: class index order must match config.CLASS_ORDER for the
    # kappa metric's ordinal weighting to mean anything
    assert train_dataset.classes == sorted(train_dataset.classes)
    class_names = train_dataset.classes

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

    accuracy = 100 * np.mean(np.array(all_preds) == np.array(all_labels))
    kappa = cohen_kappa_score(all_labels, all_preds, weights="quadratic")
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)

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
