"""
Tests for the properties this pipeline most depends on being correct.

Run with: python -m pytest tests/ -v
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

import config
import data_pipeline
from AppModel import AlzhiNet
from gradcam_cli import apply_gradcam, get_last_conv_layer
from preprocessing import gaussian_denoise
from train import OrderedImageFolder, compute_class_weights


def _make_images(root: Path, class_names, per_class=1):
    for cls in class_names:
        (root / cls).mkdir(parents=True, exist_ok=True)
        for i in range(per_class):
            Image.new("RGB", (32, 32), color="white").save(root / cls / f"img{i}.jpg")


# --- Standard CDR labels -----------------------------------------------------

def test_cdr_mapping_follows_standard_scale():
    m = config.ALZHEIMERS_CATEGORY
    assert m[0] == "Normal"
    assert m[0.5] == "MCI"
    assert m[1] == "Mild"
    assert m[2] == "Moderate"
    assert m[3] == "Severe"


def test_class_order_is_clinical_severity_order():
    assert config.CLASS_ORDER == ["Normal", "MCI", "Mild", "Moderate", "Severe"]


# --- Class index ordering (kappa depends on this) ----------------------------

def test_class_indices_follow_clinical_order_not_alphabetical(tmp_path):
    _make_images(tmp_path, ["Normal", "MCI", "Mild"])
    ds = OrderedImageFolder(root=str(tmp_path))
    # alphabetical would give MCI=0, Mild=1, Normal=2 -- wrong for ordinal metrics
    assert ds.class_to_idx == {"Normal": 0, "MCI": 1, "Mild": 2}


def test_empty_class_folder_does_not_crash_and_keeps_global_indices(tmp_path):
    _make_images(tmp_path, ["Normal", "Mild"])
    (tmp_path / "Severe").mkdir()  # exists but empty
    ds = OrderedImageFolder(root=str(tmp_path))
    assert "Severe" not in ds.classes
    assert ds.class_to_idx["Mild"] == 2  # index unaffected by missing MCI


def test_class_weights_cover_all_five_classes_even_if_some_are_missing(tmp_path):
    _make_images(tmp_path, ["Normal", "Mild"], per_class=2)
    ds = OrderedImageFolder(root=str(tmp_path))
    weights = compute_class_weights(ds)
    assert weights.shape[0] == len(config.CLASS_ORDER)


# --- Patient-level split (no leakage) ----------------------------------------

def test_patient_level_split_never_puts_one_patient_in_two_splits(tmp_path, monkeypatch):
    source = tmp_path / "source"
    for cls in ["Normal", "Mild"]:
        (source / cls).mkdir(parents=True)
        for p in range(20):
            for scan in range(3):  # multiple scans per patient
                (source / cls / f"{cls}P{p}_scan{scan}.jpg").write_bytes(b"x")

    split_dir = tmp_path / "dataset"
    monkeypatch.setattr(config, "SPLIT_DIR", str(split_dir))
    data_pipeline.patient_level_stratified_split(source_dir=str(source))

    patients_by_split = {}
    for split in ["train", "val", "test"]:
        ids = set()
        for f in (split_dir / split).rglob("*.jpg"):
            ids.add(f.name.split("_")[0])
        patients_by_split[split] = ids

    assert not (patients_by_split["train"] & patients_by_split["val"])
    assert not (patients_by_split["train"] & patients_by_split["test"])
    assert not (patients_by_split["val"] & patients_by_split["test"])
    assert all(len(ids) > 0 for ids in patients_by_split.values())


# --- Gaussian preprocessing --------------------------------------------------

def test_gaussian_denoise_preserves_shape_and_reduces_noise():
    rng = np.random.default_rng(0)
    noisy = np.clip(128 + rng.normal(0, 25, (64, 64, 3)), 0, 255).astype(np.uint8)
    smoothed = gaussian_denoise(noisy, sigma=1.5)
    assert smoothed.shape == noisy.shape
    assert smoothed.std() < noisy.std()


@pytest.mark.parametrize("sigma", [1.0, 1.5, 2.0])
def test_gaussian_denoise_accepts_full_sigma_range(sigma):
    img = np.full((32, 32, 3), 100, dtype=np.uint8)
    assert gaussian_denoise(img, sigma=sigma).shape == img.shape


# --- Grad-CAM ----------------------------------------------------------------

def test_gradcam_finds_the_last_conv_layer_despite_batchnorm():
    layer = get_last_conv_layer(AlzhiNet(num_classes=5))
    assert layer.out_channels == 256


def test_gradcam_returns_normalized_heatmap():
    model = AlzhiNet(num_classes=5)
    x = torch.randn(1, 3, config.IMAGE_SIZE, config.IMAGE_SIZE)
    cam = apply_gradcam(model, x, class_index=2)
    assert cam.shape == (config.IMAGE_SIZE, config.IMAGE_SIZE)
    assert cam.min() >= 0.0 and cam.max() <= 1.0 + 1e-5
