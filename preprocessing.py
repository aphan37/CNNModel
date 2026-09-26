"""
preprocessing.py

Step between harmonization and model input: Gaussian smoothing to suppress
high-frequency scanner noise in the MRI slices before they reach the CNN.

Why this step, and why sigma 1.0-2.0:
  - Raw MRI slices carry high-frequency acquisition noise that can act as
    a distraction for early conv layers, especially on a model trained
    from scratch (no ImageNet pretraining to fall back on for robust
    low-level filters).
  - A small Gaussian sigma (1-2 px) suppresses that noise while leaving
    anatomical boundaries (ventricles, cortical folds, hippocampal
    atrophy) intact — sigma much above 2 starts blurring the edges the
    model actually needs to see.
  - This is applied identically to every image regardless of split or
    label, so it introduces no train/test leakage — it's a deterministic,
    label-independent transform, safe to precompute once and cache to disk.

Usage:
    python preprocessing.py sweep       # visually compare candidate sigmas
    python preprocessing.py apply       # apply GAUSSIAN_SIGMA to all organized images
    python preprocessing.py stats       # compute dataset mean/std for Normalize()
"""

import os
import sys

import cv2
import numpy as np
from tqdm import tqdm

import config


def gaussian_denoise(image: np.ndarray, sigma: float = config.GAUSSIAN_SIGMA) -> np.ndarray:
    """
    Apply Gaussian smoothing to a single image (as a numpy array, any
    channel count). Kernel size is derived from sigma so it scales
    correctly instead of being a fixed, sigma-mismatched kernel.
    """
    # ~3 sigma radius on each side, forced odd (required by cv2.GaussianBlur)
    k = int(2 * round(3 * sigma) + 1)
    return cv2.GaussianBlur(image, (k, k), sigmaX=sigma, sigmaY=sigma)


def apply_to_directory(src_dir: str, dst_dir: str, sigma: float = config.GAUSSIAN_SIGMA):
    """
    Walks src_dir (expects class subfolders, e.g. organizedNACCImages/Mild/),
    applies gaussian_denoise to every .jpg, and writes the result to the
    same relative path under dst_dir. Skips files that already exist in
    dst_dir so this is safe to re-run (idempotent).
    """
    os.makedirs(dst_dir, exist_ok=True)
    class_dirs = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]

    total_processed, total_skipped = 0, 0

    for class_name in class_dirs:
        src_class_dir = os.path.join(src_dir, class_name)
        dst_class_dir = os.path.join(dst_dir, class_name)
        os.makedirs(dst_class_dir, exist_ok=True)

        files = [f for f in os.listdir(src_class_dir) if f.lower().endswith(".jpg")]
        for file_name in tqdm(files, desc=f"Denoising [{class_name}]"):
            dst_path = os.path.join(dst_class_dir, file_name)
            if os.path.exists(dst_path):
                total_skipped += 1
                continue

            src_path = os.path.join(src_class_dir, file_name)
            image = cv2.imread(src_path, cv2.IMREAD_UNCHANGED)
            if image is None:
                print(f"  [WARN] could not read {src_path}, skipping")
                continue

            denoised = gaussian_denoise(image, sigma=sigma)
            cv2.imwrite(dst_path, denoised)
            total_processed += 1

    print(f"\nDone. Processed {total_processed} images, skipped {total_skipped} already-cached images.")
    print(f"Output: {dst_dir}")


def sweep_sigma_and_save_grid(sample_class: str = None, n_samples: int = 4,
                               out_path: str = None):
    """
    Builds a comparison image grid (rows = sample images, columns = each
    candidate sigma from config.GAUSSIAN_SIGMA_SWEEP_CANDIDATES) so you can
    visually justify your final sigma choice instead of guessing.

    Requires matplotlib. Saves the grid to results/sigma_sweep.png by default.
    """
    import matplotlib.pyplot as plt

    out_path = out_path or os.path.join(config.RESULTS_DIR, "sigma_sweep.png")

    class_dirs = [d for d in os.listdir(config.ORGANIZED_DIR)
                  if os.path.isdir(os.path.join(config.ORGANIZED_DIR, d))]
    if not class_dirs:
        raise RuntimeError(f"No class folders found under {config.ORGANIZED_DIR}. Run data_pipeline.py first.")

    sample_class = sample_class or class_dirs[0]
    sample_dir = os.path.join(config.ORGANIZED_DIR, sample_class)
    files = [f for f in os.listdir(sample_dir) if f.lower().endswith(".jpg")][:n_samples]
    if not files:
        raise RuntimeError(f"No .jpg files found in {sample_dir}")

    sigmas = config.GAUSSIAN_SIGMA_SWEEP_CANDIDATES
    fig, axes = plt.subplots(len(files), len(sigmas) + 1,
                              figsize=(3 * (len(sigmas) + 1), 3 * len(files)))
    if len(files) == 1:
        axes = axes[None, :]

    for row, file_name in enumerate(files):
        image = cv2.imread(os.path.join(sample_dir, file_name), cv2.IMREAD_UNCHANGED)
        axes[row, 0].imshow(image, cmap="gray")
        axes[row, 0].set_title("Original" if row == 0 else "")
        axes[row, 0].axis("off")

        for col, sigma in enumerate(sigmas, start=1):
            denoised = gaussian_denoise(image, sigma=sigma)
            axes[row, col].imshow(denoised, cmap="gray")
            axes[row, col].set_title(f"sigma={sigma}" if row == 0 else "")
            axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved sigma comparison grid to {out_path}")
    print("Pick the sigma where noise looks suppressed but edges (ventricles, "
          "cortical folds) are still sharp — then set GAUSSIAN_SIGMA in config.py.")


def compute_dataset_mean_std(image_dir: str):
    """
    Computes per-channel mean/std over every image in image_dir (recursively,
    across class subfolders) — use this instead of ImageNet stats, since
    AlzhiNet is trained from scratch rather than fine-tuned from an
    ImageNet-pretrained backbone.
    """
    pixel_sum = np.zeros(3, dtype=np.float64)
    pixel_sq_sum = np.zeros(3, dtype=np.float64)
    n_pixels = 0

    for root, _, files in os.walk(image_dir):
        for file_name in files:
            if not file_name.lower().endswith(".jpg"):
                continue
            image = cv2.imread(os.path.join(root, file_name))
            if image is None:
                continue
            image = image.astype(np.float64) / 255.0
            pixel_sum += image.sum(axis=(0, 1))
            pixel_sq_sum += (image ** 2).sum(axis=(0, 1))
            n_pixels += image.shape[0] * image.shape[1]

    mean = pixel_sum / n_pixels
    std = np.sqrt(pixel_sq_sum / n_pixels - mean ** 2)

    # cv2 reads BGR — flip to RGB order to match torchvision's convention
    mean, std = mean[::-1], std[::-1]
    print(f"Dataset mean (RGB): {mean.tolist()}")
    print(f"Dataset std  (RGB): {std.tolist()}")
    print("Paste these into config.py / train.py's transforms.Normalize(...) call.")
    return mean.tolist(), std.tolist()


if __name__ == "__main__":
    command = sys.argv[1] if len(sys.argv) > 1 else "apply"

    if command == "sweep":
        sweep_sigma_and_save_grid()
    elif command == "apply":
        apply_to_directory(config.ORGANIZED_DIR, config.PREPROCESSED_DIR, sigma=config.GAUSSIAN_SIGMA)
    elif command == "stats":
        compute_dataset_mean_std(config.PREPROCESSED_DIR)
    else:
        print("Usage: python preprocessing.py [sweep|apply|stats]")
