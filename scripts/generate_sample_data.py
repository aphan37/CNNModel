"""
generate_sample_data.py

Generates a small SYNTHETIC dataset (fake patients, fake MRI-like images)
that exercises the exact same file layout and CSV schema the real pipeline
expects (NACCID, CDRGLOB, filename convention). This lets you run
data_pipeline.py -> preprocessing.py -> train.py -> gradcam_cli.py end to
end with zero real patient data, to confirm the pipeline itself works
before ever touching NACC data.

The images are NOT real MRIs. Each one is a simple synthetic "brain slice"
(an ellipse with a central "ventricle" blob) where ventricle size and
texture noise scale with severity, so the model has *something* real to
learn -- enough to prove the training loop, class weighting, and metrics
all function correctly. Don't read anything clinical into the accuracy
numbers from this synthetic set.

Usage:
    python scripts/generate_sample_data.py
    python scripts/generate_sample_data.py --patients 80 --scans-per-patient 3
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# Roughly mimics a realistic skew: most patients are Normal/MCI, few Severe.
CDR_WEIGHTS = {0: 0.35, 0.5: 0.30, 1: 0.20, 2: 0.10, 3: 0.05}

RNG = np.random.default_rng(config.RANDOM_SEED)


def _make_synthetic_slice(cdr_score: float, size=256) -> Image.Image:
    """A crude synthetic 'brain slice': an ellipse (brain) containing a
    central blob (ventricle) whose size grows with CDR severity, plus
    per-image noise. This is a stand-in for real anatomy, purely so the
    pipeline has a learnable signal to test against."""
    img = Image.new("L", (size, size), color=20)
    draw = ImageDraw.Draw(img)

    margin = size // 8
    draw.ellipse([margin, margin, size - margin, size - margin], fill=140)

    # ventricle size scales with severity (0 -> smallest, 3 -> largest)
    severity_frac = cdr_score / 3.0
    vent_radius = int(size * (0.06 + 0.10 * severity_frac))
    cx, cy = size // 2, size // 2
    jitter = int(size * 0.03)
    cx += int(RNG.integers(-jitter, jitter + 1))
    cy += int(RNG.integers(-jitter, jitter + 1))
    draw.ellipse([cx - vent_radius, cy - vent_radius, cx + vent_radius, cy + vent_radius], fill=40)

    arr = np.array(img).astype(np.float32)
    noise_strength = 8 + 10 * severity_frac  # more severe -> slightly noisier, arbitrary but consistent
    arr += RNG.normal(0, noise_strength, arr.shape)
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr).convert("RGB")
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))  # mild acquisition-like blur
    return img


def generate(n_patients: int, scans_per_patient: int, out_dir: str, csv_path: str):
    os.makedirs(out_dir, exist_ok=True)

    cdr_values = list(CDR_WEIGHTS.keys())
    probs = list(CDR_WEIGHTS.values())

    rows = []
    for i in range(n_patients):
        naccid = f"SIM{i:05d}"
        cdr = RNG.choice(cdr_values, p=probs)
        rows.append({"NACCID": naccid, "CDRGLOB": cdr})

        n_scans = scans_per_patient if RNG.random() > 0.3 else 1  # some patients have only 1 visit
        for scan_idx in range(n_scans):
            img = _make_synthetic_slice(cdr)
            img.save(os.path.join(out_dir, f"{naccid}_{scan_idx}.jpg"), quality=90)

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    total_images = len(os.listdir(out_dir))
    print(f"Generated {len(df)} synthetic patients, {total_images} images -> {out_dir}")
    print(f"Wrote synthetic clinical CSV -> {csv_path}")
    print("\nCDR distribution (synthetic):")
    print(df["CDRGLOB"].value_counts().sort_index())
    print("\nThis is SYNTHETIC data for pipeline smoke-testing only -- not real "
          "patient data, and not something to draw any clinical conclusion from.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate synthetic sample data for pipeline smoke testing")
    parser.add_argument("--patients", type=int, default=60)
    parser.add_argument("--scans-per-patient", type=int, default=2)
    parser.add_argument("--out-dir", default=config.LABELLED_IMAGE_FOLDER)
    parser.add_argument("--csv-path", default=config.COMMERCIAL_FILE_PATH)
    args = parser.parse_args()

    generate(args.patients, args.scans_per_patient, args.out_dir, args.csv_path)
