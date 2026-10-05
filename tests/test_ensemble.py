"""
End-to-end integration test for the sigma ensemble (ensemble.py), run
against synthetic data exactly like test_integration.py does for the
single-model pipeline. Verifies:
  1. The full prepare -> train -> evaluate flow runs without error.
  2. The "hard requirement" from ensemble.py's docstring actually holds:
     all 4 sigma variants get the IDENTICAL set of patients in each split.
  3. ensemble_results.json is produced with the expected shape.

Run with: python -m pytest tests/test_ensemble.py -v
"""

import json
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def _run(cmd, cwd, timeout=250):
    result = subprocess.run(
        [sys.executable] + cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout
    )
    assert result.returncode == 0, (
        f"Command failed: {' '.join(cmd)}\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    return result


def _patient_ids_in(split_dir: Path):
    return {re.split(r"_\d+\.jpg$", f.name)[0] for f in split_dir.rglob("*.jpg")}


def test_sigma_ensemble_runs_end_to_end_with_identical_patient_splits(tmp_path):
    cwd = str(tmp_path)

    _run([str(REPO_ROOT / "scripts" / "generate_sample_data.py"),
          "--patients", "30", "--scans-per-patient", "1"], cwd)

    _run([str(REPO_ROOT / "ensemble.py"), "prepare"], cwd, timeout=120)

    # --- the key correctness property: every sigma variant's dataset must
    # contain the EXACT same patients in train/val/test, so the 4 models'
    # test accuracy is comparable and the ensemble's combined evaluation
    # is valid ---
    sigma_dirs = sorted((tmp_path / "ensemble").glob("sigma_*/dataset"))
    assert len(sigma_dirs) == 4, "expected 4 sigma variants (config.ENSEMBLE_SIGMAS)"

    for split in ["train", "val", "test"]:
        patient_sets = [_patient_ids_in(d / split) for d in sigma_dirs]
        reference = patient_sets[0]
        for i, other in enumerate(patient_sets[1:], start=1):
            assert other == reference, (
                f"{split}: sigma variant {sigma_dirs[i]} has different patients "
                f"than {sigma_dirs[0]} — the shared patient_split_assignment.json "
                f"was not applied consistently."
            )

    _run([str(REPO_ROOT / "ensemble.py"), "train", "--epochs", "1", "--patience", "1"], cwd, timeout=250)

    for sigma_dir in sigma_dirs:
        model_path = sigma_dir.parent / "model" / "best_model.pth"
        pred_path = sigma_dir.parent / "results" / "test_predictions.json"
        assert model_path.exists(), f"missing trained model for {sigma_dir.parent.name}"
        assert pred_path.exists(), f"missing test_predictions.json for {sigma_dir.parent.name}"

    _run([str(REPO_ROOT / "ensemble.py"), "evaluate"], cwd, timeout=60)

    results_path = tmp_path / "ensemble" / "ensemble_results.json"
    assert results_path.exists()
    results = json.loads(results_path.read_text())

    assert len(results["individual_models"]) == 4
    assert "quadratic_weighted_kappa" in results["ensemble"]
    assert "confusion_matrix" in results["ensemble"]
    assert len(results["ensemble"]["confusion_matrix"]) == 5  # full 5-class matrix
    assert results["n_test_files_used"] > 0
