"""
End-to-end integration test: runs the ACTUAL pipeline scripts (as
subprocesses, exactly as a person would from the command line) against
synthetic data, in an isolated temp directory.

Unlike test_pipeline.py's unit tests, which call individual functions
directly, this catches pipeline-LEVEL regressions that unit tests miss:
a script writing to the wrong path, an argument that silently stopped
matching between two scripts, a step that only works if run in a certain
order, etc. This is the automated version of the manual smoke test used
to validate the pipeline before each release.

Runs in ~10-20 seconds (tiny synthetic dataset, 1 training epoch) and
needs no real NACC data.

Run with: python -m pytest tests/test_integration.py -v
"""

import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def _run(cmd, cwd):
    result = subprocess.run(
        [sys.executable] + cmd, cwd=cwd, capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, (
        f"Command failed: {' '.join(cmd)}\n--- stdout ---\n{result.stdout}\n"
        f"--- stderr ---\n{result.stderr}"
    )
    return result


def test_full_pipeline_runs_end_to_end_on_synthetic_data(tmp_path):
    # every script resolves its paths relative to cwd via config.py,
    # so running with cwd=tmp_path keeps this fully isolated from the
    # real repo's data/results directories
    cwd = str(tmp_path)

    _run([str(REPO_ROOT / "scripts" / "generate_sample_data.py"),
          "--patients", "24", "--scans-per-patient", "1"], cwd)

    _run([str(REPO_ROOT / "data_pipeline.py"), "organize"], cwd)
    _run([str(REPO_ROOT / "preprocessing.py"), "apply"], cwd)
    _run([str(REPO_ROOT / "preprocessing.py"), "stats"], cwd)
    assert (tmp_path / "results" / "dataset_stats.json").exists()

    _run([str(REPO_ROOT / "data_pipeline.py"), "split"], cwd)
    assert (tmp_path / "dataset" / "train").exists()
    assert (tmp_path / "dataset" / "val").exists()
    assert (tmp_path / "dataset" / "test").exists()

    _run([str(REPO_ROOT / "train.py"), "--epochs", "1", "--patience", "1"], cwd)

    model_path = tmp_path / "models" / "best_model.pth"
    report_path = tmp_path / "results" / "test_report.json"
    assert model_path.exists(), "training did not produce a saved model"
    assert report_path.exists(), "evaluation did not produce a test report"

    report = json.loads(report_path.read_text())
    assert "accuracy" in report
    assert "quadratic_weighted_kappa" in report
    assert len(report["confusion_matrix"]) == 5  # full 5-class matrix, even on tiny data

    # pick any one test image and confirm gradcam_cli.py runs against the trained model
    test_dir = tmp_path / "dataset" / "test"
    sample_image = next(test_dir.rglob("*.jpg"))
    _run([str(REPO_ROOT / "gradcam_cli.py"), "--image", str(sample_image), "--gradcam"], cwd)

    gradcam_outputs = list((tmp_path / "results").glob("gradcam_*.png"))
    assert len(gradcam_outputs) == 1, "gradcam_cli.py did not produce a heatmap image"
