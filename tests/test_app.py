"""Smoke tests for the Streamlit demo (app.py)."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from streamlit.testing.v1 import AppTest

import config
from AppModel import AlzhiNet

APP = str(Path(__file__).parent.parent / "app.py")


def test_app_shows_friendly_warning_when_no_model_is_trained(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BEST_MODEL_PATH", str(tmp_path / "missing.pth"))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert len(at.warning) == 1


def test_app_loads_cleanly_when_a_model_exists(tmp_path, monkeypatch):
    model_path = tmp_path / "best_model.pth"
    torch.save(AlzhiNet(num_classes=len(config.CLASS_ORDER)).state_dict(), model_path)
    monkeypatch.setattr(config, "BEST_MODEL_PATH", str(model_path))
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value.startswith("AlzhiNet")
