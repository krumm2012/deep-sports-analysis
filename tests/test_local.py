"""Existing-GPU worker orchestration, without loading any model."""
from types import SimpleNamespace
import sys

import numpy as np
import pytest

from dsa.gpu import local


def test_phone_worker_bounds_frames_and_uses_project_options(tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))
    monkeypatch.setattr(local, "video_dir", lambda _: tmp_path / "video")
    monkeypatch.setattr(local, "WASB", tmp_path / "wasb")
    monkeypatch.setitem(local.PROJECTS, "demo", {"gpu": {"near_only": True}})
    for f in [tmp_path / "video/source.mp4", tmp_path / "wasb/src/configs/model/wasb.yaml",
              tmp_path / "wasb/pretrained_weights/wasb_tennis_best.pth.tar"]:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.touch()
    monkeypatch.setattr(local, "plan", lambda spec, seconds: ("source.mp4", [60], 30))
    calls = []
    def process(video, out, **kw):
        calls.append((video, out, kw))
        return {"frames": 60}
    monkeypatch.setattr(local, "process", process)
    assert local.run("demo/rally", seconds=2) == {"frames": 60}
    _, out, opts = calls[0]
    assert out.endswith("test/raw.npz") and opts["limit"] == 60 and opts["near_only"]
    assert not opts["tv"] and opts["breaks"] == []


def test_worker_reports_cuda_requirement_before_loading_models(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)))
    with pytest.raises(RuntimeError, match="CUDA-enabled"):
        local.run("demo/rally")
