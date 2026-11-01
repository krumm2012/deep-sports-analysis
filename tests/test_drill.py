"""Single-person drill export must preserve body geometry and missing-data evidence."""
import json

import numpy as np
import pytest

from dsa.measure.drill import BODY, export, normalize


def test_body_normalization_removes_global_scale_and_offset_without_erasing_motion():
    xy = np.zeros((5, 19, 2), float)
    xy[:, 1:3, 1] = -30
    xy[:, 5, 0] = np.arange(5) * 10
    a, _ = normalize(xy)
    b, _ = normalize(xy * 2 + [200, 400])
    np.testing.assert_allclose(a, b)
    assert np.diff(a[:, 5, 0]).min() > 0
    with pytest.raises(ValueError, match="torso"):
        normalize(np.zeros((5, 19, 2)))


def test_export_ignores_small_background_track_and_marks_imputed_point_uncertain(tmp_path):
    pose = np.full((8, 65, 3), np.nan)
    for f in range(4):
        pose[f * 2, BODY, :2] = [75, 50]
        pose[f * 2, BODY, 2] = 4
        pose[f * 2, BODY[1:3], 1] = 30
        pose[f * 2, BODY[7:9], 1] = 60
    pose[2, BODY[5], :2] = np.nan
    boxes = np.tile([[55, 10, 95, 90, .9], [10, 0, 20, 15, .8]], (4, 1))
    np.savez(tmp_path / "raw.npz", n=np.full(4, 2), boxes=boxes, track=np.tile([0, 1], 4),
             kp=pose, fps=30, width=100, height=100)
    (tmp_path / "source-provenance.json").write_text(json.dumps({"original_pts_seconds": [0, 1 / 30, 2 / 30, .1]}))
    path = export(tmp_path)
    z = np.load(path, allow_pickle=False)
    assert z["coordinates"].shape == (4, 19, 2)
    assert np.isfinite(z["coordinates"]).all()
    assert z["confidences"][1, 5] == 0
    assert not z["raw_observed"][1, 5]
    assert json.loads(str(z["metadata"]))["subject"]["roi_frames"] == 4
