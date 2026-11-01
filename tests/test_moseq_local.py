"""MoSeq input provenance and temporal diagnostics without importing JAX."""
import numpy as np
import pytest

from dsa.measure.moseq_local import diagnose, read_input, run


def write_input(path, **changes):
    data = {"coordinates": np.ones((30, 3, 2)), "confidences": np.ones((30, 3)),
            "bodyparts": np.array(["nose", "l_hip", "r_hip"]), "fps": 30,
            "frame_indices": np.arange(30)}
    data.update(changes)
    np.savez(path, **data)


def test_input_preserves_low_confidence_and_source_frame_mapping(tmp_path):
    path = tmp_path / "input.npz"
    conf = np.ones((30, 3));conf[3, 1] = 0
    xy = np.ones((30, 3, 2));xy[3, 1] = np.nan
    write_input(path, coordinates=xy, confidences=conf, frame_indices=np.arange(30) / 2)
    data = read_input(path)
    assert np.isnan(data["coordinates"][3, 1]).all()
    assert data["confidences"][3, 1] == 0
    assert np.array_equal(data["frame_indices"], np.arange(30) / 2)


def test_untracked_or_unobserved_input_is_rejected(tmp_path):
    path = tmp_path / "bad.npz"
    write_input(path, confidences=np.ones((30, 2)))
    with pytest.raises(ValueError, match="dimensions"):
        read_input(path)
    xy = np.ones((30, 3, 2));xy[:, 1] = np.nan
    write_input(path, coordinates=xy)
    with pytest.raises(ValueError, match="observed"):
        read_input(path)
    write_input(path, frame_indices=np.arange(30)[::-1])
    with pytest.raises(ValueError, match="nondecreasing"):
        read_input(path)


def test_diagnostics_use_contiguous_runs_not_state_occupancy():
    d = diagnose(np.array([4, 4, 1, 1, 1, 4]), 10)
    assert d["used_states"] == 2 and d["transition_count"] == 2
    assert d["state_frames"] == {"1": 3, "4": 3}
    assert d["median_duration_seconds"] == pytest.approx(.2)
    assert [(r["state"], r["start_frame"], r["end_frame"]) for r in d["runs"]] == [(4, 0, 2), (1, 2, 5), (4, 5, 6)]
    assert diagnose(np.zeros(30, dtype=int), 30)["collapsed"]


def test_existing_fit_is_never_overwritten(tmp_path):
    source = tmp_path / "input.npz"
    write_input(source)
    output = tmp_path / "fit"
    output.mkdir()
    config = output / "config.yml"
    config.write_text("saved training configuration")
    with pytest.raises(FileExistsError, match="preserve existing"):
        run(source, output)
    assert config.read_text() == "saved training configuration"
    assert not (output / "summary.json").exists()
