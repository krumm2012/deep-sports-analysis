"""Dataset diagnostics retain temporal boundaries and held-out provenance."""
import json
from pathlib import Path

import numpy as np
import pytest

from dsa.pages.moseq_dataset_report import aggregate, compare_runs, distribution_shift, frozen_parameter_status, load_inputs, load_run
from dsa.pages.moseq_report import run_statistics


def record(labels, clip_id="clip", split="train"):
    labels = np.array(labels, dtype=int)
    return {"id": clip_id, "split": split, "labels": labels, "frame_indices": np.arange(len(labels)),
            "statistics": run_statistics(labels, 10)}


def test_aggregate_preserves_video_boundaries_and_reports_heldout_states():
    a = record([0, 0, 1, 1], "a")
    b = record([2, 2, 2, 0], "b")
    result = aggregate([a, b], {0, 1})
    assert result["frames"] == 8
    assert result["transition_count"] == 2  # No synthetic 1->2 at the clip boundary.
    assert result["segment_count"] == 4
    assert result["states_absent_from_training"] == [2]
    assert result["unseen_state_frame_fraction"] == pytest.approx(3 / 8)


def test_invalid_numeric_output_is_not_pooled_with_valid_segmentation():
    valid = record([0, 1, 1], "a")
    invalid = record([2, 2, 2], "b")
    invalid["statistics"]["technical_output_valid"] = False
    result = aggregate([valid, invalid])
    assert result["frames"] == 3
    assert result["invalid_output_clips"] == 1
    assert result["technical_output_valid"] is False
    assert 2 not in result["state_frames"]


def test_distribution_shift_uses_common_label_space_and_handles_zeros():
    a = aggregate([record([0, 0])])
    same = distribution_shift(a, a)
    assert same["jensen_shannon_divergence_bits"] == pytest.approx(0)
    assert same["total_variation"] == pytest.approx(0)
    opposite = distribution_shift(a, aggregate([record([1, 1])]))
    assert opposite["jensen_shannon_divergence_bits"] == pytest.approx(1)
    assert opposite["total_variation"] == pytest.approx(1)
    assert distribution_shift(a, aggregate([]))["total_variation"] is None


def test_heldout_freeze_requires_all_three_hashes_and_explicit_success():
    check = {"before_sha256": "abc", "after_sha256": "abc", "original_model_after_sha256": "abc", "unchanged": True}
    summary = {"trained_parameter_sha256": "abc", "frozen_parameter_checks": {"train": dict(check), "validation": dict(check)}}
    assert frozen_parameter_status(summary) == {"train": True, "validation": True}
    summary["frozen_parameter_checks"]["validation"]["original_model_after_sha256"] = "mutated"
    assert frozen_parameter_status(summary) == {"train": True, "validation": False}
    assert frozen_parameter_status({}) == {"train": False, "validation": False}


def test_cross_seed_agreement_is_global_and_split_specific():
    left = {"name": "seed0", "records": {"a": record([0, 0, 1, 1], "a"),
                                          "b": record([0, 1, 0, 1], "b", "validation")}}
    right = {"name": "seed1", "records": {"a": record([9, 9, 4, 4], "a"),
                                           "b": record([9, 4, 9, 4], "b", "validation")}}
    result = compare_runs(left, right, "validation")
    assert result["clips"] == 1
    assert result["per_clip"][0]["id"] == "b"
    assert result["pooled"]["adjusted_rand_index"] == pytest.approx(1)
    right["records"]["b"]["frame_indices"][1] = 20
    with pytest.raises(ValueError, match="alignment"):
        compare_runs(left, right, "validation")


def make_input(tmp_path):
    path = tmp_path / "input.npz"
    np.savez(path, coordinates=np.zeros((6, 19, 2)), image_coordinates=np.zeros((6, 19, 2)),
             confidences=np.ones((6, 19)), fps=10, frame_indices=np.arange(6), bodyparts=np.array([f"p{i}" for i in range(19)]))
    clip = {"id": "a", "split": "validation", "input": path.name, "video": "source.mp4"}
    return clip, load_inputs(clip, tmp_path)


def test_stale_input_hash_and_misaligned_results_are_rejected(tmp_path):
    clip, source = make_input(tmp_path)
    folder = tmp_path / "seed0"
    (folder / "results").mkdir(parents=True)
    np.savez(folder / "results/a.npz", syllable=np.array([0, 0, 1, 1, 0, 0]), latent_state=np.zeros((6, 2)),
             centroid=np.zeros((6, 2)), heading=np.zeros(6), fps=10, frame_indices=np.arange(6), bodyparts=source["bodyparts"])
    summary = {"status": "complete", "per_clip": {"a": {"input_sha256": "different"}}}
    (folder / "summary.json").write_text(json.dumps(summary))
    result = load_run(folder, [clip], {"a": source}, tmp_path / "report")
    assert not result["records"]
    assert "SHA256" in result["failures"][0]["error"]
    summary["per_clip"]["a"]["input_sha256"] = source["sha256"]
    summary["status"] = "failed"
    (folder / "summary.json").write_text(json.dumps(summary))
    result = load_run(folder, [clip], {"a": source}, tmp_path / "report")
    assert result["records"]["a"]["statistics"]["technical_output_valid"] is False
    assert result["aggregate"]["validation"]["frames"] == 0
