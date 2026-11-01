"""Diagnostics must distinguish reproducible labels from meaningful segmentation."""
import json

import numpy as np
import pytest

from dsa.pages.moseq_report import compare_labels, run_statistics, segments


def test_segments_use_frame_durations_and_do_not_count_exactly_point_one_as_tiny():
    labels = np.array([4, 4, 4, 8, 8, 4])
    pieces = segments(labels, 30)
    assert [(p["state"], p["start"], p["end"]) for p in pieces] == [(4, 0, 3), (8, 3, 5), (4, 5, 6)]
    stats = run_statistics(labels, 30, expected_frames=6)
    assert stats["transition_count"] == 2
    assert stats["duration_seconds"] == pytest.approx(.2)
    assert stats["tiny_segments_under_0_1_seconds"] == 2
    assert stats["active_states"] == 2
    assert json.loads(json.dumps(stats))["frequencies"][0]["segments"] == 2


def test_label_permutation_keeps_partition_agreement():
    a = np.array([0, 0, 1, 1, 2, 2])
    b = np.array([7, 7, 4, 4, 9, 9])
    result = compare_labels(a, b)
    assert result["adjusted_rand_index"] == pytest.approx(1)
    assert result["normalized_mutual_information"] == pytest.approx(1)
    assert result["degenerate_comparison"] is False


def test_identical_single_state_is_flagged_as_collapse_despite_perfect_ari():
    labels = np.zeros(30, dtype=int)
    stats = run_statistics(labels, 30)
    comparison = compare_labels(labels, labels + 5)
    assert stats["single_state_collapse"] is True
    assert stats["dominant_state_over_95_percent"] is True
    assert comparison["adjusted_rand_index"] == pytest.approx(1)
    assert comparison["degenerate_comparison"] is True
    assert "不能" in comparison["interpretation"]


def test_misaligned_or_invalid_outputs_are_not_silently_visualized():
    with pytest.raises(ValueError, match="Label/input length mismatch"):
        run_statistics(np.zeros(5, int), 30, expected_frames=6)
    with pytest.raises(ValueError, match="latent_state/label length mismatch"):
        run_statistics(np.zeros(5, int), 30, arrays={"latent_state": np.zeros((4, 2))})
    with pytest.raises(ValueError, match="nonnegative integers"):
        segments(np.array([0, .5, 1]), 30)
    stats = run_statistics(np.array([0, 1, 1]), 30, arrays={"latent_state": np.array([[0.], [np.nan], [1.]])})
    assert stats["technical_output_valid"] is False
