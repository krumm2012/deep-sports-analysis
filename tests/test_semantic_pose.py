"""Conservative pose contexts and exclusion boundaries."""
import numpy as np
import pytest

from dsa.measure.semantic_pose import apply_anchors, frame_features, suppress_boundaries


def test_features_align_each_frame_and_reject_long_static_pose():
    xy = np.zeros((40, 19, 2))
    xy[:, [1, 2], 1] = -100
    data = {'coordinates': xy, 'confidences': np.ones((40, 19)), 'fps': 25}
    x, quality, frozen = frame_features(data)
    assert x.shape == (40, 192) and np.isfinite(x).all()
    assert quality.all() and frozen.all()
    data['confidences'][10, 5] = .1
    assert not frame_features(data)[1][10]


def test_boundary_guard_and_short_islands_stay_out_of_targets():
    label = np.repeat([0, 1, 2], 10)
    accepted = suppress_boundaries(label, np.ones(30, dtype=bool))
    assert not accepted[8:12].any() and not accepted[18:22].any()
    assert accepted[12:18].all()
    mask = np.zeros(30, dtype=bool); mask[3:5] = True
    assert not suppress_boundaries(label, mask).any()


def test_anchors_cannot_use_validation_or_overlap():
    clips = {'a': {'split': 'train', 'frames': 20}, 'b': {'split': 'validation', 'frames': 20}}
    ann = {'a': {'target': np.full(20, -1), 'weight': np.zeros(20), 'origin': np.full(20, '', dtype='<U32')}}
    record = {'clip_id': 'b', 'start_frame': 0, 'end_frame_exclusive': 10, 'phase': '准备'}
    with pytest.raises(ValueError, match='training clip'):
        apply_anchors(ann, [record], clips, False)
    record['clip_id'] = 'a'
    apply_anchors(ann, [record], clips, False)
    assert (ann['a']['origin'][:10] == 'agent_visual_pose_anchor').all()
    with pytest.raises(ValueError, match='Overlapping'):
        apply_anchors(ann, [record], clips, True)
