"""Motion boundaries are proposals; they must align and remain outside training truth."""
import numpy as np
import pytest

from dsa.measure.semantic_phases import PHASES
from dsa.pages.semantic_autosegment import cycle_window, motion_descriptors, partition_motion, portable_scores, prior_phase_order, proposed_label


def test_partition_finds_pose_change_and_keeps_continuous_frame_coverage():
    x = np.r_[np.zeros((12, 3)), np.full((12, 3), 5.)]
    assert partition_motion(x, 0, 24, 2) == [0, 12, 24]
    boundaries = partition_motion(x, 3, 23, 20)
    assert boundaries[0] == 3 and boundaries[-1] == 23
    assert min(np.diff(boundaries)) >= 4
    with pytest.raises(ValueError, match='Invalid motion'):
        partition_motion(x, 0, 3, 2)


def test_exported_binary_and_multiclass_score_rules_and_unknown_fallback():
    binary = {'classes': ['forehand', 'volley'], 'mean': [0., 0.], 'scale': [1., 1.],
              'coef': [[1., -1.]], 'intercept': [0.]}
    assert portable_scores(binary, [0., 0.]) == {'forehand': .5, 'volley': .5}
    assert proposed_label(binary, [0., 0.])[0] == '无法判断'
    assert proposed_label(binary, [0., 4.])[0] == 'forehand'
    multiclass = {**binary, 'classes': ['ready', 'recovery', 'swing'],
                  'coef': [[1., 0.], [0., 1.], [-1., -1.]], 'intercept': [0., 0., 0.]}
    scores = portable_scores(multiclass, [2., 0.])
    assert np.isclose(sum(scores.values()), 1.)
    assert max(scores, key=scores.get) == 'ready'


def test_visual_order_preserves_ambiguity_and_does_not_invent_stages():
    assert prior_phase_order({'phase_description': '挥拍 → 随挥 → 回位'}) == ['挥拍', '随挥', '回位/移动']
    assert prior_phase_order({'phase_description': '引拍/挥拍 → 随挥'}) == ['无法判断', '随挥']
    assert prior_phase_order({'phase_description': '前移 + 低位伸拍'}) == []


def test_five_stage_cycle_contains_motion_anchor_and_omits_missing_clip_prefix():
    def data_at(peak):
        t = np.arange(60)
        xy = np.zeros((60, 19, 2))
        xy[:, 6, 0] = 100/(1+np.exp(-(t-peak)/2))
        return {'coordinates': xy, 'fps': 25}
    task = {'start_frame': 25, 'end_frame_exclusive': 35, 'target': '挥拍',
            'allowed_start_frame': 0, 'allowed_end_frame': 60}
    data = data_at(30)
    start, end, order, peak = cycle_window(data, task)
    assert order == list(PHASES)
    boundaries = partition_motion(motion_descriptors(data), start, end, len(order), phase_order=order, swing_frame=peak)
    swing = order.index('挥拍')
    assert boundaries[swing] <= peak < boundaries[swing+1]
    task.update(start_frame=0, end_frame_exclusive=8)
    _, _, partial, _ = cycle_window(data_at(1), task)
    assert partial[0] == '挥拍' and '准备' not in partial
    zero = {'coordinates': np.zeros((60, 19, 2)), 'fps': 25}
    assert cycle_window(zero, task)[2] == ['无法判断']
