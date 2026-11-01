import numpy as np
import pytest

from dsa.measure.semantic_pose_round import human_targets, review_segments
from dsa.measure.semantic_strokes import stroke_label
from dsa.pages.semantic_clip_labeling import annotation_registry


def test_corrected_stages_are_authoritative_and_unknown_is_not_a_target():
    clips = {'a': {'frames': 12}}
    records = [{'clip_id': 'a', 'split': 'train', 'reviewed': True, 'phase_usable': True,
                'phase': '挥拍', 'start_frame': 0, 'end_frame_exclusive': 5},
               {'clip_id': 'a', 'split': 'train', 'reviewed': True, 'phase_usable': False,
                'phase': None, 'start_frame': 5, 'end_frame_exclusive': 12}]
    a = human_targets(clips, records)['a']
    np.testing.assert_array_equal(a['target'], [2]*5+[-1]*7)
    assert (a['weight'][:5] == 2).all() and (a['origin'][:5] == 'human').all()


def test_continuous_review_is_only_a_copy_of_scored_prediction():
    rows = [{'phase': p, 'start_frame': a, 'end_frame_exclusive': b, 'start_seconds': a/25,
             'end_seconds': b/25, 'fps': 25, 'mean_score': .8}
            for a, b, p in [(0, 20, '引拍'), (20, 24, None), (24, 36, '挥拍'),
                            (36, 40, None), (40, 60, '随挥')]]
    result = review_segments(rows)
    assert len(rows) == 5 and rows[1]['phase'] is None
    assert [s['phase'] for s in result] == ['引拍', '挥拍', '随挥']
    assert all(result[i-1]['end_frame_exclusive'] == result[i]['start_frame'] for i in range(1, 3))
    assert result[0]['start_frame'] == 0 and result[-1]['end_frame_exclusive'] == 60


def test_generic_backhand_does_not_establish_grip():
    assert stroke_label({'shot_type': '反手', 'shot_type_usable': True}) == '无法判断'
    assert stroke_label({'stroke_type': '双手反手', 'stroke_type_reviewed': True}) == '双手反手'
    assert stroke_label({'shot_type': '正手', 'shot_type_usable': True,
                         'stroke_type': '无法判断', 'stroke_type_reviewed': True}) == '无法判断'


def test_registry_accepts_registered_later_rounds_but_not_path_escape(tmp_path):
    assert annotation_registry(tmp_path, {'candidate_registry': 'semantic-pose-v3/edit/candidates.json'}).is_relative_to(tmp_path)
    with pytest.raises(ValueError):
        annotation_registry(tmp_path, {'candidate_registry': '../outside.json'})
