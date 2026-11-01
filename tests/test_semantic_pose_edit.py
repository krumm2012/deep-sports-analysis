from copy import deepcopy

import pytest

from dsa.pages.semantic_pose_edit import prediction_draft


def fixture():
    task = {'clip_id': 'a', 'fps': 25, 'clip_frames': 20}
    pred = {'clip_id': 'a', 'split': 'train', 'fps': 25, 'frames': 20,
            'segments': [{'start_frame': 0, 'end_frame_exclusive': 10, 'phase': '准备', 'mean_score': .9},
                         {'start_frame': 10, 'end_frame_exclusive': 20, 'phase': None, 'mean_score': .4}]}
    return task, pred


def test_pose_drafts_stay_unconfirmed_and_unknown_is_editable():
    task, pred = fixture()
    result = prediction_draft(task, pred, {})
    assert result['done'] is False
    assert [s['phase'] for s in result['segments']] == ['准备', '无法判断']
    assert all(s['auto_suggested'] and s['suggestion_source'] == 'pose-v2-refitted-model' for s in result['segments'])


@pytest.mark.parametrize('marker', ['done', 'manual_edit'])
def test_human_confirmation_and_manual_drafts_are_preserved(marker):
    task, pred = fixture()
    prior = {'done': marker == 'done', marker: True,
             'segments': [{'start_frame': 0, 'end_frame_exclusive': 20, 'phase': '无效动作', 'notes': 'mine'}]}
    expected = deepcopy(prior)
    result = prediction_draft(task, pred, prior)
    assert result == expected
    result['segments'][0]['notes'] = 'new'
    assert prior == expected


def test_historical_boundary_survives_new_model_segmentation():
    task, pred = fixture()
    result = prediction_draft(task, pred, {'seeded_from_prior': True, 'segments': [
        {'start_frame': 0, 'end_frame_exclusive': 7}, {'start_frame': 7, 'end_frame_exclusive': 20}]})
    assert [(s['start_frame'], s['end_frame_exclusive']) for s in result['segments']] == [(0, 7), (7, 10), (10, 20)]


def test_validation_and_gaps_cannot_seed_training_editor():
    task, pred = fixture()
    pred['split'] = 'validation'
    with pytest.raises(ValueError, match='source differs'):
        prediction_draft(task, pred, {})
    pred['split'] = 'train'
    pred['segments'][1]['start_frame'] = 11
    with pytest.raises(ValueError, match='continuous'):
        prediction_draft(task, pred, {})
