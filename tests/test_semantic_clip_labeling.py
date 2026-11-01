"""Full-video labels preserve time coverage and keep invalid regions out of targets."""
from copy import deepcopy
import json

import pytest

from dsa.measure.semantic_pilot import sha
from dsa.pages.semantic_clip_labeling import seed_clip, validate_export


def test_prior_human_hints_and_manual_boundaries_stay_pending():
    clip = {'id': 'a', 'frames': 100}
    records = [{'id': 'human', 'clip_id': 'a', 'start_frame': 20, 'end_frame_exclusive': 40,
                'phase': '准备', 'shot_type': '正手', 'shot_type_usable': True, 'notes': 'old note'}]
    state = seed_clip(clip, records, {'a': {30}}, {'a': [
        {'start_frame': 20, 'end_frame_exclusive': 30, 'phase': '引拍', 'notes': 'edited'}]})
    assert state['done'] is False
    assert [(s['start_frame'], s['end_frame_exclusive']) for s in state['segments']] == [(0, 20), (20, 30), (30, 40), (40, 100)]
    assert [s['phase'] for s in state['segments']] == ['', '引拍', '准备', '']
    assert state['segments'][1]['notes'] == 'edited'
    assert seed_clip(clip, [], {})['segments'][0]['phase'] == ''


@pytest.fixture
def full_export(tmp_path):
    manifest = tmp_path / 'manifest.json'
    manifest.write_text('{"fps":25}')
    task = {'id': 'clip-a', 'clip_id': 'a', 'split': 'train', 'fps': 25, 'clip_frames': 10,
            'source_sha256': 'raw', 'normalized_video_sha256': 'normalized'}
    registry = tmp_path / 'candidates.json'
    registry.write_text(json.dumps({'fps': 25, 'items': [task]}))
    records = []
    for i, (a, b, kind, phase) in enumerate([(0, 1, 'single', '挥拍'), (1, 6, 'invalid', None), (6, 10, 'uncertain', None)]):
        records.append({**task, 'id': f'clip-a--0{i+1}', 'parent_candidate_id': 'clip-a',
                        'start_frame': a, 'end_frame_exclusive': b, 'start_seconds': a/25, 'end_seconds': b/25,
                        'annotation_type': kind, 'phase': phase, 'phase_usable': kind == 'single',
                        'excluded': kind == 'invalid', 'reviewed': True, 'review_status': 'human_reviewed'})
    payload = {'schema_version': 3, 'annotation_mode': 'complete_clip_intervals', 'manifest_sha256': sha(manifest),
               'candidate_registry_sha256': sha(registry), 'fps': 25, 'completed_tasks': ['clip-a'], 'records': records}
    return tmp_path, registry, payload


def test_one_frame_annotation_is_valid_and_invalid_regions_are_retained(full_export):
    root, registry, payload = full_export
    assert validate_export(root, registry, payload) == {'completed_clips': 1, 'intervals': 3, 'total_training_clips': 1}
    assert payload['records'][1]['phase'] is None and payload['records'][1]['excluded']


@pytest.mark.parametrize('change', ['gap', 'validation', 'invalid_as_phase', 'time', 'pending', 'wrong_source'])
def test_bad_coverage_provenance_or_confirmation_is_rejected(full_export, change):
    root, registry, original = full_export
    payload = deepcopy(original)
    if change == 'gap':
        payload['records'][1].update(start_frame=2, start_seconds=.08)
    elif change == 'validation':
        payload['records'][0]['split'] = 'validation'
    elif change == 'invalid_as_phase':
        payload['records'][1].update(phase='回位/移动', phase_usable=True)
    elif change == 'time':
        payload['records'][0]['end_seconds'] = .08
    elif change == 'pending':
        payload['records'][0]['reviewed'] = False
    elif change == 'wrong_source':
        payload['records'][0]['normalized_video_sha256'] = 'another'
    with pytest.raises(ValueError):
        validate_export(root, registry, payload)


def test_partial_progress_carries_no_confirmed_intervals(full_export):
    root, registry, payload = full_export
    payload.update(completed_tasks=[], records=[])
    assert validate_export(root, registry, payload)['completed_clips'] == 0


def test_stroke_subtype_needs_explicit_review_and_a_core_phase(full_export):
    root, registry, payload = full_export
    payload['records'][0].update(stroke_type='双手反手', stroke_type_reviewed=True, stroke_type_usable=True)
    assert validate_export(root, registry, payload)['completed_clips'] == 1
    payload['records'][1].update(stroke_type='双手反手', stroke_type_reviewed=True, stroke_type_usable=True)
    with pytest.raises(ValueError, match='core stroke phase'):
        validate_export(root, registry, payload)
