from copy import deepcopy
import json

import pytest

from dsa.measure.semantic_curation import apply_clip_type, load_curation
from dsa.measure.semantic_pilot import sha


def test_forehand_confirmation_does_not_confirm_or_change_phase_boundaries():
    draft = {'done': False, 'manual_edit': True, 'segments': [
        {'start_frame': 0, 'end_frame_exclusive': 10, 'phase': '引拍', 'notes': 'mine'},
        {'start_frame': 10, 'end_frame_exclusive': 20, 'phase': '无法判断', 'notes': ''}]}
    before = deepcopy(draft)
    apply_clip_type(draft, {'stroke_type': '正手', 'action_validity': 'normal_action'})
    assert draft['done'] is False and draft['manual_edit'] is True
    assert [(s['start_frame'], s['end_frame_exclusive'], s['phase'], s['notes']) for s in draft['segments']] == [
        (s['start_frame'], s['end_frame_exclusive'], s['phase'], s['notes']) for s in before['segments']]
    assert all(s['stroke_type'] == '正手' and s['stroke_type_reviewed'] for s in draft['segments'])


@pytest.mark.parametrize('bad', ['validation', 'duplicate', 'overlap', 'phase_confirmed'])
def test_curation_rejects_invalid_membership_and_over_confirmation(tmp_path, bad):
    manifest = {'clips': [{'id': 'a', 'split': 'train'}, {'id': 'b', 'split': 'train'},
                          {'id': 'v', 'split': 'validation'}]}
    (tmp_path/'manifest.json').write_text(json.dumps(manifest))
    data = {'manifest_sha256': sha(tmp_path/'manifest.json'), 'excluded_clips': ['a'],
            'clip_overrides': {'b': {'stroke_type': '正手', 'phase_boundaries_reviewed': False}}}
    if bad == 'validation': data['excluded_clips'] = ['v']
    elif bad == 'duplicate': data['excluded_clips'] = ['a', 'a']
    elif bad == 'overlap': data['excluded_clips'] = ['b']
    else: data['clip_overrides']['b']['phase_boundaries_reviewed'] = True
    (tmp_path/'semantic-curation.json').write_text(json.dumps(data))
    with pytest.raises(ValueError): load_curation(tmp_path, manifest)
