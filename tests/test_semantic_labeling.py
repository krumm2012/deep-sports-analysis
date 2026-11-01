"""Annotation rounds preserve source membership and keep validation out of fitting."""
from copy import deepcopy
import json

import numpy as np
import pytest

from dsa.measure.semantic_pilot import JOINTS, sha, validate_labels
from dsa.measure.semantic_phases import PHASES
from dsa.measure.semantic_weak import assemble_samples
from dsa.pages.semantic_autosegment import generate
from dsa.pages.semantic_labeling import build


@pytest.fixture
def round_root(tmp_path):
    root = tmp_path / 'project'
    for d in ['semantic-labeling', 'semantic-pilot/predictions', 'models/seed0/model']:
        (root / d).mkdir(parents=True)
    checkpoint = root / 'models/seed0/model/checkpoint.h5'
    checkpoint.write_bytes(b'frozen')
    clips = [{'id': 'old', 'split': 'train', 'sha256': 'old', 'frames': 50, 'fps': 25, 'start_seconds': 0}]
    for i, target in enumerate(['截击']*3 + ['准备']*4 + ['挥拍']*5):
        clip = f'new-{i:02d}'
        clips.append({'id': clip, 'split': 'train', 'sha256': clip, 'frames': 50, 'fps': 25, 'start_seconds': i*10})
        head = 'shot_type' if target == '截击' else 'phase'
        (root / 'semantic-pilot/predictions' / f'{clip}.json').write_text(json.dumps({
            'clip_id': clip, 'split': 'train', 'intervals': [
                {'start_frame': 8, 'end_frame_exclusive': 20, head: {'label': target, 'score': .9}}]}))
    clips += [{'id': f'heldout-{i}', 'split': 'validation', 'sha256': f'heldout-{i}', 'frames': 50, 'fps': 25,
               'start_seconds': 1000+i*10} for i in range(3)]
    (root / 'manifest.json').write_text(json.dumps({'fps': 25, 'clips': clips}))
    candidate = {'id': 'candidate-01', 'clip_id': 'old', 'split': 'train', 'source_sha256': 'old',
                 'start_frame': 4, 'end_frame_exclusive': 24, 'fps': 25, 'clip_frames': 50,
                 'video': '../clips/old/source.mp4'}
    options = {'shot_type_labels': ['正手', '截击', '无法判断'],
               'phase_labels': ['准备', '挥拍', '回位/移动', '混合/需调整区间', '无法判断']}
    (root / 'semantic-labeling/candidates.json').write_text(json.dumps({'fps': 25, **options, 'items': [candidate]}))
    label = {**candidate, 'start_seconds': .16, 'end_seconds': .96, 'shot_type': '正手',
             'phase': '混合/需调整区间', 'shot_type_usable': True, 'phase_usable': False,
             'reviewed': True, 'review_status': 'human_reviewed'}
    baseline = {'schema_version': 1, 'manifest_sha256': sha(root / 'manifest.json'),
                'baseline_checkpoint_sha256': sha(checkpoint), 'fps': 25, 'records': [label]}
    (root / 'semantic-pilot/labels.json').write_text(json.dumps(baseline))
    return root


def test_queue_uses_new_training_recordings_and_blind_validation(round_root):
    output = round_root / 'semantic-labeling-round2'
    summary = build(round_root, output)
    assert summary['new_selection_targets'] == {'截击': 3, '准备': 4, '挥拍': 5}
    training = json.loads((output / 'queue.json').read_text())
    validation = json.loads((output / 'validation-queue.json').read_text())
    new = [t for t in training['tasks'] if t['kind'] == 'new']
    assert len({t['clip_id'] for t in new}) == 12
    assert all(t['split'] == 'train' and not t['initial_shot_type'] for t in new)
    assert all(t['split'] == 'validation' and 'selection_score' not in t for t in validation['tasks'])
    assert not {t['clip_id'] for t in training['tasks']} & {t['clip_id'] for t in validation['tasks']}
    assert training['baseline_records'][0]['reviewed']
    old_id = training['queue_id']
    build(round_root, output)
    assert json.loads((output / 'queue.json').read_text())['queue_id'] == old_id


def test_supplemental_children_require_registry_and_nonoverlapping_targets(round_root):
    output = round_root / 'semantic-labeling-round2'
    build(round_root, output)
    queue = json.loads((output / 'queue.json').read_text())
    original = queue['baseline_records'][0]
    children = [{**original, 'id': f'candidate-01--0{i+1}', 'parent_candidate_id': 'candidate-01',
                 'start_frame': a, 'end_frame_exclusive': b, 'start_seconds': a/25, 'end_seconds': b/25,
                 'phase': '准备' if i == 0 else '挥拍', 'phase_usable': True}
                for i, (a, b) in enumerate([(4, 12), (12, 24)])]
    labels = {**queue, 'records': children, 'scope': 'training-only supplemental labels'}
    registry = output / 'candidates.json'
    validate_labels(round_root, labels, registry)
    with pytest.raises(ValueError, match='registry hash'):
        validate_labels(round_root, labels)
    invalid = deepcopy(labels)
    invalid['records'][1].update(start_frame=8, start_seconds=.32)
    with pytest.raises(ValueError, match='Overlapping'):
        validate_labels(round_root, invalid, registry)
    invalid = deepcopy(labels)
    invalid['records'][0].update(start_frame=0, start_seconds=0)
    with pytest.raises(ValueError, match='annotation range'):
        validate_labels(round_root, invalid, registry)


def test_validation_exports_cannot_be_trained(round_root):
    with pytest.raises(ValueError, match='Validation labels must not enter training'):
        validate_labels(round_root, {'scope': 'heldout-validation-only; do not train'})


def test_auto_proposals_preserve_queue_identity_and_leave_validation_blind(round_root):
    root = round_root
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    rng = np.random.default_rng(6)
    for c in manifest['clips']:
        c['input'] = c['id'] + '.npz'
        if c['split'] == 'train':
            np.savez(root / c['input'], coordinates=rng.normal(size=(50, 19, 2)),
                     confidences=np.ones((50, 19)), bodyparts=JOINTS, fps=25,
                     frame_indices=np.arange(50), metadata=json.dumps({'source': {'source_sha256': c['sha256']}}))
    manifest_path.write_text(json.dumps(manifest))
    baseline_path = root / 'semantic-pilot/labels.json'
    baseline = json.loads(baseline_path.read_text())
    baseline['manifest_sha256'] = sha(manifest_path)
    baseline_path.write_text(json.dumps(baseline))
    parameters = {}
    for head, classes in [('shot_type', ['截击', '正手']), ('phase', ['准备', '回位/移动', '挥拍'])]:
        k = 1 if len(classes) == 2 else len(classes)
        parameters[head] = {'classes': classes, 'coef': np.zeros((k, 192)).tolist(),
                            'intercept': [0.]*k, 'mean': [0.]*192, 'scale': [1.]*192}
    (root / 'semantic-pilot/models.json').write_text(json.dumps(parameters))
    (root / 'semantic-labeling/recommendations.json').write_text(json.dumps({
        'candidate-01': {'phase_description': '准备 → 挥拍'}}))
    output = root / 'semantic-labeling-round2'
    build(root, output)
    before = json.loads((output / 'queue.json').read_text())
    candidate_hash = sha(output / 'candidates.json')
    result = generate(root, output)
    after = json.loads((output / 'queue.json').read_text())
    assert before['queue_id'] == after['queue_id']
    assert sha(output / 'candidates.json') == candidate_hash
    assert result['tasks'] == len(before['tasks']) and result['all_pending']
    proposals = json.loads((output / 'auto-segments.json').read_text())
    for t in before['tasks']:
        p = proposals['tasks'][t['id']]
        assert p['reviewed'] is False
        for i, s in enumerate(p['segments']):
            assert s['auto_suggested'] and s['end_frame_exclusive']-s['start_frame'] >= 4
            assert t['allowed_start_frame'] <= s['start_frame'] < s['end_frame_exclusive'] <= t['allowed_end_frame']
            if i:
                assert s['start_frame'] == p['segments'][i-1]['end_frame_exclusive']
    validation_html = (output / 'validation.html').read_text()
    assert '<script id="auto-suggestions" type="application/json">{}</script>' in validation_html
    assert proposals['revision'] in (output / 'index.html').read_text()


def test_weak_dataset_preserves_human_truth_and_drops_overlapping_auto_intervals(round_root):
    root = round_root
    output = root / 'semantic-labeling-round2'
    build(root, output)
    queue = json.loads((output / 'queue.json').read_text())
    human = {**queue['baseline_records'][0], 'end_frame_exclusive': 12, 'end_seconds': .48,
             'phase': '准备', 'phase_usable': True}
    drafts = {t['id']: {'done': False, 'auto_revision': 'proposal-test', 'segments': [
        {'start_frame': t['start_frame'], 'end_frame_exclusive': t['end_frame_exclusive'],
         'phase': PHASES[i % 5]}]} for i, t in enumerate(queue['tasks'])}
    snapshot = {**queue, 'records': [human], 'completed_tasks': [], 'annotation_progress': {'tasks': drafts}}
    _, _, samples, excluded = assemble_samples(root, snapshot, output / 'candidates.json')
    assert excluded['overlaps_human_reference'] == 1
    assert sum(s['human_reviewed'] for s in samples) == 1
    assert all(s['source_weight'] == .25 and not s['human_reviewed'] for s in samples if s['label_source'] != 'human')
    assert set(s['phase'] for s in samples) == set(PHASES)
