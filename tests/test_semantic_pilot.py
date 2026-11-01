"""Check pilot leakage boundaries and the portable linear model parameters."""
import json

import numpy as np
import pytest

from dsa.measure.semantic_pilot import JOINTS, features, fit_head, sha, validate_labels


def test_features_do_not_read_outside_labeled_interval():
    xy = np.random.default_rng(5).normal(size=(30, 19, 2))
    data = {'coordinates': xy.copy(), 'bodyparts': JOINTS, 'fps': 25}
    expected = features(data, 8, 17)
    data['coordinates'][:8] = 10000
    data['coordinates'][17:] = -10000
    np.testing.assert_array_equal(features(data, 8, 17), expected)
    # Camera translation is removed, including the movement feature.
    data['coordinates'][8:17] += [123, -456]
    np.testing.assert_allclose(features(data, 8, 17), expected, atol=1e-12)
    with pytest.raises(ValueError, match='four frames'):
        features(data, 8, 10)


def test_whole_recordings_are_held_out_and_singleton_class_is_counted():
    records = [{'id': f'r{i}', 'clip_id': group, 'phase': label, 'phase_usable': True}
               for i, (group, label) in enumerate([
                   ('a', 'recovery'), ('a', 'ready'), ('b', 'recovery'),
                   ('c', 'ready'), ('d', 'swing')])]
    x = np.random.default_rng(8).normal(size=(5, 6))
    model, params, summary = fit_head(records, x, 'phase')
    for fold in summary['folds']:
        assert not set(fold['train_clips']) & set(fold['test_clips'])
    a_fold = next(f for f in summary['folds'] if f['test_clips'] == ['a'])
    assert set(a_fold['test_ids']) == {'r0', 'r1'}
    singleton = next(p for p in summary['out_of_fold'] if p['truth'] == 'swing')
    assert singleton['class_available_in_fold'] is False
    assert singleton['prediction'] != 'swing'
    assert summary['group_check']['per_class']['swing']['support'] == 1
    assert summary['group_check']['per_class']['swing']['recall'] == 0
    # Exported parameters reproduce inference without loading executable pickle.
    scores = ((x - params['mean']) / params['scale']) @ np.array(params['coef']).T + params['intercept']
    probability = np.exp(scores - scores.max(axis=1, keepdims=True))
    probability /= probability.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(probability, model.predict_proba(x))


def test_unknown_targets_are_excluded_per_head_and_binary_export_matches():
    records = [{'id': str(i), 'clip_id': str(i), 'shot_type': label,
                'shot_type_usable': label != '无法判断'}
               for i, label in enumerate(['正手', '截击', '正手', '截击', '无法判断'])]
    x = np.random.default_rng(1).normal(size=(5, 4))
    model, params, summary = fit_head(records, x, 'shot_type')
    assert summary['intervals'] == 4
    assert '无法判断' not in params['classes']
    assert '4' not in {p['id'] for p in summary['out_of_fold']}
    logits = ((x-params['mean']) / params['scale']) @ np.array(params['coef']).T + params['intercept']
    positive = 1 / (1 + np.exp(-logits[:, 0]))
    np.testing.assert_allclose(np.c_[1-positive, positive], model.predict_proba(x))


@pytest.mark.parametrize('problem', ['validation_clip', 'source', 'seconds', 'bounds', 'review', 'usable', 'manifest'])
def test_label_provenance_and_target_boundaries_are_enforced(tmp_path, problem):
    (tmp_path / 'semantic-labeling').mkdir()
    checkpoint = tmp_path / 'models/seed0/model/checkpoint.h5'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b'frozen')
    manifest = {'fps': 25, 'clips': [{'id': 'a', 'split': 'train', 'frames': 50, 'sha256': 'source'}]}
    if problem == 'validation_clip':
        manifest['clips'][0]['split'] = 'validation'
    (tmp_path / 'manifest.json').write_text(json.dumps(manifest))
    candidates = {'fps': 25, 'items': [{'id': 'r', 'clip_id': 'a', 'source_sha256': 'source'}],
                  'shot_type_labels': ['正手'], 'phase_labels': ['无法判断']}
    (tmp_path / 'semantic-labeling/candidates.json').write_text(json.dumps(candidates))
    record = {'id': 'r', 'clip_id': 'a', 'split': 'train', 'source_sha256': 'source',
              'reviewed': True, 'review_status': 'human_reviewed', 'fps': 25,
              'start_frame': 0, 'end_frame_exclusive': 10, 'start_seconds': 0, 'end_seconds': .4,
              'shot_type': '正手', 'shot_type_usable': True, 'phase': '无法判断', 'phase_usable': False}
    labels = {'fps': 25, 'manifest_sha256': sha(tmp_path / 'manifest.json'),
              'baseline_checkpoint_sha256': sha(checkpoint), 'records': [record]}
    changes = {'source': ('source_sha256', 'different'), 'seconds': ('end_seconds', .5),
               'bounds': ('end_frame_exclusive', 100), 'review': ('reviewed', False),
               'usable': ('phase_usable', True)}
    if problem in changes:
        key, value = changes[problem]
        record[key] = value
    elif problem == 'manifest':
        labels['manifest_sha256'] = 'different'
    with pytest.raises(ValueError):
        validate_labels(tmp_path, labels)
