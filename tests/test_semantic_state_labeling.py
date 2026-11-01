"""State naming translates frozen IDs and cannot silently cross model versions."""
from copy import deepcopy
import json

import numpy as np
import pytest

from dsa.measure.semantic_pilot import sha
from dsa.pages.semantic_state_labeling import apply_mapping, normalize_annotations, state_runs, validate_mapping


def example_config():
    return {'mapping_key': 'example:seed0:pinned', 'dataset': 'camera05/2025_1830', 'model': 'seed0',
            'checkpoint_sha256': 'pinned', 'fps': 30, 'mode': 'single', 'states': [{'id': 0}, {'id': 1}, {'id': 2}]}


def example_mapping(config):
    return {**{k: config[k] for k in ['mapping_key', 'dataset', 'model', 'checkpoint_sha256', 'fps']},
            'mappings': [{'state': 0, 'phase': '准备', 'reviewed': True},
                         {'state': 1, 'phase': '准备', 'reviewed': True},
                         {'state': 2, 'phase': None, 'reviewed': False}]}


def test_existing_boundaries_and_many_to_one_user_names_are_preserved():
    labels = np.array([2, 2, 0, 0, 0, 1, 1, 2])
    runs = state_runs(labels)
    assert runs == [{'state': 2, 'start_frame': 0, 'end_frame_exclusive': 2},
                    {'state': 0, 'start_frame': 2, 'end_frame_exclusive': 5},
                    {'state': 1, 'start_frame': 5, 'end_frame_exclusive': 7},
                    {'state': 2, 'start_frame': 7, 'end_frame_exclusive': 8}]
    config = example_config()
    assert validate_mapping(config, example_mapping(config)) == {0: '准备', 1: '准备'}


@pytest.mark.parametrize('field,value', [('checkpoint_sha256', 'another'), ('fps', 25),
                                        ('dataset', 'camera05_custom'), ('model', 'seed1')])
def test_mapping_cannot_cross_model_or_frame_rate(field, value):
    config = example_config()
    mapping = example_mapping(config)
    mapping[field] = value
    with pytest.raises(ValueError, match='does not match'):
        validate_mapping(config, mapping)


def test_missing_duplicate_or_confirmed_empty_names_are_rejected():
    config = example_config()
    mapping = example_mapping(config)
    duplicate = deepcopy(mapping)
    duplicate['mappings'][1]['state'] = 0
    with pytest.raises(ValueError, match='state IDs'):
        validate_mapping(config, duplicate)
    mapping['mappings'][2]['reviewed'] = True
    with pytest.raises(ValueError, match='needs a phase'):
        validate_mapping(config, mapping)


def test_apply_only_translates_ids_and_keeps_partial_states_unknown(tmp_path):
    root = tmp_path / 'data/projects/camera05/2025_1830'
    model = root / 'moseq/reference_seed0/model'
    model.mkdir(parents=True)
    checkpoint = model / 'checkpoint.h5'
    checkpoint.write_bytes(b'frozen-model')
    results = model.parent / 'results.npz'
    labels = np.array([0]*6 + [2]*4 + [1]*7, dtype=int)
    np.savez(results, syllable=labels, fps=30)
    config = example_config()
    config.update(checkpoint_sha256=sha(checkpoint), result_files={'2025_1830': sha(results)})
    mapping = example_mapping(config)
    config_path, mapping_path = tmp_path / 'states.json', tmp_path / 'mapping.json'
    config_path.write_text(json.dumps(config))
    mapping_path.write_text(json.dumps(mapping))
    destination = tmp_path / 'mapped'
    original_hash = sha(results)
    apply_mapping(config_path, mapping_path, tmp_path / 'data', destination)
    out = json.loads((destination / '2025_1830.json').read_text())
    assert out['runs'][1]['phase'] is None and out['runs'][1]['state_name_confirmed'] is False
    assert [r['state'] for r in out['runs']] == [0, 2, 1]
    assert [r['end_frame_exclusive'] for r in out['runs']] == [6, 10, 17]
    assert sha(results) == original_hash
    assert json.loads((destination / 'summary.json').read_text())['model_retrained'] is False
    config['result_files']['2025_1830'] = 'changed'
    config_path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match='state output changed'):
        apply_mapping(config_path, mapping_path, tmp_path / 'data', tmp_path / 'bad-output')
    assert not (tmp_path / 'bad-output').exists()


def test_mixed_invalid_and_uncertain_states_do_not_become_single_stage_targets():
    config = example_config()
    mapping = example_mapping(config)
    mapping['mappings'] = [
        {'state': 0, 'phase': None, 'annotation_type': 'mixed', 'mixed_phases': ['挥拍', '随挥'], 'reviewed': True},
        {'state': 1, 'phase': None, 'annotation_type': 'invalid', 'mixed_phases': [], 'reviewed': True},
        {'state': 2, 'phase': None, 'annotation_type': 'uncertain', 'mixed_phases': [], 'reviewed': True}]
    records = normalize_annotations(config, mapping)
    assert validate_mapping(config, mapping) == {}
    assert all(not m['usable_for_single_phase'] for m in records)
    assert [m['excluded'] for m in records] == [False, True, False]
    assert records[0]['mixed_phases'] == ['挥拍', '随挥']
    mapping['mappings'][0]['mixed_phases'] = ['挥拍']
    with pytest.raises(ValueError, match='at least two'):
        normalize_annotations(config, mapping)


def test_invalid_annotation_cannot_keep_a_phase_or_override_exclusion_flags():
    config = example_config()
    mapping = example_mapping(config)
    mapping['mappings'][0].update(annotation_type='invalid')
    with pytest.raises(ValueError, match='single-stage'):
        normalize_annotations(config, mapping)
    mapping['mappings'][0].update(phase=None, excluded=False)
    with pytest.raises(ValueError, match='inclusion flags'):
        normalize_annotations(config, mapping)


@pytest.mark.parametrize('kind,candidates', [('mixed', ['挥拍', '随挥']), ('invalid', []), ('uncertain', [])])
def test_applying_special_state_labels_preserves_frames_and_masks_training(tmp_path, kind, candidates):
    root = tmp_path / 'data/projects/camera05/2025_1830'
    model = root / 'moseq/reference_seed0/model'
    model.mkdir(parents=True)
    checkpoint = model / 'checkpoint.h5'
    checkpoint.write_bytes(b'frozen')
    results = model.parent / 'results.npz'
    np.savez(results, syllable=np.array([0]*5+[2]*7+[1]*4), fps=30)
    config = example_config()
    config.update(checkpoint_sha256=sha(checkpoint), result_files={'2025_1830': sha(results)})
    mapping = example_mapping(config)
    for m in mapping['mappings']:
        m.update(phase=None, reviewed=False)
    mapping['mappings'][2].update(annotation_type=kind, mixed_phases=candidates, reviewed=True)
    config_path, mapping_path = tmp_path / 'config.json', tmp_path / 'names.json'
    config_path.write_text(json.dumps(config))
    mapping_path.write_text(json.dumps(mapping))
    output = tmp_path / 'output'
    apply_mapping(config_path, mapping_path, tmp_path / 'data', output)
    record = json.loads((output / '2025_1830.json').read_text())['runs'][1]
    assert record['annotation_type'] == kind and record['phase'] is None
    assert record['phase_candidates'] == candidates and record['usable_for_single_phase'] is False
    assert record['excluded'] == (kind == 'invalid')
    assert record['start_frame'] == 5 and record['end_frame_exclusive'] == 12
