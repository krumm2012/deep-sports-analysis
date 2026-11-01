"""No GPU needed to check split leakage, sequence boundaries or frozen inference."""
from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest

from dsa.measure.moseq_dataset import aggregate, apply_frozen, fixed_heading_resampling, load_manifest, save_results, tree_sha256
from dsa.measure.moseq_local import diagnose


def write_clip(path, value, original_sha):
    xy = np.full((40, 3, 2), float(value), dtype=np.float32)
    np.savez(path, coordinates=xy, confidences=np.ones((40, 3)), bodyparts=['nose', 'l_hip', 'r_hip'],
             fps=30, frame_indices=np.arange(40),
             metadata=json.dumps({'source': {'source_sha256': original_sha}}))


def write_manifest(tmp_path):
    records = []
    for key, split, value in [('a', 'train', 1), ('b', 'train', 2), ('c', 'validation', 3)]:
        write_clip(tmp_path / f'{key}.npz', value, key)
        records.append({'id': key, 'split': split, 'input': f'{key}.npz'})
    records.append({'id': 'gap', 'split': 'excluded', 'reason': 'temporal guard'})
    path = tmp_path / 'manifest.json'
    path.write_text(json.dumps({'clips': records}))
    return path, records


def test_recordings_stay_independent_and_excluded_files_are_not_loaded(tmp_path):
    path, _ = write_manifest(tmp_path)
    manifest, clips = load_manifest(path)
    assert set(clips) == {'a', 'b', 'c'}
    assert len(manifest['clips']) == 4
    assert all(c['data']['coordinates'].shape == (40, 3, 2) for c in clips.values())
    assert clips['c']['record']['split'] == 'validation'
    assert clips['a']['input'] == tmp_path / 'a.npz'


def test_duplicate_source_cannot_cross_split_even_if_npz_metadata_differs(tmp_path):
    path, _ = write_manifest(tmp_path)
    write_clip(tmp_path / 'c.npz', 99, 'a')
    with pytest.raises(ValueError, match='Duplicate recording'):
        load_manifest(path)


def test_mismatched_joint_order_or_duplicate_ids_are_rejected(tmp_path):
    path, records = write_manifest(tmp_path)
    records[2]['id'] = 'a'
    path.write_text(json.dumps({'clips': records}))
    with pytest.raises(ValueError, match='duplicate clip id'):
        load_manifest(path)


def test_mixed_frame_rates_are_rejected_before_training(tmp_path):
    path, _ = write_manifest(tmp_path)
    with np.load(tmp_path / 'c.npz', allow_pickle=False) as z:
        data = {k: z[k].copy() for k in z.files}
    data['fps'] = 25
    np.savez(tmp_path / 'c.npz', **data)
    with pytest.raises(ValueError, match='fps differs'):
        load_manifest(path)


def test_duration_statistics_do_not_join_same_state_across_clip_boundaries():
    values = {'a': diagnose(np.zeros(30, dtype=int), 30), 'b': diagnose(np.zeros(60, dtype=int), 30)}
    result = aggregate(values, 30)
    assert result['segment_count'] == 2
    assert result['transition_count'] == 0
    assert result['median_duration_seconds'] == 1.5
    assert result['seconds'] == 3


def test_frozen_application_passes_config_and_verifies_every_parameter(tmp_path):
    model = {'params': {'a': np.array([1., 2.]), 'nested': {'b': np.eye(2)}}}
    calls = []
    def apply_model(current, data, metadata, **kwargs):
        calls.append(kwargs)
        return {'clip': {}}, deepcopy(current)
    result, check = apply_frozen(SimpleNamespace(apply_model=apply_model), model, {}, (), num_iters=5,
                                 results_path=tmp_path/'heldout.h5', config={'fix_heading': True},
                                 resampling_module=SimpleNamespace(_wrapped_resample=lambda: None))
    assert 'clip' in result and check['unchanged']
    assert check['before_sha256'] == tree_sha256(model['params'])
    assert calls[0]['fix_heading'] is True and calls[0]['return_model'] is True


def test_fixed_heading_reaches_sampler_when_apply_model_drops_options(tmp_path):
    calls = []
    def original(*args, **kwargs):
        calls.append(kwargs)
    fitting = SimpleNamespace(_wrapped_resample=original)
    model = {'params': {'a': np.array([1.])}}
    def apply_model(current, *args, **kwargs):
        # Match the installed API: initialize with config, but omit it here.
        fitting._wrapped_resample('sampler', {}, current, states_only=True)
        return {}, deepcopy(current)
    _, check = apply_frozen(SimpleNamespace(apply_model=apply_model), model, {}, (), num_iters=2,
                            results_path=tmp_path/'result.h5', config={'fix_heading': True},
                            resampling_module=fitting)
    assert calls == [{'states_only': True, 'fix_heading': True}]
    assert fitting._wrapped_resample is original
    assert check['unchanged']


def test_fixed_heading_hook_restores_sampler_after_failure():
    original = lambda *args, **kwargs: None
    fitting = SimpleNamespace(_wrapped_resample=original)
    with pytest.raises(RuntimeError):
        with fixed_heading_resampling(fitting):
            assert fitting._wrapped_resample is not original
            raise RuntimeError('inference failed')
    assert fitting._wrapped_resample is original


@pytest.mark.parametrize('mutate_original', [False, True])
def test_changed_frozen_parameters_are_rejected(tmp_path, mutate_original):
    model = {'params': {'a': np.array([1., 2.])}}
    def apply_model(current, *args, **kwargs):
        changed = current if mutate_original else deepcopy(current)
        changed['params']['a'][0] = 99
        return {}, changed
    with pytest.raises(RuntimeError, match='parameters changed'):
        apply_frozen(SimpleNamespace(apply_model=apply_model), model, {}, (), num_iters=2,
                     results_path=tmp_path/'heldout.h5', config={})


def test_nonfinite_or_misaligned_model_outputs_are_not_published_as_valid(tmp_path):
    path, _ = write_manifest(tmp_path)
    _, clips = load_manifest(path)
    source = {'a': clips['a']}
    values = {'a': {'syllable': np.zeros(40), 'latent_state': np.zeros((40, 2)),
                    'centroid': np.zeros((40, 2)), 'heading': np.zeros(39)}}
    with pytest.raises(ValueError, match='length mismatch'):
        save_results(values, source, tmp_path, 'results')
    values['a']['heading'] = np.full(40, np.nan)
    with pytest.raises(ValueError, match='Nonfinite'):
        save_results(values, source, tmp_path, 'results')
