"""Train a custom MoSeq model on independent clips and apply frozen parameters.

Manifest paths are relative to the manifest. Each clip has id, split, input and
optionally video. Splits are train, validation, or excluded; exclusion reasons
and chronological guard intervals belong in the versioned manifest.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import inspect
import json
import os
from pathlib import Path
import re
import time
import traceback

import numpy as np

from dsa.measure.moseq_local import _json, diagnose, read_input


def tree_sha256(tree) -> str:
    """Stable host-side digest, including keys, shapes and dtypes."""
    h = hashlib.sha256()

    def visit(value):
        if isinstance(value, dict):
            h.update(b'dict')
            for key in sorted(value):
                h.update(str(key).encode()); h.update(b'\0'); visit(value[key])
        elif isinstance(value, (list, tuple)):
            h.update(type(value).__name__.encode())
            for item in value:
                visit(item)
        else:
            a = np.asarray(value)
            if a.dtype.hasobject:
                raise ValueError('Cannot hash object-valued model parameters')
            h.update(str(a.dtype).encode()); h.update(str(a.shape).encode())
            h.update(np.ascontiguousarray(a).tobytes())
    visit(tree)
    return h.hexdigest()


def load_manifest(path: Path) -> tuple[dict, dict]:
    manifest = json.loads(path.read_text(encoding='utf-8'))
    records = manifest.get('clips')
    if not isinstance(records, list) or not records:
        raise ValueError('Manifest needs a nonempty clips list')
    clips, ids, split_hashes = {}, set(), {'train': set(), 'validation': set()}
    bodyparts, fps = None, None
    for record in records:
        key, split = record.get('id'), record.get('split')
        if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', key) or key in ids:
            raise ValueError(f'Invalid or duplicate clip id: {key!r}')
        ids.add(key)
        if split not in ('train', 'validation', 'excluded'):
            raise ValueError(f'Unknown split for {key}: {split}')
        if split == 'excluded':
            continue
        source = Path(record['input'])
        source = source if source.is_absolute() else path.parent / source
        data = read_input(source)
        if bodyparts is None:
            bodyparts, fps = data['bodyparts'], data['fps']
        if data['bodyparts'] != bodyparts or data['fps'] != fps:
            raise ValueError(f'Bodypart order/fps differs for {key}')
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        # Prefer original source SHA when available: per-clip metadata can make
        # duplicated recordings have different NPZ file hashes.
        metadata = json.loads(str(data.get('metadata', '{}')))
        original_hash = metadata.get('source', {}).get('source_sha256', digest)
        split_hashes[split].add(original_hash)
        clips[key] = {'record': record, 'input': source, 'data': data,
                      'input_sha256': digest, 'original_source_sha256': original_hash}
    if len([v for v in clips.values() if v['record']['split'] == 'train']) < 2:
        raise ValueError('At least two independent training clips are required')
    if not any(v['record']['split'] == 'validation' for v in clips.values()):
        raise ValueError('At least one held-out validation clip is required')
    if split_hashes['train'] & split_hashes['validation']:
        raise ValueError('Duplicate recording appears in training and validation')
    return manifest, clips


def aggregate(diagnostics: dict[str, dict], fps: float) -> dict:
    """Aggregate independent sequences, never count a transition at a cut."""
    counts, lengths = Counter(), []
    transitions, frames = 0, 0
    for d in diagnostics.values():
        counts.update(d['state_frames'])
        frames += d['frames']; transitions += d['transition_count']
        lengths.extend(r['end_frame'] - r['start_frame'] for r in d['runs'])
    if not frames:
        return {'clips': 0, 'frames': 0}
    return {'clips': len(diagnostics), 'frames': frames, 'seconds': frames / fps,
            'used_states': len(counts), 'state_frames': dict(sorted(counts.items())),
            'dominant_state_fraction': max(counts.values()) / frames,
            'transition_count': transitions, 'transitions_per_second': transitions / (frames / fps),
            'segment_count': len(lengths), 'median_duration_seconds': float(np.median(lengths) / fps),
            'duration_percentiles_seconds': (np.percentile(lengths, [10, 50, 90]) / fps).tolist(),
            'fraction_runs_under_0_1_seconds': float((np.asarray(lengths) / fps < .1 - 1e-9).mean()),
            'collapsed': len(counts) < 2,
            'boundary_note': 'Durations at recording boundaries may be truncated; cuts are never transitions.'}


def save_results(results: dict, clips: dict, output: Path, subdir: str) -> dict:
    target = output / subdir
    target.mkdir(parents=True, exist_ok=True)
    diagnostics = {}
    for key, result in results.items():
        source = clips[key]['data']
        mapped = {'syllable': np.asarray(result['syllable'], dtype=np.int16),
                  'latent_state': np.asarray(result['latent_state'], dtype=np.float32),
                  'centroid': np.asarray(result['centroid'], dtype=np.float32),
                  'heading': np.asarray(result['heading'], dtype=np.float32)}
        if any(len(v) != len(source['coordinates']) for v in mapped.values()):
            raise ValueError(f'Output length mismatch for {key}')
        if not all(np.isfinite(v).all() for v in mapped.values()):
            raise ValueError(f'Nonfinite output for {key}')
        np.savez_compressed(target / f'{key}.npz', **mapped, fps=source['fps'],
                            frame_indices=source['frame_indices'], bodyparts=np.asarray(source['bodyparts']))
        diagnostics[key] = diagnose(mapped['syllable'], source['fps'])
    if set(diagnostics) != set(clips):
        raise ValueError('Missing or unexpected output clips')
    return diagnostics


@contextmanager
def fixed_heading_resampling(resampling_module):
    """Forward fixed-heading through v0.6.6 apply_model's sampling loop.

    That release passes kwargs to initialization but omits them when sampling.
    Scope the compatibility hook to this synchronous application and restore it
    even when inference raises. State and parameter outputs are never rewritten.
    """
    original = resampling_module._wrapped_resample

    def fixed(*args, **kwargs):
        kwargs['fix_heading'] = True
        return original(*args, **kwargs)

    resampling_module._wrapped_resample = fixed
    try:
        yield
    finally:
        resampling_module._wrapped_resample = original


def apply_frozen(kpms, model, data, metadata, *, num_iters: int, results_path: Path, config: dict,
                 resampling_module=None):
    """Use official state-only inference and verify every learned parameter."""
    before = tree_sha256(model['params'])
    if config.get('fix_heading') and resampling_module is None:
        from keypoint_moseq import fitting
        resampling_module = fitting
    scope = fixed_heading_resampling(resampling_module) if config.get('fix_heading') else nullcontext()
    with scope:
        result, applied = kpms.apply_model(model, data, metadata, num_iters=num_iters,
                                           ar_only=False, save_results=True, results_path=str(results_path),
                                           return_model=True, **config)
    after, original_after = tree_sha256(applied['params']), tree_sha256(model['params'])
    if after != before or original_after != before:
        raise RuntimeError('Model parameters changed during frozen application')
    return result, {'before_sha256': before, 'after_sha256': after,
                    'original_model_after_sha256': original_after, 'unchanged': True}


def run(manifest_path: Path, output: Path, *, seed: int = 0, num_states: int = 8,
        latent_dim: int = 6, ar_iters: int = 100, full_iters: int = 500,
        apply_iters: int = 200, kappa_ar: float = 1e6, kappa_full: float = 1e5,
        allow_cpu: bool = False) -> dict:
    for name, value in {'num_states': num_states, 'latent_dim': latent_dim, 'ar_iters': ar_iters,
                        'full_iters': full_iters, 'apply_iters': apply_iters}.items():
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f'{name} must be a positive integer')
    if not all(np.isfinite(v) and v > 0 for v in (kappa_ar, kappa_full)):
        raise ValueError('Kappas must be finite and positive')
    manifest, clips = load_manifest(manifest_path)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Use a new empty directory to preserve previous fits: {output}')
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    first = next(iter(clips.values()))['data']; names, fps = first['bodyparts'], first['fps']
    summary = {'status': 'running', 'started_at_utc': datetime.now(timezone.utc).isoformat(),
               'manifest': str(manifest_path.resolve()),
               'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               'config': {'seed': seed, 'num_states': num_states, 'latent_dim_requested': latent_dim,
                          'ar_iters': ar_iters, 'full_iters': full_iters, 'apply_iters': apply_iters,
                          'kappa_ar': kappa_ar, 'kappa_full': kappa_full, 'fps': fps, 'fix_heading': True},
               'per_clip': {key: {'split': c['record']['split'], 'input': str(c['input'].resolve()),
                                  'input_sha256': c['input_sha256'], 'frames': len(c['data']['coordinates']),
                                  'results': f'results/{key}.npz'} for key, c in clips.items()},
               'excluded': [c for c in manifest['clips'] if c['split'] == 'excluded'],
               'timing': {}, 'iteration_seconds': {'ar': [], 'full': [], 'apply_train': [], 'apply_validation': []},
               'limitations': ['Held-out clips are from the same recording session/camera, not new subjects/cameras.',
                               'Unsupervised state IDs have no built-in tennis action or quality semantics.',
                               'No manual labels are available: cross-seed agreement is not recognition accuracy.']}
    _json(output / 'summary.json', summary)
    _json(output / 'dataset-manifest.json', manifest)
    try:
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
        os.environ.setdefault('MPLBACKEND', 'Agg')
        from dsa.paths import ROOT
        os.environ.setdefault('JAX_COMPILATION_CACHE_DIR', str(ROOT / 'cache/jax-moseq'))
        import jax
        import h5py
        import keypoint_moseq as kpms
        from keypoint_moseq import fitting
        from jax_moseq.models.keypoint_slds import alignment
        devices = jax.devices()
        if not allow_cpu and not any(d.platform == 'gpu' for d in devices):
            raise RuntimeError(f'GPU required; detected {devices}')
        summary['backend'] = {'default': jax.default_backend(), 'devices': [str(d) for d in devices],
                              'jax': jax.__version__, 'keypoint_moseq': importlib.metadata.version('keypoint-moseq'),
                              'numpy': np.__version__, 'allow_cpu': allow_cpu}
        apply_source = inspect.getsource(kpms.apply_model)
        if 'states_only=True' not in apply_source:
            raise RuntimeError('Installed apply_model source does not explicitly freeze model parameters')
        (output / 'installed-apply-model.py').write_text(apply_source)
        summary['apply_model_implementation'] = {
            'source_sha256': hashlib.sha256(apply_source.encode()).hexdigest(),
            'freezing': 'Official apply_model calls resampling with states_only=True; parameter SHA verified separately.',
            'documentation': 'https://keypoint-moseq.readthedocs.io/en/latest/_modules/keypoint_moseq/fitting.html'}
        alignment_source = inspect.getsource(alignment.preprocess_for_pca)
        (output / 'installed-pca-alignment.py').write_text(alignment_source)
        summary['alignment'] = {
            'fix_heading': True,
            'source_sha256': hashlib.sha256(alignment_source.encode()).hexdigest(),
            'pca': 'Subtract per-frame mean keypoint centroid and center-embed; no egocentric rotation.',
            'sampling': 'fix_heading=True is passed to training and application resampling; h is fixed at zero.',
            'interpretation': 'Preserves image-plane roll/sideways lean; heading is not physical court yaw.',
            'anterior_posterior': 'nose/hips configuration is retained but does not rotate poses in fixed-heading mode.'}
        summary['alignment']['application_compatibility'] = (
            'keypoint-moseq 0.6.6 apply_model omits kwargs during resampling; a scoped wrapper '
            'forwards fix_heading=True to every sampling step and restores the original function. '
            'No output headings or latent trajectories are overwritten.')
        summary['alignment']['compatibility_hook_sha256'] = hashlib.sha256(
            inspect.getsource(fixed_heading_resampling).encode()).hexdigest()
        skeleton = [[a, b] for a, b in [('l_shoulder', 'r_shoulder'), ('l_hip', 'r_hip'),
                    ('l_shoulder', 'l_elbow'), ('l_elbow', 'l_wrist'), ('r_shoulder', 'r_elbow'),
                    ('r_elbow', 'r_wrist'), ('l_hip', 'l_knee'), ('l_knee', 'l_ankle'),
                    ('r_hip', 'r_knee'), ('r_knee', 'r_ankle')] if a in names and b in names]
        kpms.setup_project(str(output), overwrite=True, bodyparts=names, use_bodyparts=names,
                           anterior_bodyparts=['nose'], posterior_bodyparts=['l_hip', 'r_hip'],
                           skeleton=skeleton, fps=fps)
        kpms.update_config(str(output), num_states=num_states, latent_dim=latent_dim, fix_heading=True)
        config = kpms.load_config(str(output))
        splits = {s: {k: v for k, v in clips.items() if v['record']['split'] == s}
                  for s in ('train', 'validation')}

        def format_split(split, format_seed):
            np.random.seed(format_seed)
            subset = splits[split]
            return kpms.format_data({k: c['data']['coordinates'] for k, c in subset.items()},
                                    {k: c['data']['confidences'] for k, c in subset.items()}, **config)

        # Identical regularizing input noise across model seeds; only sampler
        # initialization changes. Held-out coordinates never enter fit_pca.
        data, metadata = format_split('train', 0)
        pca = kpms.fit_pca(**data, **config)
        kpms.save_pca(pca, str(output))
        used_dim = min(latent_dim, int(pca.n_components_))
        kpms.update_config(str(output), latent_dim=used_dim)
        config = kpms.load_config(str(output))
        model = kpms.init_model(data, pca=pca, **{**config, 'seed': seed})
        summary['initial_model_prng_key'] = np.asarray(jax.device_get(model['seed'])).tolist()
        summary['config']['latent_dim'] = used_dim
        summary['pca_explained_variance'] = float(pca.explained_variance_ratio_[:used_dim].sum())
        summary['pca_training_clips'] = list(splits['train'])
        summary['timing']['setup_seconds'] = time.perf_counter() - started
        original_resample, phase = fitting._wrapped_resample, ['ar']

        def timed_resample(*args, **kwargs):
            t = time.perf_counter()
            result = original_resample(*args, **kwargs)
            jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, 'block_until_ready') else x, result)
            summary['iteration_seconds'][phase[0]].append(time.perf_counter() - t)
            return result

        def require_snapshot(iteration):
            with h5py.File(output / 'model/checkpoint.h5') as f:
                if str(iteration) not in f['model_snapshots']:
                    raise RuntimeError(f'Training stopped before requested iteration {iteration}')

        fitting._wrapped_resample = timed_resample
        try:
            model = kpms.update_hypparams(model, kappa=kappa_ar)
            t = time.perf_counter()
            model, _ = kpms.fit_model(model, data, metadata, str(output), 'model', ar_only=True,
                                      num_iters=ar_iters, generate_progress_plots=False,
                                      save_every_n_iters=100, fix_heading=True)
            require_snapshot(ar_iters)
            summary['timing']['ar_seconds_including_jit'] = time.perf_counter() - t
            _json(output / 'summary.json', summary)
            phase[0] = 'full'; model = kpms.update_hypparams(model, kappa=kappa_full)
            t = time.perf_counter()
            model, _ = kpms.fit_model(model, data, metadata, str(output), 'model', ar_only=False,
                                      start_iter=ar_iters, num_iters=ar_iters + full_iters,
                                      generate_progress_plots=False, save_every_n_iters=100, fix_heading=True)
            require_snapshot(ar_iters + full_iters)
            summary['timing']['full_seconds_including_jit'] = time.perf_counter() - t
            fitted = kpms.extract_results(model, metadata, str(output), 'model')
            fit_diagnostics = save_results(fitted, splits['train'], output, 'training_fit_results')
            summary['training_fit_aggregate'] = aggregate(fit_diagnostics, fps)
            # Keep train/application and validation/application method identical.
            # Official docs warn against comparing fitted train states directly
            # to new-data application states.
            parameter_digest = tree_sha256(model['params'])
            summary['trained_parameter_sha256'] = parameter_digest
            summary['aggregate'], summary['frozen_parameter_checks'] = {}, {}
            for split in ('train', 'validation'):
                phase[0] = f'apply_{split}'; t = time.perf_counter()
                apply_data, apply_metadata = format_split(split, 0)
                result, frozen_check = apply_frozen(kpms, model, apply_data, apply_metadata,
                                                    num_iters=apply_iters,
                                                    results_path=output / f'{split}-applied.h5', config=config,
                                                    resampling_module=fitting)
                if len(summary['iteration_seconds'][phase[0]]) != apply_iters:
                    raise RuntimeError(f'Application stopped early for {split}')
                summary['frozen_parameter_checks'][split] = frozen_check
                if any(not np.allclose(r['heading'], 0, atol=1e-6) for r in result.values()):
                    raise RuntimeError('Fixed-heading application produced nonzero heading')
                diagnostics = save_results(result, splits[split], output, 'results')
                for key, d in diagnostics.items():
                    summary['per_clip'][key]['diagnostics'] = d
                    summary['per_clip'][key]['maximum_absolute_heading'] = float(np.max(np.abs(result[key]['heading'])))
                summary['aggregate'][split] = aggregate(diagnostics, fps)
                summary['timing'][f'apply_{split}_seconds_including_jit'] = time.perf_counter() - t
                _json(output / 'summary.json', summary)
            # A final-100-iteration comparison is a convergence diagnostic only,
            # not evidence that the posterior has fully mixed.
            previous_iter = ar_iters + full_iters - 100
            if previous_iter >= ar_iters:
                previous, _, previous_metadata, _ = kpms.load_checkpoint(str(output), 'model', iteration=previous_iter)
                previous_results = kpms.extract_results(previous, previous_metadata, save_results=False)
                from sklearn.metrics import adjusted_rand_score
                pairs = [(np.asarray(previous_results[k]['syllable']), np.asarray(fitted[k]['syllable']))
                         for k in sorted(fitted)]
                per_clip_ari = {k: float(adjusted_rand_score(previous_results[k]['syllable'], fitted[k]['syllable']))
                                for k in sorted(fitted)}
                summary['sampling_diagnostic'] = {
                    'iterations_compared': [previous_iter, ar_iters + full_iters],
                    'global_train_ARI': float(adjusted_rand_score(np.concatenate([a for a, _ in pairs]),
                                                                np.concatenate([b for _, b in pairs]))),
                    'per_clip_ARI': per_clip_ari,
                    'note': 'Two training snapshots, not a posterior convergence proof or action accuracy.'}
        finally:
            fitting._wrapped_resample = original_resample
        summary['checkpoint'] = 'model/checkpoint.h5'
        summary['results_h5'] = {'train': 'train-applied.h5', 'validation': 'validation-applied.h5'}
        summary['status'] = 'complete'
        summary['timing']['total_seconds'] = time.perf_counter() - started
        summary['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        _json(output / 'summary.json', summary)
        return summary
    except Exception as exc:
        summary.update(status='failed', error=f'{type(exc).__name__}: {exc}', elapsed_seconds=time.perf_counter() - started)
        _json(output / 'summary.json', summary)
        (output / 'failure.txt').write_text(traceback.format_exc())
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', required=True, type=Path)
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--num-states', type=int, default=8)
    p.add_argument('--latent-dim', type=int, default=6)
    p.add_argument('--ar-iters', type=int, default=100)
    p.add_argument('--full-iters', type=int, default=500)
    p.add_argument('--apply-iters', type=int, default=200)
    p.add_argument('--kappa-ar', type=float, default=1e6)
    p.add_argument('--kappa-full', type=float, default=1e5)
    p.add_argument('--allow-cpu', action='store_true')
    a = p.parse_args()
    result = run(a.manifest, a.output, seed=a.seed, num_states=a.num_states, latent_dim=a.latent_dim,
                 ar_iters=a.ar_iters, full_iters=a.full_iters, apply_iters=a.apply_iters,
                 kappa_ar=a.kappa_ar, kappa_full=a.kappa_full, allow_cpu=a.allow_cpu)
    print(json.dumps({k: result[k] for k in ('status', 'config', 'timing', 'aggregate')}, indent=2))


if __name__ == '__main__':
    main()
