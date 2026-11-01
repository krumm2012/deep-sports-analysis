"""Explicit weakly supervised five-phase trial, with label provenance retained."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import socket
import time

import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dsa.measure.semantic_phases import PHASES, phase_schema
from dsa.measure.semantic_pilot import checked_input, features, metrics, sha, validate_labels, write_json


def assemble_samples(root, snapshot, registry, weak_weight=.25):
    """Keep human truth; exclude unresolved intervals and overlapping weak targets."""
    if not 0 < weak_weight < 1:
        raise ValueError('Weak-label weight must be between zero and one')
    manifest, clips = validate_labels(root, snapshot, registry)
    queue = json.loads((registry.parent / 'queue.json').read_text())
    if snapshot['queue_id'] != queue['queue_id']:
        raise ValueError('Annotation queue differs from the local registry')
    samples = []
    for r in snapshot['records']:
        if r['phase_usable'] and r['phase'] in PHASES:
            samples.append({'id': r['id'], 'clip_id': r['clip_id'], 'source_sha256': r['source_sha256'],
                            'start_frame': r['start_frame'], 'end_frame_exclusive': r['end_frame_exclusive'],
                            'phase': r['phase'], 'label_source': 'human', 'human_reviewed': True,
                            'source_weight': 1.})
    dropped = Counter()
    tasks = {t['id']: t for t in queue['tasks']}
    registered = {r['id']: r for r in json.loads(registry.read_text())['items']}
    for task_id, draft in snapshot['annotation_progress']['tasks'].items():
        if task_id not in tasks or task_id not in registered:
            raise ValueError('Unknown draft annotation task')
        task = tasks[task_id]
        if task['split'] != 'train' or clips[task['clip_id']]['split'] != 'train':
            raise ValueError('Validation drafts must not enter weak training')
        if draft['done']:
            if task_id not in snapshot['completed_tasks']:
                raise ValueError('Completed draft is absent from confirmed export records')
            continue
        if not draft.get('auto_revision'):
            dropped['draft_without_proposal_provenance'] += len(draft['segments'])
            continue
        for i, s in enumerate(draft['segments']):
            if s['phase'] not in PHASES:
                dropped['unresolved_phase'] += 1
                continue
            a, b = s['start_frame'], s['end_frame_exclusive']
            if type(a) is not int or type(b) is not int or b-a < 4 or a < task['allowed_start_frame'] or b > task['allowed_end_frame']:
                raise ValueError('Weak interval exceeds its registered annotation range')
            if any(h['clip_id'] == task['clip_id'] and max(a, h['start_frame']) < min(b, h['end_frame_exclusive'])
                   for h in samples if h['human_reviewed']):
                dropped['overlaps_human_reference'] += 1
                continue
            samples.append({'id': f'weak/{task_id}/{i+1:02d}', 'parent_candidate_id': task_id,
                            'clip_id': task['clip_id'], 'source_sha256': task['source_sha256'],
                            'start_frame': a, 'end_frame_exclusive': b, 'phase': s['phase'],
                            'label_source': 'weak_pending_user_adjustment' if draft.get('manual_edit') else 'weak_auto',
                            'human_reviewed': False, 'source_weight': float(weak_weight),
                            'proposal_revision': draft['auto_revision'],
                            'proposal_source': s.get('suggestion_source', 'unconfirmed_draft')})
    for clip_id in {s['clip_id'] for s in samples}:
        intervals = sorted((s['start_frame'], s['end_frame_exclusive']) for s in samples if s['clip_id'] == clip_id)
        if any(a < intervals[i-1][1] for i, (a, _) in enumerate(intervals) if i):
            raise ValueError('Overlapping phase training intervals')
    if set(s['phase'] for s in samples) != set(PHASES):
        raise ValueError('Five-phase training needs examples for all five stages')
    return manifest, clips, samples, dict(dropped)


def balanced_source_weights(y, source_weights):
    y, source_weights = np.asarray(y), np.asarray(source_weights, dtype=float)
    mass = {c: float(source_weights[y == c].sum()) for c in np.unique(y)}
    target = source_weights.sum()/len(mass)
    return np.array([w*target/mass[c] for c, w in zip(y, source_weights)])


def fit_model(x, y, source_weights):
    if len(set(y)) < 2:
        return DummyClassifier(strategy='most_frequent').fit(x, y, sample_weight=source_weights)
    model = make_pipeline(StandardScaler(), LogisticRegression(C=.1, max_iter=3000, random_state=0))
    model.fit(x, y, standardscaler__sample_weight=source_weights,
              logisticregression__sample_weight=balanced_source_weights(y, source_weights))
    return model


def group_evaluate(x, samples):
    y = np.array([s['phase'] for s in samples])
    groups = np.array([s['clip_id'] for s in samples])
    weights = np.array([s['source_weight'] for s in samples])
    oof, folds = [], []
    for train, test in LeaveOneGroupOut().split(x, y, groups):
        if set(groups[train]) & set(groups[test]):
            raise AssertionError('Recording leakage')
        model = fit_model(x[train], y[train], weights[train])
        baseline = DummyClassifier(strategy='most_frequent').fit(x[train], y[train], sample_weight=weights[train])
        prediction, majority = model.predict(x[test]), baseline.predict(x[test])
        folds.append({'train_clips': sorted(set(groups[train])), 'test_clips': sorted(set(groups[test])),
                      'train_ids': [samples[i]['id'] for i in train], 'test_ids': [samples[i]['id'] for i in test],
                      'missing_training_phases': [p for p in PHASES if p not in set(y[train])]})
        for i, pred, base in zip(test, prediction, majority):
            oof.append({'id': samples[i]['id'], 'clip_id': str(groups[i]), 'reference': str(y[i]),
                        'prediction': str(pred), 'majority_prediction': str(base),
                        'human_reviewed': samples[i]['human_reviewed'], 'label_source': samples[i]['label_source']})
    evaluation = {'folds': folds, 'out_of_fold': oof}
    for kind, human in [('human_reference_check', True), ('weak_label_agreement', False)]:
        subset = [p for p in oof if p['human_reviewed'] == human]
        covered = [p for p in PHASES if any(v['reference'] == p for v in subset)]
        if not subset:
            evaluation[kind] = {'intervals': 0, 'reference_phases': [], 'missing_reference_phases': list(PHASES),
                                'metrics': None, 'majority_baseline': None}
            continue
        evaluation[kind] = {'intervals': len(subset), 'reference_phases': covered,
                            'missing_reference_phases': [p for p in PHASES if p not in covered],
                            'metrics': metrics([p['reference'] for p in subset], [p['prediction'] for p in subset], list(PHASES)),
                            'majority_baseline': metrics([p['reference'] for p in subset], [p['majority_prediction'] for p in subset], list(PHASES))}
        if subset:
            evaluation[kind]['macro_f1_on_reference_phases'] = metrics(
                [p['reference'] for p in subset], [p['prediction'] for p in subset], covered)['macro_f1']
    return evaluation, fit_model(x, y, weights)


def decode_ordered_phases(probability, minimum_frames=4, transition_penalty=1.):
    """Cyclic five-stage smoothing; this prior is not a boundary accuracy claim."""
    probability = np.asarray(probability, dtype=float)
    if probability.ndim != 2 or probability.shape[1] != len(PHASES) or len(probability) < minimum_frames:
        raise ValueError('Expected T>=4 scores for the five ordered phases')
    n, phases = probability.shape
    states = phases*minimum_frames
    emission = -np.log(np.clip(probability, 1e-12, 1))
    previous = np.full((n, states), -1, dtype=np.int16)
    cost = np.full(states, np.inf)
    cost[::minimum_frames] = emission[0]
    for t in range(1, n):
        next_cost = np.full(states, np.inf)
        for phase in range(phases):
            offset = phase*minimum_frames
            for duration in range(1, minimum_frames):
                destination = offset+duration
                origin = destination-1
                next_cost[destination] = cost[origin]+emission[t, phase]
                previous[t, destination] = origin
            last = offset+minimum_frames-1
            if cost[last]+emission[t, phase] < next_cost[last]:
                next_cost[last] = cost[last]+emission[t, phase]
                previous[t, last] = last
            origin = ((phase-1) % phases)*minimum_frames+minimum_frames-1
            next_cost[offset] = cost[origin]+transition_penalty+emission[t, phase]
            previous[t, offset] = origin
        cost = next_cost
    path = np.empty(n, dtype=np.int16)
    endings = np.arange(minimum_frames-1, states, minimum_frames)
    state = int(endings[np.argmin(cost[endings])])
    for t in range(n-1, -1, -1):
        path[t] = state//minimum_frames
        if t:
            state = int(previous[t, state])
    return path


def predict_clip(model, data):
    n = len(data['coordinates'])
    vectors = []
    for frame in range(n):
        start = max(0, min(n-11, frame-5))
        vectors.append(features(data, start, min(n, start+11)))
    raw = model.predict_proba(np.stack(vectors))
    order = [list(model.classes_).index(p) for p in PHASES]
    scores = raw[:, order]
    phases = decode_ordered_phases(scores)
    bounds = np.r_[0, np.flatnonzero(np.diff(phases))+1, n]
    segments = [{'start_frame': int(a), 'end_frame_exclusive': int(b),
                 'start_seconds': float(a/data['fps']), 'end_seconds': float(b/data['fps']),
                 'phase': PHASES[int(phases[a])],
                 'mean_model_score': float(scores[a:b, int(phases[a])].mean()), 'human_reviewed': False}
                for a, b in zip(bounds[:-1], bounds[1:])]
    return phases, scores, segments


def run(root, annotation_path, registry, output, weak_weight=.25):
    started = time.monotonic()
    if output.exists():
        raise ValueError('Use a fresh output directory; existing models are preserved')
    snapshot = json.loads(annotation_path.read_text())
    manifest, clips, samples, dropped = assemble_samples(root, snapshot, registry, weak_weight)
    inputs = {c: checked_input(root, clips[c]) for c in {s['clip_id'] for s in samples}}
    x = np.stack([features(inputs[s['clip_id']], s['start_frame'], s['end_frame_exclusive']) for s in samples])
    evaluation, model = group_evaluate(x, samples)
    if set(model.classes_) != set(PHASES):
        raise ValueError('Final classifier does not cover the five stages')
    output.mkdir(parents=True)
    (output / 'predictions').mkdir()
    (output / 'annotation-snapshot.json').write_bytes(annotation_path.read_bytes())
    write_json(output / 'training-samples.json', {'scope': 'weak-supervision-training-only; not human ground truth',
               'phase_schema': phase_schema(), 'records': samples, 'excluded': dropped})
    scaler, linear = model.steps[0][1], model.steps[1][1]
    parameters = {'classes': linear.classes_.tolist(), 'mean': scaler.mean_.tolist(), 'scale': scaler.scale_.tolist(),
                  'coef': linear.coef_.tolist(), 'intercept': linear.intercept_.tolist(), 'scores_calibrated': False,
                  'probability_rule': 'softmax', 'training_type': 'weakly_supervised'}
    write_json(output / 'models.json', {'phase': parameters})
    counts = {kind: {p: sum(s['phase'] == p and s['human_reviewed'] == human for s in samples) for p in PHASES}
              for kind, human in [('human', True), ('weak', False)]}
    prediction_counts = Counter()
    summary = {'schema_version': 1, 'status': 'running', 'training_type': 'weakly_supervised',
               'created_at_utc': datetime.now(timezone.utc).isoformat(), 'host': socket.gethostname(),
               'python': platform.python_version(), 'numpy': np.__version__, 'sklearn': sklearn.__version__,
               'phase_schema': phase_schema(), 'manifest_sha256': sha(root / 'manifest.json'),
               'annotation_sha256': sha(annotation_path), 'registry_sha256': sha(registry), 'code_sha256': sha(__file__),
               'source_interval_counts': counts, 'training_intervals': len(samples),
               'training_recordings': len(inputs), 'excluded_samples': dropped,
               'configuration': {'C': .1, 'seed': 0, 'human_source_weight': 1., 'weak_source_weight': weak_weight,
                                 'class_balance': 'inverse total source weight per phase within each training fold',
                                 'feature_version': 'interval-centered-pose-motion-v1', 'features': x.shape[1],
                                 'prediction_window_frames': 11, 'decoder': 'five-stage cyclic order, minimum 4 frames, transition cost 1'},
               'evaluation': evaluation, 'heldout_semantic_accuracy': None,
               'preview_clips': [c['id'] for c in manifest['clips'] if c['split'] == 'validation'][:3],
               'limitations': ['Backswing/follow-through currently have weak references only; no human accuracy for them.',
                               'Grouped training-pool checks are not held-out validation accuracy.',
                               'Cyclic temporal smoothing imposes an action-order prior; boundaries need review.',
                               'Same subject/session, small candidate-selected set, uncalibrated scores.']}
    for c in manifest['clips']:
        if c['split'] not in {'train', 'validation'}:
            continue
        data = inputs.get(c['id'])
        if data is None:
            data = checked_input(root, c)
        ids, scores, segments = predict_clip(model, data)
        np.savez_compressed(output / 'predictions' / f'{c["id"]}.npz', phase_ids=ids,
                            probabilities=scores.astype(np.float32), phase_names=np.array(PHASES),
                            fps=data['fps'], frame_indices=np.arange(len(ids)))
        write_json(output / 'predictions' / f'{c["id"]}.json', {
            'clip_id': c['id'], 'split': c['split'], 'fps': data['fps'], 'frames': len(ids),
            'status': 'unreviewed_weak_model_prediction', 'input_sha256': sha(root / c['input']),
            'segments': segments})
        prediction_counts[c['split']] += 1
    summary.update(prediction_clips=dict(prediction_counts), model_sha256=sha(output / 'models.json'),
                   training_samples_sha256=sha(output / 'training-samples.json'),
                   duration_seconds=time.monotonic()-started)
    from dsa.pages.semantic_weak_report import report
    report(output, summary, samples)
    summary['status'] = 'complete'
    write_json(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('annotations', type=Path)
    parser.add_argument('registry', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--weak-weight', type=float, default=.25)
    args = parser.parse_args()
    result = run(args.root, args.annotations, args.registry, args.output, args.weak_weight)
    print(json.dumps({k: result[k] for k in ['status', 'training_intervals', 'training_recordings',
          'source_interval_counts', 'prediction_clips', 'duration_seconds']}, ensure_ascii=False, indent=2))
