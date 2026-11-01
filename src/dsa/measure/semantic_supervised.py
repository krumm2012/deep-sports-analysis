"""Fully supervised phase training after every active training clip is reviewed.

Whole chronological blocks are held out with a source-time guard. Original held-out
validation clips never enter fitting, expert fitting, or threshold estimation.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import joblib
import numpy as np

from dsa.measure.semantic_curation import load_curation
from dsa.measure.semantic_phases import PHASES, phase_schema
from dsa.measure.semantic_pilot import checked_input, metrics, sha, write_json
from dsa.measure.semantic_pose import PoseEnsemble, frame_features, runs, suppress_boundaries
from dsa.measure.semantic_pose_round import human_targets, review_segments
from dsa.measure.semantic_strokes import CORE_PHASES, STROKE_TYPES, stroke_label, stroke_schema
from dsa.pages.semantic_clip_labeling import annotation_registry, validate_export


def temporal_folds(clips, folds=5, guard_seconds=90.):
    if any(c.get('split') != 'train' for c in clips.values()):
        raise ValueError('Only training clips may enter temporal cross-validation')
    ordered = sorted(clips, key=lambda i: (clips[i]['start_seconds'], i))
    result = []
    for number, block in enumerate(np.array_split(ordered, min(folds, len(ordered))), 1):
        test = list(map(str, block)); train, guarded = [], []
        for i in ordered:
            if i in test:
                continue
            a = clips[i]['start_seconds']; b = a+clips[i]['frames']/clips[i]['fps']
            distance = min(max(clips[j]['start_seconds']-b,
                               a-clips[j]['start_seconds']-clips[j]['frames']/clips[j]['fps'], 0.) for j in test)
            (guarded if distance < guard_seconds else train).append(i)
        result.append({'fold': number, 'train_clips': train, 'test_clips': test,
                       'guarded_out_clips': guarded, 'guard_seconds': guard_seconds})
    return result


def interval_stroke(record, overrides):
    if record.get('stroke_type_reviewed'):
        return stroke_label(record)
    return overrides.get(record['clip_id'], {}).get('stroke_type', stroke_label(record))


class SupervisedPhaseModel:
    """Shared preparation/recovery with optional type-specific core phase experts."""
    def fit(self, x, y, weight, stroke_types, groups):
        self.shared = PoseEnsemble().fit(x, y, weight)
        self.classes_ = self.shared.classes_
        self.experts, self.expert_support = {}, {}
        for kind in STROKE_TYPES[:-1]:
            selected = (stroke_types == kind) & np.isin(y, [1, 2, 3])
            if set(y[selected]) == {1, 2, 3} and len(set(groups[selected])) >= 3:
                self.experts[kind] = PoseEnsemble().fit(x[selected], y[selected], weight[selected])
                self.expert_support[kind] = {'frames': int(selected.sum()),
                    'clips': sorted(set(groups[selected])), 'phases': list(CORE_PHASES)}
        return self

    @property
    def scaler(self):
        return self.shared.scaler

    def scores(self, x, stroke_type='无法判断'):
        pred, score, probability, agreement, distance = self.shared.scores(x)
        route = np.full(len(x), stroke_type) if isinstance(stroke_type, str) else np.asarray(stroke_type)
        if len(route) != len(x):
            raise ValueError('One stroke routing value is required per feature row')
        probability = probability.copy(); agreement = agreement.copy(); distance = distance.copy()
        for kind, expert in self.experts.items():
            ix = np.flatnonzero(route == kind)
            if not len(ix):
                continue
            _, _, core, core_agreement, core_distance = expert.scores(x[ix])
            columns = np.array([list(self.classes_).index(c) for c in expert.classes_])
            mass = probability[np.ix_(ix, columns)].sum(1, keepdims=True)
            probability[np.ix_(ix, columns)] = mass*core
            local_prediction = self.classes_[probability[ix].argmax(1)]
            use_core = np.isin(local_prediction, expert.classes_)
            agreement[ix[use_core]] &= core_agreement[use_core]
            distance[ix[use_core]] = np.maximum(distance[ix[use_core]], core_distance[use_core])
        return self.classes_[probability.argmax(1)], probability.max(1), probability, agreement, distance


def predict_frames(model, features, stroke_type, distance_limit):
    x, quality, static = features
    pred, score, probability, agreement, distance = model.scores(x, stroke_type)
    ordered = np.sort(probability, axis=1)
    accepted = quality & agreement & (score >= .7) & (ordered[:, -1]-ordered[:, -2] >= .25) & (distance <= distance_limit)
    # Human-confirmed steady preparation poses are legitimate supervised samples.
    accepted = suppress_boundaries(pred, accepted)
    return pred, score, probability, accepted, distance


def run(root, output, annotation, folds=5, guard_seconds=90.):
    started = time.monotonic()
    if output.exists():
        raise ValueError('Use a fresh fully supervised model directory')
    payload = json.loads(annotation.read_text()); registry = annotation_registry(root, payload)
    validate_export(root, registry, payload)
    manifest = json.loads((root/'manifest.json').read_text())
    curation, curation_path = load_curation(root, manifest)
    excluded = set(curation['excluded_clips']); overrides = curation.get('clip_overrides', {})
    train = {c['id']: c for c in manifest['clips'] if c['split']=='train' and c['id'] not in excluded}
    records = [r for r in payload['records'] if r['clip_id'] in train]
    confirmed = {r['clip_id'] for r in records}
    if confirmed != set(train):
        raise ValueError(f'Complete review required: {len(confirmed)}/{len(train)} clips; missing {sorted(set(train)-confirmed)}')
    clips = {**train, **{c['id']: c for c in manifest['clips'] if c['split']=='validation'}}
    f = {i: frame_features(checked_input(root, c)) for i, c in clips.items()}
    targets = human_targets(train, records)
    stroke = {i: np.full(c['frames'], '无法判断', dtype='<U16') for i, c in train.items()}
    route = {i: overrides.get(i, {}).get('stroke_type', '无法判断') for i in train}
    for r in records:
        kind = interval_stroke(r, overrides)
        stroke[r['clip_id']][r['start_frame']:r['end_frame_exclusive']] = kind
        if r['phase'] in CORE_PHASES and r.get('stroke_type_reviewed') and kind != route[r['clip_id']]:
            route[r['clip_id']] = '无法判断'
    vectors, ys, weights, groups, frames, kinds = [], [], [], [], [], []
    pose_dropped = 0
    for i, c in train.items():
        human = targets[i]['target'] >= 0
        ok = human & f[i][1]
        pose_dropped += int((human & ~ok).sum())
        if not ok.any():
            continue
        vectors.append(f[i][0][ok]); ys.extend(targets[i]['target'][ok]); weights.extend(targets[i]['weight'][ok])
        groups.extend([i]*int(ok.sum())); frames.extend(np.flatnonzero(ok)); kinds.extend(stroke[i][ok])
    x, y, w, g, frame, kind = np.vstack(vectors), np.array(ys), np.array(weights), np.array(groups), np.array(frames), np.array(kinds)
    if set(y) != set(range(5)):
        raise ValueError('The reviewed training targets must include all five phases')
    fold_reports, distances, oof_truth, oof_prediction = [], [], [], []
    for fold in temporal_folds(train, folds, guard_seconds):
        tr, te = np.isin(g, fold['train_clips']), np.isin(g, fold['test_clips'])
        if set(y[tr]) != set(range(5)) or not te.any():
            fold_reports.append({**fold, 'status': 'insufficient_training_phase_coverage'}); continue
        print(f"Fold {fold['fold']}/{folds}: train {len(fold['train_clips'])}, test {len(fold['test_clips'])}, guarded {len(fold['guarded_out_clips'])}", flush=True)
        model = SupervisedPhaseModel().fit(x[tr], y[tr], w[tr], kind[tr], g[tr])
        pred, score, probability, agreement, dist = model.scores(x[te], np.array([route[i] for i in g[te]]))
        distances.extend(dist); oof_truth.extend(y[te]); oof_prediction.extend(pred)
        fold_reports.append({**fold, 'status': 'complete', 'test_clip': f"时间组{fold['fold']}（{len(fold['test_clips'])}段）",
            'frames': int(te.sum()), 'reference_source_counts': {'human': int(te.sum())},
            'metrics': metrics(y[te], pred, list(range(5)))})
    distance_limit = float(np.quantile(distances, .95)) if distances else 1.
    final = SupervisedPhaseModel().fit(x, y, w, kind, g)
    output.mkdir(parents=True); (output/'predictions').mkdir()
    (output/'user-annotation.json').write_bytes(annotation.read_bytes())
    (output/'annotation-registry.json').write_bytes(registry.read_bytes())
    (output/'training-code.py').write_bytes(Path(__file__).read_bytes())
    if curation_path:
        (output/'curation.json').write_bytes(curation_path.read_bytes())
    joblib.dump(final, output/'model.joblib')
    joblib.dump(final.shared, output/'shared-phase-model.joblib')
    for name, expert in final.experts.items():
        joblib.dump(expert, output/('forehand-core-model.joblib' if name=='正手' else f'core-expert-{STROKE_TYPES.index(name)}.joblib'))
    np.savez_compressed(output/'training.npz', x=x, y=y, weight=w, groups=g, frames=frame,
                        stroke_types=kind, origins=np.full(len(y), 'human'))
    labeled = [{**r, 'label_source': 'human', 'human_reviewed': True} for r in records]
    write_json(output/'labels.json', {'records': labeled, 'training_type': 'fully_supervised',
                                     'phase_schema': phase_schema(), 'human_annotation_sha256': sha(annotation)})
    for i, c in clips.items():
        routing = route.get(i, '无法判断')
        pred, score, probability, accepted, dist = predict_frames(final, f[i], routing, distance_limit)
        shown = np.where(accepted, pred, -1)
        segments = [{'start_frame': a, 'end_frame_exclusive': b, 'start_seconds': a/c['fps'],
                     'end_seconds': b/c['fps'], 'fps': c['fps'], 'phase': PHASES[p] if p>=0 else None,
                     'mean_score': float(score[a:b].mean())} for a, b, p in runs(shown)]
        write_json(output/'predictions'/f'{i}.json', {'clip_id': i, 'split': c['split'], 'fps': c['fps'],
                   'frames': c['frames'], 'segments': segments, 'review_segments': review_segments(segments),
                   'stroke_type': routing, 'human_reviewed': False})
        np.savez_compressed(output/'predictions'/f'{i}.npz', prediction=pred, accepted=accepted,
                            scores=probability, fps=c['fps'], input_sha256=sha(root/c['input']))
    support = {k: {'frames': int(((kind==k)&np.isin(y,[1,2,3])).sum()),
                   'clips': sorted(set(g[(kind==k)&np.isin(y,[1,2,3])]))} for k in STROKE_TYPES[:-1]}
    summary = {'status': 'complete', 'version': output.name, 'training_type': 'fully_supervised',
        'created_at': datetime.now(timezone.utc).isoformat(), 'method': 'fully human-reviewed shared phase model and supported stroke-core experts',
        'manifest_sha256': sha(root/'manifest.json'), 'human_annotation_sha256': sha(annotation),
        'code_sha256': sha(__file__), 'phase_schema': phase_schema(), 'stroke_schema': stroke_schema(),
        'original_training_clips': sum(c['split']=='train' for c in manifest['clips']),
        'active_training_clips': sorted(train), 'excluded_clips': sorted(excluded),
        'curation_sha256': sha(curation_path) if curation_path else None, 'clip_type_confirmations': overrides,
        'user_confirmed_forehand_clips': sorted(i for i in route if route[i]=='正手'),
        'user_confirmed_forehand_pending_clips': [], 'clip_routes': route,
        'training_clips_scanned': len(train), 'training_clips_used': len(set(g)),
        'human_complete_clips': len(confirmed), 'corrected_clips': sorted(confirmed),
        'heldout_clips_predicted_only': sum(c['split']=='validation' for c in clips.values()),
        'training_frames': len(y), 'source_frame_counts': {'human': len(y)}, 'reference_conflict_exclusions': {},
        'excluded_frames': sum(c['frames'] for c in train.values())-len(y), 'pose_quality_excluded_frames': pose_dropped,
        'phase_frame_counts': {p: int((y==k).sum()) for k,p in enumerate(PHASES)},
        'weights': {'human': 2., 'automatic': 0., 'class_balance': 'within training fold only'},
        'selection': {'nearest_anchor_distance_limit': distance_limit, 'ensemble_score_min': .7,
                      'boundary_guard_frames': 2, 'scores_calibrated': False},
        'human_policy': 'All active clips are human reviewed; no automatic or agent training targets.',
        'core_experts': final.expert_support, 'stroke_support': support, 'stroke_type_classifier_trained': False,
        'evaluation': {'anchor_clip_holdouts': fold_reports, 'temporal_folds': folds, 'guard_seconds': guard_seconds,
                       'oof_metrics': metrics(oof_truth,oof_prediction,list(range(5))) if oof_truth else None,
                       'heldout_semantic_accuracy': None,
                       'note': 'Chronological block checks from the same session are not independent held-out accuracy; known clip types are supplied externally.'},
        'model_sha256': sha(output/'model.joblib'), 'elapsed_seconds': time.monotonic()-started}
    write_json(output/'summary.json', summary)
    return summary
