"""One conservative pose-labeling pass followed by weighted five-stage refitting.

Human labels, agent visual anchors and automatic labels have separate provenance.
The original held-out recordings never participate in fitting or threshold choice.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from dsa.measure.semantic_phases import PHASES, phase_schema
from dsa.measure.semantic_pilot import checked_input, metrics, sha, write_json
from dsa.measure.semantic_weak import balanced_source_weights
from dsa.pages.semantic_clip_labeling import validate_export


def smooth(x, radius=2):
    return np.stack([np.convolve(np.pad(c, (radius, radius), mode='edge'),
                     np.ones(2*radius+1)/(2*radius+1), mode='valid') for c in x.T], axis=1)


def frame_features(data):
    """Identical 9-frame contexts for training and inference, clipped to each video."""
    xy = np.asarray(data['coordinates'], dtype=float)
    if not np.isfinite(xy).all():
        raise ValueError('Pose must be finite')
    hip = xy[:, [7, 8]].mean(1)
    shoulder = xy[:, [1, 2]].mean(1)
    scale = max(float(np.median(np.linalg.norm(shoulder-hip, axis=1))), 1.)
    p = smooth(((xy-hip[:, None])/scale).reshape(len(xy), -1))
    ix = np.arange(len(p))
    velocity = (p[np.minimum(ix+2, len(p)-1)]-p[np.maximum(ix-2, 0)])*data['fps']/4
    spread = np.sqrt(np.maximum(smooth(p*p)-smooth(p)**2, 0))
    hip_velocity = np.gradient(smooth(hip/scale), axis=0)*data['fps']
    x = np.c_[p[np.maximum(0, ix-4)], p, p[np.minimum(len(p)-1, ix+4)],
              velocity, spread, hip_velocity]
    confidence = np.asarray(data['confidences'])[:, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
    observed = np.asarray(data.get('raw_observed', np.ones_like(data['confidences'], dtype=bool)))
    quality = (confidence.min(1) >= .55) & (observed.mean(1) >= .8)
    # Long nearly static poses are not enough evidence to assign an action stage.
    identical = np.r_[False, np.max(np.abs(np.diff(xy, axis=0)), axis=(1, 2))/scale < .01]
    frozen = np.zeros(len(xy), dtype=bool)
    for a, b, value in runs(identical.astype(int)):
        if value and b-a >= 25:
            frozen[max(0, a-1):b] = True
    return x.astype(np.float32), quality, frozen


def runs(labels):
    labels = np.asarray(labels)
    bounds = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1])+1, len(labels)]
    return [(int(a), int(b), int(labels[a])) for a, b in zip(bounds[:-1], bounds[1:])]


def suppress_boundaries(labels, mask, guard=2, minimum=3):
    result = mask.copy()
    for a, b, _ in runs(labels):
        if a:
            result[a:min(b, a+guard)] = False
        if b < len(labels):
            result[max(a, b-guard):b] = False
    for a, b, value in runs(result.astype(int)):
        if value and b-a < minimum:
            result[a:b] = False
    return result


class PoseEnsemble:
    def fit(self, x, y, weight):
        self.scaler = StandardScaler().fit(x, sample_weight=weight)
        z = self.scaler.transform(x)
        w = balanced_source_weights(y, weight)
        self.linear = LogisticRegression(C=.2, max_iter=3000, random_state=31).fit(z, y, sample_weight=w)
        self.trees = ExtraTreesClassifier(n_estimators=240, min_samples_leaf=3, max_features=.7,
                                          n_jobs=6, random_state=31).fit(z, y, sample_weight=w)
        self.classes_ = self.linear.classes_
        if not np.array_equal(self.classes_, self.trees.classes_):
            raise ValueError('Ensemble classes differ')
        self.neighbors = NearestNeighbors(n_neighbors=1).fit(z)
        self.feature_count = z.shape[1]
        return self

    def scores(self, x):
        z = self.scaler.transform(x)
        linear, tree = self.linear.predict_proba(z), self.trees.predict_proba(z)
        average = (linear+tree)/2
        order = np.argsort(average, axis=1)
        pred = self.classes_[order[:, -1]]
        distance = self.neighbors.kneighbors(z)[0][:, 0]/np.sqrt(self.feature_count)
        return pred, average.max(1), average, (linear.argmax(1) == tree.argmax(1)), distance


def accepted_prediction(model, x, quality, frozen, distance_limit):
    pred, score, probability, agreement, distance = model.scores(x)
    sorted_p = np.sort(probability, axis=1)
    accepted = quality & ~frozen & agreement & (score >= .70) & ((sorted_p[:, -1]-sorted_p[:, -2]) >= .25) & (distance <= distance_limit)
    accepted = suppress_boundaries(pred, accepted)
    return pred, score, probability, accepted, distance


def training_arrays(features, annotations):
    vectors, labels, weights, groups, origins = [], [], [], [], []
    for clip, a in annotations.items():
        ok = a['target'] >= 0
        if not ok.any():
            continue
        vectors.append(features[clip][0][ok]); labels.extend(a['target'][ok])
        weights.extend(a['weight'][ok]); groups.extend([clip]*ok.sum()); origins.extend(a['origin'][ok])
    return np.vstack(vectors), np.array(labels), np.array(weights), np.array(groups), np.array(origins)


def apply_anchors(annotations, anchors, clips, human):
    for record in anchors:
        clip, start, end = record['clip_id'], record['start_frame'], record['end_frame_exclusive']
        if clip not in clips or clips[clip]['split'] != 'train' or not 0 <= start < end <= clips[clip]['frames']:
            raise ValueError('Anchor must lie in a training clip')
        a = annotations[clip]
        if record['phase'] not in PHASES:
            raise ValueError('Unknown anchor phase')
        if np.any(a['target'][start:end] >= 0):
            raise ValueError('Overlapping anchors')
        a['target'][start:end] = PHASES.index(record['phase'])
        a['weight'][start:end] = 2. if human else 1.
        a['origin'][start:end] = 'human' if human else 'agent_visual_pose_anchor'


def run(root, output, annotation, anchors_path):
    started = time.monotonic()
    if (output/'summary.json').exists():
        raise ValueError('Use a fresh version; existing models are preserved')
    human = json.loads(annotation.read_text())
    validate_export(root, root/'semantic-clip-labeling/candidates.json', human)
    anchor_file = json.loads(anchors_path.read_text())
    manifest = json.loads((root/'manifest.json').read_text())
    clips = {c['id']: c for c in manifest['clips'] if c['split'] in ['train', 'validation']}
    train = {i: c for i, c in clips.items() if c['split'] == 'train'}
    data = {i: checked_input(root, c) for i, c in clips.items()}
    f = {i: frame_features(d) for i, d in data.items()}
    initial = {i: {'target': np.full(c['frames'], -1), 'weight': np.zeros(c['frames']),
                    'origin': np.full(c['frames'], 'excluded_uncertain', dtype='<U32')}
               for i, c in train.items()}
    apply_anchors(initial, [r for r in human['records'] if r['phase_usable']], train, True)
    apply_anchors(initial, anchor_file['records'], train, False)
    blocked = {i: np.zeros(c['frames'], dtype=bool) for i, c in train.items()}
    for r in [*human['records'], *anchor_file.get('exclusions', [])]:
        if r.get('annotation_type') in ['invalid', 'uncertain']:
            blocked[r['clip_id']][r['start_frame']:r['end_frame_exclusive']] = True
    for i in train:
        initial[i]['target'][blocked[i] | ~f[i][1] | f[i][2]] = -1
    x, y, w, groups, origins = training_arrays(f, initial)
    if set(y) != set(range(5)) or len(set(groups)) < 3:
        raise ValueError('Need five stages across multiple anchor clips')
    # Anchor-only group hold-outs; auto labels are not used as test truth.
    folds, distances, conflicts = [], [], []
    for group in sorted(set(groups)):
        tr, te = groups != group, groups == group
        if len(set(y[tr])) < 5:
            continue
        model = PoseEnsemble().fit(x[tr], y[tr], w[tr])
        pred, score, _, agree, dist = model.scores(x[te])
        distances.extend(dist)
        frames = np.flatnonzero(initial[str(group)]['target'] >= 0)
        for position in np.flatnonzero(agree & (score >= .80) & (pred != y[te])):
            conflicts.append({'clip_id': str(group), 'frame': int(frames[position]),
                              'original_phase': PHASES[int(y[te][position])],
                              'reference_source': str(origins[te][position]),
                              'cross_clip_prediction': PHASES[int(pred[position])],
                              'score': float(score[position]), 'action': 'exclude; do not relabel'})
        folds.append({'test_clip': str(group), 'train_clips': sorted(set(groups[tr])),
                      'frames': int(te.sum()), 'reference_source_counts': dict(Counter(origins[te])),
                      'metrics': metrics(y[te], pred, list(range(5)))})
    distance_limit = float(np.quantile(distances, .95))
    for conflict in conflicts:
        i, frame = conflict['clip_id'], conflict['frame']
        initial[i]['target'][frame] = -1
        blocked[i][frame] = True
    write_json(output/'reference-conflicts.json', {'records': conflicts,
               'method': 'leave-entire-reference-clip-out ensemble agreement, score >= .80; exclude conflicts without rewriting original annotations'})
    x, y, w, groups, origins = training_arrays(f, initial)
    if set(y) != set(range(5)):
        raise ValueError('Reference conflict screening removed a required stage')
    model = PoseEnsemble().fit(x, y, w)
    joblib.dump(model, output/'anchor-model.joblib')
    np.savez_compressed(output/'anchor-training.npz', x=x, y=y, weight=w, groups=groups, origins=origins)
    annotations, records = {}, []
    for i, c in train.items():
        pred, score, probability, accepted, distance = accepted_prediction(model, *f[i], distance_limit)
        accepted &= ~blocked[i]
        # A confirmed mixed interval constrains automatic labels to the named phases.
        for r in human['records']:
            if r['clip_id'] == i and r['annotation_type'] == 'mixed':
                allowed = [k for k, phase in enumerate(PHASES) if phase in r.get('notes', '')]
                if allowed:
                    a, b = r['start_frame'], r['end_frame_exclusive']
                    accepted[a:b] &= np.isin(pred[a:b], allowed)
        a = {k: v.copy() for k, v in initial[i].items()}
        automatic = accepted & (a['target'] < 0)
        a['target'][automatic] = pred[automatic]
        # Cap the aggregate pseudo-label mass to the anchor mass, after collection below.
        a['weight'][automatic] = .20
        a['origin'][automatic] = 'pose_auto_consensus'
        annotations[i] = a
        for start, end, phase in runs(a['target']):
            origin = a['origin'][start:end]
            for suba, subb, origin_idx in runs(np.unique(origin, return_inverse=True)[1]):
                lo, hi = start+suba, start+subb
                records.append({'clip_id': i, 'split': 'train', 'start_frame': lo,
                                'end_frame_exclusive': hi, 'start_seconds': lo/c['fps'], 'end_seconds': hi/c['fps'],
                                'phase': PHASES[phase] if phase >= 0 else None, 'phase_usable': phase >= 0,
                                'label_source': str(a['origin'][lo]), 'human_reviewed': str(a['origin'][lo]) == 'human',
                                'source_sha256': c['sha256'], 'input_sha256': sha(root/c['input']),
                                'mean_selection_score': float(score[lo:hi].mean()),
                                'reason': None if phase >= 0 else 'human/agent exclusion, pose quality, frozen tracking, disagreement, low score, distance or boundary guard'})
    xx, yy, ww, gg, oo = training_arrays(f, annotations)
    auto = oo == 'pose_auto_consensus'
    cap = min(1., w.sum()/max(ww[auto].sum(), 1e-9))
    ww[auto] *= cap
    final = PoseEnsemble().fit(xx, yy, ww)
    joblib.dump(final, output/'model.joblib')
    np.savez_compressed(output/'training.npz', x=xx, y=yy, weight=ww, groups=gg, origins=oo)
    write_json(output/'labels.json', {'schema_version': 1, 'annotation_type': 'agent_authorized_pose_round',
               'phase_schema': phase_schema(), 'records': records, 'not_human_ground_truth': True})
    (output/'predictions').mkdir(exist_ok=True)
    for i, c in clips.items():
        pred, score, probability, accepted, distance = accepted_prediction(final, *f[i], distance_limit)
        displayed = np.where(accepted, pred, -1)
        segments = [{'start_frame': a, 'end_frame_exclusive': b, 'start_seconds': a/c['fps'],
                     'end_seconds': b/c['fps'], 'phase': PHASES[p] if p >= 0 else None,
                     'mean_score': float(score[a:b].mean())} for a, b, p in runs(displayed)]
        write_json(output/'predictions'/f'{i}.json', {'clip_id': i, 'split': c['split'], 'fps': c['fps'],
                   'frames': c['frames'], 'segments': segments, 'human_reviewed': False})
        np.savez_compressed(output/'predictions'/f'{i}.npz', prediction=pred, accepted=accepted,
                            scores=probability, fps=c['fps'], input_sha256=sha(root/c['input']))
    summary = {'status': 'complete', 'created_at': datetime.now(timezone.utc).isoformat(),
        'method': 'hip-relative temporal pose ensemble; one conservative pseudo-label pass and weighted refit',
        'manifest_sha256': sha(root/'manifest.json'), 'human_annotation_sha256': sha(annotation),
        'anchors_sha256': sha(anchors_path), 'code_sha256': sha(__file__), 'phase_schema': phase_schema(),
        'training_clips_scanned': len(train), 'training_clips_used': len(set(gg)),
        'heldout_clips_predicted_only': sum(c['split']=='validation' for c in clips.values()),
        'human_complete_clips': len(human['completed_tasks']), 'anchor_clips': sorted(set(groups)),
        'training_frames': len(yy), 'source_frame_counts': dict(Counter(oo)),
        'reference_conflict_exclusions': dict(Counter(c['reference_source'] for c in conflicts)),
        'phase_frame_counts': {p: int((yy == k).sum()) for k, p in enumerate(PHASES)},
        'excluded_frames': sum(c['frames'] for c in train.values())-len(yy),
        'selection': {'ensemble_score_min': .70, 'margin_min': .25, 'both_models_agree': True,
                      'nearest_anchor_distance_limit': distance_limit, 'boundary_guard_frames': 2,
                      'minimum_auto_interval_frames': 3, 'scores_calibrated': False},
        'weights': {'human': 2., 'agent_visual_pose_anchor': 1., 'automatic': .20*cap,
                    'automatic_total_mass_capped_at_anchor_mass': True},
        'evaluation': {'anchor_clip_holdouts': folds, 'heldout_semantic_accuracy': None,
                       'note': 'Agent references are not human accuracy; human reference covers only two classes.'},
        'model_sha256': sha(output/'model.joblib'), 'elapsed_seconds': time.monotonic()-started,
        'limitations': ['Only one complete human-labeled video; mixed intervals do not supply precise five-stage boundaries.',
                       'Pose-only stages omit racket/ball contact and can confuse stance with recovery.',
                       'Automatic labels and internal anchor checks are not independent human ground truth.']}
    write_json(output/'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path); parser.add_argument('output', type=Path)
    parser.add_argument('annotation', type=Path); parser.add_argument('anchors', type=Path)
    args = parser.parse_args()
    print(json.dumps(run(args.root, args.output, args.annotation, args.anchors), ensure_ascii=False, indent=2))
