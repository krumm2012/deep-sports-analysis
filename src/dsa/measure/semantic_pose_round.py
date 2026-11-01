"""Retrain from complete human corrections, then perform one conservative screening pass.

Unlike initial bootstrapping, human corrections are not vetoed by older pose models.
Stroke categories are audited explicitly; unsupported experts remain untrained.
"""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import joblib
import numpy as np

from dsa.measure.semantic_phases import PHASES, phase_schema
from dsa.measure.semantic_curation import load_curation
from dsa.measure.semantic_pilot import checked_input, metrics, sha, write_json
from dsa.measure.semantic_pose import PoseEnsemble, accepted_prediction, frame_features, runs, training_arrays
from dsa.measure.semantic_strokes import CORE_PHASES, STROKE_TYPES, stroke_label, stroke_schema
from dsa.pages.semantic_clip_labeling import annotation_registry, validate_export


def human_targets(clips, records):
    annotations = {i: {'target': np.full(c['frames'], -1), 'weight': np.zeros(c['frames']),
                       'origin': np.full(c['frames'], 'excluded_uncertain', dtype='<U32')}
                   for i, c in clips.items()}
    for r in records:
        if r['clip_id'] not in clips or r['split'] != 'train' or not r['reviewed']:
            raise ValueError('Human targets must come from confirmed training clips')
        if not r['phase_usable']:
            continue
        a = annotations[r['clip_id']]; start, end = r['start_frame'], r['end_frame_exclusive']
        if np.any(a['target'][start:end] >= 0):
            raise ValueError('Overlapping human corrections')
        a['target'][start:end] = PHASES.index(r['phase'])
        a['weight'][start:end] = 2.
        a['origin'][start:end] = 'human'
    return annotations


def review_segments(segments, maximum_gap=10):
    """Connect short uncertainty around ordered phases for editing, never as fit truth."""
    rows = [dict(s) for s in segments]
    for i in range(len(rows)-2, -1, -1):
        if rows[i]['phase'] == rows[i+1]['phase']:
            rows[i]['end_frame_exclusive'] = rows[i+1]['end_frame_exclusive']
            rows[i]['end_seconds'] = rows[i+1]['end_seconds']; rows.pop(i+1)
    i = 0
    while i+2 < len(rows):
        left, gap, right = rows[i:i+3]
        l = PHASES.index(left['phase']) if left['phase'] in PHASES else -1
        r = PHASES.index(right['phase']) if right['phase'] in PHASES else -1
        frames = gap['end_frame_exclusive']-gap['start_frame']
        if l >= 0 and r in [l, (l+1) % len(PHASES)] and gap['phase'] is None and frames <= maximum_gap:
            if l == r:
                left['end_frame_exclusive'] = right['end_frame_exclusive']; left['end_seconds'] = right['end_seconds']
                left['continuity_proposal'] = True; rows.pop(i+2); rows.pop(i+1)
            else:
                boundary = (gap['start_frame']+gap['end_frame_exclusive']+1)//2
                fps = gap['fps'] if 'fps' in gap else (gap['end_frame_exclusive']-gap['start_frame'])/(gap['end_seconds']-gap['start_seconds'])
                left['end_frame_exclusive'] = right['start_frame'] = boundary
                left['end_seconds'] = right['start_seconds'] = boundary/fps
                left['continuity_proposal'] = right['continuity_proposal'] = True
                rows.pop(i+1)
            i = max(0, i-1)
        else:
            i += 1
    return rows


def run(root, output, annotation):
    started = time.monotonic()
    if output.exists():
        raise ValueError('Use a fresh round directory; earlier results are preserved')
    payload = json.loads(annotation.read_text()); registry = annotation_registry(root, payload)
    validate_export(root, registry, payload)
    manifest = json.loads((root/'manifest.json').read_text())
    curation, curation_path = load_curation(root, manifest)
    excluded = set(curation['excluded_clips'])
    clip_types = curation.get('clip_overrides', {})
    clips = {c['id']: c for c in manifest['clips'] if c['split'] in ['train', 'validation'] and c['id'] not in excluded}
    train = {i: c for i, c in clips.items() if c['split'] == 'train'}
    human_records = [r for r in payload['records'] if r['clip_id'] in train]
    corrected = {r['clip_id'] for r in human_records}
    if len(corrected) < 3:
        raise ValueError('This correction round needs at least three confirmed videos')
    features = {i: frame_features(checked_input(root, c)) for i, c in clips.items()}
    initial = human_targets(train, human_records)
    for i in train:
        initial[i]['target'][~features[i][1] | features[i][2]] = -1
    x, y, w, groups, origin = training_arrays(features, initial)
    if set(y) != set(range(5)):
        raise ValueError('Human corrections must cover all five phases before refitting')
    folds, distances = [], []
    for clip in sorted(set(groups)):
        tr, te = groups != clip, groups == clip
        if set(y[tr]) != set(range(5)):
            folds.append({'test_clip': str(clip), 'status': 'missing_class_in_training_fold'}); continue
        model = PoseEnsemble().fit(x[tr], y[tr], w[tr])
        pred, score, prob, agree, dist = model.scores(x[te]); distances.extend(dist)
        folds.append({'test_clip': str(clip), 'train_clips': sorted(set(groups[tr])), 'frames': int(te.sum()),
                      'reference_source_counts': {'human': int(te.sum())},
                      'metrics': metrics(y[te], pred, list(range(5)))})
    if not distances:
        raise ValueError('Insufficient phase coverage for clip-held-out distance checking')
    distance_limit = float(np.quantile(distances, .95))
    anchor = PoseEnsemble().fit(x, y, w)
    annotations = {}
    for i in train:
        a = {k: v.copy() for k, v in initial[i].items()}
        # The entire reviewed video is authoritative: no filling invalid/mixed/unknown spans.
        if i not in corrected:
            pred, score, prob, accepted, dist = accepted_prediction(anchor, *features[i], distance_limit)
            a['target'][accepted] = pred[accepted]; a['weight'][accepted] = .2
            a['origin'][accepted] = 'pose_auto_consensus'
        annotations[i] = a
    xx, yy, ww, gg, oo = training_arrays(features, annotations)
    auto = oo == 'pose_auto_consensus'
    cap = min(1., .5*w.sum()/max(ww[auto].sum(), 1e-9)); ww[auto] *= cap
    final = PoseEnsemble().fit(xx, yy, ww)
    output.mkdir(parents=True); (output/'predictions').mkdir()
    (output/'user-annotation.json').write_bytes(annotation.read_bytes())
    (output/'annotation-registry.json').write_bytes(registry.read_bytes())
    (output/'training-code.py').write_bytes(Path(__file__).read_bytes())
    if curation_path:
        (output/'curation.json').write_bytes(curation_path.read_bytes())
    joblib.dump(anchor, output/'anchor-model.joblib'); joblib.dump(final, output/'model.joblib')
    np.savez_compressed(output/'training.npz', x=xx, y=yy, weight=ww, groups=gg, origins=oo)
    records = []
    for i, c in train.items():
        a = annotations[i]
        for start, end, value in runs(a['target']):
            for lo, hi, _ in runs(np.unique(a['origin'][start:end], return_inverse=True)[1]):
                lo, hi = start+lo, start+hi
                records.append({'clip_id': i, 'split': 'train', 'start_frame': lo, 'end_frame_exclusive': hi,
                                'start_seconds': lo/c['fps'], 'end_seconds': hi/c['fps'],
                                'phase': PHASES[value] if value >= 0 else None, 'phase_usable': value >= 0,
                                'label_source': str(a['origin'][lo]), 'human_reviewed': bool(a['origin'][lo] == 'human'),
                                'source_sha256': c['sha256'], 'input_sha256': sha(root/c['input']),
                                'stroke_type': clip_types.get(i, {}).get('stroke_type', '无法判断'),
                                'stroke_type_source': 'user_clip_confirmation' if i in clip_types else None})
    write_json(output/'labels.json', {'records': records, 'schema_version': 1,
               'annotation_type': 'human-corrected-pose-round', 'phase_schema': phase_schema()})
    for i, c in clips.items():
        pred, score, prob, accepted, dist = accepted_prediction(final, *features[i], distance_limit)
        displayed = np.where(accepted, pred, -1)
        segments = [{'start_frame': a, 'end_frame_exclusive': b, 'start_seconds': a/c['fps'],
                     'end_seconds': b/c['fps'], 'fps': c['fps'], 'phase': PHASES[p] if p >= 0 else None,
                     'mean_score': float(score[a:b].mean())} for a, b, p in runs(displayed)]
        write_json(output/'predictions'/f'{i}.json', {'clip_id': i, 'split': c['split'], 'fps': c['fps'],
                   'frames': c['frames'], 'segments': segments, 'review_segments': review_segments(segments),
                   'human_reviewed': False,
                   'stroke_type': clip_types.get(i, {}).get('stroke_type', '无法判断'),
                   'stroke_type_source': 'user_clip_confirmation' if i in clip_types else None})
        np.savez_compressed(output/'predictions'/f'{i}.npz', prediction=pred, accepted=accepted,
                            scores=prob, fps=c['fps'], input_sha256=sha(root/c['input']))
    support = {stroke: {'frames': 0, 'clips': set()} for stroke in STROKE_TYPES[:-1]}
    for r in human_records:
        stroke = stroke_label(r)
        if r['phase'] in CORE_PHASES and r['phase_usable'] and stroke in support:
            support[stroke]['frames'] += r['end_frame_exclusive']-r['start_frame']
            support[stroke]['clips'].add(r['clip_id'])
    stroke_support = {k: {'frames': v['frames'], 'clips': sorted(v['clips'])} for k, v in support.items()}
    write_json(output/'stroke-support.json', {'schema': stroke_schema(), 'support': stroke_support,
               'type_classifier_trained': False, 'reason': 'Fine-grained per-cycle stroke corrections are required; legacy generic labels cannot establish backhand grip.'})
    summary = {'status': 'complete', 'version': output.name, 'created_at': datetime.now(timezone.utc).isoformat(),
        'method': 'four-clip human corrections, human-priority refit and one screened pseudo-label pass',
        'manifest_sha256': sha(root/'manifest.json'), 'human_annotation_sha256': sha(annotation),
        'code_sha256': sha(__file__), 'phase_schema': phase_schema(), 'stroke_schema': stroke_schema(),
        'training_clips_scanned': len(train), 'training_clips_used': len(set(gg)),
        'original_training_clips': sum(c['split']=='train' for c in manifest['clips']),
        'active_training_clips': sorted(train), 'excluded_clips': sorted(excluded),
        'curation_sha256': sha(curation_path) if curation_path else None,
        'clip_type_confirmations': clip_types,
        'user_confirmed_forehand_pending_clips': sorted(i for i in clip_types if i not in corrected and clip_types[i]['stroke_type']=='正手'),
        'heldout_clips_predicted_only': sum(c['split']=='validation' for c in clips.values()),
        'human_complete_clips': len(corrected), 'corrected_clips': sorted(corrected),
        'training_frames': len(yy), 'source_frame_counts': dict(Counter(oo)), 'reference_conflict_exclusions': {},
        'phase_frame_counts': {p: int((yy==k).sum()) for k, p in enumerate(PHASES)},
        'excluded_frames': sum(c['frames'] for c in train.values())-len(yy),
        'selection': {'nearest_anchor_distance_limit': distance_limit, 'ensemble_score_min': .7,
                      'boundary_guard_frames': 2, 'scores_calibrated': False},
        'weights': {'human': 2., 'automatic': .2*cap, 'automatic_mass_limit_fraction_of_human': .5},
        'human_policy': 'Never veto corrections because the old model disagrees; reviewed unknown/invalid/mixed spans stay excluded.',
        'evaluation': {'anchor_clip_holdouts': folds, 'heldout_semantic_accuracy': None},
        'stroke_type_classifier_trained': False, 'stroke_support': stroke_support,
        'model_sha256': sha(output/'model.joblib'), 'elapsed_seconds': time.monotonic()-started}
    write_json(output/'summary.json', summary)
    return summary
