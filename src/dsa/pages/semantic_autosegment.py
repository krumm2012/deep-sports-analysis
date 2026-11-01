"""Create editable proposals from pose changes; never mark them human reviewed."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from dsa.measure.semantic_pilot import checked_input, features, sha, write_json
from dsa.measure.semantic_phases import PHASES, phase_schema
from dsa.pages.semantic_labeling import build


def motion_descriptors(data):
    """Smooth hip-relative limb geometry; no MoSeq state-to-action mapping."""
    xy = np.asarray(data['coordinates'], dtype=float) / 100.
    pose = xy - xy[:, [7, 8]].mean(axis=1, keepdims=True)
    # Arm and leg configurations, with enough smoothing to suppress one-frame noise.
    pose = pose[:, [1, 2, 3, 4, 5, 6, 9, 10, 11, 12]].reshape(len(xy), -1)
    smooth = np.stack([np.convolve(np.pad(pose[:, i], (2, 2), mode='edge'),
                                  np.ones(5)/5, mode='valid') for i in range(pose.shape[1])], axis=1)
    scale = np.maximum(np.std(smooth, axis=0), .08)
    return (smooth - np.median(smooth, axis=0)) / scale


def partition_motion(descriptors, start, end, parts, minimum_frames=4, *, phase_order=None, swing_frame=None):
    """Minimum within-segment pose variation for a fixed, modest segment budget."""
    x = np.asarray(descriptors[start:end], dtype=float)
    n = len(x)
    if n < minimum_frames or not np.isfinite(x).all():
        raise ValueError('Invalid motion descriptor interval')
    parts = min(max(1, int(parts)), n // minimum_frames)
    sums = np.vstack([np.zeros(x.shape[1]), np.cumsum(x, axis=0)])
    squared = np.r_[0., np.cumsum(np.sum(x*x, axis=1))]
    costs = np.full((parts+1, n+1), np.inf)
    previous = np.full((parts+1, n+1), -1, dtype=int)
    costs[0, 0] = 0
    for k in range(1, parts+1):
        for b in range(k*minimum_frames, n+1):
            positions = np.arange((k-1)*minimum_frames, b-minimum_frames+1)
            lengths = b-positions
            totals = sums[b]-sums[positions]
            variance = squared[b]-squared[positions] - np.sum(totals*totals, axis=1)/lengths
            # A mild duration penalty discourages a series of minimum-size slivers.
            duration_penalty = .02 * ((lengths-n/parts)/(n/parts))**2 * n
            values = costs[k-1, positions] + np.maximum(variance, 0) / x.shape[1] + duration_penalty
            if phase_order and phase_order[k-1] == '挥拍' and swing_frame is not None:
                values[(positions > swing_frame-start) | (b <= swing_frame-start)] = np.inf
            j = int(np.argmin(values))
            costs[k, b] = values[j]
            previous[k, b] = positions[j]
    if not np.isfinite(costs[parts, n]):
        raise ValueError('No feasible phase boundaries for the motion anchor')
    boundaries = [n]
    for k in range(parts, 0, -1):
        boundaries.append(int(previous[k, boundaries[-1]]))
    return [int(start+b) for b in reversed(boundaries)]


def portable_scores(parameters, vector):
    x = (np.asarray(vector)-parameters['mean']) / parameters['scale']
    logits = np.asarray(parameters['coef']) @ x + parameters['intercept']
    if len(logits) == 1:
        positive = 1/(1+np.exp(-np.clip(logits[0], -700, 700)))
        scores = np.array([1-positive, positive])
    else:
        scores = np.exp(logits-np.max(logits))
        scores /= scores.sum()
    return dict(zip(parameters['classes'], map(float, scores)))


def proposed_label(parameters, vector, minimum_score=.6, minimum_margin=.15):
    scores = portable_scores(parameters, vector)
    ranking = sorted(scores, key=scores.get, reverse=True)
    first, second = ranking[:2]
    # This is an editing aid, not a calibrated rejection mechanism.
    label = first if scores[first] >= minimum_score and scores[first]-scores[second] >= minimum_margin else '无法判断'
    return label, scores


def prior_phase_order(recommendation):
    text = recommendation.get('phase_description', '')
    if '→' not in text:
        return []
    aliases = {'回位': '回位/移动', '准备': '准备', '引拍': '引拍', '挥拍': '挥拍', '随挥': '随挥'}
    return [aliases.get(part.strip(), '无法判断') for part in text.split('→')]


def cycle_window(data, task):
    """One candidate action cycle around a wrist-motion peak, clipped to the video.

    This is a five-stage editing scaffold, not a learned phase classifier or
    a contact detector. Partial cycles omit stages that have no frame support.
    """
    xy = np.asarray(data['coordinates'], dtype=float) / 100.
    pose = xy - xy[:, [7, 8]].mean(axis=1, keepdims=True)
    speed = np.linalg.norm(np.gradient(pose[:, [5, 6]], axis=0) * data['fps'], axis=-1).max(axis=1)
    speed = np.convolve(np.pad(speed, (2, 2), mode='edge'), np.ones(5)/5, mode='valid')
    a = max(task['allowed_start_frame'], task['start_frame']-10)
    b = min(task['allowed_end_frame'], task['end_frame_exclusive']+(40 if task.get('target') == '准备' else 20))
    peak = int(a+np.argmax(speed[a:b]))
    if speed[peak] < .25:
        return a, b, ['无法判断'], None
    before = np.arange(max(task['allowed_start_frame'], peak-35), max(task['allowed_start_frame'], peak-10))
    after = np.arange(min(task['allowed_end_frame']-1, peak+12), min(task['allowed_end_frame'], peak+40))
    start = int(before[np.argmin(speed[before]+.002*(before-(peak-24))**2)]) if len(before) else task['allowed_start_frame']
    end = int(after[np.argmin(speed[after]+.002*(after-(peak+24))**2)]+1) if len(after) else task['allowed_end_frame']
    left = min(2, (peak-start)//4)
    right = min(2, (end-peak-1)//4)
    order = list(PHASES[:2][-left:]) if left else []
    order += ['挥拍'] + list(PHASES[3:][:right])
    # Reserve enough observations for the swing segment as well as its neighbors.
    while (len(order)*4 > end-start) and len(order) > 1:
        if order[0] != '挥拍':
            order.pop(0)
        else:
            order.pop()
    return start, end, order, peak


def generate(root, output):
    queue = json.loads((output / 'queue.json').read_text())
    if queue['mode'] != 'training':
        raise ValueError('Semantic proposals are limited to the training annotation page')
    clips = {c['id']: c for c in json.loads((root / 'manifest.json').read_text())['clips']}
    parameters = json.loads((root / 'semantic-pilot/models.json').read_text())
    recommendations = json.loads((root / 'semantic-labeling/recommendations.json').read_text())
    inputs, descriptors, tasks = {}, {}, {}
    for task in queue['tasks']:
        clip_id = task['clip_id']
        if clips[clip_id]['split'] != 'train':
            raise ValueError('Validation recording cannot enter training proposal selection')
        if clip_id not in inputs:
            inputs[clip_id] = checked_input(root, clips[clip_id])
            descriptors[clip_id] = motion_descriptors(inputs[clip_id])
        data = inputs[clip_id]
        start, end = task['start_frame'], task['end_frame_exclusive']
        order = prior_phase_order(recommendations.get(task['id'], {})) if task['kind'] == 'mixed' else []
        peak = None
        if task['kind'] == 'new':
            start, end, order, peak = cycle_window(data, task)
        count = len(order) if order else min(5, max(2, round((end-start)/12)))
        boundaries = partition_motion(descriptors[clip_id], start, end, count,
                                      phase_order=order or None, swing_frame=peak)
        segments = []
        for i, (a, b) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            vector = features(data, a, b)
            phase, phase_scores = proposed_label(parameters['phase'], vector)
            shot, shot_scores = proposed_label(parameters['shot_type'], vector, .65, .2)
            if task['initial_shot_type']:
                shot = task['initial_shot_type']
            if order:
                phase = order[i]
            else:
                phase = '无法判断'
            segments.append({'start_frame': a, 'end_frame_exclusive': b,
                             'shot_type': shot, 'phase': phase, 'notes': '', 'auto_suggested': True,
                             'suggestion_source': ('five_phase_motion_scaffold' if task['kind'] == 'new'
                                                   else 'prior_visual_review' if order else 'pilot_classifier'),
                             'suggested_start_frame': a, 'suggested_end_frame_exclusive': b,
                             'suggested_shot_type': shot, 'suggested_phase': phase,
                             'model_scores': {'shot_type': shot_scores, 'phase': phase_scores}})
        tasks[task['id']] = {'segments': segments, 'reviewed': False,
                             'source': ('五阶段顺序＋身体运动边界候选' if task['kind'] == 'new'
                                        else '已有画面初审建议＋姿态变化边界' if order else '姿态变化边界＋小样本模型建议'),
                             'motion_peak_frame': peak,
                             'input_sha256': sha(root / clips[clip_id]['input'])}
    result = {'schema_version': 1, 'queue_id': queue['queue_id'],
              'candidate_registry_sha256': queue['candidate_registry_sha256'],
              'model_sha256': sha(root / 'semantic-pilot/models.json'), 'code_sha256': sha(__file__),
              'status': 'unreviewed_proposals', 'method': 'ordered-five-phase-pose-partition-v2',
              'phase_schema': phase_schema(),
              'minimum_segment_frames': 4, 'tasks': tasks,
              'limitations': ['Pose changes approximate semantic boundaries; all proposals need human confirmation.',
                              'Five-stage names use an ordered editing scaffold or prior visual review, not a trained five-class model.',
                              'Low model scores remain unknown; scores and thresholds are not calibrated.']}
    result['revision'] = hashlib.sha256(json.dumps(result, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
    write_json(output / 'auto-segments.json', result)
    build(root, output)
    return {'tasks': len(tasks), 'segments': sum(len(t['segments']) for t in tasks.values()),
            'all_pending': True, 'segment_counts': dict(Counter(len(t['segments']) for t in tasks.values())),
            'revision': result['revision']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = generate(args.root, args.output)
    write_json(args.output / 'auto-segments-summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
