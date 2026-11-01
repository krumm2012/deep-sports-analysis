"""Human interval annotation of complete clips; MoSeq IDs are reference-only."""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path

import numpy as np

from dsa.measure.semantic_phases import PHASES, phase_schema
from dsa.measure.semantic_pilot import sha, validate_labels, write_json
from dsa.pages.semantic_state_labeling import state_runs


SPECIAL = {'无法判断': 'uncertain', '混合/需调整区间': 'mixed', '无效动作': 'invalid'}


def annotation_registry(root, payload):
    """Resolve a registered export within this project, including later review rounds."""
    root = root.resolve()
    relative = Path(payload.get('candidate_registry') or 'semantic-clip-labeling/candidates.json')
    registry = (root / relative).resolve()
    if relative.is_absolute() or not registry.is_relative_to(root) or registry.suffix != '.json':
        raise ValueError('Annotation registry must be a JSON file inside this project')
    return registry


def seed_clip(clip, records, manual_boundaries, manual_hints=None):
    """Carry human interval hints and edited boundaries without confirming the clip."""
    selected = [r for r in records if r['clip_id'] == clip['id']]
    edits = (manual_hints or {}).get(clip['id'], [])
    bounds = {0, clip['frames'], *manual_boundaries.get(clip['id'], [])}
    for r in selected:
        bounds.update([r['start_frame'], r['end_frame_exclusive']])
    bounds = sorted(b for b in bounds if 0 <= b <= clip['frames'])
    segments = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        hints = [r for r in selected if r['start_frame'] <= a and b <= r['end_frame_exclusive']]
        hint = min(hints, key=lambda r: r['end_frame_exclusive']-r['start_frame']) if hints else None
        user_edits = [r for r in edits if r['start_frame'] <= a and b <= r['end_frame_exclusive']]
        edit = user_edits[-1] if user_edits else None
        segments.append({'start_frame': a, 'end_frame_exclusive': b,
                         'phase': edit['phase'] if edit and edit['phase'] else hint['phase'] if hint else '',
                         'shot_type': hint['shot_type'] if hint and hint['shot_type_usable'] else '无法判断',
                         'notes': edit['notes'] if edit and edit['notes'] else hint.get('notes', '') if hint else '',
                         'seed_origin': 'prior_manual_draft' if edit else 'prior_human_interval' if hint else 'manual_boundary' if a in manual_boundaries.get(clip['id'], []) or b in manual_boundaries.get(clip['id'], []) else 'unlabeled',
                         'prior_record_id': hint['id'] if hint else None})
    return {'done': False, 'segments': segments, 'seeded_from_prior': bool(selected or manual_boundaries.get(clip['id']))}


def build(root, output, prior_path):
    prior = json.loads(prior_path.read_text())
    manifest, clip_lookup = validate_labels(root, prior, root / 'semantic-labeling-round2/candidates.json')
    original_queue = json.loads((root / 'semantic-labeling-round2/queue.json').read_text())
    original_tasks = {t['id']: t for t in original_queue['tasks']}
    manual, manual_hints = {}, {}
    for task_id, draft in prior.get('annotation_progress', {}).get('tasks', {}).items():
        if draft.get('manual_edit') or draft.get('done'):
            clip_id = original_tasks[task_id]['clip_id']
            manual.setdefault(clip_id, set()).update(b for s in draft['segments'] for b in [s['start_frame'], s['end_frame_exclusive']])
            for s in draft['segments']:
                phase_edited = s.get('phase') and (not s.get('auto_suggested') or s.get('phase') != s.get('suggested_phase'))
                if phase_edited or s.get('notes'):
                    manual_hints.setdefault(clip_id, []).append({'start_frame': s['start_frame'],
                        'end_frame_exclusive': s['end_frame_exclusive'],
                        'phase': s['phase'] if phase_edited else None, 'notes': s.get('notes', '')})
    clips = sorted((c for c in manifest['clips'] if c['split'] == 'train'), key=lambda c: (c['start_seconds'], c['id']))
    if not clips:
        raise ValueError('No training clips')
    tasks, references, initial = [], {}, {'active': 0, 'tasks': {}}
    for c in clips:
        video = root / c['video']
        if not video.is_file():
            raise ValueError(f'Missing annotation video: {video}')
        task = {'id': 'clip-'+c['id'], 'clip_id': c['id'], 'split': 'train',
                'source_sha256': c['sha256'], 'normalized_video_sha256': sha(video),
                'fps': c['fps'], 'clip_frames': c['frames'], 'video': '../'+c['video'],
                'start_frame': 0, 'end_frame_exclusive': c['frames'],
                'allowed_start_frame': 0, 'allowed_end_frame': c['frames'], 'kind': 'full_clip',
                'title': '完整视频逐段标识', 'initial_shot_type': '无法判断',
                'instruction': '看完整视频，按实际画面拆分并选择阶段。允许修改任何边界；无效、混合或无法判断的区间单独保留。'}
        tasks.append(task)
        initial['tasks'][task['id']] = seed_clip(c, prior['records'], manual, manual_hints)
        with np.load(root / 'models/seed0/results' / f'{c["id"]}.npz', allow_pickle=False) as z:
            if len(z['syllable']) != c['frames'] or float(z['fps']) != manifest['fps']:
                raise ValueError('Reference state alignment differs')
            references[c['id']] = state_runs(z['syllable'])
    output.mkdir(parents=True, exist_ok=True)
    registry = {'schema_version': 3, 'scope': 'complete training clips only',
                'manifest_sha256': sha(root / 'manifest.json'), 'fps': manifest['fps'], 'items': tasks}
    write_json(output / 'candidates.json', registry)
    schema = phase_schema()
    schema['review_states'].append('无效动作')
    queue_id = hashlib.sha256(('human-full-clips-v1:'+registry['manifest_sha256']).encode()).hexdigest()
    q = {'schema_version': 3, 'export_schema_version': 3, 'full_clip': True,
         'title': f'训练视频逐段标识 · {len(tasks)}段', 'mode': 'training',
         'queue_id': queue_id, 'storage_prefix': 'camera05-human-full-clips-v1-',
         'manifest_sha256': registry['manifest_sha256'], 'fps': manifest['fps'],
         'baseline_checkpoint_sha256': sha(root / 'models/seed0/model/checkpoint.h5'),
         'baseline_labels_sha256': sha(prior_path), 'phase_schema': schema,
         'shot_type_labels': original_queue['shot_type_labels'], 'phase_labels': [*PHASES, *SPECIAL],
         'minimum_segment_frames': 1, 'require_full_coverage': True, 'phase_only': True,
         'baseline_records': [], 'tasks': tasks, 'initial_progress': initial,
         'candidate_registry': 'semantic-clip-labeling/candidates.json',
         'candidate_registry_sha256': sha(output / 'candidates.json'),
         'export_scope': 'human-complete-training-clip-intervals; original validation clips excluded',
         'raw_state_references': references, 'validation_url': '../semantic-labeling-round2/validation.html'}
    write_json(output / 'queue.json', q)
    template = Path(__file__).with_name('semantic_labeling.html').read_text()
    (output / 'index.html').write_text(template.replace('__TITLE__', escape(q['title'])).replace(
        '__QUEUE__', json.dumps(q, ensure_ascii=False).replace('<', '\\u003c')).replace('__AUTO__', '{}'))
    return {'clips': len(tasks), 'fps': manifest['fps'], 'seeded_clips': sum(d['seeded_from_prior'] for d in initial['tasks'].values()),
            'manual_boundaries': {c: sorted(v) for c, v in manual.items()}, 'automatic_phase_names': False,
            'validation_clips_excluded': sum(c['split'] == 'validation' for c in manifest['clips'])}


def validate_export(root, registry_path, payload):
    registry = json.loads(registry_path.read_text())
    if payload.get('schema_version') != 3 or payload.get('annotation_mode') != 'complete_clip_intervals':
        raise ValueError('Expected a complete-clip annotation export')
    if payload['manifest_sha256'] != sha(root / 'manifest.json') or payload['candidate_registry_sha256'] != sha(registry_path) or payload['fps'] != registry['fps']:
        raise ValueError('Annotation provenance differs')
    tasks = {t['id']: t for t in registry['items']}
    completed = payload['completed_tasks']
    if len(set(completed)) != len(completed) or not set(completed) <= set(tasks):
        raise ValueError('Unknown or duplicate completed clip')
    grouped, ids = {}, set()
    for r in payload['records']:
        parent = r['parent_candidate_id']
        if parent not in completed or r['id'] in ids:
            raise ValueError('Unconfirmed, duplicate or unknown annotation record')
        ids.add(r['id'])
        t = tasks[parent]
        if r['clip_id'] != t['clip_id'] or r['split'] != 'train' or r['source_sha256'] != t['source_sha256'] or r['fps'] != t['fps'] or r['normalized_video_sha256'] != t['normalized_video_sha256']:
            raise ValueError('Clip source or split differs')
        if r['reviewed'] is not True or r['review_status'] != 'human_reviewed':
            raise ValueError('Exported interval must be human confirmed')
        a, b = r['start_frame'], r['end_frame_exclusive']
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= t['clip_frames']:
            raise ValueError('Invalid frame interval')
        if not np.allclose([r['start_seconds'], r['end_seconds']], [a/t['fps'], b/t['fps']], atol=1e-8):
            raise ValueError('Frame and time alignment differs')
        phase, kind = r['phase'], r['annotation_type']
        if 'stroke_type' in r:
            from dsa.measure.semantic_strokes import STROKE_TYPES, CORE_PHASES
            if r['stroke_type'] not in STROKE_TYPES:
                raise ValueError('Unknown stroke category')
            if r.get('stroke_type_usable') and (not r.get('stroke_type_reviewed') or phase not in CORE_PHASES or r['stroke_type'] == '无法判断'):
                raise ValueError('Stroke target needs a reviewed type and a core stroke phase')
        if kind == 'single':
            if phase not in PHASES or r['phase_usable'] is not True or r['excluded'] is not False:
                raise ValueError('Invalid single-phase label')
        elif kind in ['invalid', 'mixed', 'uncertain']:
            if phase is not None or r['phase_usable'] is not False or r['excluded'] != (kind == 'invalid'):
                raise ValueError('Non-stage interval must not become a phase target')
        else:
            raise ValueError('Unknown interval type')
        grouped.setdefault(parent, []).append((a, b))
    for task_id in completed:
        ranges = sorted(grouped.get(task_id, []))
        if not ranges or ranges[0][0] != 0 or ranges[-1][1] != tasks[task_id]['clip_frames'] or any(ranges[i][0] != ranges[i-1][1] for i in range(1, len(ranges))):
            raise ValueError('Completed clip must have continuous full-frame coverage')
    return {'completed_clips': len(completed), 'intervals': len(ids), 'total_training_clips': len(tasks)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('build')
    create.add_argument('root', type=Path)
    create.add_argument('output', type=Path)
    create.add_argument('prior', type=Path)
    check = sub.add_parser('validate')
    check.add_argument('root', type=Path)
    check.add_argument('registry', type=Path)
    check.add_argument('export', type=Path)
    args = parser.parse_args()
    if args.command == 'build':
        result = build(args.root, args.output, args.prior)
        write_json(args.output / 'batch-summary.json', result)
    else:
        result = validate_export(args.root, args.registry, json.loads(getattr(args, 'export').read_text()))
    print(json.dumps(result, ensure_ascii=False, indent=2))
