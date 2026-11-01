"""Seed the full-clip editor with pose-v2 predictions, retaining human progress."""
from copy import deepcopy
import hashlib
from html import escape
import json
from pathlib import Path

from dsa.measure.semantic_phases import PHASES
from dsa.measure.semantic_curation import apply_clip_type
from dsa.measure.semantic_pilot import sha, write_json
from dsa.pages.semantic_clip_labeling import annotation_registry, validate_export


def prediction_draft(task, prediction, previous):
    if previous.get('done') or previous.get('manual_edit'):
        return deepcopy(previous)
    segments = prediction.get('review_segments') or prediction['segments']
    if prediction['clip_id'] != task['clip_id'] or prediction['split'] != 'train' or prediction['fps'] != task['fps'] or prediction['frames'] != task['clip_frames']:
        raise ValueError('Prediction source differs from training task')
    ends = [0]
    for s in segments:
        if s['start_frame'] != ends[-1] or s['end_frame_exclusive'] <= s['start_frame'] or s['phase'] not in [*PHASES, None]:
            raise ValueError('Prediction must provide continuous valid phase intervals')
        ends.append(s['end_frame_exclusive'])
    if ends[-1] != task['clip_frames']:
        raise ValueError('Prediction must cover complete clip')
    # Historical human boundaries remain available, including frame 214 in 00_40.
    bounds = set(ends)
    if previous.get('seeded_from_prior'):
        for s in previous['segments']:
            bounds.update([s['start_frame'], s['end_frame_exclusive']])
    bounds = sorted(bounds)
    result = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        s = next(s for s in segments if s['start_frame'] <= a < s['end_frame_exclusive'])
        phase = s['phase'] or '无法判断'
        result.append({'start_frame': a, 'end_frame_exclusive': b, 'phase': phase,
                       'shot_type': '无法判断', 'notes': '', 'auto_suggested': True,
                       'suggested_start_frame': a, 'suggested_end_frame_exclusive': b,
                       'suggested_phase': phase, 'suggested_shot_type': '无法判断',
                       'suggestion_source': prediction.get('model_version', 'pose-v2')+'-refitted-model',
                       'model_score': s['mean_score']})
    return {'done': False, 'segments': result, 'auto_opt_out': True}


def build(root, pose, output, progress):
    previous = json.loads(progress.read_text())
    validate_export(root, annotation_registry(root, previous), previous)
    q = json.loads((root/'semantic-clip-labeling/queue.json').read_text())
    summary = json.loads((pose/'summary.json').read_text())
    if summary['manifest_sha256'] != q['manifest_sha256'] or summary['status'] != 'complete':
        raise ValueError('A completed model from this dataset is required')
    model_sha = sha(pose/'model.joblib')
    version = pose.name.removeprefix('semantic-')
    fully_supervised = summary.get('training_type') == 'fully_supervised'
    if model_sha != summary['model_sha256']:
        raise ValueError('Model checksum differs')
    if 'active_training_clips' in summary:
        active = set(summary['active_training_clips'])
        q['tasks'] = [t for t in q['tasks'] if t['clip_id'] in active]
    overrides = summary.get('clip_type_confirmations', {})
    output.mkdir(parents=True, exist_ok=True)
    initial = {'active': 0, 'tasks': {}}
    proposals = {}
    for task in q['tasks']:
        if task['split'] != 'train':
            raise ValueError('Pose edit queue must contain training clips only')
        task['video'] = '../../clips/'+task['clip_id']+'/source.mp4'
        task['title'] = '微调自动标识'
        task['instruction'] = '自动分段已填入。按画面修改阶段、拆分或合并区间，逐帧调整边界；灰色区间可保留为无法判断，整段核对后确认。'
        if fully_supervised:
            task['title'] = '查看人工标定'
            task['instruction'] = '本段已由你完整确认，原始人工边界和阶段保留。可查看或修正；修改后需要重新确认并导出。新模型预测可在训练报告中切换查看。'
        pred = json.loads((pose/'predictions'/f'{task["clip_id"]}.json').read_text())
        if version != 'pose-v2':
            pred['model_version'] = version
        draft = previous['annotation_progress']['tasks'][task['id']]
        initial['tasks'][task['id']] = prediction_draft(task, pred, draft)
        if task['clip_id'] in overrides and not initial['tasks'][task['id']].get('done'):
            apply_clip_type(initial['tasks'][task['id']], overrides[task['clip_id']])
        if summary.get('stroke_schema') and not fully_supervised:
            proposals[task['id']] = {'segments': prediction_draft(task, pred, {})['segments'],
                                     'source': '4段人工校订重训后的连续阶段建议；当前手工草稿可保留或撤销恢复'}
            if task['clip_id'] in overrides:
                apply_clip_type(proposals[task['id']], overrides[task['clip_id']])
    initial['active'] = next((i for i, t in enumerate(q['tasks']) if not initial['tasks'][t['id']]['done']), 0)
    registry = {'schema_version': 3, 'scope': version+'-human-adjustment-training-only',
                'manifest_sha256': q['manifest_sha256'], 'fps': q['fps'], 'items': q['tasks']}
    write_json(output/'candidates.json', registry)
    q.update(title=f'微调Pose自动标识 · {len(q["tasks"])}段', pose_edit=True,
             queue_id=hashlib.sha256((version+'-edit:'+q['manifest_sha256']+model_sha).encode()).hexdigest(),
             storage_prefix='camera05-'+version+'-edit-', initial_progress=initial,
             baseline_labels_sha256=sha(progress), proposal_model_sha256=model_sha,
             candidate_registry=str((output/'candidates.json').relative_to(root)),
             candidate_registry_sha256=sha(output/'candidates.json'),
             validation_url='../../semantic-labeling-round2/validation.html',
             export_filename='camera05-'+version+'-adjusted-labels.json',
             export_scope='human-adjusted-'+version+'; training-clips-only; validation-excluded')
    if summary.get('stroke_schema'):
        q['stroke_types'] = summary['stroke_schema']['types']
        q['title'] = '微调阶段与击球类型 · '+version
    if summary.get('excluded_clips'):
        q['excluded_clips'] = summary['excluded_clips']
        q['clip_type_confirmations'] = overrides
        q['curation_sha256'] = summary['curation_sha256']
        q['title'] = f'校订正手动作阶段 · {len(q["tasks"])}段'
    if fully_supervised:
        q['fully_supervised'] = True
        q['title'] = f'已完成人工标定 · {len(q["tasks"])}段'
    q['raw_state_references'] = {t['clip_id']: q['raw_state_references'][t['clip_id']] for t in q['tasks']}
    write_json(output/'queue.json', q)
    (output/'human-progress.json').write_bytes(progress.read_bytes())
    template = Path(__file__).with_name('semantic_labeling.html').read_text().replace(
        '/*__TIMELINE__*/', Path(__file__).with_name('semantic_timeline.js').read_text())
    suggestions = {'queue_id': q['queue_id'], 'revision': model_sha, 'tasks': proposals} if proposals else {}
    if proposals:
        write_json(output/'auto-segments.json', suggestions)
    (output/'index.html').write_text(template.replace('__TITLE__', escape(q['title'])).replace(
        '__QUEUE__', json.dumps(q, ensure_ascii=False).replace('<', '\\u003c')).replace(
        '__AUTO__', json.dumps(suggestions, ensure_ascii=False).replace('<', '\\u003c')))
    return {'clips': len(q['tasks']), 'preserved_confirmed_clips': sum(d['done'] for d in initial['tasks'].values()),
            'model_sha256': model_sha, 'separate_browser_progress': True}


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['root', 'pose', 'output', 'progress']:
        p.add_argument(name, type=Path)
    a = p.parse_args()
    print(json.dumps(build(a.root, a.pose, a.output, a.progress), ensure_ascii=False, indent=2))
