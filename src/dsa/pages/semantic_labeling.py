"""Prepare a second annotation round without fitting or changing any models."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
from html import escape
import json
from pathlib import Path

from dsa.measure.semantic_pilot import sha, validate_labels, write_json
from dsa.measure.semantic_phases import phase_schema


def select_new_intervals(root, manifest, baseline):
    """Fixed quota, one new candidate per recording, training pool only."""
    used = {r['clip_id'] for r in baseline['records']}
    selected = []
    for head, label, quota in [('shot_type', '截击', 3), ('phase', '准备', 4), ('phase', '挥拍', 5)]:
        pool = []
        for c in manifest['clips']:
            if c['split'] != 'train' or c['id'] in used:
                continue
            predictions = json.loads((root / 'semantic-pilot/predictions' / f'{c["id"]}.json').read_text())
            for r in predictions['intervals']:
                if r.get(head, {}).get('label') == label and r['end_frame_exclusive']-r['start_frame'] >= 8:
                    pool.append((r[head]['score'], c['id'], r))
        count = 0
        for score, clip_id, r in sorted(pool, key=lambda v: (-v[0], v[1], v[2]['start_frame'])):
            if clip_id in used:
                continue
            used.add(clip_id)
            selected.append({'clip_id': clip_id, 'start_frame': r['start_frame'],
                             'end_frame_exclusive': r['end_frame_exclusive'], 'target': label,
                             'selection_head': head, 'selection_score': score})
            count += 1
            if count == quota:
                break
        if count != quota:
            raise ValueError(f'Insufficient distinct recordings for {label}: {count}/{quota}')
    return selected


def build(root, output):
    baseline_path = root / 'semantic-pilot/labels.json'
    baseline = json.loads(baseline_path.read_text())
    manifest, clips = validate_labels(root, baseline)
    original = json.loads((root / 'semantic-labeling/candidates.json').read_text())
    tasks = []
    registry = {r['id']: dict(r) for r in original['items']}
    for r in baseline['records']:
        if r['phase'] != '混合/需调整区间':
            continue
        task = {**registry[r['id']], 'start_frame': r['start_frame'],
                'end_frame_exclusive': r['end_frame_exclusive'],
                'allowed_start_frame': r['start_frame'], 'allowed_end_frame': r['end_frame_exclusive'],
                'kind': 'mixed', 'initial_shot_type': r['shot_type'],
                'replaces_candidate_id': r['id'], 'title': '拆分已有混合阶段',
                'instruction': '击球类型沿用你已确认的标签。按当前区间拆分阶段，可保留无法判断的小段。'}
        tasks.append(task)
        registry[r['id']] = task
    for i, r in enumerate(select_new_intervals(root, manifest, baseline), 1):
        c = clips[r['clip_id']]
        task = {**r, 'id': f'round2-new-{i:02d}', 'split': 'train', 'source_sha256': c['sha256'],
                'clip_frames': c['frames'], 'fps': c['fps'], 'video': f'../clips/{c["id"]}/source.mp4',
                'recording_start_seconds': c['start_seconds'], 'allowed_start_frame': 0,
                'allowed_end_frame': c['frames'], 'kind': 'new', 'initial_shot_type': '',
                'title': '补充独立视频样本',
                'instruction': '这是小样本模型筛选的候选，两个标签均需你确认。可调整时间、拆分区间，无法判断时保留未知。'}
        tasks.append(task)
        registry[task['id']] = task
    common = {'schema_version': 2, 'manifest_sha256': sha(root / 'manifest.json'),
              'baseline_checkpoint_sha256': baseline['baseline_checkpoint_sha256'], 'fps': manifest['fps'],
              'shot_type_labels': original['shot_type_labels'], 'phase_labels': original['phase_labels'],
              'baseline_labels_sha256': sha(baseline_path)}
    candidate_file = {**common, 'purpose': 'Second-round training annotation registry; model proposals need human confirmation.',
                      'items': list(registry.values())}
    # Persist exact bytes so the export can identify its local registry.
    output.mkdir(parents=True, exist_ok=True)
    candidate_path = output / 'candidates.json'
    write_json(candidate_path, candidate_file)
    training = {**common, 'mode': 'training', 'title': '第二轮：补充语义标注', 'tasks': tasks,
                'candidate_registry_sha256': sha(candidate_path),
                'candidate_registry': 'semantic-labeling-round2/candidates.json',
                'baseline_records': baseline['records']}
    validation_tasks = []
    for c in [c for c in manifest['clips'] if c['split'] == 'validation'][:3]:
        validation_tasks.append({'id': f'validation-{c["id"]}', 'clip_id': c['id'], 'split': 'validation',
                                 'source_sha256': c['sha256'], 'fps': c['fps'], 'clip_frames': c['frames'],
                                 'start_frame': 0, 'end_frame_exclusive': c['frames'],
                                 'allowed_start_frame': 0, 'allowed_end_frame': c['frames'],
                                 'video': f'../clips/{c["id"]}/source.mp4', 'kind': 'validation',
                                 'initial_shot_type': '', 'title': '独立验证：完整视频分段',
                                 'instruction': '先观看整段，再按阶段拆分。标注应覆盖整段，拿不准的部分选“无法判断”。本页隐藏模型预测。'})
    validation = {**common, 'mode': 'validation', 'title': '独立验证：第一批3段',
                  'tasks': validation_tasks, 'baseline_records': [], 'candidate_registry_sha256': None}
    template = Path(__file__).with_name('semantic_labeling.html').read_text()
    for config, filename in [(training, 'index.html'), (validation, 'validation.html')]:
        config['queue_id'] = hashlib.sha256(json.dumps(config, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        # Display taxonomy does not change task identities or the browser progress key.
        config['phase_schema'] = phase_schema()
        if (root / 'semantic-five-phase-weak-v1/summary.json').is_file():
            config['latest_report'] = '../semantic-five-phase-weak-v1/'
        write_json(output / ('queue.json' if config['mode'] == 'training' else 'validation-queue.json'), config)
        auto = {}
        if config['mode'] == 'training' and (output / 'auto-segments.json').is_file():
            proposals = json.loads((output / 'auto-segments.json').read_text())
            if proposals['queue_id'] == config['queue_id'] and proposals['candidate_registry_sha256'] == config['candidate_registry_sha256']:
                auto = proposals
        page = template.replace('__TITLE__', escape(config['title'])).replace(
            '__QUEUE__', json.dumps(config, ensure_ascii=False).replace('<', '\\u003c')).replace(
            '__AUTO__', json.dumps(auto, ensure_ascii=False).replace('<', '\\u003c'))
        (output / filename).write_text(page)
    return {'training_tasks': len(tasks), 'mixed_tasks': sum(t['kind'] == 'mixed' for t in tasks),
            'new_tasks': sum(t['kind'] == 'new' for t in tasks),
            'new_selection_targets': dict(Counter(t['target'] for t in tasks if t['kind'] == 'new')),
            'validation_clips': [t['clip_id'] for t in validation_tasks],
            'required_videos': sorted({t['clip_id'] for t in tasks + validation_tasks})}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = build(args.root, args.output)
    write_json(args.output / 'batch-summary.json', result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
