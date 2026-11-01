"""Let the user name frozen MoSeq states, without training a second classifier."""
from __future__ import annotations

import argparse
from collections import Counter
from html import escape
import json
import os
from pathlib import Path

import cv2
import numpy as np

from dsa.measure.semantic_phases import PHASES, phase_schema
from dsa.measure.semantic_pilot import sha, write_json


def state_runs(labels):
    labels = np.asarray(labels)
    bounds = np.r_[0, np.flatnonzero(np.diff(labels))+1, len(labels)]
    return [{'state': int(labels[a]), 'start_frame': int(a), 'end_frame_exclusive': int(b)}
            for a, b in zip(bounds[:-1], bounds[1:])]


def _thumbnail(video, input_path, frame, destination):
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame)
    ok, image = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f'Cannot read frame {frame} from {video}')
    with np.load(input_path, allow_pickle=False) as z:
        if 'image_coordinates' in z:
            xy = z['image_coordinates'][frame]
            xy = xy[np.isfinite(xy).all(axis=1)]
            if len(xy):
                h, w = image.shape[:2]
                lo = np.maximum(0, xy.min(0)-[130, 60]).astype(int)
                hi = np.minimum([w, h], xy.max(0)+[130, 45]).astype(int)
                if np.all(hi > lo):
                    image = image[lo[1]:hi[1], lo[0]:hi[0]]
    h, w = image.shape[:2]
    scale = min(420/w, 380/h)
    image = cv2.resize(image, (max(1, round(w*scale)), max(1, round(h*scale))))
    if not cv2.imwrite(str(destination), image):
        raise ValueError(f'Cannot write {destination}')


def build(root, output, mode):
    output.mkdir(parents=True, exist_ok=True)
    config = {'schema_version': 1, 'mode': mode, 'phase_schema': phase_schema(), 'model': 'seed0',
              'semantics_source': 'user-defined state-to-phase mapping; no automatic phase names',
              'stages_may_share_multiple_states': True}
    if mode == 'single':
        checkpoint = root / 'moseq/reference_seed0/model/checkpoint.h5'
        result_path = root / 'moseq/reference_seed0/results.npz'
        with np.load(result_path, allow_pickle=False) as z:
            labels = z['syllable'].copy()
            fps = float(z['fps'])
        runs = state_runs(labels)
        state_ids = sorted(set(map(int, labels)))
        config.update(title=f'旧示例：给已有{len(state_ids)}个状态命名', dataset='camera05/2025_1830', fps=fps,
                      title_context='你发来的8秒示例模型 · seed0 · 30fps', checkpoint_sha256=sha(checkpoint),
                      result_files={'2025_1830': sha(result_path)}, split_counts={'example': 1})
        examples = {}
        for state in state_ids:
            pool = [r for r in runs if r['state'] == state]
            median = np.median([r['end_frame_exclusive']-r['start_frame'] for r in pool])
            pool.sort(key=lambda r: (abs(r['end_frame_exclusive']-r['start_frame']-median), r['start_frame']))
            examples[state] = [{**r, 'clip_id': '2025_1830', 'video_path': root / 'source.mp4',
                                'input_path': root / 'moseq-input.npz', 'result_path': result_path} for r in pool[:3]]
        counts = Counter(map(int, labels))
        runs_count = Counter(r['state'] for r in runs)
    else:
        manifest_path = root / 'manifest.json'
        manifest = json.loads(manifest_path.read_text())
        candidates = json.loads((root / 'semantic-labeling/candidates.json').read_text())
        checkpoint = root / 'models/seed0/model/checkpoint.h5'
        counts, runs_count, result_hashes = Counter(), Counter(), {}
        split_counts = Counter()
        for clip in manifest['clips']:
            if clip['split'] not in {'train', 'validation'}:
                continue
            p = root / 'models/seed0/results' / f'{clip["id"]}.npz'
            with np.load(p, allow_pickle=False) as z:
                if float(z['fps']) != manifest['fps'] or len(z['syllable']) != clip['frames']:
                    raise ValueError('State output frame alignment differs from manifest')
                labels = z['syllable'].copy()
            counts.update(map(int, labels))
            runs_count.update(r['state'] for r in state_runs(labels))
            result_hashes[clip['id']] = sha(p)
            split_counts[clip['split']] += 1
        state_ids = sorted(counts)
        clips = {c['id']: c for c in manifest['clips']}
        examples = {state: [] for state in state_ids}
        for item in candidates['items']:
            state = item['model_state']
            if len(examples[state]) >= 3:
                continue
            clip = clips[item['clip_id']]
            if clip['split'] != 'train':
                raise ValueError('State naming examples must come from the training pool')
            p = root / 'models/seed0/results' / f'{clip["id"]}.npz'
            with np.load(p, allow_pickle=False) as z:
                if not np.all(z['syllable'][item['start_frame']:item['end_frame_exclusive']] == state):
                    raise ValueError('Example crosses a raw model-state boundary')
            examples[state].append({'state': state, 'clip_id': clip['id'], 'start_frame': item['start_frame'],
                                    'end_frame_exclusive': item['end_frame_exclusive'],
                                    'video_path': root / clip['video'], 'input_path': root / clip['input'], 'result_path': p})
        config.update(title=f'正式模型：给已有{len(state_ids)}个状态命名', dataset='camera05_custom', fps=manifest['fps'],
                      title_context='81段训练得到的正式MoSeq模型 · seed0 · 25fps',
                      checkpoint_sha256=sha(checkpoint), manifest_sha256=sha(manifest_path),
                      result_files=result_hashes, split_counts=dict(split_counts))
    states = []
    clip_runs = {}
    for state in state_ids:
        entries = []
        if not examples[state]:
            raise ValueError(f'State {state} has no representative example')
        for i, example in enumerate(examples[state]):
            a, b = example['start_frame'], example['end_frame_exclusive']
            frame = (a+b-1)//2
            name = f'state-{state}-example-{i+1}.jpg'
            _thumbnail(example['video_path'], example['input_path'], frame, output / name)
            if example['clip_id'] not in clip_runs:
                with np.load(example['result_path'], allow_pickle=False) as z:
                    clip_runs[example['clip_id']] = state_runs(z['syllable'])
            entries.append({'clip_id': example['clip_id'], 'start_frame': a, 'end_frame_exclusive': b,
                            'start_seconds': a/config['fps'], 'end_seconds': b/config['fps'],
                            'preview_frame': frame, 'image': name,
                            'video': os.path.relpath(example['video_path'], output),
                            'state_result_sha256': sha(example['result_path'])})
        states.append({'id': state, 'frames': counts[state], 'runs': runs_count[state], 'examples': entries})
    config['states'] = states
    config['clip_runs'] = clip_runs
    config['mapping_key'] = f'{config["dataset"]}:{config["model"]}:{config["checkpoint_sha256"]}'
    write_json(output / 'states.json', config)
    template = Path(__file__).with_name('semantic_state_labeling.html').read_text()
    (output / 'index.html').write_text(template.replace('__TITLE__', escape(config['title'])).replace(
        '__CONFIG__', json.dumps(config, ensure_ascii=False).replace('<', '\\u003c')))
    return {'states': len(states), 'examples': sum(len(s['examples']) for s in states),
            'fps': config['fps'], 'checkpoint_sha256': config['checkpoint_sha256']}


def normalize_annotations(config, payload):
    for key in ['mapping_key', 'checkpoint_sha256', 'fps', 'dataset', 'model']:
        if payload.get(key) != config[key]:
            raise ValueError(f'Mapping {key} does not match this frozen model')
    mappings = payload.get('mappings', [])
    if {m['state'] for m in mappings} != {s['id'] for s in config['states']} or len(mappings) != len(config['states']):
        raise ValueError('Mapping state IDs differ')
    normalized = []
    for m in mappings:
        if type(m['state']) is not int or m['phase'] not in [None, '', *PHASES] or type(m['reviewed']) is not bool:
            raise ValueError('Invalid phase definition')
        kind = m.get('annotation_type', 'single' if m['phase'] in PHASES else '')
        if kind not in ['', 'single', 'mixed', 'invalid', 'uncertain']:
            raise ValueError('Invalid state annotation type')
        mixed = m.get('mixed_phases', [])
        if not isinstance(mixed, list) or any(p not in PHASES for p in mixed) or len(set(mixed)) != len(mixed):
            raise ValueError('Invalid mixed-stage candidates')
        if kind != 'single' and m['phase'] not in [None, '']:
            raise ValueError('Only a single-stage annotation may contain a phase name')
        if kind != 'mixed' and mixed:
            raise ValueError('Only mixed annotations may contain candidate stages')
        if m['reviewed'] and kind in ['', 'single'] and m['phase'] not in PHASES:
            raise ValueError('Confirmed state needs a phase name')
        if m['reviewed'] and kind == 'mixed' and len(mixed) < 2:
            raise ValueError('Confirmed mixed state needs at least two phases')
        usable = bool(m['reviewed'] and kind == 'single' and m['phase'] in PHASES)
        excluded = bool(m['reviewed'] and kind == 'invalid')
        for name, expected in [('usable_for_single_phase', usable), ('excluded', excluded)]:
            if name in m and (type(m[name]) is not bool or m[name] != expected):
                raise ValueError('State inclusion flags differ from the reviewed annotation')
        normalized.append({'state': m['state'], 'annotation_type': kind, 'phase': m['phase'] or None,
                           'mixed_phases': mixed, 'reviewed': m['reviewed'], 'notes': str(m.get('notes', '')),
                           'usable_for_single_phase': usable, 'excluded': excluded})
    return normalized


def validate_mapping(config, payload):
    return {m['state']: m['phase'] for m in normalize_annotations(config, payload) if m['usable_for_single_phase']}


def apply_mapping(config_path, mapping_path, data_root, output):
    """Translate IDs using the user's checked dictionary; never refit a model."""
    config = json.loads(config_path.read_text())
    payload = json.loads(mapping_path.read_text())
    annotations = normalize_annotations(config, payload)
    definitions = {m['state']: m['phase'] for m in annotations if m['usable_for_single_phase']}
    confirmed = {m['state']: m for m in annotations if m['reviewed']}
    if not confirmed:
        raise ValueError('No confirmed state labels to apply')
    if output.exists():
        raise ValueError('Use a new output directory')
    dataset_root = data_root / 'projects' / config['dataset']
    checkpoint = dataset_root / ('moseq/reference_seed0/model/checkpoint.h5' if config['mode'] == 'single' else 'models/seed0/model/checkpoint.h5')
    if sha(checkpoint) != config['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint changed')
    checked_results = []
    for clip, digest in config['result_files'].items():
        p = dataset_root / ('moseq/reference_seed0/results.npz' if config['mode'] == 'single' else f'models/seed0/results/{clip}.npz')
        if sha(p) != digest:
            raise ValueError('Raw state output changed')
        with np.load(p, allow_pickle=False) as z:
            labels = z['syllable'].copy()
            if float(z['fps']) != config['fps']:
                raise ValueError('Result fps differs')
        checked_results.append((clip, digest, labels))
    output.mkdir(parents=True)
    (output / 'user-mapping.json').write_bytes(mapping_path.read_bytes())
    for clip, digest, labels in checked_results:
        write_json(output / f'{clip}.json', {'clip_id': clip, 'fps': config['fps'],
                   'checkpoint_sha256': config['checkpoint_sha256'], 'raw_state_result_sha256': digest,
                   'label_source': 'user-defined model-state mapping',
                   'runs': [{**r, 'phase': definitions.get(r['state']), 'state_name_confirmed': r['state'] in confirmed,
                             'annotation_type': confirmed.get(r['state'], {}).get('annotation_type', 'pending'),
                             'phase_candidates': confirmed.get(r['state'], {}).get('mixed_phases', []),
                             'usable_for_single_phase': r['state'] in definitions,
                             'excluded': confirmed.get(r['state'], {}).get('excluded', False),
                             'start_seconds': r['start_frame']/config['fps'], 'end_seconds': r['end_frame_exclusive']/config['fps']}
                            for r in state_runs(labels)]})
    write_json(output / 'summary.json', {'status': 'complete', 'model_retrained': False,
               'confirmed_state_names': definitions, 'clips': len(config['result_files']),
               'confirmed_state_annotations': list(confirmed.values()),
               'invalid_state_ids': [i for i, m in confirmed.items() if m['annotation_type'] == 'invalid'],
               'mixed_state_ids': [i for i, m in confirmed.items() if m['annotation_type'] == 'mixed'],
               'checkpoint_sha256': config['checkpoint_sha256'], 'mapping_sha256': sha(mapping_path)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    create = sub.add_parser('build')
    create.add_argument('root', type=Path)
    create.add_argument('output', type=Path)
    create.add_argument('--mode', choices=['single', 'dataset'], required=True)
    apply = sub.add_parser('apply')
    apply.add_argument('config', type=Path)
    apply.add_argument('mapping', type=Path)
    apply.add_argument('data_root', type=Path)
    apply.add_argument('output', type=Path)
    args = parser.parse_args()
    if args.command == 'build':
        print(json.dumps(build(args.root, args.output, args.mode), ensure_ascii=False))
    else:
        apply_mapping(args.config, args.mapping, args.data_root, args.output)
