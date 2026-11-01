"""Small, interval-level supervised pilot from reviewed training-only labels.

No MoSeq state/latent features enter the classifier. Group checks fit every
preprocessing step again, using only the other recordings in each fold.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
from html import escape
import json
from pathlib import Path
import platform
import re
import socket

import numpy as np
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from dsa.measure.moseq_local import read_input
from dsa.measure.semantic_phases import PHASES, PHASE_REVIEW_STATES, PHASE_SCHEMA_VERSION

HEADS = {'shot_type': '击球类型', 'phase': '动作阶段'}
UNKNOWN = {'无法判断', '混合/需调整区间', '', None}
JOINTS = ['nose', 'l_shoulder', 'r_shoulder', 'l_elbow', 'r_elbow', 'l_wrist',
          'r_wrist', 'l_hip', 'r_hip', 'l_knee', 'r_knee', 'l_ankle', 'r_ankle',
          'l_bigtoe', 'l_smalltoe', 'l_heel', 'r_bigtoe', 'r_smalltoe', 'r_heel']


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def validate_labels(root, labels, candidate_path=None):
    manifest = json.loads((root / 'manifest.json').read_text())
    if labels.get('scope', '').startswith('heldout-validation'):
        raise ValueError('Validation labels must not enter training')
    candidate_path = candidate_path or root / 'semantic-labeling/candidates.json'
    candidates = json.loads(candidate_path.read_text())
    if labels.get('schema_version') == 2:
        if labels.get('candidate_registry_sha256') != sha(candidate_path):
            raise ValueError('Supplemental candidate registry hash differs')
        if labels.get('baseline_labels_sha256') != sha(root / 'semantic-pilot/labels.json'):
            raise ValueError('Supplemental baseline labels hash differs')
    if labels['manifest_sha256'] != sha(root / 'manifest.json'):
        raise ValueError('Label manifest hash differs')
    if labels['baseline_checkpoint_sha256'] != sha(root / 'models/seed0/model/checkpoint.h5'):
        raise ValueError('Label checkpoint hash differs')
    if labels['fps'] != manifest['fps'] or labels['fps'] != candidates['fps']:
        raise ValueError('Label fps differs')
    clips = {c['id']: c for c in manifest['clips']}
    originals = {c['id']: c for c in candidates['items']}
    seen = set()
    for r in labels['records']:
        parent = r.get('parent_candidate_id', r['id'])
        if r['id'] in seen or parent not in originals or (
                r['id'] != parent and not re.fullmatch(re.escape(parent) + r'--\d{2,}', r['id'])):
            raise ValueError('Duplicate or unknown label ID')
        seen.add(r['id'])
        c, o = clips[r['clip_id']], originals[parent]
        if c['split'] != 'train' or r['split'] != 'train' or r['clip_id'] != o['clip_id']:
            raise ValueError('Labels must belong to the original training clip')
        if r['source_sha256'] != c['sha256'] or r['source_sha256'] != o['source_sha256']:
            raise ValueError('Label source hash differs')
        if r['reviewed'] is not True or r['review_status'] != 'human_reviewed':
            raise ValueError('Labels must be human reviewed')
        a, b = r['start_frame'], r['end_frame_exclusive']
        if type(a) is not int or type(b) is not int or not 0 <= a < b <= c['frames'] or b-a < 4:
            raise ValueError('Invalid interval bounds')
        if a < o.get('allowed_start_frame', 0) or b > o.get('allowed_end_frame', c['frames']):
            raise ValueError('Interval exceeds registered annotation range')
        if r['fps'] != manifest['fps'] or not np.allclose(
                [r['start_seconds'], r['end_seconds']], [a/r['fps'], b/r['fps']], atol=1e-8):
            raise ValueError('Interval seconds/frame mismatch')
        for head in HEADS:
            allowed = candidates['shot_type_labels' if head == 'shot_type' else 'phase_labels']
            expected_usable = r[head] not in UNKNOWN
            if head == 'phase' and r.get('phase_schema_version') == PHASE_SCHEMA_VERSION:
                allowed = (*PHASES, *PHASE_REVIEW_STATES)
                expected_usable = r[head] in PHASES
            if r[head] not in allowed or r[head + '_usable'] != expected_usable:
                raise ValueError('Unknown label or inconsistent usable flag')
    for clip_id in {r['clip_id'] for r in labels['records']}:
        for head in HEADS:
            intervals = sorted((r['start_frame'], r['end_frame_exclusive']) for r in labels['records']
                               if r['clip_id'] == clip_id and r[head + '_usable'])
            if any(a < intervals[i-1][1] for i, (a, _) in enumerate(intervals) if i):
                raise ValueError('Overlapping supervised targets in the same recording')
    return manifest, clips


def features(data, start, end):
    """Statistics inside one interval only; no frame-level target replication."""
    if end-start < 4 or start < 0 or end > len(data['coordinates']):
        raise ValueError('Feature interval needs at least four frames')
    if data['bodyparts'] != JOINTS:
        raise ValueError('Unexpected bodypart order')
    xy = data['coordinates'][start:end] / 100.
    if not np.isfinite(xy).all():
        raise ValueError('Pilot features require finite coordinates')
    hip = xy[:, [7, 8]].mean(axis=1, keepdims=True)
    pose = xy - hip
    velocity = np.diff(pose, axis=0) * data['fps']
    # Centered posture, variation, net change and joint speed; hip motion remains
    # available without using the absolute camera position or the clip ID.
    return np.concatenate([pose.mean(0).ravel(), pose.std(0).ravel(),
                           (pose[-1]-pose[0]).ravel(),
                           np.abs(velocity).mean(0).ravel(), velocity.std(0).ravel(),
                           np.abs(np.diff(hip[:, 0], axis=0) * data['fps']).mean(0)])


def classifier():
    return make_pipeline(StandardScaler(), LogisticRegression(
        C=0.1, class_weight='balanced', max_iter=3000, random_state=0))


def checked_input(root, clip):
    data = read_input(root / clip['input'])
    source = json.loads(str(np.asarray(data['metadata']).item()))['source']
    if source['source_sha256'] != clip['sha256']:
        raise ValueError('Input source hash differs from manifest')
    if data['fps'] != clip['fps'] or len(data['coordinates']) != clip['frames']:
        raise ValueError('Input frame/fps mismatch')
    return data


def metrics(y, pred, classes):
    return {'accuracy': float(accuracy_score(y, pred)),
            'macro_f1': float(f1_score(y, pred, labels=classes, average='macro', zero_division=0)),
            'confusion_matrix': confusion_matrix(y, pred, labels=classes).tolist(),
            'per_class': classification_report(y, pred, labels=classes, output_dict=True, zero_division=0)}


def fit_head(records, x, head):
    usable = [i for i, r in enumerate(records) if r[head + '_usable']]
    x = x[usable]
    rs = [records[i] for i in usable]
    y = np.array([r[head] for r in rs])
    groups = np.array([r['clip_id'] for r in rs])
    classes = sorted(set(y))
    if len(classes) < 2 or len(set(groups)) < 2:
        raise ValueError('Need at least two labels and recordings per head')
    oof, baseline, folds, predictions = [], [], [], []
    for train, test in LeaveOneGroupOut().split(x, y, groups):
        if set(groups[train]) & set(groups[test]):
            raise AssertionError('Recording leakage')
        model = classifier() if len(set(y[train])) > 1 else DummyClassifier(strategy='most_frequent')
        model.fit(x[train], y[train])
        pred = model.predict(x[test])
        dummy = DummyClassifier(strategy='most_frequent').fit(x[train], y[train])
        base = dummy.predict(x[test])
        missing = sorted(set(y[test]) - set(y[train]))
        folds.append({'test_clips': sorted(set(groups[test])), 'train_clips': sorted(set(groups[train])),
                      'train_ids': [rs[i]['id'] for i in train], 'test_ids': [rs[i]['id'] for i in test],
                      'unseen_test_classes': missing})
        for i, p, b in zip(test, pred, base):
            oof.append((str(y[i]), str(p)))
            baseline.append(str(b))
            predictions.append({'id': rs[i]['id'], 'clip_id': str(groups[i]), 'truth': str(y[i]),
                                'prediction': str(p), 'majority_prediction': str(b),
                                'class_available_in_fold': y[i] in set(y[train])})
    truth, pred = zip(*oof)
    model = classifier().fit(x, y)
    scaler, linear = model.steps[0][1], model.steps[1][1]
    parameters = {'classes': linear.classes_.tolist(), 'mean': scaler.mean_.tolist(),
                  'scale': scaler.scale_.tolist(), 'coef': linear.coef_.tolist(),
                  'intercept': linear.intercept_.tolist(),
                  'probability_rule': 'binary sigmoid for one coefficient row; otherwise softmax',
                  'scores_calibrated': False}
    summary = {'intervals': len(rs), 'recordings': len(set(groups)), 'classes': classes,
               'class_intervals': dict(Counter(y)),
               'class_recordings': {c: len(set(groups[y == c])) for c in classes},
               'group_check': metrics(truth, pred, classes),
               'majority_baseline': metrics(truth, baseline, classes),
               'folds': folds, 'out_of_fold': predictions}
    return model, parameters, summary


def report(output, summary, records):
    rows = []
    details = []
    lookup = {r['id']: r for r in records}
    for head, title in HEADS.items():
        h = summary['heads'][head]
        score, base = h['group_check'], h['majority_baseline']
        rows.append(f'<tr><td>{title}</td><td>{h["intervals"]}/{h["recordings"]}</td>'
                    f'<td>{escape(str(h["class_intervals"]))}</td><td>{score["accuracy"]:.1%}</td>'
                    f'<td>{score["macro_f1"]:.3f}</td><td>{base["accuracy"]:.1%} / {base["macro_f1"]:.3f}</td></tr>')
        detail_rows = []
        for p in sorted(h['out_of_fold'], key=lambda p: p['id']):
            r = lookup[p['id']]
            url = f'../clips/{r["clip_id"]}/source.mp4#t={r["start_seconds"]},{r["end_seconds"]}'
            detail_rows.append(f'<tr><td><a href="{url}">{r["id"]} · {r["clip_id"]}</a></td>'
                               f'<td>{escape(p["truth"])}</td><td>{escape(p["prediction"])}</td>'
                               f'<td>{"正确" if p["truth"] == p["prediction"] else "错误"}'
                               f'{"；该折训练无此类" if not p["class_available_in_fold"] else ""}</td></tr>')
        details.append(f'<h2>{title}：逐项留出结果</h2><table><tr><th>视频区间</th><th>人工标签</th>'
                       '<th>留出预测</th><th>核对</th></tr>' + ''.join(detail_rows) + '</table>')
    html = '''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>人工标注小样本训练</title>
<style>body{max-width:1150px;margin:32px auto;padding:0 24px;font:16px/1.7 system-ui;color:#192b3b}
table{width:100%;border-collapse:collapse}td,th{padding:9px;text-align:left;border-bottom:1px solid #ddd}
.notice{padding:16px;background:#fff0cf}a{color:#145b94}video{max-width:100%}</style>
<h1>人工标注小样本训练</h1><p><a href="../">项目首页</a> · <a href="../semantic-labeling/">标注核对页</a>
 · <a href="summary.json">完整指标与分折记录</a> · <a href="models.json">两个分类模型参数</a></p>
<p class="notice">本轮按24项已确认标签训练。下表是训练池内部按整段视频留出的区间检查，
不是原21段验证集准确率，也不是逐帧阶段或触球检测准确率。所有片段来自同人、同场景，且经过候选采样。</p>
<p>击球类型使用23项，阶段使用10项；混合区间仅排除阶段训练，“无法判断”按任务排除。
两个模型分别使用身体姿态和区间运动统计，标准化只在每折训练视频上拟合。
不输入MoSeq状态编号或潜变量。每个区间算一个样本，模型设置预先固定，无调参选优。</p>
<table><tr><th>任务</th><th>区间/视频数</th><th>类别区间数</th><th>留出正确率</th>
<th>宏F1</th><th>多数类基线：正确率 / 宏F1</th></tr>'''
    html = html.replace('本轮按24项已确认标签训练。', f'本轮按{len(records)}项已确认标签训练。')
    html = html.replace('击球类型使用23项，阶段使用10项；',
                        f'击球类型使用{summary["heads"]["shot_type"]["intervals"]}项，阶段使用{summary["heads"]["phase"]["intervals"]}项；')
    html += ''.join(rows) + '</table>'
    html += '''<p class="notice">截击只有3项、来自2段视频；准备只有2项，挥拍只有1项。
留出唯一挥拍视频时，训练集中没有挥拍，无法学习该类。缺失类别仍计入指标。
反手、发球、引拍、随挥等没有监督样本，当前模型不能识别。
所有模型分数未经概率校准，最高分标签也可能属于未覆盖动作。</p>
<h2>留出视频待人工检查</h2><p>21段原验证视频未用于监督拟合，也没有人工真值。
下列标签仅为候选预测，区间边界沿用冻结seed0的MoSeq分段，未训练语义边界。
过短区间（少于4帧）保留为无法判断。<a href="predictions/">102段逐区间JSON结果</a></p>'''
    for c in summary['preview_clips']:
        html += f'<h3>{c}</h3><video controls preload="metadata" src="../clips/{c}/source.mp4"></video>'
        predictions = json.loads((output / 'predictions' / f'{c}.json').read_text())
        html += '<table><tr><th>时间</th><th>击球类型候选</th><th>阶段候选</th></tr>'
        for p in predictions['intervals']:
            html += f'<tr><td>{p["start_seconds"]:.2f}–{p["end_seconds"]:.2f}s</td>'
            for head in HEADS:
                v = p.get(head)
                html += '<td>' + (escape(v['label']) + f'（模型分数 {v["score"]:.2f}）' if v else '无法判断：区间过短') + '</td>'
            html += '</tr>'
        html += '</table>'
    html += ''.join(details) + '''<h2>下一批人工标注</h2><p>优先补充独立视频中的截击、挥拍与准备，
并对11个混合区间拆分阶段边界。另行标注原留出集后才能计算语义验证指标。</p>
<p>方法依据：<a href="https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html">按组留出</a>、
<a href="https://scikit-learn.org/stable/modules/compose.html#pipeline">折内预处理流水线</a>。</p></html>'''
    if len(records) != 24:
        low = [f'{HEADS[h]}的{label}只有{count}项、来自{s["class_recordings"][label]}段视频'
               for h, s in summary['heads'].items() for label, count in s['class_intervals'].items()
               if count < 3 or s['class_recordings'][label] < 3]
        html = html.replace('截击只有3项、来自2段视频；准备只有2项，挥拍只有1项。\n留出唯一挥拍视频时，训练集中没有挥拍，无法学习该类。缺失类别仍计入指标。',
                            escape('；'.join(low)) + '。缺失训练类别的留出错误仍计入指标。')
        html = html.replace('反手、发球、引拍、随挥等没有监督样本，当前模型不能识别。',
                            '本轮可识别类别以指标表为准，未提供监督样本的动作不具备识别依据。')
        html = html.replace('并对11个混合区间拆分阶段边界。', '并继续补充尚未覆盖的类别与阶段边界。')
    (output / 'index.html').write_text(html)


def run(root, label_path, output, candidate_path=None):
    if output.exists():
        raise ValueError('Output already exists; use a new directory to preserve results')
    labels = json.loads(label_path.read_text())
    manifest, clips = validate_labels(root, labels, candidate_path)
    records = labels['records']
    inputs = {c: checked_input(root, clips[c]) for c in {r['clip_id'] for r in records}}
    x = np.stack([features(inputs[r['clip_id']], r['start_frame'], r['end_frame_exclusive']) for r in records])
    models, parameters, heads = {}, {}, {}
    for head in HEADS:
        models[head], parameters[head], heads[head] = fit_head(records, x, head)
    output.mkdir(parents=True)
    (output / 'predictions').mkdir()
    (output / 'labels.json').write_bytes(label_path.read_bytes())
    summary = {'schema_version': 1, 'status': 'running', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'host': socket.gethostname(), 'python': platform.python_version(),
               'numpy': np.__version__, 'sklearn': sklearn.__version__,
               'manifest_sha256': sha(root / 'manifest.json'), 'labels_sha256': sha(label_path),
               'code_sha256': sha(__file__), 'baseline_checkpoint_sha256': labels['baseline_checkpoint_sha256'],
               'configuration': {'C': 0.1, 'class_weight': 'balanced', 'seed': 0,
                                 'feature_version': 'interval-centered-pose-motion-v1', 'features': x.shape[1]},
               'heads': heads, 'validation_semantic_metrics': None,
               'preview_clips': [c['id'] for c in manifest['clips'] if c['split'] == 'validation'][:3],
               'limitations': ['Training-pool grouped pilot; not heldout semantic accuracy.',
                               'One subject/session; candidate-sampled intervals; no unseen-action rejection.',
                               'Uncalibrated model scores; MoSeq boundaries are not semantic boundaries.']}
    write_json(output / 'models.json', parameters)
    count = Counter()
    for c in manifest['clips']:
        if c['split'] not in {'train', 'validation'}:
            continue
        data = inputs.get(c['id'])
        if data is None:
            data = checked_input(root, c)
        if data['fps'] != manifest['fps'] or len(data['coordinates']) != c['frames']:
            raise ValueError('Prediction input frame/fps mismatch')
        with np.load(root / 'models/seed0/results' / f'{c["id"]}.npz', allow_pickle=False) as z:
            states = z['syllable']
            if len(states) != c['frames'] or float(z['fps']) != data['fps']:
                raise ValueError('MoSeq interval alignment mismatch')
        bounds = np.r_[0, np.flatnonzero(np.diff(states)) + 1, len(states)]
        intervals = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            p = {'start_frame': int(a), 'end_frame_exclusive': int(b),
                 'start_seconds': float(a/data['fps']), 'end_seconds': float(b/data['fps']),
                 'model_state': int(states[a]), 'human_reviewed': False}
            if b-a >= 4:
                f = features(data, a, b)[None]
                for head, model in models.items():
                    scores = model.predict_proba(f)[0]
                    k = int(np.argmax(scores))
                    p[head] = {'label': str(model.classes_[k]), 'score': float(scores[k]),
                               'scores': dict(zip(model.classes_.tolist(), scores.tolist()))}
            intervals.append(p)
        write_json(output / 'predictions' / f'{c["id"]}.json', {
            'clip_id': c['id'], 'split': c['split'], 'status': 'unreviewed_model_candidates',
            'input_sha256': sha(root / c['input']),
            'segmentation_sha256': sha(root / 'models/seed0/results' / f'{c["id"]}.npz'),
            'intervals': intervals})
        count[c['split']] += 1
    summary.update(prediction_clips=dict(count))
    report(output, summary, records)
    summary['status'] = 'complete'
    write_json(output / 'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('labels', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--candidates', type=Path, help='Explicit local registry for supplemental labels')
    args = parser.parse_args()
    result = run(args.root, args.labels, args.output, args.candidates)
    print(json.dumps({h: {k: v for k, v in s.items() if k in {
        'intervals', 'recordings', 'class_intervals', 'group_check', 'majority_baseline'}}
        for h, s in result['heads'].items()}, ensure_ascii=False, indent=2))
