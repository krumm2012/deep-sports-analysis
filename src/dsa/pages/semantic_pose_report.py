"""Playback report for accepted pose labels and the independently refitted model."""
from __future__ import annotations

import json
from pathlib import Path


def report(root: Path, output: Path):
    summary = json.loads((output/'summary.json').read_text())
    labels = json.loads((output/'labels.json').read_text())['records']
    manifest = json.loads((root/'manifest.json').read_text())
    clips = []
    for c in manifest['clips']:
        if c['split'] not in ['train', 'validation']:
            continue
        if c['split'] == 'train' and 'active_training_clips' in summary and c['id'] not in summary['active_training_clips']:
            continue
        clips.append({'id': c['id'], 'split': c['split'], 'frames': c['frames'], 'fps': c['fps'],
                      'video': '../'+c['video'],
                      'labels': [r for r in labels if r['clip_id'] == c['id']],
                      'prediction': json.loads((output/'predictions'/f'{c["id"]}.json').read_text())['segments']})
    payload = json.dumps({'summary': summary, 'clips': clips}, ensure_ascii=False).replace('<', '\\u003c')
    template = Path(__file__).with_suffix('.html').read_text()
    version = output.name.rsplit('-', 1)[-1]
    if summary.get('training_type') == 'fully_supervised':
        template = template.replace('姿态标识与五阶段重训 · v2', f'{summary["human_complete_clips"]}段人工监督训练 · v2')
        template = template.replace('href="../semantic-clip-labeling/">继续人工标注', 'href="edit/">查看人工标定')
        template = template.replace('编辑页已填入自动阶段和边界；可逐帧修改、确认并导出。', '编辑页保留全部人工阶段和边界；下方可切换查看新模型预测。')
    (output/'index.html').write_text(template.replace(' · v2', ' · '+version).replace('__DATA__', payload))


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path); parser.add_argument('output', type=Path)
    args = parser.parse_args(); report(args.root, args.output)
