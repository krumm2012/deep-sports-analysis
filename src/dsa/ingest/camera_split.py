"""Freeze chronological, whole-video splits after camera-drill pose extraction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def assign(clips: list[dict], train_end: float, validation_start: float) -> list[dict]:
    if not 0 < train_end < validation_start:
        raise ValueError('A positive temporal guard is required')
    result, seen = [], set()
    for source in sorted(clips, key=lambda c: (c['start_seconds'], c['id'])):
        clip = dict(source)
        digest = clip['sha256']
        if digest in seen:
            clip.update(split='excluded', exclusion_reason='Duplicate original recording SHA256')
        elif clip.get('pose_status') != 'complete':
            clip.update(split='excluded', exclusion_reason=clip.get('exclusion_reason', 'No quality-approved pose input'))
        elif float(clip.get('fps', 0)) != 25:
            raise ValueError(f"Expected 25fps: {clip['id']}")
        elif clip['start_seconds'] + clip['duration_seconds'] <= train_end:
            clip.update(split='train')
            clip.pop('exclusion_reason', None)
        elif clip['start_seconds'] >= validation_start:
            clip.update(split='validation')
            clip.pop('exclusion_reason', None)
        else:
            clip.update(split='excluded', exclusion_reason='Temporal guard between training and validation blocks')
        seen.add(digest)
        result.append(clip)
    if not any(c['split'] == 'train' for c in result) or not any(c['split'] == 'validation' for c in result):
        raise ValueError('Quality-approved recordings must exist in both temporal blocks')
    return result


def freeze(path: Path, train_end: float = 990, validation_start: float = 1080):
    if any((path.parent / 'models').glob('*/summary.json')):
        raise FileExistsError('Preserve the manifest used by existing models; use a new dataset directory')
    manifest = json.loads(path.read_text(encoding='utf-8'))
    manifest['clips'] = assign(manifest['clips'], train_end, validation_start)
    manifest['split_policy'] = {
        'unit': 'whole independent video', 'ordering': 'filename MM.SS, checked against sample wall-clock overlays',
        'train_clip_end_at_or_before_seconds': train_end,
        'validation_clip_start_at_or_after_seconds': validation_start,
        'nominal_guard_seconds': validation_start - train_end,
        'quality_gate': 'completed pose extraction; original SHA deduplication; 25fps',
        'selection': 'seed0 predefined baseline; holdout never used to select a random seed',
    }
    tmp = path.with_suffix('.json.part')
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(path)
    return manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--train-end', type=float, default=990)
    p.add_argument('--validation-start', type=float, default=1080)
    a = p.parse_args()
    manifest = freeze(a.manifest, a.train_end, a.validation_start)
    from collections import Counter
    print(json.dumps(dict(Counter(c['split'] for c in manifest['clips']))))


if __name__ == '__main__':
    main()
