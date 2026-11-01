import json

import pytest

from dsa.ingest.camera_split import assign, freeze


def clip(key, start, duration=10, **changes):
    return dict(id=key, start_seconds=start, duration_seconds=duration, sha256=key,
                pose_status='complete', fps=25, **changes)


def test_whole_clip_crossing_cut_is_guard_not_partly_training():
    records = assign([clip('a', 970), clip('b', 985), clip('c', 1075), clip('d', 1080)], 990, 1080)
    assert [c['split'] for c in records] == ['train', 'excluded', 'excluded', 'validation']


def test_duplicate_original_is_never_held_out_as_new_recording():
    repeated = clip('d', 1090); repeated['sha256'] = 'a'
    records = assign([clip('a', 20), clip('c', 1080), repeated], 990, 1080)
    assert records[-1]['split'] == 'excluded'
    assert 'Duplicate' in records[-1]['exclusion_reason']


def test_quality_failure_and_wrong_fps_cannot_enter_model():
    failed = clip('b', 30); failed.update(pose_status='failed', exclusion_reason='Empty scene')
    records = assign([clip('a', 20), failed, clip('c', 1080)], 990, 1080)
    assert records[1]['exclusion_reason'] == 'Empty scene'
    bad = clip('a', 20); bad['fps'] = 30
    with pytest.raises(ValueError, match='25fps'):
        assign([bad, clip('c', 1080)], 990, 1080)


def test_existing_model_manifest_is_preserved(tmp_path):
    path = tmp_path / 'manifest.json'; path.write_text(json.dumps({'clips': []}))
    model = tmp_path / 'models/seed0'; model.mkdir(parents=True)
    (model / 'summary.json').write_text('{}')
    before = path.read_bytes()
    with pytest.raises(FileExistsError, match='Preserve'):
        freeze(path)
    assert path.read_bytes() == before
