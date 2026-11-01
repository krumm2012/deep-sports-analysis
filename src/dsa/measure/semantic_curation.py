"""User decisions about whole clips, separate from immutable train/validation splits."""
import json

from dsa.measure.semantic_pilot import sha
from dsa.measure.semantic_strokes import STROKE_TYPES


def load_curation(root, manifest, path=None):
    path = path or root/'semantic-curation.json'
    if not path.exists():
        return {'excluded_clips': [], 'clip_overrides': {}}, None
    data = json.loads(path.read_text())
    if data['manifest_sha256'] != sha(root/'manifest.json'):
        raise ValueError('Curation source manifest differs')
    train = {c['id'] for c in manifest['clips'] if c['split'] == 'train'}
    excluded = data.get('excluded_clips', [])
    overrides = data.get('clip_overrides', {})
    if len(excluded) != len(set(excluded)) or not (set(excluded)|set(overrides)) <= train:
        raise ValueError('Curation must name unique training clips only')
    if set(excluded) & set(overrides):
        raise ValueError('Excluded clips cannot have active clip overrides')
    if any(v.get('stroke_type') not in STROKE_TYPES or v.get('phase_boundaries_reviewed') is not False for v in overrides.values()):
        raise ValueError('Clip type confirmation must not confirm phase boundaries')
    return data, path


def apply_clip_type(draft, override):
    """A user-confirmed clip type does not mark its phase intervals as reviewed."""
    for segment in draft['segments']:
        segment['stroke_type'] = override['stroke_type']
        segment['stroke_type_reviewed'] = True
        segment['stroke_type_source'] = 'user_clip_confirmation'
    draft['clip_validity'] = override.get('action_validity', 'normal_action')
    draft['clip_type_source'] = 'user_clip_confirmation'
    return draft
