"""Stroke side and backhand grip are distinct from temporal action phases."""

STROKE_TYPES = ('正手', '单手反手', '双手反手', '无法判断')
CORE_PHASES = ('引拍', '挥拍', '随挥')


def stroke_label(record):
    explicit = record.get('stroke_type')
    if explicit in STROKE_TYPES[:-1] and record.get('stroke_type_reviewed') and record.get('stroke_type_usable', True):
        return explicit
    if record.get('stroke_type_reviewed'):
        return '无法判断'
    # A legacy generic backhand label does not establish the grip subtype.
    if record.get('shot_type_usable') and record.get('shot_type') == '正手':
        return '正手'
    return '无法判断'


def stroke_schema():
    return {'types': list(STROKE_TYPES), 'shared_phases': ['准备', '回位/移动'],
            'type_specific_phases': list(CORE_PHASES),
            'annotation_scope': 'one stroke cycle, not the whole video',
            'routing': 'uncertain type stays shared; no forced branch assignment'}
