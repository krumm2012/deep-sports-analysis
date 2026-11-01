"""The user's five tennis action stages, independent of learned model classes."""

PHASE_SCHEMA_VERSION = 'tennis-five-phase-v1'
PHASES = ('准备', '引拍', '挥拍', '随挥', '回位/移动')
PHASE_DEFINITIONS = (
    ('准备', '#e6f0ff', '#244b8a', '持拍等待、站稳并准备下一次动作'),
    ('引拍', '#f0e9ff', '#62418c', '向后带拍、转体与蓄势'),
    ('挥拍', '#fff0db', '#8a4d0d', '向前加速挥动球拍'),
    ('随挥', '#e1f4ea', '#246849', '挥动后的延续、减速与收拍'),
    ('回位/移动', '#eaf0f3', '#405c6c', '恢复持拍或调整脚步与站位'),
)
PHASE_REVIEW_STATES = ('无法判断', '混合/需调整区间')


def phase_schema():
    return {'version': PHASE_SCHEMA_VERSION, 'phases': list(PHASES),
            'review_states': list(PHASE_REVIEW_STATES),
            'definitions': [{'label': label, 'background': background, 'color': color, 'description': description}
                            for label, background, color, description in PHASE_DEFINITIONS]}
