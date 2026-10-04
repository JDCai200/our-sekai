"""Lightweight settings shared by the desktop form and generation CLI."""
import math

PRESETS = {
    'EASY': (10,.25,5),
    'NORMAL': (20,.25,4),
    'HARD': (30,.4,4),
    'EXPERT': (40,.4,4),
    'MASTER': (50,.4,4),
}

FIELDS = (
    ('condition', '模型难度条件', '10 / 20 / 30 / 40 / 50；不是 PJSK 的等级'),
    ('threshold', '采音阈值', '0.01–0.99；提高通常减少采音点'),
    ('min_distance', '最小间距（帧）', '0–8 整数；每帧 32 ms，吸附后间距可能变化'),
    ('shift_ms', '时间偏移（ms）', '−100–100；负值提前，正值延后'),
    ('snap_ms', '节奏吸附上限（ms）', '0–100；0 关闭，范围内吸附到节奏网格'),
    ('snap_division', '全点网格对齐', '0 关闭；8 八分／16 十六分／32 三十二分；忽略吸附上限'),
    ('bpm', 'BPM', '0 自动估计；手动值须大于 0，最大 1000'),
    ('phase', '节拍起点（秒）', '留空自动；使用输入音频的绝对时间，包括前置'),
    ('method', '提取方式', 'peaks：峰值提取；upstream：原版相邻帧筛选'),
)


def defaults():
    return preset_settings('EXPERT')


def preset_settings(name,current=None):
    if name not in PRESETS:raise ValueError('未知的采音预设')
    condition,threshold,distance=PRESETS[name]
    values=dict(condition=condition,threshold=threshold,min_distance=distance,shift_ms=0.,
                snap_ms=0.,snap_division=32 if name=='MASTER' else 16,bpm=0.,phase=None,method='upstream')
    if current:
        values.update({key:current[key] for key in ('bpm','phase')})
    return values


def matching_preset(settings):
    for name in PRESETS:
        if settings==preset_settings(name,settings):return name
    return '自定义'


def validate(overrides=None):
    values = defaults()
    overrides = overrides or {}
    if set(overrides) - set(values):
        raise ValueError('未知的 GenéLive 参数')
    values.update(overrides)
    labels = {key: label for key, label, _ in FIELDS}
    for key in values:
        if key == 'method':
            if values[key] not in ('peaks', 'upstream'):raise ValueError('无效的提取方式')
            continue
        if key == 'phase' and (values[key] is None or str(values[key]).strip() == ''):
            values[key] = None
            continue
        try:
            number = float(values[key])
            if not math.isfinite(number):raise ValueError()
        except (TypeError, ValueError):raise ValueError(labels[key] + '须为有限数值') from None
        if key in ('condition', 'min_distance', 'snap_division'):
            if not number.is_integer():raise ValueError(labels[key] + '须为整数')
            number = int(number)
        values[key] = number
    limits = {'threshold': (.01, .99), 'min_distance': (0, 8), 'shift_ms': (-100, 100),
              'snap_ms': (0, 100), 'bpm': (0, 1000), 'phase': (0, math.inf)}
    if values['condition'] not in (10, 20, 30, 40, 50):raise ValueError('模型难度条件须为10/20/30/40/50')
    if values['snap_division'] not in (0,8,16,32):raise ValueError('全点网格对齐须为0/8/16/32')
    for key, (low, high) in limits.items():
        if values[key] is not None and not low <= values[key] <= high:
            raise ValueError(f'{labels[key]}须在 {low}–{high} 之间')
    return values


def command_arguments(settings):
    values = validate(settings)
    return [item for key, value in values.items() if value is not None
            for item in ('--genelive-' + key.replace('_', '-'), str(value))]
