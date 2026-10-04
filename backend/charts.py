"""SUS timing import, independent time-point evaluation and tap-only export."""
from fractions import Fraction
from bisect import bisect_right
import re
import numpy as np


def parse_sus(text, audio_shift=0.0):
    lengths, bpms, rows, offset = {0: 4.0}, {}, [], 0.0
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('#WAVEOFFSET '):
            offset = float(line.split()[1])
        match = re.match(r'#BPM([0-9a-z]{2}):\s*([\d.]+)', line, re.I)
        if match:
            bpms[match[1].upper()] = float(match[2])
        match = re.match(r'#(\d{3})([0-9a-z]{2,3}):\s*(\S+)', line, re.I)
        if match:
            bar, channel, data = int(match[1]), match[2].lower(), match[3]
            if channel == '02':
                lengths[bar] = float(data)
            else:
                rows.append((bar, channel, data))
    max_bar = max((row[0] for row in rows), default=0)
    bar_beats, starts, current = [], [0.0], 4.0
    for bar in range(max_bar + 1):
        current = lengths.get(bar, current)
        if current <= 0:
            raise ValueError('Invalid measure length')
        bar_beats.append(current)
        starts.append(starts[-1] + current)
    tempos, notes, masks = [], [], set()
    for bar, channel, data in rows:
        if len(data) % 2:
            raise ValueError(f'Odd SUS data length at measure {bar}')
        count = len(data) // 2
        for i in range(count):
            value = data[2*i:2*i+2].lower()
            if value == '00':
                continue
            beat = starts[bar] + bar_beats[bar] * i / count
            if channel == '08':
                if value.upper() not in bpms:
                    raise ValueError(f'Undefined BPM {value}')
                tempos.append((beat, bpms[value.upper()]))
                continue
            lane = int(channel[1], 36)
            if lane < 2 or lane > 13:
                continue  # exclude skill/fever channels
            key = (round(beat, 8), lane)
            if channel[0] == '1' and value[0] in '78':
                masks.add(key)
            kind = None
            if channel[0] == '1' and value[0] in '123':
                kind = 'tap'
            elif channel[0] == '3' and value[0] in '123':
                kind = {'1': 'hold_start', '2': 'hold_end', '3': 'hold_step'}[value[0]]
            if kind:
                notes.append((beat, lane, kind, key))
    tempos = sorted(dict(tempos).items())
    if not tempos or tempos[0][0] != 0:
        raise ValueError('SUS must define BPM at beat zero')
    if any(bpm <= 0 for _, bpm in tempos):
        raise ValueError('BPM must be positive')
    tempo_times = [0.0]
    for i in range(1, len(tempos)):
        tempo_times.append(tempo_times[-1] + (tempos[i][0] - tempos[i-1][0])*60/tempos[i-1][1])
    def seconds(beat):
        i = bisect_right([x[0] for x in tempos], beat) - 1
        # Positive WAVE offset delays audio: chart time -> audio time subtracts it.
        return tempo_times[i] + (beat-tempos[i][0])*60/tempos[i][1] - offset + audio_shift
    events = [{'time': seconds(beat), 'beat': beat, 'lane': lane, 'kind': kind}
              for beat, lane, kind, key in notes if key not in masks]
    return {'events': sorted(events, key=lambda x: x['time']),
            'tempos': [{'beat': b, 'bpm': v, 'time': seconds(b)} for b, v in tempos],
            'wave_offset': offset, 'audio_shift': audio_shift}


def unique_times(values):
    return sorted(set(round(float(t), 6) for t in values))


def match_times(predicted, reference, tolerance=0.05):
    """Maximum-cardinality ordered one-to-one matching. No offset fitting."""
    p, r = unique_times(predicted), unique_times(reference)
    i = j = 0
    pairs = []
    while i < len(p) and j < len(r):
        delta = p[i] - r[j]
        if abs(delta) <= tolerance + 1e-9:
            pairs.append((p[i], r[j], delta))
            i += 1
            j += 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    precision = len(pairs)/len(p) if p else 0.0
    recall = len(pairs)/len(r) if r else 0.0
    return {'predicted': len(p), 'reference': len(r), 'matched': len(pairs),
            'precision': precision, 'recall': recall,
            'f1': 2*precision*recall/(precision+recall) if precision+recall else 0,
            'mae_ms': float(np.mean([abs(x[2])*1000 for x in pairs])) if pairs else None,
            'tolerance_ms': tolerance*1000}


def evaluate(candidates, chart, start, end):
    # Heads/taps are primary. Tails/steps have different musical semantics.
    groups = {'tap_and_head': ['tap', 'hold_start'], 'all_visible': ['tap', 'hold_start', 'hold_end', 'hold_step']}
    result = {}
    for group, kinds in groups.items():
        ref = [e['time'] for e in chart['events'] if e['kind'] in kinds and start <= e['time'] < end]
        streams = {'combined': [e['time'] for e in candidates]}
        for source in sorted({e['source'] for e in candidates}):
            streams[source] = [e['time'] for e in candidates if e['source'] == source]
        result[group] = {key: {str(ms): match_times(times, ref, ms/1000) for ms in [25, 50, 100]}
                         for key, times in streams.items()}
    result['note'] = 'Unsnapped predictions; unique time points; no automatic offset fitting. Official chart agreement is not syllable-boundary ground truth.'
    return result


def export_sus(events, bpm, offset=0.0, division=0, snap_ms=35.0):
    if not np.isfinite(bpm) or not 10 <= bpm <= 1000:
        raise ValueError('BPM must be between 10 and 1000')
    if not np.isfinite(offset) or division not in [0, 4, 8, 12, 16, 24, 32, 48, 64]:
        raise ValueError('Invalid offset or division')
    bars = {}
    for event in events:
        beat = (float(event['time'])-offset)*bpm/60
        if division:
            target = round(beat*division/4)*4/division
            if abs(target-beat)*60/bpm*1000 <= snap_ms:
                beat = target
        tick = round(beat*480)
        if tick < 0:
            raise ValueError('Offset places a selected point before measure zero')
        bar, within = divmod(tick, 1920)
        if bar > 999:
            raise ValueError('Chart exceeds SUS measure limit')
        bars.setdefault(bar, set()).add(within)
    lines = ['#TITLE "Auto audio candidates"', '#DESIGNER "AutoPick"',
             '#WAVEOFFSET '+str(-offset), '#REQUEST "ticks_per_beat 480"',
             '#00002: 4', '#BPM01: '+str(bpm), '#00008: 01', '#TIL00: ""', '#HISPEED 00']
    for bar, ticks in sorted(bars.items()):
        fractions = [Fraction(t, 1920) for t in ticks]
        count = int(np.lcm.reduce([f.denominator for f in fractions]))
        data = ['00']*count
        for f in fractions:
            data[int(f*count)] = '12'  # lane 6, width 2: editable timing skeleton
        lines.append(f'#{bar:03d}16:'+''.join(data))
    return '\n'.join(lines)+'\n'
