"""Editable SUS preview with explicit, deterministic placeholder geometry."""
from collections import defaultdict
from fractions import Fraction
from math import lcm
from .schema import validate, parse_position_free, STARTS, ENDS


def export_preview(events, bpm, title='Mapperatorinator preview', audio_shift=0., geometry=None):
    validate(events)
    if not 10 <= bpm <= 1000:
        raise ValueError('Invalid BPM')
    rows = defaultdict(dict)
    active = {}
    changes = []
    placed = []
    cursor = 0
    occupied_at_tick = set()
    previous_tick = None
    chain_channels={};channel_release_ticks={}
    if geometry is not None and len(geometry)!=len(events):raise ValueError('Geometry/event count mismatch')

    def put(tick, channel, value):
        bar, within = divmod(tick, 1920)
        if not 0 <= bar <= 999:
            raise ValueError('Outside SUS measure range')
        key = (bar, channel)
        if within in rows[key]:
            raise ValueError('Conflicting SUS events')
        rows[key][within] = value

    for index, original in enumerate(events):
        e = dict(original)
        # OpenSekai adds project FillerSec during playback. JSON/model times
        # are absolute in the padded audio, while SUS measures start after it.
        if e['time'] < audio_shift:
            raise ValueError('Event precedes the project audio origin')
        tick = round((e['time'] - audio_shift) * bpm / 60 * 480)
        if tick != previous_tick:
            occupied_at_tick = set()
            previous_tick = tick
        kind, chain = e['kind'], e['chain']
        if kind in STARTS or chain is None:
            unavailable = occupied_at_tick | {lane for lane, _ in active.values()}
            available = [lane for lane in range(2, 14) if lane not in unavailable]
            if not available:
                raise ValueError('No free placeholder lane')
            # Constant width 1 avoids collisions with held notes. This is not a
            # learned PJSK layout and does not claim two-hand playability.
            lane = available[cursor % len(available)]
            cursor += 1
            if kind in STARTS:
                active[chain] = (lane, e['critical'])
        else:
            lane, inherited_critical = active[chain]
            if inherited_critical and not e['critical']:
                e['critical'] = True
                changes.append(dict(event=index, attribute='critical', reason='SUS inherits critical slide start'))
        width=1
        if geometry is not None:
            g=geometry[index];lane=int(g['lane'])+2;width=int(g['width'])
            if not 2<=lane<=13 or not 1<=width<=14-lane:raise ValueError('Invalid learned geometry')
            if kind in STARTS:
                used=set(chain_channels.values())
                # Near-coincident real times may quantize to one SUS tick.
                # Never reuse an ending channel at that tick: SUS sorts points
                # by lane and can otherwise read the new start before the end.
                chain_channels[chain]=next(c for c in range(12) if c not in used and channel_release_ticks.get(c)!=tick)
        occupied_at_tick.add(lane)
        x = '0123456789abcdefghijklmnopqrstuvwxyz'[lane]
        overlay = None
        if chain is None:
            overlay = (6 if e['critical'] else 5) if e['trace'] else (2 if e['critical'] else 1)
        else:
            # The placeholder lane's slide channel can be reused once closed.
            channel = '0123456789abcdefghijklmnopqrstuvwxyz'[chain_channels[chain] if geometry is not None else lane - 2]
            point = 1 if kind in STARTS else 2 if kind in ENDS else 5 if kind == 'slide_hidden' else 3
            put(tick, '3' + x + channel, str(point) + '0123456789abcdefghijklmnopqrstuvwxyz'[width])
            if kind.endswith('_hidden') and kind in STARTS | ENDS:
                overlay = 8 if e['critical'] else 7
                if e['trace']:
                    e['trace'] = False
                    changes.append(dict(event=index, attribute='trace', reason='Hidden endpoint overlay cannot also encode trace'))
            elif kind == 'slide_attach':
                overlay = 3
                if e['trace']:
                    e['trace'] = False
                    changes.append(dict(event=index, attribute='trace', reason='Attach overlay cannot also encode trace'))
                inherited = active[chain][1]
                if e['critical'] != inherited:
                    e['critical'] = inherited
                    changes.append(dict(event=index, attribute='critical', reason='Attach inherits chain critical'))
            elif e['trace']:
                overlay = 6 if e['critical'] else 5
            elif e['critical']:
                overlay = 2
            if kind in ENDS:
                del active[chain]
                if geometry is not None:
                    channel_release_ticks[chain_channels[chain]]=tick
                    del chain_channels[chain]
        if overlay is not None:
            put(tick, '1' + x, str(overlay) + '0123456789abcdefghijklmnopqrstuvwxyz'[width])
        if e['direction'] != 'none':
            put(tick, '5' + x, {'up': '1', 'left': '3', 'right': '4'}[e['direction']]+'0123456789abcdefghijklmnopqrstuvwxyz'[width])
        placed.append(dict(e, time=round(audio_shift + tick / 480 * 60 / bpm, 6),
                           **({'lane':lane-2,'width':width} if geometry is not None else {})))
    lines = [f'#TITLE "{title}"', '#DESIGNER "Mapperatorinator PJSK prototype - temporary layout"',
             '#WAVE "bgm.ogg"', '#JACKET "bg.png"', '#WAVEOFFSET 0',
             '#REQUEST "ticks_per_beat 480"', '#00002: 4', f'#BPM01: {bpm:.6f}',
             '#00008: 01', '#TIL00: ""', '#HISPEED 00']
    if geometry is not None:lines[1]='#DESIGNER "Genelive + AutoOsu NS + official-chart layout GRU; two-finger projection"'
    for (bar, channel), values in sorted(rows.items()):
        count = lcm(*(Fraction(tick, 1920).denominator for tick in values))
        data = ['00'] * count
        for tick, value in values.items():
            data[tick * count // 1920] = value
        lines.append(f'#{bar:03d}{channel}: ' + ''.join(data))
    text = '\n'.join(lines) + '\n'
    parsed = parse_position_free(text, audio_shift=audio_shift,include_geometry=geometry is not None)['events']
    # Compare graph topology after renumbering, including simultaneous notes.
    def canonical(sequence):
        groups = defaultdict(list)
        singles = []
        for e in sequence:
            # SUS beat parsing rounds fractional beats before seconds. Compare
            # the encoded tick, while still requiring exact types and topology.
            value = (round(e['time'] * bpm / 60 * 480),) + tuple(e[k] for k in ('kind', 'critical', 'trace', 'direction'))
            if geometry is not None:value+=tuple(e[k] for k in ('lane','width'))
            if e['chain'] is None:
                singles.append(value)
            else:
                groups[e['chain']].append(value)
        return sorted(singles), sorted(tuple(group) for group in groups.values())
    if canonical(parsed) != canonical(placed):
        raise ValueError('SUS round-trip changed event attributes or slide topology')
    times_by_key = defaultdict(list)
    def key(e):
        return (round(e['time'] * bpm / 60 * 480),) + tuple(e[k] for k in ('kind', 'critical', 'trace', 'direction'))
    for e in parsed:
        times_by_key[key(e)].append(e['time'])
    errors = [abs(original['time'] - times_by_key[key(expected)].pop()) * 1000
              for original, expected in zip(events, placed)]
    return text, dict(events=len(parsed), placeholder_width=1 if geometry is None else None,
                      audio_shift=audio_shift, time_reference='SUS time + project FillerSec = audio absolute time',
                      layout='learned intervals projected to two-finger constraints' if geometry is not None else 'temporary rotating free lanes; constant slide lane',
                      attribute_adjustments=changes,
                      max_timing_error_ms=max(errors, default=0),
                      round_trip_verified=True)
