"""Atomic position-free PJSK objects: a slide always owns its real tail."""
from collections import defaultdict
from mapper_pjsk.schema import validate, STARTS, ENDS, KINDS


def pack_objects(events):
    validate(events)
    chains = defaultdict(list)
    objects = []
    for event in events:
        point = {k: v for k, v in event.items() if k != 'chain'}
        if event['chain'] is None:
            objects.append(dict(object_kind='single', start=event['time'], points=[point]))
        else:
            chains[event['chain']].append(point)
    for chain, points in chains.items():
        if points[0]['kind'] not in STARTS or points[-1]['kind'] not in ENDS:
            raise ValueError('Incomplete reference slide')
        objects.append(dict(object_kind='slide', start=points[0]['time'],
                            end=points[-1]['time'], points=points))
    objects.sort(key=lambda o: (o['start'], o['object_kind']))
    return objects


def unpack_objects(objects):
    events = []
    chain = 0
    for obj in objects:
        if obj['object_kind'] not in ('single', 'slide') or not obj['points']:
            raise ValueError('Invalid object')
        is_slide = obj['object_kind'] == 'slide'
        points = obj['points']
        if points[0]['time'] != obj['start']:
            raise ValueError('Start clock mismatch')
        if is_slide:
            if points[0]['kind'] not in STARTS or points[-1]['kind'] not in ENDS or points[-1]['time'] != obj['end']:
                raise ValueError('Incomplete slide object')
        elif len(points) != 1 or points[0]['kind'].startswith('slide_'):
            raise ValueError('Invalid single object')
        events.extend(dict(p, chain=chain if is_slide else None) for p in points)
        if is_slide:
            chain += 1
    events.sort(key=lambda e: (e['time'], KINDS.index(e['kind']), e['chain'] if e['chain'] is not None else -1))
    validate(events)
    return events


def head_window(objects, start, duration):
    """Own objects by their heads; retain tails outside the input crop."""
    if duration <= 0:
        raise ValueError('Invalid window duration')
    return [o for o in objects if start <= o['start'] < start+duration]
