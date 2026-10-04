"""Keep every input head; complete endpoint-only slides, with explicit repairs."""
from collections import Counter
import numpy as np
from autoosu_pjsk.objects import unpack_objects
from .vocabulary import TYPES,DIRECTIONS


def to_objects(times,actions,bpm,audio_duration,max_active=6,snap_division=0):
    if snap_division not in (0,8,16,32):raise ValueError('Invalid judgment grid division')
    objects=[];repairs=[];active=[]
    if len(times)!=len(actions['count']):raise ValueError('Head/action count mismatch')
    for i,time in enumerate(times):
        active=[end for end in active if end>time]
        for slot in range(int(actions['count'][i])):
            kind=TYPES[int(actions['kind'][i,slot])]
            critical=bool(actions['critical'][i,slot]);direction=DIRECTIONS[int(actions['direction'][i,slot])] if kind=='flick' else 'none'
            point=dict(time=float(time),kind='slide_start' if kind=='slide' else kind,
                       critical=critical,trace=kind=='trace',direction=direction)
            if kind=='slide':
                # Span is predicted by the trained head, then represented on a
                # 1/8-quarter-beat grid. No window-boundary or end-of-song tail.
                step=4/snap_division if snap_division else .125
                beats=float(np.clip(np.expm1(min(float(actions['span'][i,slot]),np.log1p(64))),step,64))
                beats=max(step,round(beats/step)*step)
                end=round(float(time)+beats*60/bpm,6)
                reason='predicted tail outside audio' if end>=audio_duration else 'temporary preview active-slide capacity' if len(active)>=max_active else None
                if reason:
                    repairs.append(dict(head=i,slot=slot,reason=reason,original_kind='slide',replacement='tap',predicted_end=end))
                    point.update(kind='tap',trace=False)
                    objects.append(dict(object_kind='single',start=float(time),points=[point]))
                else:
                    tail=dict(time=end,kind='slide_end',critical=critical,trace=False,
                              direction=DIRECTIONS[int(actions['tail_direction'][i,slot])])
                    objects.append(dict(object_kind='slide',start=float(time),end=end,points=[point,tail]));active.append(end)
            else:
                objects.append(dict(object_kind='single',start=float(time),points=[point]))
    events=unpack_objects(objects)
    generated_heads={o['start'] for o in objects}
    if generated_heads!={float(t) for t in times}:raise AssertionError('GenéLive onsets were changed')
    return objects,events,dict(input_head_instants=len(set(times)),output_head_instants=len(generated_heads),
                               head_times_exactly_preserved=True,repairs=repairs,
                               object_counts=dict(Counter(o['object_kind'] if o['object_kind']=='slide' else o['points'][0]['kind'] for o in objects)))
