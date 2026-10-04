"""Reconcile calibrated candidate times with an explicit project origin."""
import math


def constrain_candidates(events, audio_shift, postprocessing=None):
    if not math.isfinite(audio_shift) or audio_shift<0:
        raise ValueError('Invalid project audio origin')
    post=postprocessing or {}
    budget=(abs(float(post.get('shift_ms',0)))+float(post.get('snap_ms',0)))/1000
    if not math.isfinite(budget) or not 0<=budget<=.2:
        raise ValueError('Invalid candidate time correction budget')
    adjusted=[];removed=[];merged=[];by_time={}
    for source in events:
        event=dict(source);time=float(event['time'])
        if not math.isfinite(time):raise ValueError('Invalid candidate time')
        if time<audio_shift:
            raw=event.get('raw_model_time')
            # Only undo a bounded calibration crossing. Events genuinely
            # inside the project's excluded prefix must never be piled up
            # at zero; imported events without provenance are also excluded.
            crossed=raw is not None and math.isfinite(float(raw)) and float(raw)>=audio_shift and abs(float(raw)-time)<=budget+1e-6
            if not crossed:
                removed.append(dict(time=time,raw_model_time=raw,reason='before_project_audio_origin'))
                continue
            event['time']=audio_shift
            if abs(float(event.get('end',time))-time)<1e-6:event['end']=audio_shift
            event['origin_adjustment']=dict(original_time=time,adjusted_time=audio_shift,raw_model_time=raw,reason='calibration_crossed_project_origin')
            adjusted.append(event['origin_adjustment'])
        key=event['time']
        if key in by_time:
            previous=by_time[key]
            if float(event.get('score',0))>float(previous.get('score',0)):
                by_time[key]=event;discarded=previous
            else:discarded=event
            merged.append(dict(time=key,id=discarded.get('id'),score=discarded.get('score'),reason='duplicate_after_origin_adjustment'))
        else:by_time[key]=event
    return [by_time[t] for t in sorted(by_time)],dict(audio_shift=audio_shift,correction_budget_seconds=budget,
        input_heads=len(events),retained_heads=len(by_time),adjusted=adjusted,removed=removed,merged=merged)
