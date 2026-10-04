"""Learned rhythmic decoration; sample calibrated chart-derived posteriors."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
from .paths import ROOT
from .portable_forest import Forest


def rhythm_features(times, spans, counts, index, phase, occupancy):
    time=times[index]; before=time-times[index-1] if index else 1.
    after=times[index+1]-time if index+1<len(times) else 1.
    nearby=sum(abs(t-time)<=1. for t in times)
    return [min(before,8)/8,min(after,8)/8,min(spans[index],16)/16,counts[index]/2,
            np.sin(phase*np.pi/2),np.cos(phase*np.pi/2),np.sin(phase*2*np.pi),np.cos(phase*2*np.pi),
            min(nearby,16)/16,float(occupancy),float(before<.3),float(after<.3)]


@lru_cache(maxsize=2)
def load(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    if data['schema']!=1:raise ValueError('Unsupported expression model')
    models={key:Forest(data.pop(key)) for key in ('single','slide')}
    return data,models


def apply_expression(objects,bpm,offset=0.,model_path=None):
    path=Path(model_path) if model_path else ROOT/'diagnostics/pjsk_patterns_v2/expression.json'
    if not path.is_file():raise FileNotFoundError('缺少训练后的键型/重音模型：'+str(path))
    data,models=load(str(path)); result=deepcopy(objects)
    groups={}
    for i,o in enumerate(result):groups.setdefault(o['start'],[]).append(i)
    heads=sorted(groups); beats=[(t-offset)*bpm/60 for t in heads]
    spans=[max((result[i].get('end',t)-t)*bpm/60 for i in groups[t]) for t in heads]
    counts=[len(groups[t]) for t in heads]
    seed=int.from_bytes(hashlib.sha256(np.asarray(heads,dtype='<f8').tobytes()).digest()[:8],'little')
    rng=np.random.default_rng(seed); changes=0; single=slide=critical=0; directions=('none','up','left','right')
    for n,t in enumerate(heads):
        occupation=min(sum(o['start']<t and o.get('end',o['start'])>t for o in result),2)/2
        row=np.asarray(rhythm_features(beats,spans,counts,n,beats[n],occupation))[None]
        for i in groups[t]:
            o=result[i]; key='slide' if o['object_kind']=='slide' else 'single'; model=models[key]
            if key=='single' and o['points'][0]['kind']=='trace':continue
            probabilities=model.predict_proba(row)[0]
            # Blend the learned posterior with the acoustic proposal. Sampling
            # chooses observed label combinations, never arbitrary key quotas.
            p=o['points'][0]; direction=directions.index(o['points'][-1].get('direction','none'))
            category=direction if key=='slide' else (direction if p['kind']=='flick' else 0)
            original=category*2+int(bool(p.get('critical')))
            probabilities=.75*probabilities
            match=np.flatnonzero(model.classes_==original)
            if len(match):probabilities[match[0]]+=.25
            probabilities/=probabilities.sum()
            label=int(rng.choice(model.classes_,p=probabilities)); direction=directions[label//2]; accent=bool(label%2)
            changes+=label!=original
            if key=='slide':
                o['points'][-1]['direction']=direction
                slide+=direction!='none'
                for point in o['points']:point['critical']=accent
            else:
                p.update(kind='tap' if direction=='none' else 'flick',direction=direction,critical=accent,trace=False)
                single+=direction!='none'
            critical+=accent
    return result,dict(model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),training_songs=data['training_songs'],
        training_objects=data['training_objects'],changed_objects=int(changes),single_flicks=single,slide_tail_flicks=slide,
        critical_objects=critical,policy='Seeded posterior sampling from rhythm-conditioned forests, 75% learned / 25% acoustic proposal; head times, counts and spans retained',
        limitations=['Chart-only rhythm supervision, no paired audio accent training; later constraints may repair objects.'])
