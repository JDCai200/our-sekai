"""Sample supervised hold-path families independently of single-key motifs."""
from copy import deepcopy
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import numpy as np
from .expression import rhythm_features
from .portable_forest import Forest


@lru_cache(maxsize=2)
def load(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    forest=Forest(data.pop('forest'))
    return data,forest


def apply_slide_paths(objects,bpm,offset,path):
    data,forest=load(str(path)); result=deepcopy(objects)
    groups={}
    for i,o in enumerate(result):groups.setdefault(o['start'],[]).append(i)
    heads=sorted(groups); beats=[(t-offset)*bpm/60 for t in heads]
    spans=[max((result[i].get('end',t)-t)*bpm/60 for i in groups[t]) for t in heads]
    counts=[len(groups[t]) for t in heads]
    seed=int.from_bytes(hashlib.sha256(b'slide-paths-v1'+np.asarray(heads,dtype='<f8').tobytes()).digest()[:8],'little')
    rng=np.random.default_rng(seed); usage={}; controls=0
    for n,t in enumerate(heads):
        occupancy=min(sum(o['start']<t and o.get('end',o['start'])>t for o in result),2)/2
        for i in groups[t]:
            o=result[i]
            if o['object_kind']!='slide':continue
            row=rhythm_features(beats,spans,counts,n,beats[n],occupancy)
            row[2]=min((o['end']-t)*bpm/60,16)/16
            probabilities=forest.predict_proba(np.asarray(row)[None])[0]
            label=int(rng.choice(forest.classes_,p=probabilities/probabilities.sum()))
            prototypes=data['patterns'][label]['paths']; source=prototypes[int(rng.integers(len(prototypes)))]
            start=o['points'][0]; end=o['points'][-1]
            preferred=start.get('_preferred_interval',[source[0]['lane'],source[0]['width']])
            center=preferred[0]+preferred[1]/2
            mirrored=bool(rng.integers(2))
            # The feasibility solver assigns each finger to a six-lane half.
            # Scale the complete observed shape into that space together;
            # clipping every wide node independently would erase tapering.
            edge=min(p['lane'] for p in source)
            extent=max(p['lane']+p['width'] for p in source)-edge
            scale=min(1.,6/max(extent,1))
            base=0. if center<6 else 6.
            first=(source[0]['lane']-edge+source[0]['width']/2)*scale
            shift=float(np.clip(center-base-first,0,6-extent*scale))
            def interval(point):
                width=max(1.,point['width']*scale)
                c=(point['lane']-edge+point['width']/2)*scale+shift
                if mirrored:c=6-c
                c=np.clip(c,width/2,6-width/2)
                return [float(base+c-width/2),float(width)]
            points=[dict(start,_preferred_interval=interval(source[0]))]
            for point in source[1:-1]:
                time=round(t+(o['end']-t)*point['u'],6)
                if time-t<60/bpm/480*2 or o['end']-time<60/bpm/480*2 or time-points[-1]['time']<60/bpm/480*2:continue
                points.append(dict(time=time,kind='slide_hidden',direction='none',trace=False,
                    critical=bool(start['critical']),_preferred_interval=interval(point)))
            points.append(dict(end,_preferred_interval=interval(source[-1])))
            o['points']=points; usage[label]=usage.get(label,0)+1; controls+=max(len(points)-2,0)
    return result,dict(modes=data['modes'],usage=usage,hidden_controls=controls,
        model_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
        policy='Rhythm/span-conditioned forest posterior over actual training hold-path families; fixed seed; real source knots and widths; whole-shape six-lane scaling/mirroring before constraints')
