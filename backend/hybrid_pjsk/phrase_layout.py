"""Retrieve coordinated rhythmic phrases and slide paths from training charts."""
from collections import Counter,defaultdict
from pathlib import Path
from copy import deepcopy
import hashlib,json
import numpy as np
from mapper_pjsk.schema import parse_position_free,STARTS
from .paths import ROOT,PRIOR,dump

OUT=ROOT/'diagnostics/pjsk_phrases'


def read_objects(chart):
    chains=defaultdict(list);objects=[]
    for e in chart['events']:
        if e['chain'] is None:objects.append(dict(object_kind='single',start=e['time'],points=[e]))
        else:chains[e['chain']].append(e)
    for points in chains.values():
        objects.append(dict(object_kind='slide',start=points[0]['time'],end=points[-1]['time'],points=points))
    return sorted(objects,key=lambda o:(o['start'],o['points'][0]['lane']))


def groups_of(objects):
    groups=defaultdict(list)
    for i,o in enumerate(objects):groups[o['start']].append(i)
    return sorted(groups.items())


def corpus():
    manifest=json.loads((PRIOR/'manifest.json').read_text(encoding='utf-8'));songs=[s for s in manifest['rows'] if s['split']=='train']
    sources={str(s['id']):hashlib.sha256((ROOT/'reference'/f'{s["id"]:04d}_master.sus').read_bytes()).hexdigest() for s in songs}
    path=OUT/'corpus.json'
    if path.is_file():
        data=json.loads(path.read_text(encoding='utf-8'))
        if data['sources']==sources:return data
    templates=[]
    for song in songs:
        chart=parse_position_free((ROOT/'reference'/f'{song["id"]:04d}_master.sus').read_text(encoding='utf-8-sig'),song['filler'],True)
        objects=read_objects(chart);groups=groups_of(objects)
        for a in range(0,len(groups),4):
            part=groups[a:a+8]
            if len(part)<3 or any(len(ids)>2 for _,ids in part):continue
            t0=part[0][0];tempo=max((t for t in chart['tempos'] if t['time']<=t0+1e-6),key=lambda t:t['time'],default=chart['tempos'][0]);bpm=tempo['bpm']
            sequence=[]
            for t,ids in part:
                entries=[]
                for index in ids:
                    o=objects[index];span=(o.get('end',t)-t)*bpm/60
                    pts=[dict(u=(p['time']-t)/max(o.get('end',t)-t,1e-6),lane=p['lane'],width=p['width']) for p in o['points']]
                    entries.append(dict(kind='slide' if o['object_kind']=='slide' else o['points'][0]['kind'],span=span,path=pts))
                sequence.append(dict(beat=(t-t0)*bpm/60,objects=entries))
            templates.append(dict(song=song['id'],source_start=t0,sequence=sequence))
    OUT.mkdir(parents=True,exist_ok=True);data=dict(sources=sources,templates=templates,training_song_ids=[s['id'] for s in songs]);dump(path,data);return data


def apply_phrases(objects,bpm,offset,chooser=None):
    objects=deepcopy(objects);groups=groups_of(objects)
    library=corpus() if chooser is None else None
    templates=library['templates'] if library is not None else [];selections=[]
    for a in range(0,len(groups),8):
        part=groups[a:a+8];t0=part[0][0];best=None
        for template in (templates if chooser is None else []):
            seq=template['sequence']
            if len(seq)<len(part):continue
            cost=0.
            for n,(time,ids) in enumerate(part):
                ref=seq[n];cost+=3*abs(len(ids)-len(ref['objects']))
                cost+=min(abs((time-t0)*bpm/60-ref['beat']),4)*.35
                actual=sorted('slide' if objects[i]['object_kind']=='slide' else objects[i]['points'][0]['kind'] for i in ids)
                wanted=sorted(o['kind'] for o in ref['objects'])
                cost+=sum(x!=y for x,y in zip(actual,wanted))*.7
            if best is None or cost<best[0]:best=(cost,template)
        score,template=best if chooser is None else (0.,None)
        # Mirroring follows phrase parity, not independently randomized notes.
        # The whole phrase retains the source's relative movement/width motif.
        mirror=bool(int((t0-offset)*bpm/60//8)%2)
        if chooser is not None:
            template,mirror,learned=chooser(objects,part,bpm,offset)
            score=learned['score']
        selections.append(dict(time=t0,source_song=template['song'],source_start=template['source_start'],cost=score,mirrored=mirror))
        if chooser is not None:selections[-1].update(learned)
        for n,(time,ids) in enumerate(part):
            available=list(template['sequence'][n]['objects'])
            ordered=sorted(ids,key=lambda i:('slide' if objects[i]['object_kind']=='slide' else objects[i]['points'][0]['kind'],i))
            for slot,index in enumerate(ordered):
                o=objects[index];kind='slide' if o['object_kind']=='slide' else o['points'][0]['kind']
                match=next((j for j,r in enumerate(available) if r['kind']==kind),0)
                ref=available.pop(match) if available else template['sequence'][n]['objects'][slot%len(template['sequence'][n]['objects'])]
                path=ref['path']
                def interval(u):
                    lane=float(np.interp(u,[p['u'] for p in path],[p['lane'] for p in path]))
                    width=float(np.interp(u,[p['u'] for p in path],[p['width'] for p in path]))
                    return [12-lane-width if mirror else lane,width]
                if o['object_kind']=='slide':
                    start,end=o['start'],o['end'];points=[dict(o['points'][0],_preferred_interval=interval(0))]
                    if ref['kind']=='slide':
                        # Preserve only real source path breakpoints. Hidden
                        # control points shape the hold without new judgments.
                        for p in path[1:-1]:
                            time=round(start+(end-start)*p['u'],6)
                            if time-start<60/bpm/480*2 or end-time<60/bpm/480*2:continue
                            if time-points[-1]['time']<60/bpm/480*2:continue
                            points.append(dict(time=time,kind='slide_hidden',critical=points[0]['critical'],trace=False,
                                               direction='none',_preferred_interval=interval(p['u'])))
                    points.append(dict(o['points'][-1],_preferred_interval=interval(1)));o['points']=points
                else:o['points'][0]['_preferred_interval']=interval(0)
    return objects,dict(method='Coordinated eight-head-group retrieval from training songs, whole-phrase mirroring, original internal slide path breakpoints',
        training_song_ids=library['training_song_ids'] if library is not None else [],phrases=selections,
        hidden_shape_points=sum(p['kind']=='slide_hidden' for o in objects for p in o['points']),
        limitations=['Corpus retrieval adds structural guidance; it is not a newly trained end-to-end model.',
                    'Paths are time-scaled from source motifs and constrained afterward; source identity does not prove target musical suitability.'])


def analyze_official():
    manifest=json.loads((PRIOR/'manifest.json').read_text(encoding='utf-8'));rows=[]
    for song in manifest['rows']:
        chart=parse_position_free((ROOT/'reference'/f'{song["id"]:04d}_master.sus').read_text(encoding='utf-8-sig'),song['filler'],True)
        objects=read_objects(chart);slides=[o for o in objects if o['object_kind']=='slide'];singles=[o for o in objects if o['object_kind']=='single']
        heads=[o['points'][0] for o in objects];close=overlap=curved=changing=0;gaps=[];examples=[]
        for o in slides:
            ps=o['points'];centers=[p['lane']+p['width']/2 for p in ps]
            if len(ps)>2:curved+=1
            if len({p['width'] for p in ps})>1:changing+=1
            tail=ps[-1];bpm=max((t for t in chart['tempos'] if t['time']<=tail['time']),key=lambda t:t['time'])['bpm']
            nearby=[n for n in singles if 0<=n['start']-tail['time']<=60/bpm*.25+1e-6]
            if nearby:close+=1
            same=[n for n in nearby if min(n['points'][0]['lane']+n['points'][0]['width'],tail['lane']+tail['width'])>max(n['points'][0]['lane'],tail['lane'])]
            if same:
                overlap+=1
                if len(examples)<4:examples.append(dict(tail=tail['time'],next_head=same[0]['start'],gap=same[0]['start']-tail['time'],tail_interval=[tail['lane'],tail['width']],next_interval=[same[0]['points'][0]['lane'],same[0]['points'][0]['width']]))
        rows.append(dict(id=song['id'],title=song['title'],split=song['split'],events=len(chart['events']),slides=len(slides),
            width_counts=dict(Counter(p['width'] for p in chart['events'])),head_interval_counts=dict(Counter(f"{p['lane']}:{p['width']}" for p in heads)),
            slides_with_internal_points=curved,slides_with_width_changes=changing,tail_then_single_within_quarter_beat=close,
            overlapping_tail_then_single_within_quarter_beat=overlap,overlap_examples=examples))
    OUT.mkdir(parents=True,exist_ok=True);report=dict(songs=rows,protocol='Full semantic SUS; unique object heads; tail/head rectangle overlap, time gap 0 through 1/4 quarter-note beat; simultaneous opposite-side singles counted separately.',
        interpretation='Nearby notes occur in official charts; visual overlap alone does not establish same-finger assignment or a forbidden pattern.')
    dump(OUT/'official_patterns.json',report);corpus();return report


if __name__=='__main__':
    r=analyze_official();print(json.dumps({k:sum(s[k] for s in r['songs']) for k in ('events','slides','slides_with_internal_points','slides_with_width_changes','tail_then_single_within_quarter_beat','overlapping_tail_then_single_within_quarter_beat')},ensure_ascii=False))
