"""Match fixed GenéLive heads to official object sets; exclude unlabeled noise."""
from collections import defaultdict,Counter
import json
import numpy as np
import torch
from autoosu_pjsk.backend import ROOT,TimingModel
from autoosu_pjsk.objects import pack_objects
from mapper_pjsk.schema import parse_position_free
from charts import match_times
from genelive_backend import events_from_probabilities
from .model import TYPES,DIRECTIONS,SLOTS,empty_action,acoustic_features

OUT=ROOT/'diagnostics/genelive_autoosu'
PRIOR=ROOT/'diagnostics/genelive_finetune_conservative'


def dump(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')


def placement_features(times,scores,bpm,genelive_features,context_start):
    times=np.asarray(times)
    before=np.r_[0.,np.diff(times)]*bpm/60
    after=np.r_[np.diff(times),0.]*bpm/60
    basic=np.column_stack([scores,np.minimum(before,8)/8,np.minimum(after,8)/8,np.ones(len(times))]).astype(np.float32)
    positions=np.clip((times-context_start)/.032,0,len(genelive_features)-1)
    lower=np.floor(positions).astype(int);upper=np.minimum(lower+1,len(genelive_features)-1)
    weight=(positions-lower)[:,None]
    context=genelive_features[lower]*(1-weight)+genelive_features[upper]*weight
    return np.concatenate([basic,context.astype(np.float32)],axis=1)


def labels_for_candidates(objects,times,bpm):
    groups=defaultdict(list)
    for obj in objects:
        if obj['points'][0]['kind'] not in ('slide_start_hidden',):groups[obj['start']].append(obj)
    matches=match_times(times,list(groups),.05)
    # match_times does not expose pairs; repeat its ordered one-to-one rule.
    p=sorted(set(round(float(t),6) for t in times));r=sorted(round(float(t),6) for t in groups)
    i=j=0
    assigned={}
    while i<len(p) and j<len(r):
        delta=p[i]-r[j]
        if abs(delta)<=.050000001:assigned[p[i]]=r[j];i+=1;j+=1
        elif delta<0:i+=1
        else:j+=1
    target=empty_action(len(times));mask=torch.zeros(len(times),dtype=torch.bool)
    # Unmatched candidates have no official type supervision. Teacher feedback
    # uses an explicit tap placeholder; loss is masked, not fabricated truth.
    target['count'].fill_(1);target['kind'][:,0]=0
    omitted=0
    for n,time in enumerate(times):
        ref=assigned.get(round(time,6))
        if ref is None:continue
        objs=sorted(groups[ref],key=lambda o:(TYPES.index('slide' if o['object_kind']=='slide' else o['points'][0]['kind']),o.get('end',0)))
        if len(objs)>SLOTS:raise ValueError('Object set exceeds model slots')
        mask[n]=True;target['count'][n]=len(objs)
        for slot,obj in enumerate(objs):
            head=obj['points'][0];kind='slide' if obj['object_kind']=='slide' else head['kind']
            target['kind'][n,slot]=TYPES.index(kind)
            target['critical'][n,slot]=int(head['critical'])
            target['direction'][n,slot]=DIRECTIONS.index(head['direction'])
            if kind=='slide':
                target['span'][n,slot]=np.log1p((obj['end']-obj['start'])*bpm/60)
                target['tail_direction'][n,slot]=DIRECTIONS.index(obj['points'][-1]['direction'])
                omitted+=max(len(obj['points'])-2,0)
    return target,mask,matches,omitted


def prepare(timing):
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((PRIOR/'manifest.json').read_text(encoding='utf-8'))
    old=json.loads((PRIOR/'report.json').read_text(encoding='utf-8'))
    rows=[]
    for song in manifest['rows']:
        stem=f'{song["id"]:04d}'
        data=json.loads((PRIOR/stem/'analysis.json').read_text(encoding='utf-8'))
        probs=np.load(PRIOR/f'{stem}_finetuned_probs.npy')
        candidates=events_from_probabilities(probs,max(0,song['start']-5),song['start'],song['start']+song['duration'],
            data['estimated_bpm'],data['postprocessing']['grid_origin'],threshold=old['selected']['threshold'],
            **{k:data['postprocessing'][k] for k in ('method','min_distance','shift_ms','snap_ms')})
        times=[e['time'] for e in candidates];bpm=data['estimated_bpm']
        chart=parse_position_free((ROOT/'reference'/f'{stem}_master.sus').read_text(encoding='utf-8-sig'),song['filler'])
        objects=pack_objects(chart['events'])
        objects=[o for o in objects if song['start']<=o['start']<song['start']+song['duration']]
        target,mask,agreement,mids=labels_for_candidates(objects,times,bpm)
        path=OUT/f'{stem}_acoustic.npy'
        if path.exists():acoustic=np.load(path)
        else:
            specs=torch.from_numpy(np.load(ROOT/'diagnostics/autoosu_pjsk'/f'{stem}_features.npy'))
            acoustic=acoustic_features(timing,specs,times,song['start'],bpm,data['postprocessing']['grid_origin']).numpy()
            np.save(path,acoustic)
        placement=placement_features(times,[e['score'] for e in candidates],bpm,
            np.load(PRIOR/f'{stem}_features.npy'),max(0,song['start']-5))
        row=dict(song=song,bpm=bpm,times=times,acoustic=torch.from_numpy(acoustic),placement=torch.from_numpy(placement),
                 target=target,mask=mask,onset_agreement=agreement,unmodelled_reference_midpoints=mids)
        rows.append(row)
        dump(OUT/f'{stem}_targets.json',dict(song=song,candidates=candidates,objects=objects,
             matched_type_labels=int(mask.sum()),unmatched_candidates=int((~mask).sum()),unmodelled_midpoints=mids))
        print(f'{stem}: GenéLive {len(times)} heads; {int(mask.sum())} aligned object sets',flush=True)
    return rows


def previous_actions(target):
    first=empty_action()
    return {k:torch.cat([first[k],target[k][:-1]]) for k in target}
