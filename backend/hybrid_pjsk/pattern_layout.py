"""Train-only medoid pattern vocabulary and learned sequential selection.

Patterns contain object types and real slide geometry. In this first version
the supplied acoustic head types/spans are conditions, never overwritten.
No stems or inference-time reference chart are required.
"""
from copy import deepcopy
from functools import lru_cache
from collections import Counter
from pathlib import Path
import argparse,hashlib,json
import numpy as np
from mapper_pjsk.schema import parse_position_free
from .paths import ROOT,PRIOR,dump
from .phrase_layout import read_objects,groups_of,apply_phrases,corpus

OUT=ROOT/'diagnostics/pjsk_patterns'
TYPES=('tap','flick','slide','trace')
VERSION=1


def mirror_sequence(sequence):
    seq=deepcopy(sequence)
    for group in seq:
        for obj in group['objects']:
            for p in obj['path']:p['lane']=12-p['lane']-p['width']
        group['objects'].sort(key=lambda o:(o['kind'],o['path'][0]['lane']))
    return seq


def geometry_vector(sequence):
    rows=np.zeros((8,2,12),np.float64)
    for n,group in enumerate(sequence[:8]):
        for slot,obj in enumerate(sorted(group['objects'],key=lambda o:(o['kind'],o['path'][0]['lane']))[:2]):
            row=rows[n,slot];row[0]=1;row[1+TYPES.index(obj['kind'])]=1
            ps=obj['path'];u=[p['u'] for p in ps]
            for j,t in enumerate((0.,.5,1.)):
                lane=np.interp(t,u,[p['lane'] for p in ps]);width=np.interp(t,u,[p['width'] for p in ps])
                row[5+2*j]=(lane+width/2)/12;row[6+2*j]=width/12
            row[11]=min(obj['span'],8)/8
    return rows.reshape(-1)


def ending_state(sequence):
    obj=sequence[-1]['objects']
    centers=sorted((o['path'][-1]['lane']+o['path'][-1]['width']/2)/12 for o in obj)
    return [centers[0],centers[-1],float(np.mean([o['path'][-1]['width']/12 for o in obj]))]


def context_vector(sequence,previous_mode=-1,previous_state=None,modes=24,phase=0.,occupation=(0.,0.)):
    rows=np.zeros((8,8),np.float64);previous=0.
    for n,g in enumerate(sequence[:8]):
        rows[n,0]=1;rows[n,1]=min(g['beat']-previous,8)/8;previous=g['beat']
        rows[n,2]=len(g['objects'])/2
        for o in g['objects']:rows[n,3+TYPES.index(o['kind'])]+=.5
        rows[n,7]=min(max(o['span'] for o in g['objects']),8)/8
    history=np.zeros(modes*2+1);history[previous_mode if previous_mode>=0 else -1]=1
    return np.r_[rows.reshape(-1),previous_state or [.25,.75,.25],history,np.sin(phase*np.pi/2),np.cos(phase*np.pi/2),occupation]


def occupation_at(objects,time,bpm):
    active=[o for o in objects if o['start']<time and o.get('end',o['start'])>time]
    return (min(len(active),2)/2,min(max([o['end']-time for o in active],default=0)*bpm/60,8)/8)


def source_sequence(objects,part,bpm):
    t0=part[0][0];sequence=[]
    for time,ids in part:
        entries=[]
        for i in ids:
            o=objects[i];span=o.get('end',time)-time;points=o['points']
            path=[dict(u=(p['time']-time)/max(span,1e-6),lane=p.get('lane',0),width=p.get('width',3),
                       kind=p['kind'],critical=bool(p.get('critical')),direction=p.get('direction','none')) for p in points]
            entries.append(dict(kind='slide' if o['object_kind']=='slide' else points[0]['kind'],span=span*bpm/60,path=path,
                                critical=bool(points[0].get('critical')),direction=points[0].get('direction','none'),
                                tail_direction=points[-1].get('direction','none')))
        sequence.append(dict(beat=(time-t0)*bpm/60,objects=entries))
    return sequence


def official_chunks(song):
    text=(ROOT/'reference'/f'{song["id"]:04d}_master.sus').read_text(encoding='utf-8-sig')
    chart=parse_position_free(text,song['filler'],True);objects=read_objects(chart);groups=groups_of(objects)
    def beat(t):
        tempo=max((v for v in chart['tempos'] if v['time']<=t+1e-6),key=lambda v:v['time'],default=chart['tempos'][0])
        return tempo['beat']+(t-tempo['time'])*tempo['bpm']/60
    chunks=[]
    for a in range(0,len(groups),8):
        part=groups[a:a+8]
        if len(part)<3 or any(len(ids)>2 for _,ids in part):continue
        t0=part[0][0];tempo=max((v for v in chart['tempos'] if v['time']<=t0+1e-6),key=lambda v:v['time'],default=chart['tempos'][0])
        seq=source_sequence(objects,part,tempo['bpm'])
        for g,(t,_) in zip(seq,part):g['beat']=beat(t)-beat(t0)
        chunks.append(dict(song=song['id'],source_start=t0,phase=beat(t0),sequence=seq,occupation=occupation_at(objects,t0,tempo['bpm'])))
    return chunks


def canonical(chunk):
    seq=chunk['sequence'];first=seq[0]['objects']
    mirrored=np.mean([o['path'][0]['lane']+o['path'][0]['width']/2 for o in first])>6
    return mirror_sequence(seq) if mirrored else deepcopy(seq),int(mirrored)


def compatibility(actual,reference):
    if len(reference)<len(actual):return float('inf')
    score=0.
    for a,b in zip(actual,reference):
        score+=3*abs(len(a['objects'])-len(b['objects']))+.35*min(abs(a['beat']-b['beat']),4)
        ka=sorted(o['kind'] for o in a['objects']);kb=sorted(o['kind'] for o in b['objects'])
        score+=.7*sum(x!=y for x,y in zip(ka,kb))
    return score


class PatternChooser:
    def __init__(self,bundle):
        self.bundle=bundle;self.previous=-1;self.state=None;self.uses=Counter()
        self.candidates=[]
        for cluster,p in enumerate(bundle['patterns']):
            for variant_id,variant in enumerate(p['variants']):
                for mirrored in (False,True):
                    seq=mirror_sequence(variant['sequence']) if mirrored else variant['sequence']
                    self.candidates.append((cluster,cluster*2+int(mirrored),variant_id,variant,mirrored,seq))
        self.lengths=np.array([len(c[5]) for c in self.candidates])
        self.clusters=np.array([c[0] for c in self.candidates]);self.mode_ids=np.array([c[1] for c in self.candidates])
        self.first=np.array([ending_state(c[5][:1])[:2] for c in self.candidates])
        descriptors=[self.descriptor(c[5]) for c in self.candidates]
        self.beats=np.stack([d[0] for d in descriptors]);self.counts=np.stack([d[1] for d in descriptors]);self.kinds=np.stack([d[2] for d in descriptors])

    @staticmethod
    def descriptor(seq):
        beats=np.zeros(8);counts=np.zeros(8,dtype=int);kinds=np.full((8,2),-1,dtype=int)
        for n,g in enumerate(seq[:8]):
            beats[n]=g['beat'];counts[n]=len(g['objects'])
            kinds[n,:min(2,counts[n])]=[TYPES.index(o['kind']) for o in sorted(g['objects'],key=lambda o:o['kind'])[:2]]
        return beats,counts,kinds

    def choose(self,sequence,phase=0.,occupation=(0.,0.)):
        b=self.bundle;k=b['modes'];x=context_vector(sequence,self.previous,self.state,k,phase,occupation)
        probabilities=np.full(2*k,1e-6)
        probabilities[b['selector'].classes_.astype(int)]=b['selector'].predict_proba(x[None])[0]
        transition=b['transitions'][self.previous if self.previous>=0 else -1]
        beats,counts,kinds=self.descriptor(sequence);n=len(sequence)
        fits=(3*np.abs(self.counts[:,:n]-counts[:n])+.35*np.minimum(np.abs(self.beats[:,:n]-beats[:n]),4)).sum(axis=1)
        slots=np.arange(2)[None,None,:]<np.minimum(self.counts[:,:n],counts[:n])[...,None]
        fits+=.7*((self.kinds[:,:n]!=kinds[:n])&slots).sum(axis=(1,2))
        fits[self.lengths<n]=np.inf
        join=np.zeros(len(fits)) if self.state is None else np.abs(self.first-np.asarray(self.state[:2])).sum(axis=1)
        usage=np.array([.15*max(0,self.uses[i]-2) for i in range(k)])[self.clusters]
        scores=fits-.55*np.log(probabilities[self.mode_ids]+1e-6)-.12*np.log(transition[self.mode_ids])+join*.35+usage
        index=int(scores.argmin())
        if not np.isfinite(scores[index]):raise ValueError('No compatible motif length')
        cluster,mode,variant_id,variant,mirrored,seq=self.candidates[index];score=scores[index];fit=fits[index]
        self.previous=mode;self.state=ending_state(seq[:len(sequence)]);self.uses[cluster]+=1
        return variant,mirrored,dict(pattern=cluster,mode=mode,variant=variant_id,score=float(score),compatibility=float(fit),
                                    model_probability=float(probabilities[mode]),transition_probability=float(transition[mode]))

    def __call__(self,objects,part,bpm,offset):
        return self.choose(source_sequence(objects,part,bpm),(part[0][0]-offset)*bpm/60,occupation_at(objects,part[0][0],bpm))


@lru_cache(maxsize=4)
def load_patterns(model_path=None):
    path=Path(model_path) if model_path else OUT/'patterns.joblib'
    if path.suffix == '.json':
        from .portable_forest import load_bundle
        return load_bundle(path)
    import joblib
    active=OUT/'active_model.json'
    if model_path is None and active.is_file():
        config=json.loads(active.read_text(encoding='utf-8'))
        if config.get('version')!=2:raise ValueError('Invalid active vocabulary version')
        path=OUT.parent/'pjsk_patterns_v2/patterns.joblib'
        if hashlib.sha256(path.read_bytes()).hexdigest()!=config['sha256']:raise ValueError('Active vocabulary checksum mismatch')
    bundle=joblib.load(path)
    if bundle['version'] not in (1,2):raise ValueError('Incompatible pattern model; retrain it')
    bundle['_checkpoint_path']=str(path.resolve())
    return bundle


def apply_patterns(objects,bpm,offset,model_path=None):
    bundle=load_patterns(model_path);chooser=PatternChooser(bundle)
    result,report=apply_phrases(objects,bpm,offset,chooser=chooser)
    if bundle.get('rich_path_model'):
        from .slide_paths import apply_slide_paths
        path=Path(bundle['_checkpoint_path']).parent/'slide_paths.json'
        result,paths=apply_slide_paths(result,bpm,offset,path)
        report['hold_path_model']=paths
    report.update(method='Train-only KMeans medoid vocabulary + RandomForest conditional selection + learned transitions',
                  training_song_ids=bundle['training_song_ids'],
                  modes=bundle['modes'],pattern_usage=dict(chooser.uses),
                  checkpoint_sha256=hashlib.sha256((Path(bundle['_checkpoint_path']) if '_checkpoint_path' in bundle else OUT/'patterns.joblib').read_bytes()).hexdigest(),
                  vocabulary_version=bundle.get('version',1),
                  training_chunks=bundle.get('training_chunks',527),
                  selection_policy=bundle.get('selection_policy','Cached mirrored representatives, vectorized complete candidate scoring'),
                  type_policy='Existing NS object types, counts and durations retained; mode selection conditioned on these types.',
                  limitations=['Vocabulary quality depends on corpus coverage; independent feasibility checks are not a blind human playtest.',
                               'Selector uses rhythm, object types and previous motif context; no audio similarity controller.',
                               'Pattern selection precedes independent constraints; repairs may alter the selected motif.'])
    return result,report


def assign_mode(chunk,bundle):
    seq,mirrored=canonical(chunk)
    if bundle['version']==2:
        from .pattern_training_v2 import feature_vector
        vector=feature_vector(seq);scaled=bundle['scaler'].transform(vector[None])*bundle['feature_weights']
    else:scaled=bundle['scaler'].transform(geometry_vector(seq)[None])
    cluster=int(bundle['clusterer'].predict(scaled)[0])
    if bundle['version']==2:cluster=int(bundle['cluster_remap'][cluster])
    return cluster*2+mirrored


def fit(train,modes):
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    seqs=[canonical(c)[0] for c in train];vectors=np.stack([geometry_vector(s) for s in seqs])
    scaler=StandardScaler().fit(vectors);scaled=scaler.transform(vectors)
    clusterer=KMeans(n_clusters=modes,n_init=10,random_state=731).fit(scaled)
    patterns=[]
    for label in range(modes):
        ids=np.flatnonzero(clusterer.labels_==label)
        nearest=sorted(ids,key=lambda i:np.linalg.norm(scaled[i]-clusterer.cluster_centers_[label]))[:3]
        variants=[dict(song=train[i]['song'],source_start=train[i]['source_start'],sequence=seqs[i]) for i in nearest]
        patterns.append(dict(id=label,members=len(ids),source_songs=sorted({train[i]['song'] for i in ids}),variants=variants))
    bundle=dict(version=VERSION,modes=modes,scaler=scaler,clusterer=clusterer,patterns=patterns)
    xs=[];ys=[];transitions=np.ones((2*modes+1,2*modes),np.float64)
    previous=-1;state=None;song=None
    for chunk in train:
        if chunk['song']!=song:previous=-1;state=None;song=chunk['song']
        mode=assign_mode(chunk,bundle)
        xs.append(context_vector(chunk['sequence'],previous,state,modes,chunk['phase'],chunk['occupation']));ys.append(mode)
        transitions[previous if previous>=0 else -1,mode]+=1
        previous=mode;state=ending_state(chunk['sequence'])
    selector=RandomForestClassifier(n_estimators=160,max_depth=12,min_samples_leaf=3,class_weight='balanced_subsample',n_jobs=2,random_state=731)
    selector.fit(xs,ys);bundle.update(selector=selector,transitions=transitions/transitions.sum(axis=1,keepdims=True))
    return bundle


def evaluate(chunks,bundle,method='patterns'):
    chooser=None;song=None;errors=[];type_errors=[];used=[];classified=[]
    library=corpus()['templates']
    for chunk in chunks:
        if song!=chunk['song']:chooser=PatternChooser(bundle);song=chunk['song']
        actual=chunk['sequence']
        if method=='patterns':
            variant,mirrored,record=chooser.choose(actual,chunk['phase'],chunk['occupation']);used.append(record['pattern'])
            classified.append(record['mode']==assign_mode(chunk,bundle))
        else:
            variant=min(library,key=lambda t:compatibility(actual,t['sequence']))
            mirrored=bool(int(chunk['phase']//8)%2)
        predicted=mirror_sequence(variant['sequence']) if mirrored else variant['sequence']
        for a,b in zip(actual,predicted):
            refs=list(b['objects'])
            for obj in sorted(a['objects'],key=lambda o:o['kind']):
                match=next((i for i,r in enumerate(refs) if r['kind']==obj['kind']),0)
                ref=refs.pop(match) if refs else b['objects'][0]
                type_errors.append(obj['kind']!=ref['kind'])
                for u in ((0.,.5,1.) if obj['kind']=='slide' else (0.,)):
                    def edge(o):
                        ps=o['path'];width=np.interp(u,[p['u'] for p in ps],[p['width'] for p in ps]);lane=np.interp(u,[p['u'] for p in ps],[p['lane'] for p in ps])
                        return lane+width/2,width
                    ac,aw=edge(obj);bc,bw=edge(ref);errors.append((abs(ac-bc),abs(aw-bw)))
    arr=np.asarray(errors)
    return dict(fragments=len(chunks),geometry_samples=len(arr),center_mae_lanes=float(arr[:,0].mean()),width_mae_lanes=float(arr[:,1].mean()),
                type_compatibility=float(1-np.mean(type_errors)),used_modes=len(set(used)),
                mode_exact=float(np.mean(classified)) if classified else None,
                interpretation='Geometry preference error BEFORE constraints, on fixed official types/times; not playability or human quality.')


def main():
    import joblib
    manifest=json.loads((PRIOR/'manifest.json').read_text(encoding='utf-8'))
    split={name:[c for s in manifest['rows'] if s['split']==name for c in official_chunks(s)] for name in ('train','validation','test')}
    OUT.mkdir(parents=True,exist_ok=True);trials=[];best=None
    for modes in (12,24):
        bundle=fit(split['train'],modes);validation=evaluate(split['validation'],bundle)
        score=validation['center_mae_lanes']+validation['width_mae_lanes']
        trials.append(dict(modes=modes,validation=validation))
        print(json.dumps(trials[-1]),flush=True)
        if best is None or score<best[0]:best=(score,bundle)
    bundle=best[1];bundle['training_song_ids']=[s['id'] for s in manifest['rows'] if s['split']=='train']
    joblib.dump(bundle,OUT/'patterns.joblib')
    dump(OUT/'pattern_catalog.json',dict(modes=bundle['modes'],training_song_ids=bundle['training_song_ids'],patterns=bundle['patterns']))
    report=dict(version=VERSION,training_song_ids=bundle['training_song_ids'],splits={k:len(v) for k,v in split.items()},trials=trials,
                selected_modes=bundle['modes'],results={name:{method:evaluate(split[name],bundle,method) for method in ('patterns','retrieval')} for name in ('validation','test')},
                protocol='Split by song, nonoverlapping eight-head chunks; cluster/scaler/selector/transitions fitted only on train; K selected only on validation. Sequential prediction uses previously selected motifs, never held-out previous geometry.',
                source_hashes={str(s['id']):hashlib.sha256((ROOT/'reference'/f'{s["id"]:04d}_master.sus').read_bytes()).hexdigest() for s in manifest['rows']},
                limitations=['Only 14 previously inspected songs (8 train/4 validation/2 test); not a fresh blind benchmark.',
                             'Patterns condition on existing types/counts/spans; they do not learn acoustic type selection.',
                             'No paired audio feature supervision in the first selector.',
                             'Held-out official geometry is one stylistic choice; error is not a quality score.'])
    dump(OUT/'report.json',report);print(json.dumps(report['results']),flush=True)
    write_gallery(bundle)


def write_gallery(bundle):
    """Local, standalone visual audit of real representatives, no CDN."""
    cards=[]
    for pattern in bundle['patterns']:
        seq=pattern['variants'][0]['sequence'];end=max(g['beat']+o['span'] for g in seq for o in g['objects'])+.5
        scale=350/max(end,.5);marks=[]
        for lane in range(13):marks.append(f'<line x1="{lane*20+10}" x2="{lane*20+10}" y1="10" y2="375" stroke="#ddd"/>')
        for g in seq:
            for o in g['objects']:
                ps=o['path'];color={'tap':'#1bb6b3','flick':'#e57699','slide':'#65b465','trace':'#dcc052'}[o['kind']]
                y=lambda u:20+(g['beat']+o['span']*u)*scale
                if o['kind']=='slide':
                    left=[f'{10+p["lane"]*20},{y(p["u"]):.2f}' for p in ps]
                    right=[f'{10+(p["lane"]+p["width"])*20},{y(p["u"]):.2f}' for p in reversed(ps)]
                    marks.append(f'<polygon points="{" ".join(left+right)}" fill="{color}" opacity=".4"/>')
                p=ps[0];marks.append(f'<rect x="{10+p["lane"]*20}" y="{y(0):.2f}" width="{p["width"]*20}" height="6" fill="{color}"/>')
        kinds=Counter(o['kind'] for g in seq for o in g['objects']);widths=sorted({p['width'] for g in seq for o in g['objects'] for p in o['path']})
        cards.append(f'<article><h2>模式 {pattern["id"]}</h2><p>{pattern["members"]} 个训练片段；来源曲 {pattern["source_songs"]}</p><svg viewBox="0 0 270 390">{"".join(marks)}</svg><p>类型 {dict(kinds)}<br>宽度 {widths}<br>代表曲 {pattern["variants"][0]["song"]}，{pattern["variants"][0]["source_start"]:.2f}s</p></article>')
    (OUT/'pattern_gallery.html').write_text('<!doctype html><meta charset="utf-8"><title>紫谱键型模式库</title><style>body{font-family:Arial,"Microsoft YaHei",sans-serif;background:#f4f5f7;margin:24px}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:18px}article{background:white;padding:16px;border-radius:12px}svg{width:100%;height:390px}p{font-size:13px;line-height:1.6}</style><h1>紫谱键型模式库</h1><p>仅训练曲聚类。图示为真实代表片段的规范镜像，时间从上到下；青色 Tap、粉色 Flick、绿色长条、黄色 Trace。各卡片时间缩放不同。生成器可以选择同类的其他变体及镜像，实际输出仍需约束核验。</p><main>'+''.join(cards)+'</main>',encoding='utf-8')


if __name__=='__main__':main()
