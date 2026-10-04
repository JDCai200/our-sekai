"""Larger, rhythm/path-aware motif vocabulary, trained only on training songs."""
import argparse,hashlib,json,time
from pathlib import Path
from collections import Counter
import numpy as np
import joblib
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from threadpoolctl import threadpool_limits
from .pattern_corpus import OUT,save
from .pattern_layout import canonical,context_vector,ending_state,mirror_sequence,PatternChooser,TYPES,load_patterns,write_gallery


def feature_vector(sequence):
    rows=np.zeros((8,2,22));rhythm=np.zeros((8,2));previous=0.
    for n,g in enumerate(sequence[:8]):
        rhythm[n]=[min(g['beat']-previous,8)/8,min(g['beat'],16)/16];previous=g['beat']
        for slot,o in enumerate(sorted(g['objects'],key=lambda o:(o['kind'],o['path'][0]['lane']))[:2]):
            r=rows[n,slot];r[0]=1;r[1+TYPES.index(o['kind'])]=1
            ps=o['path'];u=[p['u'] for p in ps]
            for j,t in enumerate(np.linspace(0,1,7)):
                lane=np.interp(t,u,[p['lane'] for p in ps]);width=np.interp(t,u,[p['width'] for p in ps])
                r[5+2*j]=(lane+width/2)/12;r[6+2*j]=width/12
            r[19]=min(o['span'],8)/8
            movement=np.diff([p['lane']+p['width']/2 for p in ps]);sign=np.sign(movement[np.abs(movement)>1e-6])
            r[20]=min(sum(a!=b for a,b in zip(sign,sign[1:])),6)/6
            r[21]=min(max(len(ps)-2,0),8)/8
    return np.r_[rows.reshape(-1),rhythm.reshape(-1)]


def feature_weights():
    row=np.r_[.5,np.full(4,1.1),np.full(14,.35),.7,.5,.5]
    return np.r_[np.tile(row,16),np.full(16,1.4)]


def fit(train,requested,seed=731):
    start=time.monotonic();seqs=[canonical(c)[0] for c in train]
    vectors=np.stack([feature_vector(s) for s in seqs]);scaler=StandardScaler().fit(vectors)
    weights=feature_weights();scaled=scaler.transform(vectors)*weights
    # StandardScaler and clustering see training only. Held-out songs never
    # influence pruning, representative choice, class weights or transitions.
    clusterer=KMeans(n_clusters=requested,n_init=3,random_state=seed,max_iter=150).fit(scaled)
    labels=clusterer.labels_;ids_by=[np.flatnonzero(labels==i) for i in range(requested)]
    stable=[i for i,ids in enumerate(ids_by) if len(ids)>=20 and len({train[j]['song'] for j in ids})>=3]
    if not stable:raise ValueError('No supported clusters')
    mapping=np.empty(requested,dtype=int)
    for i in range(requested):
        target=i if i in stable else min(stable,key=lambda j:np.linalg.norm(clusterer.cluster_centers_[i]-clusterer.cluster_centers_[j]))
        mapping[i]=stable.index(target)
    labels=mapping[labels];patterns=[]
    for label,raw in enumerate(stable):
        ids=np.flatnonzero(labels==label);nearest=sorted(ids,key=lambda i:np.linalg.norm(scaled[i]-clusterer.cluster_centers_[raw]))
        representatives=[];songs=set()
        for i in nearest:
            if train[i]['song'] not in songs:representatives.append(i);songs.add(train[i]['song'])
            if len(representatives)==3:break
        patterns.append(dict(id=label,members=len(ids),source_songs=sorted({train[i]['song'] for i in ids}),
            variants=[dict(song=train[i]['song'],source_start=train[i]['source_start'],sequence=seqs[i]) for i in representatives]))
    k=len(patterns);ys=[];xs=[];counts=np.zeros((2*k+1,2*k));previous=-1;state=None;song=None
    for n,chunk in enumerate(train):
        if chunk['song']!=song:previous=-1;state=None;song=chunk['song']
        mode=int(labels[n])*2+canonical(chunk)[1]
        xs.append(context_vector(chunk['sequence'],previous,state,k,chunk['phase'],chunk['occupation']));ys.append(mode)
        counts[previous if previous>=0 else -1,mode]+=1;previous=mode;state=ending_state(chunk['sequence'])
    frequencies=np.bincount(ys,minlength=2*k)+.1;prior=frequencies/frequencies.sum()
    transitions=(counts+8*prior[None])/(counts.sum(axis=1,keepdims=True)+8)
    selector=RandomForestClassifier(n_estimators=128,max_depth=14,max_leaf_nodes=192,min_samples_leaf=4,
        class_weight='balanced_subsample',n_jobs=2,random_state=seed).fit(xs,ys)
    selector.n_jobs=1
    return dict(version=2,feature_version='rhythm-path-368-v2',modes=k,requested_modes=requested,
        scaler=scaler,clusterer=clusterer,cluster_remap=mapping,feature_weights=weights,
        patterns=patterns,selector=selector,transitions=transitions,
        training_song_ids=sorted({c['song'] for c in train}),training_chunks=len(train),
        training_seconds=time.monotonic()-start,seed=seed,
        support_policy='Each surviving raw cluster has >=20 training motifs from >=3 songs; unsupported raw clusters merge to the nearest supported cluster; three representatives from different songs.',
        transition_policy='Eight observations of global training prior as shrinkage; no held-out transitions.',
        selection_policy='Cached representative mirrors; vectorized scoring of all compatible variants; no candidate pruning.')


def evaluate(chunks,bundle):
    chooser=None;song=None;errors=[];types=[];used=[];selected=[];knot_samples=0
    bundle['selector'].n_jobs=1
    for chunk in chunks:
        if song!=chunk['song']:chooser=PatternChooser(bundle);song=chunk['song']
        actual=chunk['sequence'];variant,mirrored,record=chooser.choose(actual,chunk['phase'],chunk['occupation'])
        predicted=mirror_sequence(variant['sequence']) if mirrored else variant['sequence'];used.append(record['pattern'])
        selected.append((record['pattern'],record['variant'],bool(mirrored)))
        for a,b in zip(actual,predicted):
            refs=list(b['objects'])
            for o in sorted(a['objects'],key=lambda o:o['kind']):
                match=next((i for i,r in enumerate(refs) if r['kind']==o['kind']),0)
                ref=refs.pop(match) if refs else b['objects'][0];types.append(o['kind']==ref['kind'])
                # Identical target-derived probes for every model: including
                # predicted knots would change the weighting between models.
                probes=sorted({0.,*np.linspace(0,1,7),*[p['u'] for p in o['path']]}) if o['kind']=='slide' else [0.]
                knot_samples+=len(probes)
                for u in probes:
                    def edge(x):
                        ps=x['path'];width=np.interp(u,[p['u'] for p in ps],[p['width'] for p in ps]);lane=np.interp(u,[p['u'] for p in ps],[p['lane'] for p in ps]);return lane+width/2,width
                    ac,aw=edge(o);bc,bw=edge(ref);errors.append((abs(ac-bc),abs(aw-bw)))
    arr=np.asarray(errors);counts=Counter(used);p=np.array(list(counts.values()),dtype=float)/len(used)
    return dict(fragments=len(chunks),center_mae_lanes=float(arr[:,0].mean()),width_mae_lanes=float(arr[:,1].mean()),
        type_compatibility=float(np.mean(types)),used_modes=len(counts),effective_modes=float(np.exp(-np.sum(p*np.log(p)))),
        maximum_mode_share=max(counts.values())/len(used),pattern_usage=dict(counts),unique_selected_variants=len(set(selected)),geometry_samples=knot_samples,
        interpretation='Sequential source geometry preference before constraints on fixed held-out official types/times; not onset accuracy, final hand feel or unique correct geometry.')


def main():
    args=argparse.ArgumentParser();args.add_argument('--modes',nargs='+',type=int,default=[100,128,160])
    args.add_argument('--evaluate-only',action='store_true',help='Re-evaluate existing fitted checkpoints without retraining');opts=args.parse_args()
    manifest=json.loads((OUT/'corpus/manifest.json').read_text(encoding='utf-8'))
    splits={s:json.loads((OUT/'corpus'/f'{s}_chunks.json').read_text(encoding='utf-8')) for s in ('train','validation','test')}
    # Class-count search uses a fixed song-disjoint validation subset. Final
    # winner and old baseline are evaluated on ALL validation/test fragments.
    ids=sorted({c['song'] for c in splits['validation']})
    search_ids=set(ids[::max(1,len(ids)//12)])
    search=[c for c in splits['validation'] if c['song'] in search_ids]
    trials=[];best=None
    with threadpool_limits(limits=2):
        for requested in opts.modes:
            b=joblib.load(OUT/f'patterns_{requested}.joblib') if opts.evaluate_only else fit(splits['train'],requested)
            score=evaluate(search,b)
            item=dict(requested_modes=requested,actual_modes=b['modes'],training_seconds=b['training_seconds'],validation_search=score,
                      supported_at_least_100=b['modes']>=100)
            trials.append(item);print(json.dumps(item),flush=True)
            if not opts.evaluate_only:joblib.dump(b,OUT/f'patterns_{requested}.joblib',compress=3)
            value=score['center_mae_lanes']+score['width_mae_lanes']+max(0,.78-score['type_compatibility'])*3
            if b['modes']>=100 and (best is None or value<best[0]):best=value,b
        if best is None:save(OUT/'trials.json',trials);raise ValueError('No vocabulary with at least 100 supported patterns; enlarge corpus before promotion')
        b=best[1];old=load_patterns(str(OUT.parent/'pjsk_patterns/patterns.joblib'))
        results={s:dict(expanded=evaluate(splits[s],b),baseline_24=evaluate(splits[s],old)) for s in ('validation','test')}
    # Export for review first; runtime activation is a separate verified step.
    destination=OUT/'patterns.joblib';joblib.dump(b,destination,compress=3)
    save(OUT/'pattern_catalog.json',dict(modes=b['modes'],patterns=b['patterns'],training_song_ids=b['training_song_ids']))
    report=dict(version=2,evaluation_version='target-fixed-probes-v2',dataset_splits=manifest['splits'],trials=trials,selected_modes=b['modes'],requested_modes=b['requested_modes'],
        results=results,model_bytes=destination.stat().st_size,model_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
        support=dict(minimum=min(p['members'] for p in b['patterns']),minimum_songs=min(len(p['source_songs']) for p in b['patterns'])),
        protocol='Song-disjoint splits; known development songs training only; scaler, vocabulary, pruning, representatives, RF and transitions fitted only on train; K selection on a fixed validation song subset; complete validation/test evaluated once after K selection.',
        limitations=['No audio semantic training or repeated melody controller.',
                     'Vocabulary conditions on NS types/counts/spans and does not overwrite them.',
                     'Independent motion/overlap projection can alter motifs; full pipeline regression is needed before activation.'])
    save(OUT/'report.json',report)
    from . import pattern_layout
    previous=pattern_layout.OUT
    try:pattern_layout.OUT=OUT;write_gallery(b)
    finally:pattern_layout.OUT=previous
    print(json.dumps(dict(selected_modes=b['modes'],results=results,model_mb=report['model_bytes']/1024**2)),flush=True)


if __name__=='__main__':main()
