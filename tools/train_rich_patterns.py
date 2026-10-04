"""Train song-disjoint path vocabularies and rhythmic flick/critical forests."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'backend')]
from hybrid_pjsk.pattern_training_v2 import fit,evaluate
from hybrid_pjsk.expression import rhythm_features

OUT=ROOT/'backend/diagnostics/pjsk_patterns_v3'


def write(path,data):
    path.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False,indent=2),encoding='utf-8')


def rows(chunks):
    xs={'single':[],'slide':[]}; ys={'single':[],'slide':[]}
    for chunk in chunks:
        seq=chunk['sequence']; times=[g['beat'] for g in seq]
        spans=[max(o['span'] for o in g['objects']) for g in seq]; counts=[len(g['objects']) for g in seq]
        for n,g in enumerate(seq):
            feature=rhythm_features(times,spans,counts,n,chunk['phase']+g['beat'],chunk['occupation'][0])
            for o in g['objects']:
                if o['kind']=='trace':continue
                key='slide' if o['kind']=='slide' else 'single'
                direction=o['tail_direction'] if key=='slide' else o['direction'] if o['kind']=='flick' else 'none'
                label=('none','up','left','right').index(direction)*2+int(o['critical'])
                xs[key].append(feature); ys[key].append(label)
    return {k:(np.asarray(xs[k],np.float32),np.asarray(ys[k],np.int64)) for k in xs}


def forest_json(model):
    return dict(classes=model.classes_.tolist(),trees=[{key:getattr(est.tree_,key).tolist() for key in
        ('children_left','children_right','feature','threshold','value')} for est in model.estimators_])


def main():
    start=time.monotonic()
    splits={k:json.loads((OUT/'corpus'/f'{k}_chunks.json').read_text(encoding='utf-8')) for k in ('train','validation','test')}
    manifest=json.loads((OUT/'corpus/manifest.json').read_text(encoding='utf-8'))
    ids={k:{c['song'] for c in v} for k,v in splits.items()}
    assert not ids['train'] & (ids['validation']|ids['test']) and not ids['validation']&ids['test']
    train=splits['train']; search=[c for c in splits['validation'] if c['song'] in sorted(ids['validation'])[:12]]
    trials=[]; best=None
    with threadpool_limits(limits=2):
        for k in (192,256):
            bundle=fit(train,k,seed=20261004)
            score=evaluate(search,bundle)
            value=score['center_mae_lanes']+score['width_mae_lanes']
            trials.append(dict(requested=k,modes=bundle['modes'],validation=score))
            print(json.dumps(trials[-1]),flush=True)
            joblib.dump(bundle,OUT/f'patterns_{k}.joblib',compress=3)
            if bundle['modes']>=100 and (best is None or value<best[0]):best=(value,bundle)
        if best is None:raise ValueError('No supported candidate vocabulary')
        bundle=best[1]
        joblib.dump(bundle,OUT/'patterns.joblib',compress=3)
        validation=evaluate(splits['validation'],bundle)
        test=evaluate(splits['test'],bundle)
    data={k:rows(v) for k,v in splits.items()}; expression=dict(schema=1,training_songs=len(ids['train']),
        training_objects=sum(len(y) for x,y in data['train'].values()),feature_version='rhythm12-v1')
    metrics={}
    for key,(x,y) in data['train'].items():
        model=RandomForestClassifier(n_estimators=64,max_depth=12,max_leaf_nodes=256,min_samples_leaf=30,
            n_jobs=2,random_state=20261004).fit(x,y)
        expression[key]=forest_json(model); metrics[key]={}
        frequency=np.asarray([(y==c).mean() for c in model.classes_])
        for split in ('validation','test'):
            tx,ty=data[split][key]; p=model.predict_proba(tx)
            metrics[key][split]=dict(objects=len(ty),log_loss=float(log_loss(ty,p,labels=model.classes_)),
                frequency_baseline_log_loss=float(log_loss(ty,np.tile(frequency,(len(ty),1)),labels=model.classes_)),
                reference_label_counts=dict(Counter(map(int,ty))),mean_predicted_probabilities=p.mean(axis=0).tolist(),
                classes=model.classes_.tolist())
        print(key,json.dumps(metrics[key]),flush=True)
    write(OUT/'expression.json',expression)
    report=dict(dataset_splits=manifest['splits'],trials=trials,selected_modes=bundle['modes'],
        min_cluster_support=min(p['members'] for p in bundle['patterns']),min_source_songs=min(len(p['source_songs']) for p in bundle['patterns']),
        geometry_validation=validation,geometry_test=test,expression=metrics,
        seconds=time.monotonic()-start,policy='Song-disjoint splits; train-only scaler, clusters, medoids and forests; select K on validation only; chart rhythm supervision, no audio accent supervision')
    write(OUT/'training-report.json',report)
    print('TRAINING COMPLETE',json.dumps(dict(modes=bundle['modes'],seconds=report['seconds'])),flush=True)


if __name__=='__main__':main()
