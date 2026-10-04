"""Song-disjoint training of normalized hold path families and selector."""
import base64
import json
from pathlib import Path
import sys
import zlib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'backend'))
from hybrid_pjsk.expression import rhythm_features
OUT=ROOT/'backend/diagnostics/pjsk_patterns_v3'


def prepare(chunks):
    xs=[]; shapes=[]; records=[]
    for chunk in chunks:
        seq=chunk['sequence']; times=[g['beat'] for g in seq]
        spans=[max(o['span'] for o in g['objects']) for g in seq]; counts=[len(g['objects']) for g in seq]
        for n,g in enumerate(seq):
            for o in g['objects']:
                if o['kind']!='slide':continue
                path=o['path']; u=[p['u'] for p in path]; center=[p['lane']+p['width']/2 for p in path]
                sign=1 if center[-1]>=center[0] else -1
                shape=np.r_[(np.interp(np.linspace(0,1,11),u,center)-center[0])*sign/12,
                    np.interp(np.linspace(0,1,11),u,[p['width'] for p in path])/12]
                row=rhythm_features(times,spans,counts,n,chunk['phase']+g['beat'],chunk['occupation'][0])
                row[2]=min(o['span'],16)/16
                xs.append(row); shapes.append(shape); records.append(dict(song=chunk['song'],path=path))
    return np.asarray(xs,np.float32),np.asarray(shapes),records


def compact(array):
    array=np.ascontiguousarray(array)
    return dict(dtype=array.dtype.str,shape=list(array.shape),zlib=base64.b64encode(zlib.compress(array.tobytes(),6)).decode('ascii'))


def main():
    data={k:prepare(json.loads((OUT/'corpus'/f'{k}_chunks.json').read_text(encoding='utf-8'))) for k in ('train','validation','test')}
    x,shapes,records=data['train']
    with threadpool_limits(limits=2):
        scaler=StandardScaler().fit(shapes); scaled=scaler.transform(shapes)
        cluster=KMeans(n_clusters=64,n_init=3,max_iter=150,random_state=20261004).fit(scaled)
        supported=[i for i in range(64) if sum(cluster.labels_==i)>=20 and len({records[j]['song'] for j in np.flatnonzero(cluster.labels_==i)})>=3]
        remap=np.array([supported.index(i if i in supported else min(supported,key=lambda j:np.linalg.norm(cluster.cluster_centers_[i]-cluster.cluster_centers_[j]))) for i in range(64)])
        labels=remap[cluster.labels_]; patterns=[]
        for label,raw in enumerate(supported):
            ids=np.flatnonzero(labels==label); nearest=sorted(ids,key=lambda i:np.linalg.norm(scaled[i]-cluster.cluster_centers_[raw]))
            chosen=[]; songs=set()
            for i in nearest:
                if records[i]['song'] not in songs:chosen.append(records[i]['path']);songs.add(records[i]['song'])
                if len(chosen)==3:break
            patterns.append(dict(members=len(ids),songs=len({records[i]['song'] for i in ids}),paths=chosen))
        forest=RandomForestClassifier(n_estimators=48,max_depth=14,max_leaf_nodes=256,min_samples_leaf=20,n_jobs=2,random_state=20261004).fit(x,labels)
    frequency=np.array([(labels==c).mean() for c in forest.classes_]); scores={}
    for split in ('validation','test'):
        tx,ts,unused=data[split]; targets=remap[cluster.predict(scaler.transform(ts))]
        probabilities=forest.predict_proba(tx)
        scores[split]=dict(slides=len(tx),log_loss=float(log_loss(targets,probabilities,labels=forest.classes_)),
            frequency_baseline_log_loss=float(log_loss(targets,np.tile(frequency,(len(tx),1)),labels=forest.classes_)))
    model=dict(schema=1,modes=len(patterns),patterns=patterns,forest=dict(classes=forest.classes_.tolist(),trees=[
        {key:compact(getattr(est.tree_,key)) for key in ('children_left','children_right','feature','threshold','value')} for est in forest.estimators_]))
    (OUT/'slide_paths.json').write_text(json.dumps(model,ensure_ascii=False),encoding='utf-8')
    (OUT/'slide-path-report.json').write_text(json.dumps(dict(modes=len(patterns),training_slides=len(x),scores=scores),indent=2),encoding='utf-8')
    print(json.dumps(dict(modes=len(patterns),training_slides=len(x),scores=scores)),flush=True)


if __name__=='__main__':main()
