"""Inference-only JSON random forest. No pickle or scikit-learn on mobile."""
import json
import base64
import zlib
from pathlib import Path
import numpy as np


class Forest:
    def __init__(self, data):
        self.classes_=np.asarray(data['classes'])
        self.trees=[]
        for tree in data['trees']:
            arrays={}
            for key,value in tree.items():
                if isinstance(value,dict) and 'zlib' in value:
                    arrays[key]=np.frombuffer(zlib.decompress(base64.b64decode(value['zlib'])),dtype=value['dtype']).reshape(value['shape'])
                else:arrays[key]=np.asarray(value)
            self.trees.append(arrays)

    def predict_proba(self, rows):
        # sklearn converts forest input to float32 before testing tree thresholds.
        rows=np.asarray(rows,dtype=np.float32)
        result=np.zeros((len(rows),len(self.classes_)),dtype=np.float64)
        for tree in self.trees:
            nodes=np.zeros(len(rows),dtype=np.int64)
            while True:
                active=tree['children_left'][nodes]!=-1
                if not np.any(active):break
                indices=np.flatnonzero(active); current=nodes[indices]
                left=rows[indices,tree['feature'][current].astype(int)]<=tree['threshold'][current]
                nodes[indices]=np.where(left,tree['children_left'][current],tree['children_right'][current])
            values=tree['value'][nodes,0,:]
            result+=values/np.maximum(values.sum(axis=1,keepdims=True),1e-30)
        return result/len(self.trees)


def load_bundle(path):
    bundle=json.loads(Path(path).read_text(encoding='utf-8'))
    if bundle['version'] not in (1,2):raise ValueError('Invalid portable vocabulary')
    bundle['selector']=Forest(bundle.pop('forest'))
    bundle['transitions']=np.asarray(bundle['transitions'],np.float64)
    bundle['_checkpoint_path']=str(Path(path).resolve())
    return bundle
