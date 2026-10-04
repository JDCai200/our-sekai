"""Update trained forests/paths; keep the unchanged parity-verified ONNX stages."""
import json
import base64
import zlib
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from hybrid_pjsk.pattern_layout import load_patterns


def plain(value):
    if isinstance(value,np.ndarray):return value.tolist()
    if isinstance(value,np.generic):return value.item()
    if isinstance(value,dict):return {key:plain(item) for key,item in value.items()}
    if isinstance(value,(tuple,list)):return [plain(item) for item in value]
    return value


def main():
    output=ROOT/'artifacts/portable-models'; bundle=load_patterns(); forest=bundle['selector']
    pattern={key:value for key,value in bundle.items() if key not in ('selector','_checkpoint_path','scaler','clusterer')}
    pattern['scaler']=dict(mean=bundle['scaler'].mean_,scale=bundle['scaler'].scale_)
    pattern['cluster_centers']=bundle['clusterer'].cluster_centers_
    def compact(array):
        array=np.ascontiguousarray(array)
        return dict(dtype=array.dtype.str,shape=list(array.shape),zlib=base64.b64encode(zlib.compress(array.tobytes(),6)).decode('ascii'))
    pattern['forest']=dict(classes=forest.classes_,trees=[{key:compact(getattr(est.tree_,key)) for key in
        ('children_left','children_right','feature','threshold','value')} for est in forest.estimators_])
    pattern['forest_encoding']='lossless-zlib-numpy-v1'
    (output/'patterns.json').write_text(json.dumps(plain(pattern),ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print('Portable path vocabulary updated:',bundle['modes'])


if __name__=='__main__':main()
