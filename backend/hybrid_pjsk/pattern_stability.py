"""Training/validation-only seed sensitivity audit; never selects on test."""
import json
import numpy as np
import joblib
from sklearn.metrics import adjusted_rand_score
from threadpoolctl import threadpool_limits
from .pattern_corpus import OUT,save
from .pattern_training_v2 import fit,evaluate,feature_vector
from .pattern_layout import canonical


def main():
    train=json.loads((OUT/'corpus/train_chunks.json').read_text(encoding='utf-8'))
    validation=json.loads((OUT/'corpus/validation_chunks.json').read_text(encoding='utf-8'))
    ids=sorted({c['song'] for c in validation});selected=set(ids[::max(1,len(ids)//12)])
    validation=[c for c in validation if c['song'] in selected]
    vectors=np.stack([feature_vector(canonical(c)[0]) for c in train]);records=[]
    with threadpool_limits(limits=2):
        for requested in (128,160):
            baseline=joblib.load(OUT/f'patterns_{requested}.joblib');alternate=fit(train,requested,2027)
            def labels(b):return b['cluster_remap'][b['clusterer'].predict(b['scaler'].transform(vectors)*b['feature_weights'])]
            metric=evaluate(validation,alternate)
            record=dict(requested_modes=requested,primary_seed=731,alternate_seed=2027,
                primary_modes=baseline['modes'],alternate_modes=alternate['modes'],
                training_cluster_ari=float(adjusted_rand_score(labels(baseline),labels(alternate))),
                alternate_validation=metric,training_seconds=alternate['training_seconds'])
            joblib.dump(alternate,OUT/f'patterns_{requested}_seed2027.joblib',compress=3)
            records.append(record);print(json.dumps(record),flush=True)
    save(OUT/'stability_report.json',dict(records=records,
        policy='Training-only clustering/RF; validation-only sensitivity check. Primary seed fixed to 731; secondary seed not used to choose deployment; test data never accessed.'))


if __name__=='__main__':main()
