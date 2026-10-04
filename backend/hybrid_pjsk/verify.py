"""Actual trained generation, whole-object validity, exact onset retention."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from autoosu_pjsk.backend import TimingModel,CHECKPOINT,SHA256
from autoosu_pjsk.objects import unpack_objects
from mapper_pjsk.schema import validate,parse_position_free
from mapper_pjsk.sus_export import export_preview
from .model import ObjectSelector,free_select,acoustic_features
from .data import OUT,prepare,dump
from .decode import to_objects


def main():
    torch.set_num_threads(2)
    base=TimingModel()
    def forbidden(*a,**kw):raise AssertionError('AutoOsu NP was called in hybrid generation')
    base.model.np_forward=forbidden
    initialized=ObjectSelector(base.model)
    for k,v in base.model.ns_gru.state_dict().items():
        assert torch.equal(v,initialized.memory.state_dict()[k])
    # Exercise the actual acoustic branch with NP explicitly forbidden.
    specs=torch.from_numpy(np.load(OUT.parent/'autoosu_pjsk/0001_features.npy'))
    acoustic_features(base,specs,[10.1,10.2],10.,150.,10.)
    selector_path=OUT/'object_selector.pth'
    state=torch.load(selector_path,map_location='cpu',weights_only=True)
    selector=ObjectSelector(base.model,state['placement_size']);selector.restore(state);selector.eval()
    rows=prepare(base);summaries=[]
    for row in rows:
        actions=free_select(selector,row['acoustic'],row['placement'])
        # A crop boundary is not an audio boundary; let complete objects extend
        # beyond excerpts. 1000s is only this structural test's container bound.
        objects,events,checks=to_objects(row['times'],actions,row['bpm'],1000.)
        validate(events);assert unpack_objects(objects)==events
        assert len({o['start'] for o in objects})==len(row['times'])
        _,sus=export_preview(events,row['bpm'])
        assert sus['round_trip_verified']
        summaries.append(dict(song=row['song']['id'],heads=len(row['times']),objects=len(objects),repairs=len(checks['repairs']),exact_head_times=True,sus_roundtrip=True))
    assert hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest()==SHA256
    changed=[k for k,v in base.model.ns_gru.state_dict().items() if not torch.equal(v,selector.memory.state_dict()[k])]
    report=dict(autoosu_np_not_called=True,pretrained_ns_initialization_exact=True,
                ns_gru_tensors_finetuned=changed,original_checkpoint_unchanged=True,
                selector_sha256=hashlib.sha256(selector_path.read_bytes()).hexdigest(),songs=summaries)
    dump(OUT/'verification.json',report)
    print(json.dumps(dict(verified_songs=len(summaries),total_fixed_heads=sum(s['heads'] for s in summaries),ns_changed_tensors=len(changed),selector_sha256=report['selector_sha256']),ensure_ascii=False))


if __name__=='__main__':main()
