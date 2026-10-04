"""Check all pretrained tensors remain unchanged and exercise the saved adapter."""
import json,hashlib
import numpy as np
import torch
from safetensors import safe_open
from .model import ROOT,WEIGHTS,load_base,PositionFreeAdapter
from .schema import Codec,validate

def main():
    torch.set_num_threads(2);out=ROOT/'diagnostics/mapper_pjsk'
    state=torch.load(out/'adapter.pth',map_location='cpu',weights_only=True)
    model=PositionFreeAdapter(load_base(),Codec(state['duration'],state['max_chains']));model.restore(state);model.eval()
    dataset=json.loads((out/'dataset.json').read_text(encoding='utf-8'))
    row=next(r for r in dataset['rows'] if r['song']['split']=='test')
    enc=torch.from_numpy(np.load(row['feature']));tokens=torch.tensor([row['tokens']],dtype=torch.long)
    with torch.no_grad():
        logits=model(enc,tokens[:,:-1]);loss=torch.nn.functional.cross_entropy(logits.transpose(1,2),tokens[:,1:])
    assert torch.isfinite(loss)
    # Compare every original tensor, not just requires_grad flags.
    current=model.base.state_dict();checked=0
    with safe_open(WEIGHTS/'model.safetensors',framework='pt',device='cpu') as original:
        for key in original.keys():
            assert torch.equal(original.get_tensor(key),current[key]),f'Pretrained tensor changed: {key}'
            checked+=1
    generated=json.loads((out/'generated_test.json').read_text(encoding='utf-8'));validate(generated['events'])
    # Exercise the separately supported existing-onset workflow with saved weights.
    sample=next(r for r in dataset['rows'] if r['song']['id']==1)
    candidates=json.loads((ROOT/'results/unipjsk_tell_your_world_genelive_tuned/analysis.json').read_text(encoding='utf-8'))
    schedule=sorted(e['time']-sample['start'] for e in candidates['events'] if e.get('selected',True) and sample['start']<=e['time']<sample['start']+state['duration'])
    locked=model.generate_events(torch.from_numpy(np.load(sample['feature'])),candidate_times=schedule)
    validate(locked['events'])
    quantized={round(t*100)/100 for t in schedule}
    assert all(e['time'] in quantized for e in locked['events'])
    locked.update(start=sample['start'],duration=state['duration'])
    (out/'typed_candidates_smoke.json').write_text(json.dumps(locked,ensure_ascii=False,indent=2),encoding='utf-8')
    assert not any(p.requires_grad for p in model.base.parameters())
    checks=dict(pretrained_tensors_unchanged=checked,saved_adapter_forward_loss=float(loss),
                generated_events=len(generated['events']),generated_structure_valid=True,
                candidate_schedule_requested=len(schedule),candidate_schedule_kept=len(locked['events']),
                checkpoint_sha256=hashlib.sha256((out/'adapter.pth').read_bytes()).hexdigest())
    (out/'verification.json').write_text(json.dumps(checks,indent=2),encoding='utf-8');print(json.dumps(checks))
if __name__=='__main__':main()
