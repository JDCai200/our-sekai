"""Transfer AutoOsu's NS memory; independent PJSK attribute heads."""
import copy
import torch
from torch import nn
import numpy as np
from autoosu_pjsk.backend import TimingModel

TYPES = ['tap', 'flick', 'slide', 'trace']
DIRECTIONS = ['none', 'up', 'left', 'right']
SLOTS = 4


class ObjectSelector(nn.Module):
    def __init__(self, upstream,placement_size=772):
        super().__init__()
        # Input remains 448-dimensional; replace the 32-dimensional NP context
        # by frozen GenéLive audio context, probability and inter-onset gaps.
        self.placement_size=placement_size
        self.placement = nn.Sequential(nn.LayerNorm(placement_size),nn.Linear(placement_size,32),nn.GELU())
        self.memory = copy.deepcopy(upstream.ns_gru)
        self.hidden_projection = copy.deepcopy(upstream.ns_proj_1)
        for p in self.memory.parameters():p.requires_grad_(True)
        for p in self.hidden_projection.parameters():p.requires_grad_(True)
        self.previous_count = nn.Embedding(SLOTS+1,32)
        self.previous_type = nn.Embedding(len(TYPES)+1,32,padding_idx=len(TYPES))
        self.previous_critical = nn.Embedding(2,32)
        self.previous_direction = nn.Embedding(len(DIRECTIONS),32)
        self.previous_span = nn.Linear(1,32,bias=False)
        self.count = nn.Linear(256,SLOTS)
        self.kind = nn.Linear(256,SLOTS*len(TYPES))
        self.critical = nn.Linear(256,SLOTS*2)
        self.direction = nn.Linear(256,SLOTS*len(DIRECTIONS))
        self.tail_direction = nn.Linear(256,SLOTS*len(DIRECTIONS))
        self.span = nn.Linear(256,SLOTS)
        self.allowed_counts = [1,2]
        self.allowed_types = [0,1,2]
        with torch.no_grad():
            self.previous_count.weight.zero_()
            self.previous_count.weight[0].copy_(upstream.action_emb.weight[0])
            for i in range(1,SLOTS+1):
                token=sum(4**k for k in range(i))
                self.previous_count.weight[i].copy_(upstream.action_emb.weight[token])
            self.previous_type.weight.zero_()
            # Semantic initialization only; no horizontal 4K columns in targets.
            for i,token in enumerate((1,1,2,1)):
                self.previous_type.weight[i].copy_((upstream.action_emb.weight[token]-upstream.action_emb.weight[0])*.1)
            self.previous_critical.weight.zero_();self.previous_direction.weight.zero_()
            self.previous_span.weight.zero_()
            self.span.bias.fill_(np.log1p(1.))

    def action_embedding(self, actions):
        count=actions['count'];kind=actions['kind']
        mask=torch.arange(SLOTS,device=count.device)[None,:]<count[...,None]
        embedding=self.previous_count(count)
        attributes=self.previous_type(kind)+self.previous_critical(actions['critical'])+self.previous_direction(actions['direction'])
        attributes=attributes+self.previous_span(actions['span'][...,None])
        return embedding+(attributes*mask[...,None]).sum(-2)/count.clamp_min(1)[...,None]

    def forward(self, acoustic, placement, previous, hidden=None):
        action=self.action_embedding(previous)
        x=torch.cat([acoustic,self.placement(placement),action],dim=-1)[None]
        out,hidden=self.memory(x,hidden)
        h=torch.nn.functional.gelu(self.hidden_projection(out[0]))
        return dict(count=self.count(h),kind=self.kind(h).reshape(-1,SLOTS,len(TYPES)),
                    critical=self.critical(h).reshape(-1,SLOTS,2),
                    direction=self.direction(h).reshape(-1,SLOTS,len(DIRECTIONS)),
                    tail_direction=self.tail_direction(h).reshape(-1,SLOTS,len(DIRECTIONS)),
                    span=torch.nn.functional.softplus(self.span(h))),hidden

    def decode(self, logits):
        allowed=[c-1 for c in self.allowed_counts]
        ci=logits['count'][:,allowed].argmax(-1)
        count=torch.tensor(self.allowed_counts,device=ci.device)[ci]
        kinds=logits['kind'][...,self.allowed_types].argmax(-1)
        kind=torch.tensor(self.allowed_types,device=ci.device)[kinds]
        direction=logits['direction'].argmax(-1)
        flick=kind==TYPES.index('flick')
        direction=torch.where(flick,logits['direction'][...,1:].argmax(-1)+1,torch.zeros_like(direction))
        return dict(count=count,kind=kind,critical=logits['critical'].argmax(-1),direction=direction,
                    tail_direction=logits['tail_direction'].argmax(-1),span=logits['span'])

    def save_state(self):
        return dict(state_dict=self.state_dict(),allowed_counts=self.allowed_counts,allowed_types=self.allowed_types,
                    schema='genelive-autoosu-object-selector-v1',slots=SLOTS,placement_size=self.placement_size)

    def restore(self,state):
        self.load_state_dict(state['state_dict'],strict=True)
        self.allowed_counts=state['allowed_counts'];self.allowed_types=state['allowed_types']


def empty_action(n=1):
    return dict(count=torch.zeros(n,dtype=torch.long),kind=torch.full((n,SLOTS),len(TYPES),dtype=torch.long),
                critical=torch.zeros((n,SLOTS),dtype=torch.long),direction=torch.zeros((n,SLOTS),dtype=torch.long),
                tail_direction=torch.zeros((n,SLOTS),dtype=torch.long),span=torch.zeros((n,SLOTS)))


@torch.inference_mode()
def acoustic_features(timing_model, specs, times, start, bpm, offset, condition=3.):
    """Use CNN/beat features only: never call AutoOsu NP/NS selection here."""
    base=timing_model.model
    conv=base.stack(specs[None]).permute(0,2,1,3).flatten(2,3)[0]
    ids=torch.tensor([min(max(round((t-start)*100),0),len(conv)-1) for t in times],dtype=torch.long)
    beat=(torch.tensor(times,dtype=torch.float64)-offset)*bpm/60
    phases=((beat%1)/round(1/48,5)).round().long();nums=(beat%4).floor().long()
    difficulty=base.difficulty_proj(torch.full((len(times),1),condition))
    # Exactly the original 320+32+16+16 static NS inputs; NP feature is replaced.
    return torch.cat([conv[ids],base.beat_phase_emb(phases),base.beat_num_emb(nums),difficulty],dim=-1)


@torch.inference_mode()
def free_select(model,acoustic,placement):
    previous=empty_action();hidden=None;out=[]
    for i in range(len(acoustic)):
        logits,hidden=model(acoustic[i:i+1],placement[i:i+1],previous,hidden)
        action=model.decode(logits);out.append(action);previous=action
    return {k:torch.cat([a[k] for a in out]) for k in out[0]} if out else empty_action(0)
