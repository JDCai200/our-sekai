"""Reuse verified Mapperatorinator encoder AND decoder, replace position-free I/O."""
from pathlib import Path
import json,sys,os
os.environ.setdefault('USE_TF','0')
os.environ.setdefault('USE_FLAX','0')
from unittest.mock import patch
import torch
from torch import nn
from safetensors.torch import load_file
from .schema import Codec,KINDS,DIRECTIONS,STARTS,ENDS,validate

ROOT=Path(__file__).resolve().parents[1]
WEIGHTS=ROOT/'models/mapperatorinator-v32'

def load_base():
    sys.path.insert(0,str(ROOT/'models/mapperatorinator/osuT5'))
    from osuT5.model import Mapperatorinator
    from osuT5.model.configuration_mapperatorinator import MapperatorinatorConfig
    from osuT5.model.custom_transformers.configuration_varwhisper import VarWhisperConfig
    stored=json.loads((WEIGHTS/'config.json').read_text(encoding='utf-8'))
    # The saved backbone config is complete. Do not download unrelated Whisper weights/config.
    backbone=VarWhisperConfig(**stored['backbone_config'])
    with patch.object(VarWhisperConfig,'from_pretrained',return_value=backbone):
        cfg=MapperatorinatorConfig(**stored,vocab_size_out=stored['vocab_size'],src_seq_len=2048,tgt_seq_len=2560)
    cfg._attn_implementation='sdpa'
    if isinstance(cfg.backbone_config,dict):cfg.backbone_config['_attn_implementation']='sdpa'
    else:cfg.backbone_config._attn_implementation='sdpa'
    base=Mapperatorinator(cfg)
    base.load_state_dict(load_file(str(WEIGHTS/'model.safetensors')),strict=True)
    base.eval()
    for p in base.parameters():p.requires_grad_(False)
    return base

class PositionFreeAdapter(nn.Module):
    def __init__(self,base,codec=None):
        super().__init__();self.base=base;self.codec=codec or Codec()
        dim=base.config.hidden_size
        self.embedding=nn.Embedding(len(self.codec.names),dim)
        self.output=nn.Linear(dim,len(self.codec.names),bias=True)
        self.trained_kinds=set(KINDS)
        self._initialize()
    def _initialize(self):
        data=json.loads((WEIGHTS/'tokenizer.json').read_text(encoding='utf-8'))
        offset=data['offset'];ranges={}
        for r in data['event_ranges']:
            ranges[r['type']]=(offset,r['min_value'],r['max_value']);offset+=r['max_value']-r['min_value']+1
        old=self.base.decoder_embedder.weight;out=self.base.get_output_embeddings().weight
        aliases={'tap':'circle','flick':'circle','trace':'circle','slide_start':'hold_note',
                 'slide_start_hidden':'hold_note','slide_end':'hold_note_end','slide_end_hidden':'hold_note_end',
                 'slide_tick':'hold_note_sustain','slide_attach':'hold_note_sustain','slide_hidden':'hold_note_sustain'}
        with torch.no_grad():
            self.embedding.weight.normal_(std=.02);self.output.weight.normal_(std=.01);self.output.bias.zero_()
            for i,name in enumerate(self.codec.names):
                source=None
                if i<3:source=i
                elif name.startswith('TIME_'):source=ranges['t'][0]+int(name[5:])-ranges['t'][1]
                elif name.startswith('kind:'):source=ranges[aliases[name[5:]]][0]
                if source is not None:
                    self.embedding.weight[i].copy_(old[source]);self.output.weight[i].copy_(out[source])
    def forward(self,encoder,tokens):
        d=self.base.get_decoder()(inputs_embeds=self.embedding(tokens),encoder_hidden_states=encoder,
                                 use_cache=False,return_dict=True)
        logits=self.output(d.last_hidden_state)
        # The next-token field is known from the grammar. Do not make types,
        # booleans and time bins compete as if they were interchangeable labels.
        mask=torch.full((tokens.shape[1],len(self.codec.names)),float('-inf'),device=logits.device)
        fields=[list(range(3,self.codec.time_end))+[2]]+[list(g.values()) for g in self.codec.groups.values()]
        for phase,ids in enumerate(fields):
            mask[phase::6,ids]=0.
        return logits+mask
    def adapter_state(self):
        return dict(embedding=self.embedding.state_dict(),output=self.output.state_dict(),
                    duration=self.codec.duration,max_chains=self.codec.max_chains,trained_kinds=sorted(self.trained_kinds))
    def restore(self,state):
        self.embedding.load_state_dict(state['embedding']);self.output.load_state_dict(state['output'])
        self.trained_kinds=set(state.get('trained_kinds',KINDS))
    @torch.inference_mode()
    def encode_audio(self,wave):
        return self.base.get_encoder()(frames=wave[None],return_dict=True).last_hidden_state
    @torch.inference_mode()
    def generate_events(self,encoder,max_events=70,candidate_times=None):
        """Constrained autoregressive generation; never invent missing slide tails."""
        from transformers.cache_utils import EncoderDecoderCache,DynamicCache
        codec=self.codec;tokens=[1];past=EncoderDecoderCache(DynamicCache(),DynamicCache())
        active={};used=set();events=[];last_tick=0;simultaneous=0
        inverse={g:{v:k for k,v in ids.items()} for g,ids in codec.groups.items()}
        partial={};terminated=False
        schedule=None if candidate_times is None else [round(t*100) for t in candidate_times]
        if schedule is not None and (schedule!=sorted(schedule) or any(not 0<=t<codec.duration for t in candidate_times)):
            raise ValueError('Candidate times must be sorted and inside the audio window')
        for step in range(max_events*6+1):
            phase=step%6
            if phase==0:
                if schedule is not None and len(events)>=len(schedule):
                    terminated=not active;break
                lo=last_tick+(1 if simultaneous>=4 else 0)
                allowed=[3+schedule[len(events)]] if schedule is not None else list(range(3+lo,codec.time_end-1))
                if schedule is None and not active:allowed.append(2)
            elif phase==1:
                kinds=['tap','flick','trace']
                if len(used)<codec.max_chains and len(active)<4:kinds+=list(STARTS)
                if any(t<partial['tick'] for t in active.values()):
                    kinds+=['slide_tick','slide_attach','slide_hidden']+list(ENDS)
                allowed=[codec.groups['kind'][k] for k in kinds if k in self.trained_kinds]
            elif phase==2:allowed=list(codec.groups['critical'].values())
            elif phase==3:
                kind=partial['kind']
                values=[True] if kind=='trace' else [False] if kind in ('tap','flick') else [False,True]
                allowed=[codec.groups['trace'][v] for v in values]
            elif phase==4:
                kind=partial['kind'];directions=DIRECTIONS if kind in ('trace','slide_end') else DIRECTIONS[1:] if kind=='flick' else ['none']
                allowed=[codec.groups['direction'][d] for d in directions]
            else:
                kind=partial['kind']
                if kind in STARTS:refs=[next(c for c in range(codec.max_chains) if c not in used)]
                elif kind.startswith('slide_'):refs=[c for c,t in active.items() if t<partial['tick']]
                else:refs=[None]
                allowed=[codec.groups['chain'][c] for c in refs]
            if not allowed:break
            input_ids=torch.tensor([tokens[-1:]],dtype=torch.long,device=encoder.device)
            d=self.base.get_decoder()(inputs_embeds=self.embedding(input_ids),encoder_hidden_states=encoder,
                        past_key_values=past,use_cache=True,return_dict=True)
            past=d.past_key_values;logits=self.output(d.last_hidden_state[0,-1])
            token=allowed[int(logits[allowed].argmax())];tokens.append(token)
            if token==2:terminated=True;break
            if phase==0:
                tick=token-3;simultaneous=simultaneous+1 if tick==last_tick else 1;last_tick=tick
                partial=dict(time=tick/100,tick=tick)
            else:
                group=list(codec.groups)[phase-1];partial[group]=inverse[group][token]
                if phase==5:
                    e={k:v for k,v in partial.items() if k!='tick'};events.append(e)
                    if e['kind'] in STARTS:active[e['chain']]=partial['tick'];used.add(e['chain'])
                    elif e['kind'] in ENDS:del active[e['chain']]
        # Truncation is explicit. Remove whole unfinished chains rather than fabricate notes.
        removed=sum(e['chain'] in active for e in events)
        events=[e for e in events if e['chain'] not in active]
        validate(events)
        return dict(schema='pjsk-position-free-v1',time_reference='window_relative',events=events,positions_generated=False,widths_generated=False,
                    terminated=terminated,unfinished_chain_events_removed=removed,
                    time_mode='candidate_schedule' if schedule is not None else 'free_generation',
                    candidate_events_requested=len(schedule) if schedule is not None else None,
                    warnings=['Experimental PJSK adaptation; output has no playable horizontal layout.',
                              'Candidate schedule is quantized to 10ms; unfinished chains may be removed.' if schedule is not None else
                              'Times generated by Mapperatorinator adapter, not locked to the existing GenéLive detector.'])
