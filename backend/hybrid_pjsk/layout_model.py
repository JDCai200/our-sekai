"""Autoregressive joint lane/width model supervised by full official SUS charts."""
from pathlib import Path
import argparse,json,random
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from mapper_pjsk.schema import parse_position_free,KINDS,DIRECTIONS
from .data import ROOT,PRIOR,dump

INTERVALS=[(lane,width) for lane in range(12) for width in range(1,13-lane)]
INDEX={pair:i for i,pair in enumerate(INTERVALS)}
OUT=ROOT/'diagnostics/pjsk_layout'


def features(events,bpm=143.,offset=0.,tempos=None):
    times=np.array([e['time'] for e in events]);n=len(times)
    if tempos:
        origin=np.array([t['time'] for t in tempos]);idx=np.searchsorted(origin,times,side='right')-1
        idx=np.maximum(idx,0)
        beats=np.array([tempos[i]['beat'] for i in idx])+(times-origin[idx])*np.array([tempos[i]['bpm'] for i in idx])/60
        speed=np.array([tempos[i]['bpm'] for i in idx])
    else:beats=(times-offset)*bpm/60;speed=np.full(n,bpm)
    before=np.r_[0,np.diff(times)];after=np.r_[np.diff(times),0]
    x=np.zeros((n,len(KINDS)+len(DIRECTIONS)+8),np.float32)
    for i,e in enumerate(events):
        x[i,KINDS.index(e['kind'])]=1;x[i,len(KINDS)+DIRECTIONS.index(e['direction'])]=1
        x[i,-8:]=[e['critical'],e['trace'],min(before[i]*speed[i]/60,8)/8,
                  min(after[i]*speed[i]/60,8)/8,np.sin(beats[i]*np.pi/2),np.cos(beats[i]*np.pi/2),
                  np.sin(beats[i]*np.pi*2),np.cos(beats[i]*np.pi*2)]
    return torch.from_numpy(x)


class LayoutModel(nn.Module):
    def __init__(self):
        super().__init__();self.previous=nn.Embedding(len(INTERVALS)+1,24)
        self.memory=nn.GRU(len(KINDS)+len(DIRECTIONS)+8+24,96,2,batch_first=True)
        self.output=nn.Linear(96,len(INTERVALS))

    def forward(self,x,previous,hidden=None):
        y,h=self.memory(torch.cat([x,self.previous(previous)],-1)[None],hidden)
        return self.output(y[0]),h

    @torch.inference_mode()
    def predict(self,x):
        if isinstance(x,np.ndarray):x=torch.from_numpy(x)
        self.eval();previous=torch.tensor([len(INTERVALS)]);hidden=None;out=[]
        for frame in x:
            logits,hidden=self(frame[None],previous,hidden);out.append(logits[0])
            previous=logits.argmax(-1)
        return torch.stack(out) if out else torch.zeros((0,len(INTERVALS)))


def load_layout(path=None):
    model=LayoutModel();state=torch.load(path or OUT/'layout_model.pth',map_location='cpu',weights_only=True)
    model.load_state_dict(state['state_dict'],strict=True);model.eval();return model


def evaluate(model,rows,baseline=None):
    correct=width_ok=0;center=[];width_error=[];total=0
    for row in rows:
        pred=model.predict(row['x']).argmax(-1).tolist() if baseline is None else [baseline]*len(row['y'])
        for a,b in zip(pred,row['y'].tolist()):
            la,wa=INTERVALS[a];lb,wb=INTERVALS[b]
            correct+=a==b;width_ok+=wa==wb;center.append(abs((la+wa/2)-(lb+wb/2)));width_error.append(abs(wa-wb));total+=1
    return dict(events=total,joint_exact=correct/max(total,1),width_exact=width_ok/max(total,1),
                center_mae_lanes=float(np.mean(center)),width_mae_lanes=float(np.mean(width_error)))


def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=24);args=p.parse_args()
    torch.set_num_threads(2);torch.manual_seed(731);random.seed(731)
    OUT.mkdir(parents=True,exist_ok=True);manifest=json.loads((PRIOR/'manifest.json').read_text(encoding='utf-8'))
    rows=[]
    for song in manifest['rows']:
        path=ROOT/'reference'/f'{song["id"]:04d}_master.sus'
        chart=parse_position_free(path.read_text(encoding='utf-8-sig'),song['filler'],include_geometry=True)
        events=chart['events'];y=torch.tensor([INDEX[(e['lane'],e['width'])] for e in events])
        rows.append(dict(song=song,x=features(events,tempos=chart['tempos']),y=y))
    train=[r for r in rows if r['song']['split']=='train'];val=[r for r in rows if r['song']['split']=='validation'];test=[r for r in rows if r['song']['split']=='test']
    counts=torch.bincount(torch.cat([r['y'] for r in train]),minlength=len(INTERVALS));baseline=int(counts.argmax())
    model=LayoutModel();optimizer=torch.optim.AdamW(model.parameters(),lr=.0008,weight_decay=.005)
    history=[];best=float('inf');chosen=0
    for epoch in range(1,args.epochs+1):
        random.shuffle(train);model.train();losses=[]
        for row in train:
            teacher=torch.cat([torch.tensor([len(INTERVALS)]),row['y'][:-1]])
            if epoch>=4:
                pred=model.predict(row['x']).argmax(-1)
                prev=torch.cat([torch.tensor([len(INTERVALS)]),pred[:-1]])
                teacher=torch.where(torch.rand(len(teacher))<min(.65,epoch/24),prev,teacher)
                model.train()
            hidden=None
            for a in range(0,len(teacher),192):
                optimizer.zero_grad();logits,hidden=model(row['x'][a:a+192],teacher[a:a+192],hidden)
                loss=F.cross_entropy(logits,row['y'][a:a+192]);loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step();hidden=hidden.detach();losses.append(float(loss.detach()))
        metric=evaluate(model,val);score=metric['center_mae_lanes']+metric['width_mae_lanes']
        if score<best:
            chosen=epoch;best=score;torch.save(dict(schema='pjsk-layout-gru-v1',state_dict=model.state_dict()),OUT/'layout_model.pth')
        history.append(dict(epoch=epoch,loss=float(np.mean(losses)),validation=metric))
        print(json.dumps(history[-1]),flush=True)
    model=load_layout()
    report=dict(training='Full official chart event sequences; previous predicted interval feedback up to 65%; validation center MAE + width MAE selection',
                source_songs=[dict(id=r['song']['id'],split=r['song']['split'],events=len(r['y'])) for r in rows],chosen_epoch=chosen,
                parameters=sum(p.numel() for p in model.parameters()),history=history,
                results={s:evaluate(model,rs) for s,rs in [('train',train),('validation',val),('test',test)]},
                baseline={s:evaluate(model,rs,baseline) for s,rs in [('validation',val),('test',test)]},
                baseline_interval=INTERVALS[baseline],limitations=['Previously inspected 14-song split; exploratory, not blind generalization.',
                'Positions are stylistic choices, so exact official geometry is not a unique correctness criterion.',
                'Event-type/time context only; no direct audio-to-position supervision.',
                'Raw model geometry has no playability guarantee; independent projection and witness checker required.'])
    dump(OUT/'report.json',report);print('FINAL '+json.dumps(report['results']),flush=True)


if __name__=='__main__':main()
