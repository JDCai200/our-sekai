"""Small reproducible training pilot; frozen original backbone, new PJSK I/O."""
import argparse,json,hashlib,time,random
from pathlib import Path
from collections import Counter
import numpy as np
import torch
import librosa
from engine import decode
from .schema import parse_position_free,complete_window,Codec
from .model import ROOT,WEIGHTS,load_base,PositionFreeAdapter

OUT=ROOT/'diagnostics/mapper_pjsk'

def dump(path,data):path.write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def prepare(codec):
    OUT.mkdir(parents=True,exist_ok=True)
    manifest=json.loads((ROOT/'diagnostics/genelive_finetune_conservative/manifest.json').read_text(encoding='utf-8'))
    rows=[];counts=Counter();excluded=0
    for song in manifest['rows']:
        i=song['id'];text=(ROOT/'reference'/f'{i:04d}_master.sus').read_text(encoding='utf-8-sig')
        score=parse_position_free(text,song['filler']);dump(OUT/f'{i:04d}_reference.json',score)
        counts.update(e['kind'] for e in score['events'])
        # Prespecified excerpts/windows; retain previous song-disjoint split.
        for offset in [0.,8.,16.]:
            start=song['start']+offset;events=complete_window(score['events'],start,codec.duration)
            excluded+=sum(start<=e['time']<start+codec.duration for e in score['events'])-len(events)
            tokens=codec.encode(events);decoded=codec.decode(tokens)
            assert len(events)==len(decoded)
            for a,b in zip(events,decoded):
                assert all(a[k]==b[k] for k in codec.groups)
                assert abs(a['time']-b['time'])<=.005001
            rows.append(dict(song=song,start=start,events=events,tokens=tokens,
                             feature=str(OUT/f'{i:04d}_{int(offset)}_{codec.duration:g}s_encoder.npy')))
    dump(OUT/'dataset.json',dict(rows=rows,kind_counts_full_charts=dict(counts),
         boundary_chain_events_excluded=excluded,roundtrip='Exact attributes/counts/links; time quantization <=5ms',
         positions_in_targets=False,widths_in_targets=False))
    return rows

def main():
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=3);p.add_argument('--duration',type=float,default=4)
    p.add_argument('--kind-balanced',action='store_true');p.add_argument('--kind-weight',type=float,default=1.)
    p.add_argument('--time-weight',type=float,default=1.)
    args=p.parse_args()
    if args.epochs<1 or not 0<args.duration<=16 or args.kind_weight<=0 or args.time_weight<0:raise ValueError('Invalid training options')
    torch.set_num_threads(2);torch.manual_seed(20261002);random.seed(20261002)
    codec=Codec(args.duration);rows=prepare(codec)
    print('Loading verified pretrained encoder and decoder',flush=True)
    base=load_base();model=PositionFreeAdapter(base,codec)
    original_sha=json.loads((WEIGHTS/'provenance.json').read_text())['files'][-1]['sha256']
    for row in rows:
        file=Path(row['feature'])
        if not file.exists():
            song=row['song'];wave=decode(ROOT/'reference'/f'{song["id"]:04d}_{song["bundle"]}.mp3',row['start'],args.duration).mean(axis=0)
            wave=librosa.resample(wave,orig_sr=44100,target_sr=16000)
            enc=model.encode_audio(torch.from_numpy(wave.astype(np.float32))).cpu().numpy()
            np.save(file,enc)
            print(f'Cached {song["id"]} {row["start"]}',flush=True)
    # All baseline parameters remain frozen. New embeddings/output are trained.
    params=list(model.embedding.parameters())+list(model.output.parameters())
    optimizer=torch.optim.AdamW(params,lr=.0003,weight_decay=.01)
    train=[r for r in rows if r['song']['split']=='train'];val=[r for r in rows if r['song']['split']=='validation'];test=[r for r in rows if r['song']['split']=='test']
    model.trained_kinds={e['kind'] for r in train for e in r['events']}
    kind_counts=Counter(e['kind'] for r in train for e in r['events'])
    class_weights=torch.ones(len(codec.names))
    if args.kind_balanced:
        inverse={k:1/np.sqrt(v) for k,v in kind_counts.items()};normalizer=np.mean(list(inverse.values()))
        for k,v in inverse.items():class_weights[codec.groups['kind'][k]]=v/normalizer
    def batch(row):
        return torch.from_numpy(np.load(row['feature'])),torch.tensor([row['tokens']],dtype=torch.long)
    def evaluate(selected):
        losses=[];correct=total=0;fields=Counter();matches=Counter();truth=Counter();guess=Counter();hit=Counter()
        with torch.no_grad():
            for row in selected:
                enc,tokens=batch(row);logits=model(enc,tokens[:,:-1]);target=tokens[:,1:]
                losses.append(float(torch.nn.functional.cross_entropy(logits.transpose(1,2),target)))
                pred=logits.argmax(-1);correct+=int((pred==target).sum());total+=target.numel()
                for index in range(1,target.shape[1]-1,6):
                    a=int(target[0,index]);b=int(pred[0,index]);truth[a]+=1;guess[b]+=1
                    if a==b:hit[a]+=1
                for index in range(target.shape[1]-1):
                    field=['time']+list(codec.groups);f=field[index%6];fields[f]+=1;matches[f]+=int(pred[0,index]==target[0,index])
        f1s=[2*hit[k]/(truth[k]+guess[k]) for k in truth]
        inverse_kind={v:k for k,v in codec.groups['kind'].items()}
        return dict(loss=float(np.mean(losses)),teacher_forced_token_accuracy=correct/total,
                    kind_macro_f1=float(np.mean(f1s)),kind_confusion_counts={group:{inverse_kind[k]:v for k,v in values.items()} for group,values in [('reference',truth),('predicted',guess),('matched',hit)]},
                    field_accuracy={k:matches[k]/v for k,v in fields.items()},tokens=total,
                    warning='Reference times and preceding reference events supplied; NOT free-generation quality or onset recall.')
    history=[];initial=evaluate(val);best=float('inf')
    print('INITIAL '+json.dumps(initial),flush=True)
    for epoch in range(1,args.epochs+1):
        random.shuffle(train);losses=[]
        for row in train:
            enc,tokens=batch(row);optimizer.zero_grad();logits=model(enc,tokens[:,:-1])
            per_token=torch.nn.functional.cross_entropy(logits.transpose(1,2),tokens[:,1:],weight=class_weights,reduction='none')
            field_weights=torch.ones(tokens.shape[1]-1);field_weights[0::6]=args.time_weight;field_weights[1::6]=args.kind_weight
            loss=(per_token*field_weights).sum()/field_weights.sum();loss.backward()
            torch.nn.utils.clip_grad_norm_(params,1.);optimizer.step();losses.append(float(loss.detach()))
        metric=evaluate(val);history.append(dict(epoch=epoch,training_loss=float(np.mean(losses)),validation=metric))
        print(json.dumps(history[-1]),flush=True)
        selection=-metric['kind_macro_f1'] if args.kind_balanced else metric['loss']
        if selection<best:
            best=selection;torch.save(model.adapter_state(),OUT/'adapter.pth')
    model.restore(torch.load(OUT/'adapter.pth',weights_only=True,map_location='cpu'))
    final_test=evaluate(test)
    generated=[]
    for row in test[:1]:
        enc,_=batch(row);result=model.generate_events(enc,max_events=35)
        result.update(audio=row['song']['bundle'],start=row['start'],duration=args.duration)
        dump(OUT/'generated_test.json',result);generated.append(dict(song=row['song']['title'],count=len(result['events']),terminated=result['terminated'],removed=result['unfinished_chain_events_removed']))
    report=dict(model='Mapperatorinator-v32 position-free PJSK I/O adapter',pretrained_sha256=original_sha,
        trainable_parameters=sum(p.numel() for p in params),frozen_encoder_and_decoder=True,
        epochs=args.epochs,training_options=vars(args),initial_validation=initial,history=history,test=final_test,
        free_generation=generated,train_songs=8,validation_songs=4,test_songs=2,
        adapter=str(OUT/'adapter.pth'),production_changed=False,
        limitations=['14 early Master charts, short excerpts; not enough to validate chart quality.',
                     'Only complete slides inside each crop used; no cross-window continuation learning.',
                     'No positions, widths or guide decorations; not playable in OpenSekai yet.',
                     'Teacher-forced accuracy is not onset recall; free-generation must be evaluated separately.',
                     'Trace/easing coverage depends on training samples; unobserved types are not learned.'])
    dump(OUT/'report.json',report);print(json.dumps(report,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
