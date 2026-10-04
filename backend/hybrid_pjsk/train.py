"""Small song-disjoint PJSK NS transfer pilot with fixed GenéLive heads."""
import argparse
from collections import Counter
import hashlib
import json
import random
import numpy as np
import torch
import torch.nn.functional as F
from autoosu_pjsk.backend import TimingModel,SHA256,COMMIT
from .model import ObjectSelector,TYPES,SLOTS,free_select,empty_action
from .data import OUT,PRIOR,prepare,previous_actions,dump


def type_metrics(reference,predicted,hits):
    kinds=sorted(reference)
    return dict(kind_macro_f1=float(np.mean([2*hits[k]/max(reference[k]+predicted[k],1) for k in kinds])),
                kind_counts=dict(reference),predicted_kind_counts=dict(predicted),matched_kind_counts=dict(hits))


def evaluate(model,rows,baseline=False):
    truth=Counter();guess=Counter();hit=Counter();correct_count=exact=total=0
    spans=[];direction_correct=direction_total=critical_correct=critical_total=0
    onset=dict(predicted=0,reference=0,matched=0)
    with torch.no_grad():
        for row in rows:
            for k in onset:onset[k]+=row['onset_agreement'][k]
            y=row['target'];p=free_select(model,row['acoustic'],row['placement']) if not baseline else None
            for n in row['mask'].nonzero().flatten().tolist():
                count=int(y['count'][n]);pred_count=int(p['count'][n]) if p is not None else 1
                total+=1;correct_count+=count==pred_count
                a=Counter(TYPES[int(k)] for k in y['kind'][n,:count])
                b=Counter(TYPES[int(k)] for k in p['kind'][n,:pred_count]) if p is not None else Counter(tap=1)
                truth.update(a);guess.update(b);hit.update(a&b);exact+=a==b
                if p is not None:
                    for slot in range(min(count,pred_count)):
                        if y['kind'][n,slot]==2:
                            sec_true=np.expm1(float(y['span'][n,slot]))*60/row['bpm']
                            sec_pred=np.expm1(min(float(p['span'][n,slot]),np.log1p(64)))*60/row['bpm']
                            spans.append(abs(sec_true-sec_pred))
                        if y['kind'][n,slot]==1:
                            direction_total+=1;direction_correct+=int(y['direction'][n,slot]==p['direction'][n,slot])
                        critical_total+=1;critical_correct+=int(y['critical'][n,slot]==p['critical'][n,slot])
    result=type_metrics(truth,guess,hit)
    onset.update(extra=onset['predicted']-onset['matched'],missed=onset['reference']-onset['matched'])
    onset.update(recall=onset['matched']/max(onset['reference'],1),extra_ratio=onset['extra']/max(onset['predicted'],1))
    result.update(count_accuracy=correct_count/max(total,1),kind_set_exact_accuracy=exact/max(total,1),matched_head_sets=total,
                  slide_span_mae_seconds=float(np.mean(spans)) if spans else None,
                  flick_direction_accuracy=None if baseline else direction_correct/max(direction_total,1),
                  critical_accuracy=None if baseline else critical_correct/max(critical_total,1),
                  slide_span_evaluated_slots=len(spans),genelive_onsets=onset,
                  metric_scope='Free autoregressive feedback; types evaluated only at matched GenéLive heads; span reads true-slide slots, not full-object accuracy.')
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--epochs',type=int,default=24)
    args=parser.parse_args()
    if args.epochs<1:raise ValueError('Epochs must be positive')
    torch.set_num_threads(2);torch.manual_seed(20261002);random.seed(20261002)
    timing=TimingModel();rows=prepare(timing);model=ObjectSelector(timing.model)
    train=[r for r in rows if r['song']['split']=='train'];val=[r for r in rows if r['song']['split']=='validation'];test=[r for r in rows if r['song']['split']=='test']
    counts=Counter();kinds=Counter()
    for r in train:
        for i in r['mask'].nonzero().flatten().tolist():
            counts[int(r['target']['count'][i])]+=1
            kinds.update(int(k) for k in r['target']['kind'][i,:r['target']['count'][i]])
    model.allowed_counts=sorted(counts);model.allowed_types=sorted(kinds)
    kind_weights=torch.ones(len(TYPES));count_weights=torch.ones(SLOTS)
    for k,v in kinds.items():kind_weights[k]=1/np.sqrt(v)
    for k,v in counts.items():count_weights[k-1]=1/np.sqrt(v)
    kind_weights/=kind_weights[model.allowed_types].mean();count_weights/=count_weights[[c-1 for c in model.allowed_counts]].mean()
    optimizer=torch.optim.AdamW(model.parameters(),lr=.00015,weight_decay=.001)
    baselines={split:evaluate(model,selected,baseline=True) for split,selected in [('validation',val),('test',test)]}
    history=[];best=-1.;chosen_epoch=0
    print('Training '+str(sum(p.numel() for p in model.parameters()))+' NS parameters; fixed GenéLive; no AutoOsu NP',flush=True)
    for epoch in range(1,args.epochs+1):
        random.shuffle(train);losses=[];model.train()
        for row in train:
            optimizer.zero_grad()
            previous=previous_actions(row['target'])
            if epoch>=3:
                predicted=free_select(model,row['acoustic'],row['placement'])
                predicted_previous={k:torch.cat([empty_action()[k],predicted[k][:-1]]) for k in predicted}
                choose=(torch.rand(len(row['times']))<min(.65,epoch/24))
                previous={k:torch.where(choose.reshape((-1,)+(1,)*(v.ndim-1)),predicted_previous[k],v) for k,v in previous.items()}
            logits,_=model(row['acoustic'],row['placement'],previous)
            y=row['target'];mask=row['mask']
            slot_mask=(torch.arange(SLOTS)[None,:]<y['count'][:,None])&mask[:,None]
            loss=F.cross_entropy(logits['count'][mask],y['count'][mask]-1)
            loss+=2*F.cross_entropy(logits['kind'][slot_mask],y['kind'][slot_mask],weight=kind_weights)
            loss+=.2*F.cross_entropy(logits['critical'][slot_mask],y['critical'][slot_mask])
            flick=slot_mask&(y['kind']==1);slide=slot_mask&(y['kind']==2)
            if flick.any():loss+=.5*F.cross_entropy(logits['direction'][flick],y['direction'][flick])
            if slide.any():
                loss+=.5*F.cross_entropy(logits['tail_direction'][slide],y['tail_direction'][slide])
                loss+=F.smooth_l1_loss(logits['span'][slide],y['span'][slide])
            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            losses.append(float(loss.detach()))
        model.eval();metric=evaluate(model,val)
        score=.4*metric['kind_macro_f1']+.4*metric['kind_set_exact_accuracy']+.2*metric['count_accuracy']
        if score>best:
            best=score;chosen_epoch=epoch;torch.save(model.save_state(),OUT/'object_selector.pth')
        history.append(dict(epoch=epoch,training_loss=float(np.mean(losses)),validation=metric))
        print(json.dumps(dict(epoch=epoch,loss=float(np.mean(losses)),kind_f1=metric['kind_macro_f1'],count_accuracy=metric['count_accuracy'])),flush=True)
    model.restore(torch.load(OUT/'object_selector.pth',map_location='cpu',weights_only=True));model.eval()
    prior=json.loads((PRIOR/'report.json').read_text(encoding='utf-8'))
    report=dict(architecture='Fixed GenéLive heads + AutoOsu CNN and transferred/fine-tuned NS GRU + factorized PJSK object heads',
                source_commit=COMMIT,autoosu_checkpoint_sha256=SHA256,genelive_checkpoint_sha256=prior['sha256'],
                trainable_parameters=sum(p.numel() for p in model.parameters()),chosen_epoch=chosen_epoch,
                selection='Validation 0.4 type macro-F1 + 0.4 type-set exact accuracy + 0.2 count accuracy; test not used to choose',history=history,
                results={s:evaluate(model,rs) for s,rs in [('train',train),('validation',val),('test',test)]},all_tap_baselines=baselines,
                supported_types=[TYPES[i] for i in model.allowed_types],supported_counts=model.allowed_counts,
                reference_midpoints_not_predicted=sum(r['unmodelled_reference_midpoints'] for r in rows),
                training_feedback='Mixed reference/predicted previous actions, up to 65% predicted; GenéLive 768D context replaces AutoOsu NP context',
                limitations=['14 song excerpts of 30 seconds; existing inspected split, not a new blind evaluation.',
                             'No horizontal positions/widths learned; no Trace or hidden-endpoint examples.',
                             'Phase one predicts slide endpoints only; intermediate points not yet modelled.',
                             'Sampling the transferred NS GRU at fixed onsets differs from original 10ms cadence; GRU is fine-tuned.',
                             'Unmatched GenéLive candidates retained; no invented official type labels or deletion.',
                             'Type scores are conditional on matched heads; overall chart accuracy and end-time accuracy not implied.'])
    dump(OUT/'report.json',report)
    print('FINAL '+json.dumps(dict(test=report['results']['test'],baseline=baselines['test'],chosen_epoch=chosen_epoch),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
