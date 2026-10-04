"""Frozen audio-only gate on two existing development excerpts, no refitting."""
import json
import numpy as np
import soundfile as sf
from charts import match_times
from mapper_pjsk.schema import parse_position_free,STARTS
from genelive_backend import events_from_probabilities
from .data import ROOT,PRIOR,dump
from .layout_model import OUT
from .salience import select,NAMES


def main():
    prior=json.loads((PRIOR/'report.json').read_text(encoding='utf-8'));manifest=json.loads((PRIOR/'manifest.json').read_text(encoding='utf-8'));samples=[]
    for song in manifest['rows']:
        if song['id'] not in (1,3):continue
        stem=f'{song["id"]:04d}';name='unipjsk_tell_your_world' if song['id']==1 else 'unipjsk_teo';directory=ROOT/'results'/name
        data=json.loads((PRIOR/stem/'analysis.json').read_text(encoding='utf-8'));post=data['postprocessing']
        events=events_from_probabilities(np.load(PRIOR/f'{stem}_finetuned_probs.npy'),max(0,song['start']-5),song['start'],song['start']+song['duration'],
            data['estimated_bpm'],post['grid_origin'],threshold=prior['selected']['threshold'],**{k:post[k] for k in ('method','min_distance','shift_ms','snap_ms')})
        stems={}
        for source in NAMES:
            y,sr=sf.read(directory/f'{source}.wav',dtype='float32',always_2d=True)
            if sr!=44100:raise ValueError('Unexpected stem sample rate')
            stems[source]=y.T
        kept,selection=select(events,stems,song['start'])
        chart=parse_position_free((ROOT/'reference'/f'{stem}_master.sus').read_text(encoding='utf-8-sig'),song['filler'])
        ref=[e['time'] for e in chart['events'] if (e['chain'] is None or e['kind'] in STARTS) and song['start']<=e['time']<song['start']+song['duration']]
        before=match_times([e['time'] for e in events],ref);after=match_times([e['time'] for e in kept],ref)
        sample=dict(id=song['id'],title=song['title'],before=before,after=after,selection=selection)
        samples.append(sample);print(json.dumps({k:v for k,v in sample.items() if k!='selection'},ensure_ascii=False),flush=True)
    OUT.mkdir(parents=True,exist_ok=True)
    dump(OUT/'salience_validation.json',dict(samples=samples,protocol='Two previously inspected training/development excerpts; fixed thresholds; full semantic unique heads; ±50ms ordered matching; no chart used during audio selection.',
        limitations='Not a held-out evaluation; chart mismatch is not proof of inaudible music, and matched heads do not prove the intended stem.'))


if __name__=='__main__':main()
