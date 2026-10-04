"""Position-free experimental generation and official-chart inspection."""
import argparse,json
from pathlib import Path
import numpy as np
from .schema import parse_position_free,Codec

def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    s=sub.add_parser('parse');s.add_argument('chart');s.add_argument('--audio-shift',type=float,default=0)
    s.add_argument('--output',required=True)
    g=sub.add_parser('generate');g.add_argument('audio');g.add_argument('--start',type=float,default=25)
    g.add_argument('--duration',type=float,default=4);g.add_argument('--output',required=True)
    g.add_argument('--adapter');g.add_argument('--max-events',type=int,default=60)
    g.add_argument('--candidates',help='Existing analysis.json: use its selected time points instead of predicting times')
    a=p.parse_args()
    if a.command=='parse':result=parse_position_free(Path(a.chart).read_text(encoding='utf-8-sig'),a.audio_shift)
    else:
        import torch,librosa
        from engine import decode
        from .model import ROOT,load_base,PositionFreeAdapter
        torch.set_num_threads(2)
        path=Path(a.adapter) if a.adapter else ROOT/'diagnostics/mapper_pjsk/adapter.pth'
        state=torch.load(path,map_location='cpu',weights_only=True)
        if abs(a.duration-state['duration'])>1e-6:raise ValueError('Duration must match the trained adapter window')
        model=PositionFreeAdapter(load_base(),Codec(state['duration'],state['max_chains']))
        model.restore(state);model.eval()
        wave=decode(Path(a.audio),a.start,a.duration).mean(axis=0)
        wave=librosa.resample(wave,orig_sr=44100,target_sr=16000)
        enc=model.encode_audio(torch.from_numpy(wave.astype(np.float32)))
        schedule=None
        if a.candidates:
            candidate=json.loads(Path(a.candidates).read_text(encoding='utf-8'))
            schedule=sorted(e['time']-a.start for e in candidate['events'] if e.get('selected',True) and a.start<=e['time']<a.start+a.duration)
        result=model.generate_events(enc,a.max_events,schedule);result.update(audio=str(Path(a.audio).resolve()),start=a.start,duration=a.duration)
        result['events']=[dict(e,time=round(e['time']+a.start,6)) for e in result['events']]
        result['time_reference']='audio_absolute'
    Path(a.output).write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'Saved {len(result["events"])} position-free events')

if __name__=='__main__':main()
