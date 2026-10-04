"""Audio-only tempo/phase fit and bounded binary-grid quantization."""
from copy import deepcopy
from pathlib import Path
import json
import numpy as np
import soundfile as sf
import librosa
from charts import evaluate, unique_times


def estimate_grid(stem_dir, data):
    times, weights = [], []
    for name, source_weight in [('mix',1.),('drums',2.)]:
        file = Path(stem_dir)/f'{name}.wav'
        if not file.is_file():
            continue
        y,sr = sf.read(file,dtype='float32',always_2d=True)
        y = librosa.resample(y=y.mean(axis=1),orig_sr=sr,target_sr=22050)
        envelope = librosa.onset.onset_strength(y=y,sr=22050,hop_length=128)
        frames = librosa.onset.onset_detect(onset_envelope=envelope,sr=22050,hop_length=128)
        if not len(frames):
            continue
        scale = max(float(np.percentile(envelope[frames],90)),1e-6)
        times.extend((frames*128/22050+data['start']).tolist())
        weights.extend((np.minimum(envelope[frames]/scale,1)*source_weight).tolist())
    if len(times)<8:
        raise ValueError('有效起音不足，无法自动确定稳定网格；请手动指定 BPM 和起点')
    t,w = np.asarray(times),np.asarray(weights)
    initial = data.get('estimated_bpm')
    if not initial:
        raise ValueError('缺少初步 BPM；请手动指定')
    candidates = set()
    for factor in (.5,1,2):
        center = initial*factor
        if 100<=center<=240:
            candidates.update(np.round(np.arange(max(100,center*.96),min(240,center*1.04),.02),4))
    # A declared tempo-range prior resolves metrical octave ambiguity. This does
    # not prove a unique musical BPM, and is adjustable through manual BPM.
    if not candidates:
        raise ValueError('BPM 不在默认 100–240 范围；请手动指定')
    scores = []
    for bpm in sorted(candidates):
        step = 60/bpm/4
        z = np.sum(w*np.exp(2j*np.pi*(t-data['start'])/step))/np.sum(w)
        scores.append((float(abs(z)),float(bpm),float(np.angle(z)%(2*np.pi)/(2*np.pi)*step)))
    score,bpm,phase = max(scores)
    rounded = round(bpm)
    rounded_row = min(scores,key=lambda x:abs(x[1]-rounded))
    if abs(bpm-rounded)<.15 and rounded_row[0]>=score-.01:
        bpm = float(rounded)
        step = 60/bpm/4
        z = np.sum(w*np.exp(2j*np.pi*(t-data['start'])/step))/np.sum(w)
        score,phase = float(abs(z)),float(np.angle(z)%(2*np.pi)/(2*np.pi)*step)
    return {'bpm':bpm,'offset':data['start']+phase,'method':'librosa onset + weighted periodic phase fit',
            'phase_coherence':score,'onsets':len(t),'tempo_prior':[100,240],
            'initial_bpm':initial,'note':'相位为十六分网格锚点，不保证是第零拍/小节起点。BPM 范围先验可能选错半倍/双倍。'}


def quantize(data, bpm, offset, grid='binary', max_shift_ms=20, basis='manual'):
    if not np.isfinite(bpm) or not 10<=bpm<=1000 or not np.isfinite(offset):
        raise ValueError('无效 BPM 或网格起点')
    if grid not in ('binary','16','32') or not np.isfinite(max_shift_ms) or not 0<=max_shift_ms<=100:
        raise ValueError('无效量化规则')
    result = deepcopy(data)
    moves, divisions = [], {}
    for e in result['events']:
        e.setdefault('unquantized_time',e['time'])
        e.setdefault('unquantized_end',e['end'])
        e['time'],e['end'] = e['unquantized_time'],e['unquantized_end']
        if not e['selected']:
            continue
        choices = (4,8,16,32) if grid=='binary' else (int(grid),)
        for division in choices:
            step = 60/bpm*4/division
            target = offset+round((e['time']-offset)/step)*step
            delta = target-e['time']
            if abs(delta)*1000<=max_shift_ms+1e-6 and result['start']<=target<result['start']+result['duration']:
                e['time'] = round(target,6)
                e['end'] = round(min(result['start']+result['duration'],max(e['time'],e['end']+delta)),6)
                e['quantization_division'] = division
                moves.append(abs(delta)*1000)
                divisions[str(division)] = divisions.get(str(division),0)+1
                break
    active = [e for e in result['events'] if e['selected']]
    result['rhythm'] = {'bpm':float(bpm),'offset':float(offset),'basis':basis,'grid':grid,
                        'max_shift_ms':max_shift_ms,'moved_events':len(moves),'divisions':divisions,
                        'max_actual_shift_ms':max(moves,default=0),'mean_shift_ms':float(np.mean(moves)) if moves else 0,
                        'remaining_unsnapped':len(active)-len(moves),
                        'coalesced_time_points':len(unique_times(e['time'] for e in data['events'] if e['selected']))-len(unique_times(e['time'] for e in active)),
                        'note':'不补点；四分/八分/十六分优先，三十二分作为细网格，三连音不自动吸附；超出移动上限保留原时间。'}
    chart = result.get('reference')
    if chart:
        result.setdefault('evaluation',{})['pre_quantization'] = evaluate([e for e in data['events'] if e['selected']],chart,data['start'],data['start']+data['duration'])
        result['evaluation']['selected_models'] = evaluate(active,chart,data['start'],data['start']+data['duration'])
    return result


def save_quantized(source, destination, bpm=None, offset=None, basis='audio', grid='binary', max_shift_ms=20):
    import shutil,csv
    source,destination = Path(source),Path(destination)
    data = json.loads((source/'analysis.json').read_text(encoding='utf-8'))
    estimate = None
    if bpm is None or offset is None:
        estimate = estimate_grid(source,data)
        bpm,offset = estimate['bpm'],estimate['offset']
    result = quantize(data,bpm,offset,grid,max_shift_ms,basis)
    result['rhythm']['audio_estimate'] = estimate
    result['parent_run'] = source.name
    destination.mkdir(parents=True,exist_ok=False)
    for file in source.glob('*.wav'):
        shutil.copyfile(file,destination/file.name)
    (destination/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    with (destination/'candidates.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        fields = ['id','time','unquantized_time','source','reason','pitch','score','selected','quantization_division']
        writer=csv.DictWriter(handle,fieldnames=fields,extrasaction='ignore');writer.writeheader();writer.writerows(result['events'])
    return result
