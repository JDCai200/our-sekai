"""Inference-only adapter for the official KLab GenéLive! checkpoint."""
from pathlib import Path
import hashlib
import json
import sys
import time
import numpy as np

ROOT = Path(__file__).resolve().parent
UPSTREAM = ROOT/'models/genelive'


def filter_inaudible_events(events, audio, sr=44100, context_start=0., radius=.064):
    """Reject model hallucinations in near silence without moving any onset.

    Use all channels, a gain-relative -50 dB peak gate, and a 64 ms margin
    for the frame clock and bounded snapping. This is an audibility gate,
    not an onset detector or a filter selecting one musical part.
    """
    audio=np.asarray(audio)
    if audio.ndim==1:audio=audio[None,:]
    level=max(1e-5,float(np.max(np.abs(audio),initial=0))*.003)
    kept=[];removed=[]
    for e in events:
        center=(e['time']-context_start)*sr
        a=max(0,round(center-radius*sr));b=min(audio.shape[-1],round(center+radius*sr))
        peak=float(np.max(np.abs(audio[:,a:b]),initial=0)) if b>a else 0.
        if peak>=level:kept.append(e)
        else:removed.append(dict(time=e['time'],score=e['score'],peak=peak,reason='near_silence'))
    return kept,dict(threshold_peak=level,radius_seconds=radius,removed=removed)


def extract_frames(probs, threshold=.4, min_distance=4, method='upstream'):
    probs=np.asarray(probs).reshape(-1)
    if not 0<=min_distance<=8 or not .01<=threshold<=.99:
        raise ValueError('无效的阈值或最小帧间距')
    if method=='peaks':
        from scipy.signal import find_peaks
        return find_peaks(probs,height=threshold,distance=max(1,min_distance))[0].tolist()
    if method!='upstream': raise ValueError('未知放点提取方法')
    frames=np.flatnonzero(probs>=threshold)
    excludes=set();previous=0;previous_prob=None
    for frame in frames:
        prob=float(probs[frame])
        if previous_prob and frame-previous<=min_distance:
            excludes.add(int(previous if prob>previous_prob else frame))
        previous,previous_prob=int(frame),prob
    return [int(f) for f in frames if f not in excludes]


def events_from_probabilities(probs, context_start, start, end, bpm, origin,
                             threshold=.4, min_distance=4, method='upstream', shift_ms=0, snap_ms=0,
                             snap_division=0, grid_min_time=0):
    if not np.isfinite(shift_ms) or abs(shift_ms)>100 or not 0<=snap_ms<=100:
        raise ValueError('无效的时间校正或吸附上限')
    frames=extract_frames(probs,threshold,min_distance,method)
    times=np.asarray(frames)*.032+context_start+shift_ms/1000
    done=np.zeros(len(times),bool)
    if snap_division not in (0,8,16,32):raise ValueError('全点网格对齐须为0/8/16/32')
    if snap_division:
        if not np.isfinite(bpm) or bpm<=0 or not np.isfinite(origin) or not np.isfinite(grid_min_time):
            raise ValueError('无效的网格时间轴')
        step=240/bpm/snap_division
        lower=max(start,grid_min_time)
        first=int(np.ceil((lower-origin)/step-1e-9))
        last=int(np.ceil((end-origin)/step-1e-9))-1
        if first>last:raise ValueError('分析范围内没有可用节奏网格位置')
        # Keep context/padding predictions out before moving valid candidates.
        valid=(times>=lower)&(times<end)
        frames=np.asarray(frames)[valid].tolist();times=times[valid]
        indices=np.floor((times-origin)/step+.5)
        times=origin+np.clip(indices,first,last)*step
    elif snap_ms:
        for division in (4,8,16,32):
            step=240/bpm/division
            targets=origin+np.round((times-origin)/step)*step
            mask=(np.abs(targets-times)<=snap_ms/1000+1e-9)&~done
            times[mask]=targets[mask];done[mask]=True
    events={}
    for frame,t in zip(frames,times):
        if not start<=t<end:continue
        t=round(float(t),6);score=float(probs[frame])
        if t in events and events[t]['score']>=score:continue
        events[t]=dict(time=t,end=t,pitch=None,score=score,source='genelive',reason='pretrained_chart_onset',
                       selected=True,raw_model_time=context_start+frame*.032)
    result=sorted(events.values(),key=lambda e:e['time'])
    for i,e in enumerate(result):e['id']=i
    return result


def infer(mono, bpm, beat_origin, condition=40, threshold=.4, checkpoint_path=None, feature_path=None):
    import librosa
    import torch
    sys.path.insert(0,str(UPSTREAM))
    from notes_generator.models.onsets import SimpleOnsets
    from notes_generator.constants import ConvStackType, NMELS
    from notes_generator.models.beats import gen_beats_array
    torch.set_num_threads(2)
    model = SimpleOnsets(NMELS,1,enable_condition=True,enable_beats=True,
                         conv_stack_type=ConvStackType.v7,num_layers=2,onset_weight=64,dropout=.5)
    checkpoint=Path(checkpoint_path) if checkpoint_path else UPSTREAM/'pretrained_model/model.pth'
    model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True),strict=True)
    model.eval()
    if feature_path:
        def save_features(module, inputs):
            np.save(feature_path, inputs[0][0].detach().cpu().numpy())
        model.onset_linear.register_forward_pre_hook(save_features)
    audio=librosa.resample(mono,orig_sr=44100,target_sr=16000)
    mel=librosa.feature.melspectrogram(y=audio,sr=16000,hop_length=512,fmin=30.,n_mels=229,htk=True)
    mel=np.log(np.clip(mel,1e-5,None)).T.astype(np.float32)
    # Official model sees 0/1/2 beat flags. With no known bar origin, auto mode
    # estimates the phase but cannot establish a true first beat of the bar.
    beat_origin=max(0.,float(beat_origin))
    beats=gen_beats_array(len(mel),[(float(bpm),beat_origin*1000,4)],len(mel))
    with torch.inference_mode():
        probs=model(torch.from_numpy(mel)[None],torch.full((1,len(mel),1),float(condition)),
                    torch.from_numpy(beats.astype(np.float32))[None])[0,:,0].numpy()
    # Same nearby-frame filtering as the upstream predictor; no chart-driven
    # threshold tuning or global note-count adjustment.
    frames=extract_frames(probs,threshold)
    return frames,probs,hashlib.sha256(checkpoint.read_bytes()).hexdigest()


def analyze_genelive(audio_path, output, start=0., duration=30., bpm=0., phase=None,
                     condition=40, threshold=.4, input_mode='mix', progress=lambda m,p:None,
                     postprocess_config=None, checkpoint_path=None, feature_path=None):
    import librosa
    import soundfile as sf
    from engine import decode, separate
    from rhythm import estimate_grid
    started=time.time()
    if condition not in (10,20,30,40,50) or not .01<=threshold<=.99:
        raise ValueError('GenéLive 难度须为10/20/30/40/50，阈值须为0.01–0.99')
    if input_mode not in ('mix','priority'): raise ValueError('无效的模型输入模式')
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    audio_path=Path(audio_path).resolve()
    progress('读取音频及前后5秒上下文',3)
    context_start=max(0.,float(start)-5.)
    y=decode(audio_path,context_start,0 if duration==0 else duration+start-context_start+5)
    target_a=round((start-context_start)*44100)
    target_n=len(y[0])-target_a if duration==0 else min(round(duration*44100),len(y[0])-target_a)
    if target_n<=0: raise ValueError('分析范围没有音频')
    length=target_n/44100
    sf.write(output/'mix.wav',y[:,target_a:target_a+target_n].T,44100,subtype='PCM_16')
    target= y[:,target_a:target_a+target_n].mean(axis=0)
    progress('确定节拍时间轴',12)
    mono22=librosa.resample(target,orig_sr=44100,target_sr=22050)
    tempo,beats=librosa.beat.beat_track(y=mono22,sr=22050,hop_length=256,units='time')
    initial=float(np.asarray(tempo).ravel()[0])
    if not bpm:
        fitted=estimate_grid(output,{'start':start,'estimated_bpm':initial})
        bpm=fitted['bpm']
    else: fitted={'method':'manual','bpm':bpm}
    if phase is None:
        beat_origin=start+(float(beats[0]) if len(beats) else 0.)
        origin_basis='audio beat tracking; bar origin uncertain'
    else:
        beat_origin=float(phase);origin_basis='manual'
    # Preserve the metrical position when the inference crop starts mid-bar.
    bar=240/bpm
    local_origin=(beat_origin-context_start)%bar
    selected_windows=[]
    input_wave=y.mean(axis=0)
    if input_mode=='priority':
        progress('实验模式：分离声部，构建人声优先输入',20)
        stems=separate(y,progress)
        from engine import SESSIONS
        SESSIONS.clear()
        v=stems['vocals'].mean(axis=0); other=stems['other'].mean(axis=0)
        weight=np.zeros(len(v),np.float32)
        for pos in range(0,len(v),5*44100):
            end=min(len(v),pos+5*44100)
            n=(end-pos)//4410
            if not n: continue
            vr=np.sqrt(np.mean(v[pos:pos+n*4410].reshape(n,4410)**2,axis=1))
            mr=np.sqrt(np.mean(input_wave[pos:pos+n*4410].reshape(n,4410)**2,axis=1))
            active=float(np.mean((vr>=.003)&(vr/np.maximum(mr,1e-6)>=.12)))>=.2
            weight[pos:end]=float(active)
            selected_windows.append({'start':context_start+pos/44100,'end':context_start+end/44100,
                                     'source':'vocals' if active else 'other'})
        # Short crossfades reduce synthetic onsets at source changes.
        from scipy.ndimage import uniform_filter1d
        weight=uniform_filter1d(weight,size=8821,mode='nearest')
        input_wave=v*weight+other*(1-weight)
        sf.write(output/'model_input.wav',input_wave,44100,subtype='PCM_16')
    progress('GenéLive 官方预训练放点模型推理',65)
    frames,probs,sha=infer(input_wave,bpm,local_origin,condition,threshold,checkpoint_path,feature_path)
    post=postprocess_config or {}
    musical_origin=beat_origin if phase is not None else fitted.get('offset',beat_origin)
    # Full-grid quantization must share the exported SUS zero, rather than
    # silently using an audio phase that the SUS file cannot represent.
    origin=post.get('grid_min_time',0.) if post.get('snap_division',0) else musical_origin
    events=events_from_probabilities(probs,context_start,start,start+length,bpm,origin,threshold,
              post.get('min_distance',4),post.get('method','upstream'),post.get('shift_ms',0),post.get('snap_ms',0),
              post.get('snap_division',0),post.get('grid_min_time',0))
    events,audibility=filter_inaudible_events(events,y,context_start=context_start)
    np.save(output/'frame_probabilities.npy',probs)
    data={'schema':1,'audio':str(audio_path),'start':start,'duration':length,'events':events,
          'estimated_bpm':bpm,'estimated_beat_offset':beat_origin,'elapsed_seconds':round(time.time()-started,2),
          'model_versions':{'genelive':'Genelive custom checkpoint' if checkpoint_path else 'KLab official StepMania checkpoint','sha256':sha,
                            'checkpoint':str(Path(checkpoint_path).resolve()) if checkpoint_path else str(UPSTREAM/'pretrained_model/model.pth'),
                            'upstream_commit':'43806e50e2a252b2c74acdb5283fdf59002f645b'},
          'parameters':{'condition':condition,'threshold':threshold,'input_mode':input_mode,
                        'frame_ms':32,'minimum_distance_frames':post.get('min_distance',4),'context_seconds':5,
                        'requested_bpm':bpm if fitted.get('method')=='manual' else 0.,'requested_phase':phase},
          'rhythm':{'bpm':bpm,'offset':beat_origin,'basis':origin_basis,'audio_estimate':fitted,'grid':'model_frames'},
          'source_windows':selected_windows,'audibility_filter':audibility,
          'warnings':['现成权重训练于StepMania，不保证PJSK官谱或人声优先。',
                      '主声部输入为未经重新训练的分布外实验；与原始混音模式分别评价。',
                      '输出为32ms帧上的采音时间点，未预测键型；未按官谱调阈值或平移。']}
    data['postprocessing']=dict(method=post.get('method','upstream'),min_distance=post.get('min_distance',4),
                                shift_ms=post.get('shift_ms',0),snap_ms=post.get('snap_ms',0),grid_origin=origin,
                                snap_division=post.get('snap_division',0),grid_min_time=post.get('grid_min_time',0))
    data['postprocessing']['musical_grid_origin']=musical_origin
    data['postprocessing']['grid_reference']='SUS origin' if post.get('snap_division',0) else 'audio phase'
    if post:
        data['warnings'][-1]='输出经开发官谱调参后的峰值提取、时间校正和有限吸附；不是未经调参的官方输出。'
        data['calibration_basis']=post.get('basis','development chart calibration')
    if checkpoint_path:
        data['warnings'][0]='使用自定义微调权重；请以该权重的独立验证报告判断适用范围。'
    (output/'analysis.json').write_text(json.dumps(data,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    progress('现成模型采音完成',100)
    return data
