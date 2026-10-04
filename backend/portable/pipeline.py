"""Local ONNX chain: audio -> onsets -> NS -> motif/layout -> verified SUS."""
import hashlib
import json
from pathlib import Path
import numpy as np
from genelive_backend import events_from_probabilities,filter_inaudible_events
from hybrid_pjsk.genelive_settings import validate,preset_settings
from hybrid_pjsk.audio_origin import constrain_candidates
from hybrid_pjsk.decode import to_objects
from hybrid_pjsk.layout_features import INTERVALS
from hybrid_pjsk.two_finger import project,verify_exported_witness
from hybrid_pjsk.grid_verification import verify_sus_grid
from mapper_pjsk.sus_export import export_preview
from worker import timeline,finite
from companion.song_package import load_json,write_json,finalize_project
from .dsp import Features,read_wave,write_wave,beat_flags,estimate_tempo
from .runtime import Runtime


def verify_models(models):
    models=Path(models)
    registry=load_json(models/'model-lock.json')
    for row in registry['files']:
        path=models/row['path']
        if not path.is_file() or path.stat().st_size!=row['size'] or hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:
            raise ValueError('离线模型缺失或校验失败：'+row['path'])


def onset_context(runtime,mel,condition,bpm,origin,check=lambda:None):
    frames=len(mel); flags=beat_flags(frames,bpm,origin)
    features=np.empty((1,frames,770),np.float32)
    # A halo exceeding the exported CNN temporal receptive field keeps borders identical.
    for start in range(0,frames,640):
        check(); end=min(start+640,frames); a=max(0,start-128); b=min(frames,end+128)
        stack=runtime.run('genelive_stack',dict(mel=mel[None,a:b],
            condition=np.full((1,b-a,1),condition,np.float32),beats=flags[None,a:b]))[0]
        features[:,start:end]=stack[:,start-a:end-a]
    runtime.release('genelive_stack')
    result=np.zeros((1,frames,768),np.float32)
    starts=list(range(0,frames,640))
    for reverse in (False,True):
        hidden=np.zeros((4,1,384),np.float32); cell=hidden.copy()
        for start in reversed(starts) if reverse else starts:
            check(); end=min(start+640,frames)
            out,hidden,cell=runtime.run('genelive_recurrent',dict(features=features[:,start:end],hidden=hidden,cell=cell))
            if reverse:result[:,start:end,384:]=out[:,:,384:]
            else:result[:,start:end]=out
    runtime.release('genelive_recurrent')
    probabilities=runtime.run('genelive_head',dict(context=result))[0][0,:,0]
    runtime.release('genelive_head')
    return result[0],probabilities


def placement_features(times,scores,bpm,context):
    times=np.asarray(times); before=np.r_[0.,np.diff(times)]*bpm/60; after=np.r_[np.diff(times),0.]*bpm/60
    basic=np.column_stack([scores,np.minimum(before,8)/8,np.minimum(after,8)/8,np.ones(len(times))]).astype(np.float32)
    positions=np.clip(times/.032,0,len(context)-1); lower=np.floor(positions).astype(int); upper=np.minimum(lower+1,len(context)-1)
    weight=(positions-lower)[:,None]
    return np.concatenate([basic,(context[lower]*(1-weight)+context[upper]*weight).astype(np.float32)],axis=1)


def acoustic(runtime,specs,times,bpm,offset,check=lambda:None):
    frames=specs.shape[1]; times=np.asarray(times)
    indices=np.clip(np.rint(times*100).astype(np.int64),0,frames-1)
    beats=(times-offset)*bpm/60
    phases=np.rint((beats%1)/round(1/48,5)).astype(np.int64); numbers=np.floor(beats%4).astype(np.int64)
    result=np.empty((len(times),384),np.float32)
    for start in range(0,frames,640):
        check(); end=min(start+640,frames); selected=np.flatnonzero((indices>=start)&(indices<end))
        if not len(selected):continue
        a=max(0,start-128); b=min(frames,end+128)
        result[selected]=runtime.run('acoustic',dict(specs=specs[None,:,a:b],indices=indices[selected]-a,
            phases=phases[selected],numbers=numbers[selected],conditions=np.full((len(selected),1),3.,np.float32)))[0]
    runtime.release('acoustic')
    return result


def select(runtime,sound,placement,report,check=lambda:None):
    allowed_counts=np.asarray(report['selector_allowed_counts'],np.int64)
    allowed_types=np.asarray(report['selector_allowed_types'],np.int64)
    previous=dict(previous_count=np.zeros(1,np.int64),previous_kind=np.full((1,4),4,np.int64),
        previous_critical=np.zeros((1,4),np.int64),previous_direction=np.zeros((1,4),np.int64),previous_span=np.zeros((1,4),np.float32))
    hidden=np.zeros((2,1,256),np.float32); actions=[]
    for index in range(len(sound)):
        check()
        count,kind,critical,direction,tail,span,hidden=runtime.run('selector_step',dict(
            acoustic=sound[index:index+1],placement=placement[index:index+1],hidden=hidden,**previous))
        action=dict(count=allowed_counts[np.argmax(count[:,allowed_counts-1],axis=-1)],
            kind=allowed_types[np.argmax(kind[:,:,allowed_types],axis=-1)],critical=np.argmax(critical,axis=-1).astype(np.int64),
            direction=np.argmax(direction,axis=-1).astype(np.int64),tail_direction=np.argmax(tail,axis=-1).astype(np.int64),span=span)
        action['direction']=np.where(action['kind']==1,np.argmax(direction[:,:,1:],axis=-1)+1,0).astype(np.int64)
        actions.append(action)
        previous={f'previous_{key}':action[key] for key in ('count','kind','critical','direction','span')}
    runtime.release('selector_step')
    return {key:np.concatenate([action[key] for action in actions],axis=0) for key in actions[0]}


class Layout:
    def __init__(self,runtime,check):self.runtime=runtime; self.check=check
    def predict(self,features):
        hidden=np.zeros((2,1,96),np.float32); previous=np.asarray([len(INTERVALS)],np.int64); result=[]
        for row in features:
            self.check()
            logits,hidden=self.runtime.run('layout_step',dict(event=row[None],previous_interval=previous,hidden=hidden))
            result.append(logits[0]); previous=np.argmax(logits,axis=-1).astype(np.int64)
        self.runtime.release('layout_step')
        return np.asarray(result,np.float32)


def run(request,models,job,android=False,progress=lambda message,percent:None):
    job=Path(job); job.mkdir(parents=True,exist_ok=True)
    def check():
        if (job/'cancel').exists():raise InterruptedError('已取消生成')
    progress('正在校验本地模型',1); verify_models(models); check()
    settings=preset_settings(request.get('difficulty','EXPERT').upper()); settings.update(request.get('parameters') or {}); settings=validate(settings)
    audio=Path(request['audio']); samples=read_wave(audio); old=request.get('existing_manifest')
    if old:
        original=load_json(old); filler=finite(original['fillerSec'],'原前置',0,120)
        times=timeline(len(samples)/44100-filler,filler); times['previewStartTimeSec']=float(original.get('previewStartTimeSec',filler))
        phase=settings['phase']+filler if settings['phase'] is not None else None
    else:
        start=finite(request.get('crop_start',0),'裁剪起点'); duration=finite(request.get('crop_duration',0),'裁剪时长')
        a=round(start*44100); b=round((start+duration)*44100) if duration else len(samples)
        samples=samples[a:min(b,len(samples))]
        times=timeline(len(samples)/44100,request.get('filler',9),request.get('preview_start',0)); filler=times['fillerSec']
        phase=finite(settings['phase']-start,'裁剪后节拍起点',0,len(samples)/44100)+filler if settings['phase'] is not None else None
        samples=np.concatenate([np.zeros((round(filler*44100),2),np.float32),samples])
    stage=job/'project'; stage.mkdir(exist_ok=False); write_wave(stage/'audio.wav',samples)
    import secrets
    manifest=dict(formatVersion=1,id=secrets.token_hex(6),title=request.get('title') or audio.stem,
        scoreTitle=request.get('title') or audio.stem,userName='Our Sekai',audioFileName='audio.wav',scoreFileName='score.json',
        jacketFileName='jacket.png',videoFileName='',musicDifficultyType=request.get('difficulty','EXPERT').lower(),playLevel=0,
        fillerSec=filler,secForMusicScoreMaker=times['secForMusicScoreMaker'],previewStartTimeSec=times['previewStartTimeSec'])
    write_json(stage/'manifest.json',manifest); mono=samples.mean(axis=1)
    progress('正在分析音乐节奏',5); check()
    bpm=settings['bpm']
    if not bpm:
        bpm,estimated_phase=estimate_tempo(mono,models)
        if phase is None:phase=estimated_phase
    elif phase is None:
        # The exact downbeat is unknown; choose the declared project music origin.
        phase=filler
    dsp=Features(models); progress('正在提取音频特征',10)
    mel=dsp.onset(mono); check(); runtime=Runtime(models,android)
    try:
        progress('正在生成采音点',20)
        context,probabilities=onset_context(runtime,mel,settings['condition'],bpm,phase%(240/bpm),check)
        origin=filler if settings['snap_division'] else phase
        candidates=events_from_probabilities(probabilities,0.,0.,len(mono)/44100,bpm,origin,
            settings['threshold'],settings['min_distance'],settings['method'],settings['shift_ms'],settings['snap_ms'],settings['snap_division'],filler)
        candidates,audibility=filter_inaudible_events(candidates,samples.T)
        candidates,adjustments=constrain_candidates(candidates,filler,settings)
        if not candidates:raise ValueError('未找到有效采音点，请调整难度或高级参数')
        times_head=[float(event['time']) for event in candidates]
        placement=placement_features(times_head,[event['score'] for event in candidates],bpm,context)
        progress('正在生成音符类型和长条',55)
        sound=acoustic(runtime,dsp.acoustic(mono),times_head,bpm,origin,check)
        actions=select(runtime,sound,placement,load_json(Path(models)/'parity-report.json'),check)
        objects,events,agreement=to_objects(times_head,actions,bpm,len(mono)/44100,snap_division=settings['snap_division'])
        from hybrid_pjsk.expression import apply_expression
        objects,expression=apply_expression(objects,bpm,filler,Path(models)/'expression.json')
        progress('正在安排音符位置和路径',75)
        objects,events,geometry,playability=project(objects,Layout(runtime,check),bpm,filler,
            phrase_style=False,pattern_style=True,pattern_model=Path(models)/'patterns.json')
        check(); progress('正在核验实际导出谱面',94)
        sus,verification=export_preview(events,bpm,manifest['title'],audio_shift=filler,geometry=geometry)
        if settings['snap_division']:
            verification['grid_verification']=verify_sus_grid(sus,bpm,settings['snap_division'])
            if not verification['grid_verification']['passed']:raise ValueError('节奏网格核验失败')
        playability['exported_verification']=verify_exported_witness(objects,playability['witness'],sus,bpm,filler)
        if not playability['exported_verification']['passed']:raise ValueError('实际 SUS 可游玩核验失败')
        result=dict(schema='pjsk-layout-v1',events=[dict(event,**position) for event,position in zip(events,geometry)],
            onset_agreement=agreement,two_finger=playability,sus_preview=verification,genelive_postprocessing=settings,
            audibility_filter=audibility,audio_origin_adjustments=adjustments,expression=expression,tempo=dict(bpm=bpm,offset=phase),runtime='portable-onnx')
        write_json(stage/'chart.json',result); (stage/'chart.sus').write_text(sus,encoding='utf-8')
        finalize_project(stage,old); check(); progress('歌曲包已生成',100)
        return stage
    finally:
        runtime.close()
