"""Audio-only dominant-part selection and evidence gate for GenéLive candidates."""
from pathlib import Path
import argparse,json
import numpy as np
import soundfile as sf
import librosa
from scipy.ndimage import median_filter,maximum_filter1d
from engine import decode,separate,SESSIONS
from genelive_backend import filter_inaudible_events

NAMES=('vocals','other','bass','drums')


def stem_features(audio,sr=44100):
    mono=audio.mean(axis=0) if audio.ndim==2 else audio
    y=librosa.resample(mono,orig_sr=sr,target_sr=16000)
    hop=160;mel=librosa.feature.melspectrogram(y=y,sr=16000,n_fft=1024,hop_length=hop,n_mels=40,power=2)
    log=librosa.power_to_db(mel,ref=np.max,top_db=70)
    # Flux covers amplitude attacks and consonant/vowel spectral changes.
    flux=np.maximum(np.diff(log,axis=1,prepend=log[:,:1]),0).mean(axis=0)
    rms=librosa.feature.rms(y=y,frame_length=1024,hop_length=hop)[0]
    pitch=librosa.yin(y,fmin=65,fmax=1000,sr=16000,frame_length=1024,hop_length=hop)
    midi=median_filter(librosa.hz_to_midi(pitch),size=7)
    before=median_filter(midi,size=9);delta=np.abs(np.roll(before,-5)-np.roll(before,5));delta[:6]=delta[-6:]=0
    scale=max(float(np.quantile(flux,.9)),.05)
    return dict(rms=rms,flux=flux/scale,pitch_change=delta,hop_seconds=.01,
                relative_activity_floor=max(float(np.quantile(rms,.9))*.04,.00015))


def select(events,stems,context_start=0.,sr=44100,window_seconds=2.):
    tracks={name:stem_features(stems[name],sr) for name in NAMES}
    n=min(len(t['rms']) for t in tracks.values());count=max(1,round(window_seconds*100));windows=[]
    previous=None;source_by_frame=np.empty(n,dtype=object)
    for a in range(0,n,count):
        b=min(n,a+count);energies={k:float(np.sqrt(np.mean(v['rms'][a:b]**2))) for k,v in tracks.items()}
        strongest=max(energies,key=energies.get)
        # Obvious vocals retain priority. Otherwise favor melodic accompaniment
        # when it is strong enough; 'other' is not a verified lead melody label.
        vocal=tracks['vocals']['rms'][a:b]
        vocal_active=float(np.mean(vocal>tracks['vocals']['relative_activity_floor']))
        if vocal_active>=.25 and energies['vocals']>=.35*max(energies.values()):choice='vocals'
        elif energies['other']>=.55*max(energies.values()):choice='other'
        else:choice=strongest
        if previous and choice!='vocals' and energies[previous]>=.85*energies[choice]:choice=previous
        previous=choice;source_by_frame[a:b]=choice
        windows.append(dict(start=context_start+a*.01,end=context_start+b*.01,source=choice,stem_rms=energies,vocal_active_fraction=vocal_active))
    kept=[];removed=[];mix=sum(stems.values())
    audible,silence=filter_inaudible_events(events,mix,sr,context_start)
    for e in audible:
        frame=min(max(round((e['time']-context_start)*100),0),n-1);source=source_by_frame[frame]
        def evidence(name):
            t=tracks[name];a=max(0,frame-5);b=min(n,frame+6)
            rms=float(np.max(t['rms'][a:b]));flux=float(np.max(t['flux'][a:b]));pitch=float(np.max(t['pitch_change'][a:b]))
            activity=rms>=t['relative_activity_floor']
            # Sustained tones can be charted on pitch/phonetic transitions.
            # Plain high energy alone is insufficient evidence for another note.
            pitch_support=name!='drums' and pitch>=.8 and flux>=.18
            support=activity and (flux>=.45 or pitch_support)
            return dict(source=name,rms=rms,flux=flux,pitch_change_semitones=pitch,supported=bool(support))
        proof=evidence(source);drums=None
        if not proof['supported'] and source!='drums':
            drums=evidence('drums')
            # Only very distinct, audible percussion may supplement a melody.
            if drums['supported'] and drums['flux']>=1.2 and drums['rms']>=.55*proof['rms']:proof=drums
        record=dict(time=e['time'],preferred_source=source,**proof)
        if proof['supported']:
            kept.append(dict(e,source=proof['source'],reason='dominant_stem_acoustic_evidence',acoustic_evidence=record))
        else:removed.append(dict(record,reason='no_supported_change_in_dominant_stem'))
    return kept,dict(policy='audio-only vocal prominence, then dominant accompaniment; strong drums may supplement',windows=windows,
                     removed=removed,silence=silence,input_heads=len(events),retained_heads=len(kept),
                     limitations=['Source separation leakage can cause false attribution.',
                     'Other includes multiple instruments; this does not prove isolated main-melody tracking.',
                     'Pitch/flux are evidence heuristics, not human audibility labels; genuine subtle syllables may be removed.',
                     'No reference chart is used to choose stems or retain notes.'])


def analyze(analysis_path,output,stem_dir=None):
    analysis_path=Path(analysis_path);output=Path(output)
    if output.exists():raise FileExistsError(output)
    data=json.loads(analysis_path.read_text(encoding='utf-8'));source=Path(data['audio']);start=float(data['start']);length=float(data['duration'])
    directory=Path(stem_dir) if stem_dir else output.parent/(output.stem+'_stems')
    if all((directory/f'{name}.wav').exists() for name in NAMES):
        stems={}
        for name in NAMES:
            audio,sr=sf.read(directory/f'{name}.wav',dtype='float32',always_2d=True)
            if sr!=44100:raise ValueError('Cached stems must be 44100 Hz')
            stems[name]=audio.T
        if abs(stems['vocals'].shape[-1]/44100-length)>.05:raise ValueError('Cached stem duration differs from analysis')
    else:
        directory.mkdir(parents=True,exist_ok=False)
        wave=decode(source,start,length)
        stems=separate(wave,lambda message,percent:print(f'{percent}% {message}',flush=True));SESSIONS.clear()
        for name,audio in stems.items():sf.write(directory/f'{name}.wav',audio.T,44100,subtype='FLOAT')
    print('Computing dominant-part acoustic evidence',flush=True)
    candidates,report=select([e for e in data['events'] if e.get('selected',True)],stems,start)
    data['events']=candidates;data['salience_selection']=report;data['salience_stem_dir']=str(directory.resolve())
    output.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print(json.dumps(dict(input=report['input_heads'],retained=report['retained_heads'],removed=len(report['removed']),silent=len(report['silence']['removed'])),ensure_ascii=False),flush=True)
    return data


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--analysis',required=True);p.add_argument('--output',required=True);p.add_argument('--stem-dir')
    args=p.parse_args();analyze(args.analysis,args.output,args.stem_dir)
