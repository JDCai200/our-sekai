"""Explicit audio features using exported desktop windows and mel filters."""
from pathlib import Path
import math
import wave
import numpy as np
from scipy.fft import rfft
from scipy.signal import find_peaks
import soxr


def read_wave(path):
    with wave.open(str(path),'rb') as source:
        if source.getsampwidth()!=2:raise ValueError('需要 PCM16 WAV 音频')
        rate=source.getframerate(); channels=source.getnchannels()
        samples=np.frombuffer(source.readframes(source.getnframes()),dtype='<i2').reshape(-1,channels).astype(np.float32)/32768
    if channels==1:samples=np.repeat(samples,2,axis=1)
    if channels>2:samples=samples[:,:2]
    if rate!=44100:samples=soxr.resample(samples,rate,44100,quality='HQ').astype(np.float32)
    return samples


def write_wave(path,samples):
    pcm=np.clip(np.asarray(samples)*32768,-32768,32767).astype('<i2')
    with wave.open(str(path),'wb') as output:
        output.setnchannels(2); output.setsampwidth(2); output.setframerate(44100)
        output.writeframes(pcm.tobytes())


def stft_power(signal,window,hop,padding):
    n=len(window); signal=np.asarray(signal,dtype=np.float32)
    padded=np.pad(signal,n//2,mode=padding)
    count=1+(len(padded)-n)//hop
    out=np.empty((count,n//2+1),np.float32)
    # Bound temporary FFT memory for long songs.
    for start in range(0,count,512):
        indices=np.arange(start,min(start+512,count))[:,None]*hop+np.arange(n)[None,:]
        frames=padded[indices]*window[None,:]
        values=rfft(frames,axis=1)
        out[start:start+len(frames)]=values.real**2+values.imag**2
    return out


class Features:
    def __init__(self,models):
        self.filters=np.load(Path(models)/'dsp.npz')

    def onset(self,mono):
        y=soxr.resample(mono,44100,16000,quality='HQ').astype(np.float32)
        power=stft_power(y,self.filters['onset_window'],512,'constant')
        mel=power@self.filters['onset_mel'].T
        return np.log(np.maximum(mel,1e-5)).astype(np.float32)

    def acoustic(self,mono):
        signal=np.asarray(mono,np.float32)+np.float32(1e-9)
        values=[]
        for n in (1024,2048,4096):
            power=stft_power(signal,self.filters[f'window_{n}'],441,'reflect')
            mel=power@self.filters[f'mel_{n}']
            logged=np.log(np.maximum(mel,np.finfo(np.float32).tiny))
            padded=np.pad(mono,n//2,mode='reflect')
            for start in range(0,len(logged),512):
                indices=np.arange(start,min(start+512,len(logged)))[:,None]*441+np.arange(n)[None,:]
                silent=~np.any(padded[indices]!=0,axis=1)
                logged[start:start+len(silent)][silent]=self.filters[f'silence_{n}']
            values.append(logged)
        result=np.stack(values).astype(np.float32)
        if not np.all(np.isfinite(result)):raise ValueError('音频特征存在非法值')
        return result


def beat_flags(frames,bpm,origin):
    result=np.zeros((frames,1),np.float32)
    times=np.arange(max(0,origin)*1000,max(0,(frames-.5)*32),60000/bpm)
    indices=np.round(times/32).astype(np.int64)
    result[indices,0]=1
    result[indices[::4],0]=2
    return result


def estimate_tempo(mono,models):
    """Audio-only spectral-flux pulse prior + weighted 16th-grid phase fit."""
    y=soxr.resample(mono,44100,22050,quality='HQ').astype(np.float32)
    filters=np.load(Path(models)/'dsp.npz')
    spectrum=stft_power(y,filters['onset_window'],128,'constant')
    mel=np.log(np.maximum(spectrum@filters['tempo_mel'].T,1e-5))
    flux=np.r_[0,np.maximum(np.diff(mel,axis=0),0).mean(axis=1)]
    peaks,_=find_peaks(flux,height=max(float(np.percentile(flux,75)),1e-5),distance=8)
    if len(peaks)<8:raise ValueError('有效起音不足，请在高级设置手动填写 BPM')
    times=peaks*128/22050; weights=np.minimum(flux[peaks]/max(np.percentile(flux[peaks],90),1e-6),1)
    # Pulse autocorrelation selects the musical octave before fitting the finer grid.
    envelope=flux-flux.mean(); n=1<<(2*len(envelope)-1).bit_length()
    fft=np.fft.rfft(envelope,n=n); correlation=np.fft.irfft(fft*np.conj(fft),n=n)[:len(envelope)]
    lag_min=math.ceil(60/240*22050/128); lag_max=min(len(correlation)-1,math.floor(60/100*22050/128))
    if lag_max<lag_min:raise ValueError('歌曲太短，请手动填写 BPM')
    lag=lag_min+int(np.argmax(correlation[lag_min:lag_max+1]))
    initial=60*22050/128/lag
    candidates=set()
    for factor in (.5,1,2):
        center=initial*factor
        if 100<=center<=240:candidates.update(np.round(np.arange(max(100,center*.96),min(240,center*1.04),.02),4))
    rows=[]
    for bpm in sorted(candidates):
        step=60/bpm/4; z=np.sum(weights*np.exp(2j*np.pi*times/step))/weights.sum()
        rows.append((float(abs(z)),float(bpm),float(np.angle(z)%(2*np.pi)/(2*np.pi)*step)))
    if not rows:raise ValueError('无法确定节奏，请手动填写 BPM')
    score,bpm,phase=max(rows)
    integer=round(bpm); nearby=min(rows,key=lambda row:abs(row[1]-integer))
    if abs(bpm-integer)<.15 and nearby[0]>=score-.01:bpm=float(integer)
    return bpm,phase
