"""Desktop parity checks for the phone's model/DSP/postprocessing chain."""
import json
from pathlib import Path
import sys
import numpy as np
import torch
import librosa
import soxr

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'backend'),str(ROOT/'backend/models/genelive')]
from portable.dsp import Features,read_wave,beat_flags
from portable.runtime import Runtime
from portable.pipeline import onset_context,acoustic
from autoosu_pjsk.backend import TimingModel
from hybrid_pjsk.pattern_layout import load_patterns,context_vector
from hybrid_pjsk.portable_forest import load_bundle
from notes_generator.models.beats import gen_beats_array


def main():
    torch.set_num_threads(2)
    models=ROOT/'artifacts/portable-models'
    sample=read_wave(ROOT/'backend/.smoke/new_bpm/project/audio.wav').mean(axis=1)
    dsp=Features(models)
    audio=librosa.resample(sample,orig_sr=44100,target_sr=16000)
    reference=np.log(np.maximum(librosa.feature.melspectrogram(y=audio,sr=16000,hop_length=512,fmin=30,n_mels=229,htk=True),1e-5)).T
    mel=dsp.onset(sample)
    onset_error=float(np.max(np.abs(reference-mel)))
    print('Onset mel max error:',onset_error,flush=True)
    timing=TimingModel(); expected_specs=timing.features(sample).numpy(); specs=dsp.acoustic(sample)
    acoustic_error=float(np.max(np.abs(expected_specs-specs)))
    print('Acoustic mel max error:',acoustic_error,flush=True)
    # Prespecified float32 DSP tolerances; do not accept large silent-band drift.
    np.testing.assert_allclose(reference,mel,atol=2e-4,rtol=2e-4)
    np.testing.assert_allclose(expected_specs,specs,atol=2e-4,rtol=2e-4)
    runtime=Runtime(models)
    try:
        # Compare bounded CNN chunks to a full ONNX call, using exactly the same features.
        flags=beat_flags(len(mel),143,9%(240/143))
        np.testing.assert_array_equal(flags,gen_beats_array(len(mel),[(143,9%(240/143)*1000,4)],len(mel)))
        full=runtime.run('genelive_stack',dict(mel=mel[None],condition=np.full((1,len(mel),1),40,np.float32),beats=flags[None]))[0]
        starts=list(range(0,len(mel),640)); context=np.zeros((1,len(mel),768),np.float32)
        for reverse in (False,True):
            h=np.zeros((4,1,384),np.float32); c=h.copy()
            for start in reversed(starts) if reverse else starts:
                end=min(start+640,len(mel)); out,h,c=runtime.run('genelive_recurrent',dict(features=full[:,start:end],hidden=h,cell=c))
                if reverse:context[:,start:end,384:]=out[:,:,384:]
                else:context[:,start:end]=out
        bounded,prob=onset_context(runtime,mel,40,143,9%(240/143))
        np.testing.assert_allclose(context[0],bounded,atol=3e-5,rtol=2e-4)
        times=np.asarray([0.,6.39,6.4,9.,12.79,12.8,14.9]); indices=np.rint(times*100).astype(np.int64)
        beats=(times-9)*143/60; phases=np.rint((beats%1)/round(1/48,5)).astype(np.int64); nums=np.floor(beats%4).astype(np.int64)
        expected=runtime.run('acoustic',dict(specs=specs[None],indices=indices,phases=phases,numbers=nums,conditions=np.full((len(times),1),3,np.float32)))[0]
        actual=acoustic(runtime,specs,times,143,9)
        np.testing.assert_allclose(expected,actual,atol=3e-5,rtol=2e-4)
        bundle=load_patterns(); portable=load_bundle(models/'patterns.json')
        row=context_vector(bundle['patterns'][0]['variants'][0]['sequence'],modes=bundle['modes'])
        p=bundle['selector'].predict_proba(row[None]); q=portable['selector'].predict_proba(row[None])
        np.testing.assert_allclose(p,q,atol=1e-12)
        report=dict(onset_mel_max_absolute_error=onset_error,acoustic_logmel_max_absolute_error=acoustic_error,
            recurrent_and_cnn_chunk_parity_passed=True,acoustic_chunk_parity_passed=True,forest_probability_parity_passed=True,
            android_device_tested=False)
        (ROOT/'artifacts/portable-runtime-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))
    finally:runtime.close()


if __name__=='__main__':main()
