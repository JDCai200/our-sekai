"""Add exact audio windows/filterbanks and hashes to exported ONNX assets."""
import hashlib
import json
from pathlib import Path
import sys
import shutil
import numpy as np
import librosa
import torch
import torchaudio

ROOT=Path(__file__).resolve().parents[1]


def main():
    out=ROOT/'artifacts/portable-models'
    filters=dict(onset_window=torch.hann_window(2048).numpy(),
        onset_mel=librosa.filters.mel(sr=16000,n_fft=2048,fmin=30,n_mels=229,htk=True),
        tempo_mel=librosa.filters.mel(sr=22050,n_fft=2048,n_mels=128))
    for n in (1024,2048,4096):
        transform=torchaudio.transforms.MelSpectrogram(sample_rate=44100,n_fft=n,hop_length=441,f_max=11000,n_mels=80,power=2)
        filters[f'window_{n}']=transform.spectrogram.window.numpy()
        filters[f'mel_{n}']=transform.mel_scale.fb.numpy()
        # Float32 FFT kernels differ most in the epsilon-only silent bands.
        # Preserve the desktop model's exact silence feature, not FFT roundoff.
        silence=transform(torch.full((44100,),1e-9,dtype=torch.float32))
        filters[f'silence_{n}']=torch.log(silence.clamp_min(torch.finfo(torch.float32).tiny))[:,30].numpy()
    np.savez(out/'dsp.npz',**filters)
    shutil.copy2(ROOT/'backend/diagnostics/pjsk_patterns_v2/expression.json',out/'expression.json')
    shutil.copy2(ROOT/'backend/diagnostics/pjsk_patterns_v2/slide_paths.json',out/'slide_paths.json')
    names=['dsp.npz','patterns.json','parity-report.json','expression.json','slide_paths.json']+[row['file'] for row in json.loads((out/'parity-report.json').read_text())['neural_stages']]
    registry=dict(schema=1,files=[dict(path=name,size=(out/name).stat().st_size,
        sha256=hashlib.sha256((out/name).read_bytes()).hexdigest()) for name in names])
    (out/'model-lock.json').write_text(json.dumps(registry,indent=2),encoding='utf-8')
    print('Portable model assets checksum-locked:',len(names))


if __name__=='__main__':main()
