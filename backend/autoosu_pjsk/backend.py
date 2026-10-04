"""Verified upstream NP model; ten-ms frames, contextual overlapping inference."""
import hashlib
import importlib.util
from pathlib import Path
import numpy as np
import torch
import torchaudio

ROOT = Path(__file__).resolve().parents[1]
UPSTREAM = ROOT / 'models/autoosu'
CHECKPOINT = UPSTREAM / 'checkpoints/osu_model_v2.pt'
SHA256 = '6cd6c5b65a2ee926ca74de9bad2ebfa6d766e46bb65ae887577b7c2367d3045f'
COMMIT = 'b81dc6f43f6274eb37b6cffa0945e3e5e23cf1dd'
HYPERPARAMS = dict(bp_emb_dim=32, bn_emb_dim=16, diff_emb_dim=16,
                  np_hidden_size=256, np_num_layers=2, ns_pre_proj_size=32,
                  ns_hidden_size=256, ns_num_layers=2, action_emb_dim=32)


class TimingModel:
    def __init__(self):
        if hashlib.sha256(CHECKPOINT.read_bytes()).hexdigest() != SHA256:
            raise ValueError('AutoOsu checkpoint checksum mismatch')
        spec = importlib.util.spec_from_file_location('autoosu_upstream_models', UPSTREAM/'models.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.model = module.OsuModel(HYPERPARAMS)
        state = torch.load(CHECKPOINT, map_location='cpu', weights_only=True)
        self.model.load_state_dict(state['model_state_dict'], strict=True)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.mels = [torchaudio.transforms.MelSpectrogram(sample_rate=44100, n_fft=n,
                    hop_length=441, f_max=11000, n_mels=80, power=2) for n in (1024, 2048, 4096)]

    @torch.inference_mode()
    def features(self, mono):
        if len(mono) < 4096:
            raise ValueError('Audio shorter than one analysis window')
        y = torch.as_tensor(np.asarray(mono).copy(), dtype=torch.float32)
        # Match convert.py exactly: three STFT scales, natural log, no new
        # normalization or alternative mel filter conventions.
        # A silent band can be exactly zero despite upstream waveform epsilon.
        # Only floor zero/subnormal energies; nonzero musical features match.
        features = torch.stack([torch.log(mel(y + 1e-9).T.clamp_min(torch.finfo(torch.float32).tiny)) for mel in self.mels])
        if not torch.isfinite(features).all():
            raise ValueError('Non-finite audio features')
        return features

    @torch.inference_mode()
    def predict(self, specs, bpm, offset=0., difficulty=3., start=0., core_seconds=24., context_seconds=3.):
        if not 10 <= bpm <= 1000 or not np.isfinite(offset) or not 0 < difficulty <= 12:
            raise ValueError('Invalid timing/difficulty input')
        n = specs.shape[1]
        # offset/start are seconds on the source audio timeline; 49 phase bins
        # include the phase-one rounding bin used by upstream convert.py.
        beat = (torch.arange(n, dtype=torch.float64)*.01 + start-offset)*bpm/60
        phases = ((beat % 1)/round(1/48, 5)).round().long()
        numbers = ((beat % 4).floor()).long()
        diffs = torch.full((n, 1), difficulty)
        core = round(core_seconds*100); context = round(context_seconds*100)
        if core < 1 or context < 0:
            raise ValueError('Invalid context window')
        result = np.zeros(n, np.float32)
        for left in range(0, n, core):
            right = min(left+core, n)
            lo, hi = max(0, left-context), min(n, right+context)
            probability = self.model.np_forward(specs[:, lo:hi][None], phases[None, lo:hi],
                            numbers[None, lo:hi], diffs[None, lo:hi])[-1].reshape(-1)
            # Commit central frames only; same global spectrogram/clock in all
            # windows. No EOS and no independent four-second event sequences.
            result[left:right] = probability[left-lo:right-lo].cpu().numpy()
        if not np.isfinite(result).all():
            raise ValueError('Non-finite onset probabilities')
        return result
