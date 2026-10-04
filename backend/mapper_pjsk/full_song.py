"""Run the saved four-second adapter over an entire song without reloading it."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np
from .schema import Codec, validate
from .sus_export import export_preview


def estimate_tempo(y, sr):
    import librosa
    hop = 128
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    initial = float(np.asarray(librosa.feature.tempo(onset_envelope=env, sr=sr, hop_length=hop)).ravel()[0])
    frames = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=hop)
    times = frames * hop / sr
    weights = np.minimum(env[frames] / max(float(np.percentile(env[frames], 90)), 1e-6), 1)
    candidates = set()
    for factor in (.5, 1, 2):
        center = initial * factor
        if 100 <= center <= 240:
            candidates.update(np.round(np.arange(max(100, center*.96), min(240, center*1.04), .02), 4))
    if not candidates or len(times) < 8:
        raise ValueError('Insufficient audio evidence to estimate tempo')
    scores = []
    for bpm in sorted(candidates):
        step = 60 / bpm / 4
        z = np.sum(weights * np.exp(2j*np.pi*times/step)) / weights.sum()
        scores.append((float(abs(z)), float(bpm), float(np.angle(z) % (2*np.pi) / (2*np.pi) * step)))
    score, bpm, phase = max(scores)
    nearest = min(scores, key=lambda row: abs(row[1]-round(bpm)))
    if abs(bpm-round(bpm)) < .15 and nearest[0] >= score-.01:
        bpm = float(round(bpm))
    return dict(bpm=bpm, initial_bpm=initial, phase_seconds=phase,
                coherence=score, method='audio-only onset periodicity, 100-240 BPM prior',
                note='Tempo is an audio estimate; export uses fine ticks, not forced beat snapping.')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('audio'); p.add_argument('--output', required=True)
    p.add_argument('--title', default='Mapperatorinator preview')
    p.add_argument('--max-events', type=int, default=60)
    a = p.parse_args()
    output = Path(a.output)
    if output.with_suffix('.json').exists() or output.with_suffix('.sus').exists():
        raise FileExistsError('Output already exists; use a new output name')
    import torch, librosa
    from engine import decode
    from .model import ROOT, load_base, PositionFreeAdapter
    torch.set_num_threads(2)
    np.random.seed(20261002); torch.manual_seed(20261002)
    audio = Path(a.audio).resolve()
    wave = decode(audio).mean(axis=0)
    duration = len(wave)/44100
    y = librosa.resample(wave, orig_sr=44100, target_sr=16000).astype(np.float32)
    tempo = estimate_tempo(y, 16000)
    print(f'Audio {duration:.3f}s; estimated BPM {tempo["bpm"]}; loading model', flush=True)
    adapter_path = ROOT/'diagnostics/mapper_pjsk/adapter.pth'
    state = torch.load(adapter_path, map_location='cpu', weights_only=True)
    model = PositionFreeAdapter(load_base(), Codec(state['duration'], state['max_chains']))
    model.restore(state); model.eval()
    window = state['duration']; samples = round(window*16000)
    events, windows = [], []
    next_chain = 0
    for offset in range(0, len(y), samples):
        start = offset/16000
        clip = y[offset:offset+samples]
        length = len(clip)/16000
        if np.max(np.abs(clip)) < .001:
            windows.append(dict(start=start, duration=length, skipped='silence'))
            print(f'{start:.1f}s: silence skipped', flush=True)
            continue
        padded = np.pad(clip, (0, samples-len(clip)))
        encoded = model.encode_audio(torch.from_numpy(padded))
        result = model.generate_events(encoded, max_events=a.max_events)
        # Remove complete chains extending into padded audio, never invent a tail.
        beyond = {e['chain'] for e in result['events'] if e['chain'] is not None and e['time'] >= length}
        chosen = [dict(e) for e in result['events'] if e['time'] < length and e['chain'] not in beyond]
        ids = {c: next_chain+n for n,c in enumerate(sorted({e['chain'] for e in chosen if e['chain'] is not None}))}
        next_chain += len(ids)
        for e in chosen:
            e['time'] = round(e['time']+start, 6)
            if e['chain'] is not None: e['chain'] = ids[e['chain']]
        events.extend(chosen)
        windows.append(dict(start=start, duration=length, events=len(chosen),
                            terminated=result['terminated'],
                            unfinished_chain_events_removed=result['unfinished_chain_events_removed'],
                            padding_events_removed=len(result['events'])-len(chosen)))
        print(f'{start:.1f}-{start+length:.1f}s: {len(chosen)} events; EOS {result["terminated"]}; incomplete removed {result["unfinished_chain_events_removed"]}', flush=True)
        # Keep a recovery file while inference runs; final JSON is written after verification.
        output.with_suffix('.progress.json').write_text(json.dumps(dict(events=events, windows=windows), ensure_ascii=False, indent=2), encoding='utf-8')
    validate(events)
    sus, export = export_preview(events, tempo['bpm'], a.title)
    result = dict(schema='pjsk-position-free-v1', time_reference='audio_absolute',
                  audio=str(audio), audio_sha256=hashlib.sha256(audio.read_bytes()).hexdigest(), duration=duration,
                  adapter=str(adapter_path), adapter_sha256=hashlib.sha256(adapter_path.read_bytes()).hexdigest(),
                  generation='Mapperatorinator PJSK adapter free generation; independent four-second windows',
                  positions_generated=False, widths_generated=False, events=events,
                  kind_counts=dict(Counter(e['kind'] for e in events)), windows=windows, tempo=tempo,
                  sus_preview=export,
                  warnings=['Experimental draft: timing quality has not been validated for this song.',
                            'Independent windows cannot predict cross-window slides.',
                            'SUS geometry is a temporary preview, not model-predicted PJSK layout.',
                            'Unfinished chains are removed; no artificial endpoints or onset-detector fallback.'])
    output.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    output.with_suffix('.sus').write_text(sus, encoding='utf-8')
    print(json.dumps(dict(events=len(events), kinds=result['kind_counts'], export=export, output=str(output)), ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
