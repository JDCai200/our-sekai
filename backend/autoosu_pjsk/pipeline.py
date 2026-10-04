"""Timing-only runnable stage; PJSK types/layout require separate training."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.signal import find_peaks
from mapper_pjsk.full_song import estimate_tempo
from mapper_pjsk.sus_export import export_preview
from .objects import pack_objects


def select_times(probability, duration, threshold=.3, min_distance_ms=40, shift_ms=0,
                 bpm=None, offset=0., snap_ms=20, start=0.):
    if not 0 <= threshold <= 1 or min_distance_ms < 10 or not 0 <= snap_ms <= 100:
        raise ValueError('Invalid selection settings')
    if bpm is not None and not 10 <= bpm <= 1000:
        raise ValueError('Invalid BPM')
    peaks, _ = find_peaks(probability, height=threshold, distance=max(1, round(min_distance_ms/10)))
    chosen = {}
    for frame in peaks:
        raw = start + frame*.01 + shift_ms/1000
        t, division = raw, None
        if bpm is not None:
            for div in (4, 8, 16, 32):
                step = 60/bpm*4/div
                target = offset + round((raw-offset)/step)*step
                if abs(target-raw)*1000 <= snap_ms+1e-8:
                    t, division = target, div
                    break
        if not start <= t < start+duration:
            continue
        t = round(t, 6)
        candidate = dict(time=t, raw_time=round(raw, 6), score=float(probability[frame]),
                         source='AutoOsu-NP mix', grid_division=division)
        # Timing stage counts distinct instants; it never invents chords by
        # counting neighboring peaks at a single snapped time.
        if t not in chosen or chosen[t]['score'] < candidate['score']:
            chosen[t] = candidate
    return [chosen[t] for t in sorted(chosen)]


def audio_grid(mono, start=0., bpm=None):
    import librosa
    short = librosa.resample(mono, orig_sr=44100, target_sr=16000)
    if bpm is None:
        grid = estimate_tempo(short, 16000)
    else:
        if not 10 <= bpm <= 1000:
            raise ValueError('Invalid manual BPM')
        env = librosa.onset.onset_strength(y=short, sr=16000, hop_length=128)
        frames = librosa.onset.onset_detect(onset_envelope=env, sr=16000, hop_length=128)
        step = 60/bpm/4
        z = np.sum(env[frames]*np.exp(2j*np.pi*(frames*128/16000)/step)) / max(float(env[frames].sum()),1e-6)
        grid = dict(bpm=bpm,phase_seconds=float(np.angle(z)%(2*np.pi)/(2*np.pi)*step),
                    coherence=float(abs(z)),method='manual BPM; audio-estimated phase')
    _, beats = librosa.beat.beat_track(y=short, sr=16000, hop_length=128,
                                      bpm=grid['bpm'], units='time')
    phase = grid['phase_seconds']
    step = 60/grid['bpm']/4
    anchor = float(beats[0]) if len(beats) else phase
    anchor = phase + round((anchor-phase)/step)*step
    grid['offset'] = start + anchor
    grid['downbeat_verified'] = False
    return grid


def main():
    p = argparse.ArgumentParser()
    p.add_argument('audio'); p.add_argument('--output', required=True)
    p.add_argument('--title', default='AutoOsu timing draft')
    p.add_argument('--settings'); p.add_argument('--bpm', type=float)
    p.add_argument('--offset', type=float)
    p.add_argument('--difficulty', type=float)
    a = p.parse_args()
    output = Path(a.output)
    for suffix in ('.json', '.sus', '.np.npy'):
        if output.with_suffix(suffix).exists():
            raise FileExistsError('Output exists; choose a new output name')
    import torch
    from engine import decode
    from .backend import TimingModel, SHA256, COMMIT
    torch.set_num_threads(2)
    settings = dict(difficulty=3., threshold=.3, min_distance_ms=40, shift_ms=0, snap_ms=20)
    if a.settings:
        settings.update(json.loads(Path(a.settings).read_text(encoding='utf-8'))['selected'])
    if a.difficulty is not None:
        settings['difficulty'] = a.difficulty
    y = decode(Path(a.audio)).mean(axis=0)
    duration = len(y)/44100
    grid = audio_grid(y, bpm=a.bpm)
    if a.offset is not None:
        grid['offset'] = a.offset
    print(f'Audio {duration:.3f}s, BPM {grid["bpm"]}, anchor {grid["offset"]:.3f}s; loading AutoOsu', flush=True)
    model = TimingModel()
    specs = model.features(y)
    probabilities = model.predict(specs, grid['bpm'], grid['offset'], settings['difficulty'])
    times = select_times(probabilities, duration, bpm=grid['bpm'], offset=grid['offset'],
                         **{k: settings[k] for k in ('threshold', 'min_distance_ms', 'shift_ms', 'snap_ms')})
    # Do not export hallucinations inside digitally silent lead-in/tail audio.
    before_silence_filter = len(times)
    times = [c for c in times if np.max(np.abs(y[max(0,round(c['raw_time']*44100)-882):min(len(y),round(c['raw_time']*44100)+882)]), initial=0)>1e-5]
    events = [dict(time=c['time'], kind='tap', critical=False, trace=False, direction='none', chain=None) for c in times]
    sus, verification = export_preview(events, grid['bpm'], a.title)
    result = dict(schema='pjsk-position-free-v1', time_reference='audio_absolute',
                  audio=str(Path(a.audio).resolve()), duration=duration, events=events,
                  objects=pack_objects(events), candidates=times, tempo=grid, settings=settings,
                  model=dict(source='https://github.com/issyun/AutoOsu', commit=COMMIT, checkpoint_sha256=SHA256,
                             stage='pretrained note placement; 4K note selection NOT used', frame_ms=10),
                  type_mode='all taps as explicit timing placeholders; PJSK structure model not trained',
                  silence_filtered_candidates=before_silence_filter-len(times),
                  positions_generated=False, widths_generated=False, sus_preview=verification,
                  warnings=['Only the timing stage is implemented and pretrained.',
                            'Mixture audio is used; vocal/melody source selection is not yet implemented.',
                            'osu! difficulty is a model conditioning value, not the PJSK level.',
                            'Beat anchor is estimated from audio and has unresolved metrical ambiguity.',
                            'Tap type and geometry are placeholders, not model predictions.'])
    output.with_suffix('.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    output.with_suffix('.sus').write_text(sus, encoding='utf-8')
    np.save(output.with_suffix('.np.npy'), probabilities)
    print(f'Saved {len(events)} timing candidates, round-trip {verification["round_trip_verified"]}', flush=True)


if __name__ == '__main__':
    main()
