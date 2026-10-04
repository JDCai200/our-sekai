"""Song-disjoint calibration; audio-only tempo; test never selects settings."""
import itertools
import json
from collections import Counter
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
from charts import match_times
from mapper_pjsk.schema import parse_position_free
from .backend import ROOT, TimingModel, SHA256, COMMIT
from .pipeline import audio_grid, select_times
from .objects import pack_objects, unpack_objects, head_window
from genelive_backend import events_from_probabilities

OUT = ROOT/'diagnostics/autoosu_pjsk'


def dump(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def summarize(metrics):
    totals = {k: sum(m[k] for m in metrics) for k in ('predicted', 'reference', 'matched')}
    totals['extra'] = totals['predicted']-totals['matched']
    totals['missed'] = totals['reference']-totals['matched']
    totals['recall'] = totals['matched']/max(totals['reference'], 1)
    totals['extra_ratio'] = totals['extra']/max(totals['predicted'], 1)
    totals['false_negative_ratio'] = totals['missed']/max(totals['reference'], 1)
    totals['precision'] = totals['matched']/max(totals['predicted'], 1)
    totals['f1'] = 2*totals['matched']/max(totals['predicted']+totals['reference'], 1)
    return totals


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    manifest = json.loads((ROOT/'diagnostics/genelive_finetune_conservative/manifest.json').read_text(encoding='utf-8'))
    baseline = json.loads((ROOT/'diagnostics/genelive_finetune_conservative/report.json').read_text(encoding='utf-8'))
    rows = []
    print('Loading verified AutoOsu NP checkpoint', flush=True)
    model = TimingModel()
    full_counts = Counter(); cross_boundary = 0
    difficulties = (2., 3., 4., 5.)
    for song in manifest['rows']:
        stem = f'{song["id"]:04d}'
        chart = parse_position_free((ROOT/'reference'/f'{stem}_master.sus').read_text(encoding='utf-8-sig'), song['filler'])
        objects = pack_objects(chart['events'])
        restored = pack_objects(unpack_objects(objects))
        assert restored == objects
        full_counts.update(e['kind'] for e in chart['events'])
        for offset in (0., 8., 16.):
            start = song['start']+offset
            owned = head_window(objects, start, 4.)
            cross_boundary += sum(o['object_kind']=='slide' and o['end']>=start+4 for o in owned)
            # A long object crossing a window retains every original point.
            for obj in owned:
                if obj['object_kind']=='slide':
                    assert obj['points'][-1]['time']==obj['end']
        dump(OUT/f'{stem}_objects.json', dict(schema='pjsk-atomic-objects-v1', objects=objects))
        cached = ROOT/'diagnostics/genelive_finetune_conservative'/stem/'mix.wav'
        if cached.exists():
            y, sr = sf.read(cached, dtype='float32', always_2d=True)
            assert sr == 44100
            y = y.mean(axis=1)
        else:
            from engine import decode
            y = decode(ROOT/'reference'/f'{stem}_{song["bundle"]}.mp3', song['start'], song['duration']).mean(axis=0)
        grid_path = OUT/f'{stem}_grid.json'
        if grid_path.exists():
            grid = json.loads(grid_path.read_text(encoding='utf-8'))
        else:
            grid = audio_grid(y, song['start']); dump(grid_path, grid)
        specs_path = OUT/f'{stem}_features.npy'
        if specs_path.exists():
            specs = torch.from_numpy(np.load(specs_path))
        else:
            specs = model.features(y); np.save(specs_path, specs.numpy())
        probs = {}
        for difficulty in difficulties:
            path = OUT/f'{stem}_d{difficulty:g}_np.npy'
            if not path.exists():
                probability = model.predict(specs, grid['bpm'], grid['offset'], difficulty, start=song['start'])
                np.save(path, probability)
            probs[difficulty] = np.load(path)
        kinds = {'tap', 'flick', 'trace', 'slide_start'}
        head_times = [e['time'] for e in chart['events'] if e['kind'] in kinds and song['start']<=e['time']<song['start']+song['duration']]
        visible_times = [e['time'] for e in chart['events'] if e['kind'] not in {'slide_start_hidden', 'slide_end_hidden', 'slide_hidden', 'slide_attach'} and song['start']<=e['time']<song['start']+song['duration']]
        previous = ROOT/'diagnostics/genelive_finetune_conservative'
        data = json.loads((previous/stem/'analysis.json').read_text(encoding='utf-8'))
        previous_events = events_from_probabilities(np.load(previous/f'{stem}_finetuned_probs.npy'),
            max(0,song['start']-5),song['start'],song['start']+song['duration'],
            data['estimated_bpm'], data['postprocessing']['grid_origin'],
            threshold=baseline['selected']['threshold'],
            **{k:data['postprocessing'][k] for k in ('method','min_distance','shift_ms','snap_ms')})
        rows.append(dict(song=song, grid=grid, probs=probs, reference=head_times, all_visible=visible_times,
                         previous_times=[e['time'] for e in previous_events]))
        print(f'{stem}: audio BPM {grid["bpm"]:.3f}, head instants {len(set(head_times))}', flush=True)
    def run(row, settings):
        song, grid = row['song'], row['grid']
        selected = select_times(row['probs'][settings['difficulty']], song['duration'],
                               start=song['start'], bpm=grid['bpm'], offset=grid['offset'],
                               **{k: settings[k] for k in ('threshold','min_distance_ms','shift_ms','snap_ms')})
        return [c['time'] for c in selected]
    validation = [r for r in rows if r['song']['split']=='validation']
    trials = []
    for difficulty, threshold, distance, shift, snap in itertools.product(difficulties,
            (.01,.03,.05,.1,.15,.16,.17,.18,.19,.2,.25,.3,.4,.5,.6,.7,.8),
            (40,80,120), (-20.,0.,20.), (0.,20.)):
        settings = dict(difficulty=difficulty, threshold=threshold, min_distance_ms=distance, shift_ms=shift, snap_ms=snap)
        metric = summarize([match_times(run(r, settings), r['reference'], .05) for r in validation])
        trials.append(dict(settings=settings, validation=metric))
    # Reusing an already inspected test split cannot constitute a new blind test.
    comparable_prior = {split:summarize([match_times(r['previous_times'],r['reference'],.05)
                       for r in rows if r['song']['split']==split]) for split in ('train','validation','test')}
    required = comparable_prior['validation']['recall']
    feasible = [t for t in trials if t['validation']['recall'] >= required]
    chosen = min(feasible, key=lambda t: (t['validation']['extra_ratio'], -t['validation']['f1'])) if feasible else max(trials, key=lambda t: t['validation']['f1'])
    settings = chosen['settings']
    results = {}; per_song = []
    for split in ('train','validation','test'):
        selected_rows = [r for r in rows if r['song']['split']==split]
        for r in selected_rows:
            prediction = run(r, settings)
            per_song.append(dict(song=r['song'], audio_grid=r['grid'],
                                 head_onsets=match_times(prediction, r['reference'], .05),
                                 all_visible=match_times(prediction, r['all_visible'], .05)))
        results[split] = summarize([match_times(run(r, settings), r['reference'], .05) for r in selected_rows])
    report = dict(selected=settings, selection='Lowest validation extra ratio meeting prior validation recall; otherwise best validation F1',
                  recall_constraint=required, recall_constraint_feasible=bool(feasible), results=results,
                  prior_genelive=comparable_prior, historical_genelive_old_parser=baseline['finetuned'], per_song=per_song,
                  architecture=dict(model='AutoOsu pretrained NP only', commit=COMMIT, checkpoint_sha256=SHA256,
                                    audio='mix; three STFT scales; 10ms frames', tempo='audio-only per excerpt',
                                    context='24-second cores, three-second left/right context'),
                  object_representation=dict(full_reference_events=sum(full_counts.values()),
                                             counts=dict(full_counts), atomic_object_roundtrip=True,
                                             cross_boundary_slides_retained=cross_boundary),
                  limitations=['14 songs, one 30-second excerpt each; no full-chart quality claim.',
                               'Existing held-out songs have previously been inspected; exploratory comparison only.',
                               'No reference tempo, per-song offset fitting or test-selected settings.',
                               'Head onsets count distinct tap/flick/trace/visible slide starts, not chords or all acoustic notes.',
                               'NP was trained on osu!mania action times including tails; PJSK types/source selection not trained.',
                               'Comparison to GenéLive is historical and uses the same excerpts/tolerance; both pipelines differ.'])
    report['limitations'].append('Both predictions re-evaluated using the full SUS parser: tick-removal overlays are no longer mistaken for tap heads; historical scores use different labels.')
    dump(OUT/'report.json', report); dump(OUT/'settings.json', dict(selected=settings))
    dump(OUT/'trials.json', trials)
    dump(OUT/'provenance.json', dict(source='https://github.com/issyun/AutoOsu',commit=COMMIT,checkpoint='osu_model_v2.pt',sha256=SHA256))
    print(json.dumps(dict(selected=settings,results=results,object_representation=report['object_representation']),ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
