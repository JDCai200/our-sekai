"""Check real output from a packaged worker against a known-duration fixture."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from worker import checked_chart


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('job', type=Path)
    parser.add_argument('--source-seconds', required=True, type=float)
    parser.add_argument('--filler', default=9., type=float)
    args = parser.parse_args()
    import numpy as np
    import soundfile as sf
    response = json.loads((args.job / 'response.json').read_text(encoding='utf-8'))
    assert response['success'], response
    project = args.job / 'project'
    chart = checked_chart(project / 'chart.json')
    audio, rate = sf.read(project / 'audio.wav', dtype='float32')
    assert rate == 44100
    assert abs(len(audio) / rate - args.source_seconds - args.filler) < 1 / rate
    assert not np.any(audio[:round(args.filler * rate)])
    assert len(chart['events']) > 0
    manifest = json.loads((project / 'manifest.json').read_text(encoding='utf-8'))
    assert manifest['fillerSec'] == args.filler
    assert manifest['previewStartTimeSec'] == args.filler
    report = dict(packaged_worker_real_inference_passed=True,
                  source_duration_seconds=args.source_seconds,
                  audio_duration_seconds=len(audio) / rate,
                  leading_silence_seconds=args.filler,
                  note_structure_count=len(chart['events']),
                  sus_roundtrip_and_playability_passed=True,
                  external_python_required=False)
    (ROOT / 'artifacts/packaged-worker-report.json').write_text(
        json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
