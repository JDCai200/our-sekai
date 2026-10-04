"""Exercise the mobile inference chain on desktop, including package regeneration."""
import json
from pathlib import Path
import sys
import uuid
import wave

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'backend')]
from portable.pipeline import run
from companion.song_package import publish, export_zip, extract_existing, load_json


def main():
    home = ROOT / 'artifacts/portable-test' / uuid.uuid4().hex
    models = ROOT / 'artifacts/portable-models'
    request = dict(audio=str(ROOT/'backend/smoke.wav'), difficulty='EXPERT', parameters=dict(bpm=143, phase=0))
    stage = run(request, models, home/'new', progress=lambda message, percent: print(percent, message, flush=True))
    project = publish(stage, home/'Songs')
    with wave.open(str(project/'audio.wav'), 'rb') as audio:
        assert audio.getnframes() == 15 * 44100
        assert not any(audio.readframes(9 * 44100))
    package = project.with_suffix('.zip'); export_zip(project, package)
    manifest, audio = extract_existing(package, home/'existing')
    request.update(audio=str(audio), existing_manifest=str(manifest), filler=40, crop_start=3)
    regenerated = run(request, models, home/'regenerated')
    assert (regenerated/'audio.wav').read_bytes() == (project/'audio.wav').read_bytes()
    assert load_json(regenerated/'manifest.json')['fillerSec'] == 9
    cancelled = home/'cancelled'; cancelled.mkdir(); (cancelled/'cancel').touch()
    try:
        run(request, models, cancelled)
    except InterruptedError:
        pass
    else:
        raise AssertionError('Cancel was ignored')
    assert not (cancelled/'project').exists()
    report = dict(success=True, project=str(project), package=str(package), note_count=len(load_json(project/'score.json')['NoteList']),
        leading_silence_seconds=9, output_duration_seconds=15, existing_package_audio_unchanged=True,
        cancellation_before_generation_passed=True, android_device_tested=False)
    (ROOT/'artifacts/portable-job-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__': main()
