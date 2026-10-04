"""Integration tests against real inference and cooperative cancellation."""
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND = ROOT / 'backend'

def main():
    previous = BACKEND / '.smoke/new_bpm/project'
    original = json.loads((previous / 'manifest.json').read_text(encoding='utf-8'))
    job = BACKEND / '.smoke/existing'
    job.mkdir(parents=True, exist_ok=True)
    request = dict(schema=1, audio=str(previous/'audio.wav'), existing_manifest=str(previous/'manifest.json'),
                   difficulty='EXPERT', parameters={'bpm':143})
    (job/'request.json').write_text(json.dumps(request), encoding='utf-8')
    subprocess.run([sys.executable,'-s','-X','utf8',str(BACKEND/'worker.py'),str(job/'request.json')],check=True)
    response = json.loads((job/'response.json').read_text(encoding='utf-8'))
    import soundfile as sf
    import numpy as np
    before, sr = sf.read(previous/'audio.wav', dtype='float32')
    after, sr2 = sf.read(job/'project/audio.wav', dtype='float32')
    assert sr == sr2 == 44100
    assert len(before) == len(after) == 15 * sr
    assert not np.any(after[:9*sr])
    assert original['fillerSec'] == response['timeline']['fillerSec'] == 9
    assert response['timeline']['previewStartTimeSec'] == original['previewStartTimeSec']
    assert np.max(np.abs(before-after)) <= 1/32768
    cancel = BACKEND / '.smoke/cancel'
    cancel.mkdir(parents=True, exist_ok=True)
    (cancel/'request.json').write_text(json.dumps(request),encoding='utf-8')
    process = subprocess.Popen([sys.executable,'-s','-X','utf8',str(BACKEND/'worker.py'),str(cancel/'request.json')])
    try:
        deadline = time.monotonic() + 15
        while not (cancel/'status.json').exists() and time.monotonic() < deadline and process.poll() is None:
            time.sleep(.1)
        (cancel/'cancel').touch()
        assert process.wait(timeout=5) == 130
        assert not (cancel/'response.json').exists()
    finally:
        if process.poll() is None: process.kill(); process.wait()
    summary = dict(existing_package_audio_seconds=15, filler_seconds=9,
                   no_duplicate_padding=True, pcm_round_trip_error=float(np.max(np.abs(before-after))),
                   cancel_exit_code=130, cancel_published_project=False)
    (ROOT/'artifacts/integration-report.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))

if __name__ == '__main__': main()
