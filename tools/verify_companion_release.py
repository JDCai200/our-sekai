"""Exercise the packaged desktop UI's job/export path without a Unity editor."""
import json
from pathlib import Path
import subprocess
import wave
import uuid

ROOT=Path(__file__).resolve().parents[1]


def main():
    executable=ROOT/'artifacts/companion/OurSekai/OurSekai.exe'
    output=ROOT/'artifacts/companion-test'/('frozen-'+uuid.uuid4().hex)
    output.mkdir(parents=True)
    request=output/'request.json'; report=output/'report.json'
    request.write_text(json.dumps(dict(schema=1,audio=str(ROOT/'backend/smoke.wav'),difficulty='EXPERT',
        parameters=dict(bpm=143,phase=0))),encoding='utf-8')
    selected=output/'Custom output folder'
    subprocess.run([str(executable),'--generate',str(request),'--output-directory',str(output),'--songs-directory',str(selected),'--report',str(report)],check=True,timeout=300)
    response=json.loads(report.read_text(encoding='utf-8'))
    assert response['success'],response
    project=Path(response['project']); score=json.loads((project/'score.json').read_text(encoding='utf-8'))
    assert project.parent == selected
    jobs=list((output/'Jobs').iterdir()); assert len(jobs)==1
    chart=json.loads((jobs[0]/'project/chart.json').read_text(encoding='utf-8'))
    assert chart['expression']['training_songs']==427
    style=chart['two_finger']['phrase_style']
    assert style['modes']==236 and style['hold_path_model']['modes']==64
    assert 'whole-shape six-lane' in style['hold_path_model']['policy']
    assert len(score['NoteList'])==response['note_count']>0
    with wave.open(str(project/'audio.wav'),'rb') as audio:
        assert audio.getframerate()==44100 and audio.getnframes()==15*44100
        assert not any(audio.readframes(9*44100))
    package=Path(response['package'])
    import zipfile
    with zipfile.ZipFile(package) as archive:
        assert archive.testzip() is None
        assert {'manifest.json','score.json','audio.wav','chart.sus'}.issubset(archive.namelist())
        assert not any('generation.log' in name or 'request.json' in name for name in archive.namelist())
    response.update(packaged_gui_to_worker_to_song_zip_passed=True,leading_silence_seconds=9,
        source_duration_seconds=6,output_duration_seconds=15,unity_required=False,custom_output_directory_passed=True)
    (ROOT/'artifacts/companion-integration-report.json').write_text(json.dumps(response,indent=2),encoding='utf-8')
    print(json.dumps(response,indent=2))


if __name__=='__main__':main()
