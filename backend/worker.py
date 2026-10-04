"""Offline job protocol for the Unity front end. No network calls at runtime."""
import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import re
import sys
import threading
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if getattr(sys, 'frozen', False):
    ROOT = Path(sys._MEIPASS)
sys.path.insert(0, str(ROOT))

def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + f'.{os.getpid()}.{threading.get_ident()}.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    os.replace(temporary, path)

def finite(value, name, lower=0, upper=math.inf):
    number = float(value)
    if not math.isfinite(number) or not lower <= number <= upper:
        raise ValueError(f'{name} 必须在 {lower}–{upper} 之间')
    return number

def timeline(source_duration, filler=9., preview=0.):
    source_duration = finite(source_duration, '音乐时长', .1)
    filler = finite(filler, '前置空白', 0, 120)
    preview = finite(preview, '试听起点', 0, source_duration)
    return dict(fillerSec=filler, audioDurationSec=source_duration + filler,
                sourceDurationSec=source_duration,
                secForMusicScoreMaker=math.ceil(source_duration + 2),
                previewStartTimeSec=preview + filler)

def verify_models():
    registry = json.loads((ROOT / 'model-lock.json').read_text(encoding='utf-8'))
    for row in registry['files']:
        path = ROOT / row['path']
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError('模型文件缺失或校验失败：' + row['path'])

def checked_chart(path):
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    check = data['two_finger']['exported_verification']
    if not (check.get('passed') and check.get('visual_overlap_checked') and check.get('visual_overlap_count') == 0):
        raise ValueError('实际导出谱面的双指／无重叠核验失败')
    if data.get('genelive_postprocessing', {}).get('snap_division', 0):
        grid = data['sus_preview']['grid_verification']
        if not grid.get('passed') or grid.get('off_grid_count') != 0:
            raise ValueError('实际 SUS 节奏网格核验失败')
    return data

class ProgressLog(io.TextIOBase):
    def __init__(self, file, progress):
        self.file, self.progress, self.buffer = file, progress, ''
        self.percent = 15
    def write(self, text):
        self.file.write(text)
        self.file.flush()
        self.buffer += text
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            if line.strip():
                match = re.search(r'GenéLive (\d+)%', line)
                if match:
                    self.percent = max(self.percent, 15 + int(match.group(1)) * .65)
                    message = line[match.end():].strip()
                elif line.startswith('NS:'):
                    self.percent, message = 85, '正在生成音符类型和长条'
                elif line.startswith('Learned geometry'):
                    self.percent, message = 90, '正在安排音符位置和长条路径'
                else:
                    continue
                self.progress(message, self.percent)
        return len(text)
    def flush(self):
        self.file.flush()

def run(request_path):
    request_path = Path(request_path).resolve()
    job = request_path.parent
    status, response = job / 'status.json', job / 'response.json'
    cancel = job / 'cancel'
    stop = threading.Event()
    def progress(message, percent):
        if cancel.exists():
            raise InterruptedError('已取消生成')
        atomic_json(status, dict(message=message, percent=percent))
    def watch_cancel():
        while not stop.wait(.25):
            if cancel.exists():
                atomic_json(status, dict(message='已取消生成', percent=-1))
                os._exit(130)  # Worker is disposable; Unity owns and cleans its staging directory.
    threading.Thread(target=watch_cancel, daemon=True).start()
    try:
        request = json.loads(request_path.read_text(encoding='utf-8-sig'))
        if request.get('schema') != 1:
            raise ValueError('不支持的生成请求版本')
        from hybrid_pjsk.genelive_settings import preset_settings, validate, command_arguments
        difficulty = request.get('difficulty', 'EXPERT').upper()
        values = preset_settings(difficulty)
        values.update(request.get('parameters') or {})
        settings = validate(values)
        layout = request.get('layout', 'patterns')
        if layout not in ('patterns', 'patterns24', 'model'):
            raise ValueError('不支持的排键方式')
        audio = Path(request['audio']).resolve()
        if not audio.is_file():
            raise FileNotFoundError('找不到歌曲文件')
        stage = job / 'project'
        stage.mkdir(exist_ok=False)
        progress('正在校验离线模型', 1)
        verify_models()
        from engine import decode
        import soundfile as sf
        progress('正在读取音乐', 5)
        existing = request.get('existing_manifest')
        if existing:
            old_path = Path(existing).resolve()
            old = json.loads(old_path.read_text(encoding='utf-8-sig'))
            expected = (old_path.parent / old['audioFileName']).resolve()
            if audio != expected:
                if hashlib.sha256(audio.read_bytes()).digest() != hashlib.sha256(expected.read_bytes()).digest():
                    raise ValueError('所选音频与歌曲包不一致')
            filler = finite(old['fillerSec'], '已有包前置空白', 0, 120)
            samples = decode(audio)
            duration = samples.shape[1] / 44100 - filler
            times = timeline(duration, filler)
            # Existing music already has its padding. Never add or subtract it again.
            times['previewStartTimeSec'] = finite(old.get('previewStartTimeSec', filler), '试听起点')
            settings['phase'] = settings['phase'] + filler if settings['phase'] is not None else None
            padded = stage / 'audio.wav'
            sf.write(padded, samples.T, 44100, subtype='PCM_16')
        else:
            start = finite(request.get('crop_start', 0), '原曲裁剪起点')
            length = finite(request.get('crop_duration', 0), '原曲裁剪长度')
            samples = decode(audio, start, length)
            duration = samples.shape[1] / 44100
            times = timeline(duration, request.get('filler', 9), request.get('preview_start', 0))
            filler = times['fillerSec']
            if settings['phase'] is not None:
                settings['phase'] = finite(settings['phase'] - start, '裁剪后节拍起点', 0, duration) + filler
            padded = stage / 'audio.wav'
            import numpy as np
            with sf.SoundFile(padded, 'w', samplerate=44100, channels=2, subtype='PCM_16') as output:
                output.write(np.zeros((round(filler * 44100), 2), dtype=np.float32))
                output.write(samples.T)
        title = request.get('title') or audio.stem
        manifest = dict(formatVersion=1, id=request.get('id') or os.urandom(6).hex(),
                        title=title, scoreTitle=title, userName='Our Sekai',
                        audioFileName='audio.wav', scoreFileName='score.json',
                        jacketFileName='jacket.png', videoFileName='',
                        musicDifficultyType=difficulty.lower(), playLevel=0,
                        fillerSec=filler, secForMusicScoreMaker=times['secForMusicScoreMaker'],
                        previewStartTimeSec=times['previewStartTimeSec'])
        atomic_json(stage / 'manifest.json', manifest)
        atomic_json(stage / 'generation-settings.json', dict(request=request, effective_parameters=settings,
                    timeline=times, source_sha256=hashlib.sha256(audio.read_bytes()).hexdigest()))
        progress('正在生成谱面', 15)
        from hybrid_pjsk.cli import main
        command = ['hybrid_pjsk.cli', '--audio', str(padded), '--output', str(stage / 'chart'),
                   '--title', title, '--project-manifest', str(stage / 'manifest.json'),
                   '--audio-shift', str(filler), '--layout-style', 'patterns' if layout == 'patterns24' else layout]
        if layout == 'patterns24':
            command += ['--pattern-model', str(ROOT / 'diagnostics/pjsk_patterns/patterns.joblib')]
        command += command_arguments(settings)
        old_argv = sys.argv
        try:
            sys.argv = command
            with (stage / 'generation.log').open('w', encoding='utf-8') as log:
                writer = ProgressLog(log, progress)
                with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                    main()
        finally:
            sys.argv = old_argv
        progress('正在核验实际导出谱面', 96)
        chart = checked_chart(stage / 'chart.json')
        if not chart['events']:
            raise ValueError('未生成可游玩的音符')
        atomic_json(response, dict(schema=1, success=True, project=str(stage), sus=str(stage / 'chart.sus'),
                    note_count=len(chart['events']), timeline=times, error=''))
        progress('谱面已生成，等待加入歌曲库', 100)
        return 0
    except Exception as exception:
        (job / 'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        atomic_json(response, dict(schema=1, success=False, error=str(exception)))
        atomic_json(status, dict(message='生成失败：' + str(exception), percent=-1))
        return 1
    finally:
        stop.set()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('request', nargs='?', type=Path)
    parser.add_argument('--check-models', action='store_true')
    args = parser.parse_args()
    if args.check_models:
        verify_models()
        print('Offline model checksums passed')
    elif args.request:
        raise SystemExit(run(args.request))
    else:
        parser.error('request.json is required')
