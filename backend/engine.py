"""Runtime audio decoding, deliberately independent of the former editor tool."""
import os
import subprocess
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
FFMPEG = Path(os.environ.get('OUR_SEKAI_FFMPEG', str(ROOT / 'bin/ffmpeg.exe')))

def decode(path, start=0, duration=0):
    if not FFMPEG.is_file():
        raise FileNotFoundError('FFmpeg is missing; run the offline runtime setup first')
    command = [str(FFMPEG), '-nostdin', '-v', 'error', '-ss', str(start), '-i', str(path)]
    if duration:
        command += ['-t', str(duration)]
    command += ['-f', 'f32le', '-acodec', 'pcm_f32le', '-ar', '44100', '-ac', '2', 'pipe:1']
    result = subprocess.run(command, capture_output=True, check=True,
                            creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
    samples = np.frombuffer(result.stdout, np.float32).reshape(-1, 2).T.copy()
    if samples.shape[1] < 4410:
        raise ValueError('音频不足 0.1 秒，或开始时间超出音频长度')
    return samples

def separate(*args, **kwargs):
    raise ValueError('Our Sekai 使用完整音乐采音，不提供声部分离模式')
