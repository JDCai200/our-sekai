"""Compile changed runtime C# against an existing community build, without Unity.

This checks API compatibility only; it is not a Unity game build or play test.
"""
import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--compiler', required=True, type=Path)
    parser.add_argument('--managed', required=True, type=Path)
    parser.add_argument('--defines', default='UNITY_STANDALONE_WIN,UNITY_STANDALONE')
    args = parser.parse_args()
    output = ROOT / '.build-tools/csharp-check'
    output.mkdir(parents=True, exist_ok=True)
    sources = list((ROOT / 'Assets/OurSekai/Runtime').rglob('*.cs'))
    sources += list((ROOT / 'Assets/CustomMusicScoreManager/Runtime/UI').glob('ScreenLayerCustomMusicScoreManager*.cs'))
    sources.append(ROOT / 'Assets/Scripts/Assembly-CSharp/Sekai/MusicScoreMaker/Ingame/Presenters/MusicScoreMakerEntryPoint.cs')
    references = [f'/reference:"{p.resolve()}"' for p in args.managed.glob('*.dll')]
    options = ['/nologo', '/nostdlib', '/nowarn:0436', '/target:library', '/langversion:latest', '/codepage:65001',
               '/define:' + args.defines, f'/out:"{output / "OurSekai.RuntimeCheck.dll"}"']
    response = output / 'compile.rsp'
    response.write_text('\n'.join(options + references + [f'"{p}"' for p in sources]), encoding='utf-8-sig')
    return subprocess.run([str(args.compiler.resolve()), '/noconfig', '@' + str(response)], check=False).returncode

if __name__ == '__main__':
    raise SystemExit(main())
