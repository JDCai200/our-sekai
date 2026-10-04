"""Verify our output using an installed community player's JSON deserializer."""
import argparse
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--managed', required=True, type=Path)
    parser.add_argument('--compiler', required=True, type=Path)
    parser.add_argument('--score', required=True, type=Path)
    args = parser.parse_args()
    output = ROOT / '.build-tools/CommunityScoreCheck.exe'
    response = output.with_suffix('.rsp')
    options = ['/nologo','/nostdlib','/target:exe',f'/out:"{output}"']
    options += [f'/reference:"{path.resolve()}"' for path in args.managed.glob('*.dll')]
    options += [f'"{ROOT / "tools/CommunityScoreCheck.cs"}"']
    response.write_text('\n'.join(options), encoding='utf-8-sig')
    subprocess.run([str(args.compiler.resolve()), '/noconfig', '@'+str(response)], check=True)
    subprocess.run([str(output),str(args.managed.resolve()),str(args.score.resolve())],check=True)


if __name__ == '__main__':
    main()
