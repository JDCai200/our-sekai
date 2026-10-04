"""Create a source ZIP using Git's tracked/untracked-but-not-ignored inventory."""
import argparse
import subprocess
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'artifacts/our-sekai-source.zip')
    parser.add_argument('--companion', action='store_true', help='Export independent companion sources only')
    args = parser.parse_args()
    if args.companion and args.output == ROOT/'artifacts/our-sekai-source.zip':
        args.output = ROOT/'artifacts/our-sekai-companion-source.zip'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(['git','-C',str(ROOT),'ls-files','-z','--cached','--others','--exclude-standard'],
                            capture_output=True,check=True)
    files = sorted(set(p.decode('utf-8') for p in result.stdout.split(b'\0') if p))
    if args.companion:
        files = [p for p in files if p.startswith(('companion/','android/','backend/','tools/','docs/'))
                 or p in ('README.md','LICENSE','.gitignore')]
    included = 0
    with zipfile.ZipFile(args.output,'w',zipfile.ZIP_DEFLATED) as archive:
        for relative in files:
            path = ROOT / relative
            if path.is_file(): archive.write(path,('our-sekai-companion/' if args.companion else 'our-sekai/')+relative); included += 1
    print(f'Exported {included} source/resource files to {args.output}')

if __name__ == '__main__': main()
