"""Export/restore a checksum-locked model pack, separate from the Git source."""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1] / 'backend'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('pack', type=Path)
    parser.add_argument('--export', action='store_true')
    args = parser.parse_args()
    registry = json.loads((ROOT / 'model-lock.json').read_text(encoding='utf-8'))
    if args.export:
        args.pack.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.pack, 'x', zipfile.ZIP_DEFLATED) as archive:
            for item in registry['files']:
                data = (ROOT / item['path']).read_bytes()
                if hashlib.sha256(data).hexdigest() != item['sha256']: raise ValueError(item['path'])
                archive.writestr(item['path'], data)
        return
    with zipfile.ZipFile(args.pack) as archive:
        # Only allow the fixed registry paths, and verify the entire pack before writing.
        contents = []
        for item in registry['files']:
            data = archive.read(item['path'])
            if len(data) != item['size'] or hashlib.sha256(data).hexdigest() != item['sha256']:
                raise ValueError('模型校验失败：' + item['path'])
            contents.append((item['path'], data))
        for relative, data in contents:
            target = ROOT / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    print('Model pack restored; all checksums passed')

if __name__ == '__main__': main()
