"""Make relocatable release archives and verify their complete ZIP payloads."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def archive_directory(source, destination, prefix):
    temporary = destination.with_suffix('.tmp')
    files = sorted(path for path in source.rglob('*') if path.is_file())
    with zipfile.ZipFile(temporary, 'w', zipfile.ZIP_DEFLATED, compresslevel=3) as archive:
        for path in files:
            archive.write(path, str(Path(prefix)/path.relative_to(source)))
    with zipfile.ZipFile(temporary) as archive:
        assert archive.testzip() is None
        assert len(archive.infolist()) == len(files)
    temporary.replace(destination)
    print(destination.name, destination.stat().st_size, 'bytes', flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest-only', action='store_true')
    args = parser.parse_args()
    artifacts = ROOT/'artifacts'
    release = artifacts/'companion/OurSekai'
    assert (release/'OurSekai.exe').is_file()
    assert (release/'AutoChart/OurSekai.Worker.exe').is_file()
    if not args.manifest_only:
        shutil.copy2(ROOT/'docs/companion-user-guide.md', release/'使用说明.md')
        archive_directory(release, artifacts/'OurSekai-Companion-Windows.zip', 'OurSekai')
        archive_directory(artifacts/'portable-models', artifacts/'our-sekai-mobile-models-v3.zip', '')
    rows = []
    for name in ('OurSekai-Companion-Windows.zip', 'OurSekai-Companion-Android-arm64.apk',
                 'our-sekai-companion-source.zip', 'our-sekai-mobile-models-v3.zip', 'our-sekai-models-v3.zip','our-sekai-training-corpus-v3.zip'):
        path = artifacts/name
        assert path.is_file(), name
        digest = hashlib.sha256()
        with path.open('rb') as source:
            for block in iter(lambda: source.read(4*1024*1024), b''): digest.update(block)
        rows.append(dict(path=name, size=path.stat().st_size, sha256=digest.hexdigest()))
    (artifacts/'companion-release-manifest.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')


if __name__ == '__main__': main()
