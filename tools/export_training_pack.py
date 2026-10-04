"""Backup the exact chart-only training corpus with portable provenance."""
import hashlib
import json
from pathlib import Path
import zipfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    corpus=ROOT/'backend/diagnostics/pjsk_patterns_v3/corpus'
    manifest=json.loads((corpus/'manifest.json').read_text(encoding='utf-8'))
    output=ROOT/'artifacts/our-sekai-training-corpus-v3.zip'
    temporary=output.with_suffix('.tmp')
    with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as archive:
        for row in manifest['rows']:
            relative=f'charts/{row["id"]:04d}_master.sus'; path=corpus/relative
            if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError(relative)
            archive.write(path,'corpus/'+relative); row['path']=relative
        for name in ('musics.json','musicDifficulties.json','train_chunks.json','validation_chunks.json','test_chunks.json'):
            archive.write(corpus/name,'corpus/'+name)
        archive.writestr('corpus/manifest.json',json.dumps(manifest,ensure_ascii=False,indent=2))
    with zipfile.ZipFile(temporary) as archive:assert archive.testzip() is None
    temporary.replace(output)
    print('Training corpus backup:',output.stat().st_size,'bytes; no audio or local paths')


if __name__=='__main__':main()
