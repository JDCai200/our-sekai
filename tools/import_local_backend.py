"""One-time migration of our existing inference source and trusted model files.

Usage: python tools/import_local_backend.py PATH_TO_EXISTING_TOOL
Copies inference dependencies only. Never copies songs, training corpora or venvs.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PACKAGES = ('hybrid_pjsk', 'autoosu_pjsk', 'mapper_pjsk')
MODEL_FILES = (
    'diagnostics/genelive_finetune_conservative/model_head_finetuned.pth',
    'diagnostics/genelive_autoosu/object_selector.pth',
    'diagnostics/pjsk_layout/layout_model.pth',
    'diagnostics/pjsk_patterns/patterns.joblib',
    'diagnostics/pjsk_patterns/active_model.json',
    'diagnostics/pjsk_patterns_v2/patterns.joblib',
    'models/autoosu/checkpoints/osu_model_v2.pt',
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    source = args.source.resolve()
    destination = ROOT / 'backend'
    destination.mkdir(exist_ok=True)
    for package in SOURCE_PACKAGES:
        for file in (source / package).glob('*.py'):
            target = destination / package / file.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, target)
    for name in ('charts.py', 'rhythm.py', 'genelive_backend.py'):
        shutil.copy2(source / name, destination / name)
    for package in ('genelive', 'autoosu'):
        upstream = source / 'models' / package
        for file in upstream.rglob('*'):
            relative = file.relative_to(upstream)
            if '.git' in relative.parts or '__pycache__' in relative.parts:
                continue
            if file.is_file() and (file.suffix == '.py' or 'LICENSE' in file.name.upper()):
                target = destination / 'models' / package / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(file, target)
    registry = []
    for relative in MODEL_FILES:
        file = source / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, target)
        registry.append(dict(path=relative, size=file.stat().st_size,
                             sha256=hashlib.sha256(file.read_bytes()).hexdigest()))
    (destination / 'model-lock.json').write_text(
        json.dumps(dict(schema=1, files=registry), indent=2), encoding='utf-8')
    # The legacy CLI read an absolute developer path out of a training report.
    # In this independent project, use the checkpoint beside the runtime code.
    cli = destination / 'hybrid_pjsk/cli.py'
    text = cli.read_text(encoding='utf-8')
    text = text.replace("prior=json.loads((PRIOR/'report.json').read_text(encoding='utf-8'))",
                        "prior={'checkpoint':str(PRIOR/'model_head_finetuned.pth')}")
    cli.write_text(text, encoding='utf-8')
    print(f'Imported inference source and {len(registry)} model assets to {destination}')

if __name__ == '__main__':
    main()
