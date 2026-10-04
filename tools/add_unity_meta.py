"""Generate stable metadata for new, code-only Unity assets."""
from pathlib import Path
import uuid

ROOT = Path(__file__).resolve().parents[1]
paths = list((ROOT/'Assets/OurSekai').rglob('*')) + [ROOT/'Assets/OurSekai',
    ROOT/'Assets/CustomMusicScoreManager/Runtime/UI/ScreenLayerCustomMusicScoreManager.Generation.cs']
for path in paths:
    if path.suffix == '.meta': continue
    meta = path.with_name(path.name + '.meta')
    if meta.exists(): continue
    guid = uuid.uuid5(uuid.NAMESPACE_URL, 'org.oursekai.game/' + path.relative_to(ROOT).as_posix()).hex
    if path.is_dir():
        content = f'fileFormatVersion: 2\nguid: {guid}\nfolderAsset: yes\nDefaultImporter:\n  externalObjects: {{}}\n  userData: \n  assetBundleName: \n  assetBundleVariant: \n'
    else:
        content = f'fileFormatVersion: 2\nguid: {guid}\nMonoImporter:\n  externalObjects: {{}}\n  serializedVersion: 2\n  defaultReferences: []\n  executionOrder: 0\n  icon: {{instanceID: 0}}\n  userData: \n  assetBundleName: \n  assetBundleVariant: \n'
    meta.write_text(content,encoding='utf-8')
