"""Pin inherited Git dependencies to the commits in the upstream lock file."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
manifest_path = ROOT/'Packages/manifest.json'
lock_path = ROOT/'Packages/packages-lock.json'
manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
lock = json.loads(lock_path.read_text(encoding='utf-8'))
manifest['dependencies'].pop('com.coplaydev.unity-mcp',None)
lock['dependencies'].pop('com.coplaydev.unity-mcp',None)
manifest['dependencies']['com.unity.nuget.newtonsoft-json'] = '3.0.2'
for name, value in list(manifest['dependencies'].items()):
    record = lock['dependencies'].get(name,{})
    if record.get('source') == 'git' and record.get('hash'):
        pinned = value.split('#',1)[0] + '#' + record['hash']
        manifest['dependencies'][name] = pinned
        record['version'] = pinned
if 'com.unity.nuget.newtonsoft-json' in lock['dependencies']:
    lock['dependencies']['com.unity.nuget.newtonsoft-json']['depth'] = 0
manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
lock_path.write_text(json.dumps(lock,indent=2)+'\n',encoding='utf-8')
