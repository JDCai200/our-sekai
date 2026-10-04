"""Check signature, offline permission scope, Python payload and locked models."""
import io
import hashlib
import json
import os
from pathlib import Path
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    apk = ROOT/'artifacts/OurSekai-Companion-Android-arm64.apk'
    config = json.loads((ROOT/'.build-tools/android/build-env.json').read_text())
    build = Path(config['android_home'])/'build-tools/35.0.0'
    env = dict(os.environ, JAVA_HOME=config['java_home'])
    subprocess.run([str(build/'apksigner.bat'), 'verify', '--verbose', str(apk)], env=env, check=True)
    # Legacy aapt interprets non-ASCII file arguments using a narrow encoding.
    permissions = subprocess.check_output([str(build/'aapt.exe'), 'dump', 'permissions', str(apk.relative_to(ROOT))], cwd=ROOT, env=env).decode('utf-8')
    assert 'android.permission.INTERNET' not in permissions
    with zipfile.ZipFile(apk) as archive:
        assert archive.testzip() is None
        lock = json.loads(archive.read('assets/models/model-lock.json'))
        for row in lock['files']:
            assert hashlib.sha256(archive.read('assets/models/'+row['path'])).hexdigest() == row['sha256']
        with zipfile.ZipFile(io.BytesIO(archive.read('assets/chaquopy/app.imy'))) as python:
            modules = python.namelist()
            for module in ('portable/pipeline', 'portable/runtime', 'portable/android_bridge', 'companion/song_package','hybrid_pjsk/expression','hybrid_pjsk/slide_paths'):
                assert any(name.startswith(module+'.') for name in modules), module
        assert all(name.startswith('lib/arm64-v8a/') for name in archive.namelist() if name.startswith('lib/'))
    report = dict(apk_bytes=apk.stat().st_size, zip_integrity_passed=True, model_hashes_passed=True,
        embedded_python_pipeline=True, abi='arm64-v8a', min_sdk=24, target_sdk=35,
        signature_verified=True, debug_signing=True, internet_permission=False,
        android_device_tested=False, supports_16kb_pages=False)
    (ROOT/'artifacts/android-apk-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__': main()
