"""Build a self-contained ARM64 companion APK using a local Android toolchain."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'backend')]
from portable.pipeline import verify_models

PYTHON_FILES = ['worker.py','charts.py','genelive_backend.py',
    'portable/__init__.py','portable/dsp.py','portable/runtime.py','portable/pipeline.py','portable/android_bridge.py',
    'hybrid_pjsk/__init__.py','hybrid_pjsk/audio_origin.py','hybrid_pjsk/decode.py','hybrid_pjsk/genelive_settings.py',
    'hybrid_pjsk/grid_verification.py','hybrid_pjsk/layout_features.py','hybrid_pjsk/paths.py','hybrid_pjsk/expression.py','hybrid_pjsk/slide_paths.py',
    'hybrid_pjsk/pattern_layout.py','hybrid_pjsk/portable_forest.py','hybrid_pjsk/phrase_layout.py',
    'hybrid_pjsk/two_finger.py','hybrid_pjsk/vocabulary.py','autoosu_pjsk/__init__.py','autoosu_pjsk/objects.py',
    'mapper_pjsk/__init__.py','mapper_pjsk/schema.py','mapper_pjsk/sus_export.py']


def main():
    config=json.loads((ROOT/'.build-tools/android/build-env.json').read_text(encoding='utf-8'))
    models=ROOT/'artifacts/portable-models'; verify_models(models)
    app=ROOT/'android/app/src/main'; python=app/'python'
    for relative in PYTHON_FILES:
        destination=python/relative; destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'backend'/relative,destination)
    for name in ('__init__.py','song_package.py'):
        destination=python/'companion'/name; destination.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT/'companion'/name,destination)
    target=app/'assets/models'; target.mkdir(parents=True,exist_ok=True)
    registry=json.loads((models/'model-lock.json').read_text())
    for row in registry['files']+[{"path":"model-lock.json"}]:shutil.copy2(models/row['path'],target/row['path'])
    # Retain model code attribution in the APK alongside the embedded assets.
    licenses=app/'assets/licenses'; licenses.mkdir(parents=True,exist_ok=True)
    shutil.copy2(ROOT/'LICENSE',licenses/'OpenSekai-MIT.txt')
    shutil.copy2(ROOT/'backend/models/genelive/LICENSE',licenses/'Genelive-MIT.txt')
    env=os.environ.copy(); env['JAVA_HOME']=config['java_home']; env['ANDROID_HOME']=config['android_home']
    env['OURSEKAI_BUILD_PYTHON']=config['build_python']; env['ANDROID_USER_HOME']=str(ROOT/'.build-tools/android/android-user')
    env['PATH']=str(Path(config['java_home'])/'bin')+os.pathsep+env.get('PATH','')
    command=[config['gradle'],'--gradle-user-home',str(ROOT/'.build-tools/android/gradle-cache'),
        '--no-daemon','--console=plain',':app:assembleDebug']
    result=subprocess.run(command,cwd=ROOT/'android',env=env)
    if result.returncode:raise SystemExit(result.returncode)
    source=ROOT/'android/app/build/outputs/apk/debug/app-debug.apk'
    destination=ROOT/'artifacts/OurSekai-Companion-Android-arm64.apk'
    shutil.copy2(source,destination)
    print('APK built:',destination,flush=True)


if __name__=='__main__':main()
