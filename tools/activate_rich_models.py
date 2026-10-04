"""Promote trained chart models and update checksum-locked release inputs."""
import hashlib
import json
from pathlib import Path
import shutil
import joblib

ROOT=Path(__file__).resolve().parents[1]
BACKEND=ROOT/'backend'
TRAINING=BACKEND/'diagnostics/pjsk_patterns_v3'


def main():
    report=json.loads((TRAINING/'training-report.json').read_text(encoding='utf-8'))
    assert report['selected_modes']>=100 and report['min_source_songs']>=3
    for key in ('single','slide'):
        metric=report['expression'][key]['validation']
        if metric['log_loss']>=metric['frequency_baseline_log_loss']:
            raise ValueError('Expression model did not improve held-out probability loss: '+key)
    target=BACKEND/'diagnostics/pjsk_patterns_v2'; target.mkdir(parents=True,exist_ok=True)
    path_report=json.loads((TRAINING/'slide-path-report.json').read_text(encoding='utf-8'))
    if path_report['scores']['validation']['log_loss']>=path_report['scores']['validation']['frequency_baseline_log_loss']:
        raise ValueError('Hold path selector did not improve held-out probability loss')
    for name in ('expression.json','slide_paths.json'):shutil.copy2(TRAINING/name,target/name)
    bundle=joblib.load(TRAINING/'patterns.joblib');bundle['rich_path_model']=True
    joblib.dump(bundle,target/'patterns.joblib',compress=3)
    active=BACKEND/'diagnostics/pjsk_patterns/active_model.json'
    active.write_text(json.dumps(dict(version=2,sha256=hashlib.sha256((target/'patterns.joblib').read_bytes()).hexdigest(),
        modes=report['selected_modes'],training_version='rich-v3'),indent=2),encoding='utf-8')
    registry_path=BACKEND/'model-lock.json'; registry=json.loads(registry_path.read_text(encoding='utf-8'))
    for extra in ('diagnostics/pjsk_patterns_v2/expression.json','diagnostics/pjsk_patterns_v2/slide_paths.json'):
        if not any(row['path']==extra for row in registry['files']):registry['files'].append(dict(path=extra))
    for row in registry['files']:
        path=BACKEND/row['path']; row.update(size=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    registry_path.write_text(json.dumps(registry,indent=2),encoding='utf-8')
    audit=ROOT/'docs/training-v3-report.json'; shutil.copy2(TRAINING/'training-report.json',audit)
    shutil.copy2(TRAINING/'slide-path-report.json',ROOT/'docs/slide-path-training-report.json')
    print('Activated rich models:',report['selected_modes'],'supported path clusters')


if __name__=='__main__':main()
