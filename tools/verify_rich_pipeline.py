"""Run real audio, audit final expression/paths, compare identical NS objects."""
from pathlib import Path
import argparse
import json
import sys
import uuid
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'backend')]
from portable.pipeline import run
from companion.song_package import load_json


def statistics(objects):
    slides=[o for o in objects if o['object_kind']=='slide']
    return dict(objects=len(objects),slides=len(slides),single_flicks=sum(o['points'][0]['kind']=='flick' for o in objects),
        critical_objects=sum(bool(o['points'][0]['critical']) for o in objects),
        tail_flicks=sum(o['points'][-1]['direction']!='none' for o in slides),
        holds_with_controls=sum(len(o['points'])>2 for o in slides),
        hold_turns=int(sum(sum(a*b<0 for a,b in zip(np.diff([p['lane']+p['width']/2 for p in o['points']]),
            np.diff([p['lane']+p['width']/2 for p in o['points']])[1:])) for o in slides)),
        hold_width_changes=sum(len({p['width'] for p in o['points']})>1 for o in slides))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--crop-start',type=float,default=20)
    parser.add_argument('--report',type=Path,default=ROOT/'artifacts/rich-pipeline-report.json')
    args=parser.parse_args()
    import hybrid_pjsk.expression as expression
    original=expression.apply_expression; observed=[]
    def audit(objects,*args,**kwargs):
        result,report=original(objects,*args,**kwargs)
        assert [o['start'] for o in result]==[o['start'] for o in objects]
        assert [o.get('end') for o in result]==[o.get('end') for o in objects]
        observed.append(dict(before=objects,after=result,report=report))
        return result,report
    expression.apply_expression=audit
    fixture=ROOT/'../only my railgun/only_my_railgun_genelive_autoosu_genelive/mix.wav'
    job=ROOT/'artifacts/rich-test'/uuid.uuid4().hex
    stage=run(dict(audio=str(fixture.resolve()),crop_start=args.crop_start,crop_duration=60,difficulty='EXPERT',parameters=dict(bpm=143,phase=args.crop_start)),
        ROOT/'artifacts/portable-models',job,progress=lambda message,percent:print(percent,message,flush=True))
    from hybrid_pjsk.two_finger import project
    from portable.pipeline import Layout
    from portable.runtime import Runtime
    # Old geometry + unchanged acoustic object types, with the same candidates.
    runtime=Runtime(ROOT/'artifacts/portable-models')
    try:
        old,events,geometry,playability=project(observed[0]['before'],Layout(runtime,lambda:None),143,9,
            phrase_style=False,pattern_style=True,pattern_model=ROOT/'artifacts/rich-baseline-patterns.joblib')
        index=0
        for obj in old:
            for point in obj['points']:
                point.update(geometry[index]); index+=1
    finally:runtime.close()
    chart=load_json(stage/'chart.json')
    from mapper_pjsk.schema import parse_position_free
    from hybrid_pjsk.phrase_layout import read_objects
    final=read_objects(parse_position_free((stage/'chart.sus').read_text(encoding='utf-8'),9,True))
    report=dict(project=str(stage),old_pipeline=statistics(old),new_pipeline=statistics(final),
        expression=observed[0]['report'],same_input_onsets=True,
        actual_sus_verified=chart['two_finger']['exported_verification']['passed'],
        pattern_modes_used=len(chart['two_finger'].get('phrase_style',{}).get('pattern_usage',{})),
        hold_path_model=chart['two_finger'].get('phrase_style',{}).get('hold_path_model'),
        limits='One 60-second musical excerpt at explicit BPM; not a blind subjective playtest')
    args.report.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()
