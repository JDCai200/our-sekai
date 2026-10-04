"""Run the trained layout and independent constraints across all cached songs."""
import argparse,json,hashlib
from collections import Counter
import soundfile as sf
import torch
import numpy as np
from autoosu_pjsk.backend import TimingModel
from mapper_pjsk.sus_export import export_preview
from .data import prepare,OUT,ROOT,dump
from .model import ObjectSelector,free_select
from .decode import to_objects
from .layout_model import load_layout,OUT as LAYOUT_OUT
from .two_finger import project,verify_exported_witness
from .phrase_layout import OUT as PHRASE_OUT


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--layout-style',choices=['patterns','phrases'],default='patterns')
    parser.add_argument('--pattern-model');parser.add_argument('--output');args=parser.parse_args()
    torch.set_num_threads(2);loader=TimingModel();state=torch.load(OUT/'object_selector.pth',map_location='cpu',weights_only=True)
    ns=ObjectSelector(loader.model,state['placement_size']);ns.restore(state);ns.eval();layout=load_layout();rows=prepare(loader);report=[]
    for row in rows:
        song=row['song'];audio=ROOT/'reference'/f'{song["id"]:04d}_{song["bundle"]}.mp3'
        actions=free_select(ns,row['acoustic'],row['placement'])
        objects,_,_=to_objects(row['times'],actions,row['bpm'],sf.info(audio).duration)
        objects,events,geometry,proof=project(objects,layout,row['bpm'],song['filler'],phrase_style=args.layout_style=='phrases',pattern_style=args.layout_style=='patterns',pattern_model=args.pattern_model)
        text,sus=export_preview(events,row['bpm'],audio_shift=song['filler'],geometry=geometry)
        exported=verify_exported_witness(objects,proof['witness'],text,row['bpm'],song['filler'])
        deviations=[]
        for record in proof['witness']['paths']:
            for p,g in zip(objects[record['object']]['points'],record['points']):
                if '_preferred_interval' in p:
                    lane,width=p['_preferred_interval']
                    deviations.append((abs(g['lane']+g['width']/2-lane-width/2),abs(g['width']-width)))
        item=dict(id=song['id'],split=song['split'],heads=len(row['times']),objects=len(objects),
                  converted_slides=len(proof['repairs']),verification=proof['verification'],exported_verification=exported,head_instants_preserved=proof['head_instants_preserved'],
                  layout_style=args.layout_style,pattern_usage=proof['phrase_style'].get('pattern_usage'),
                  constraint_center_change_lanes=float(np.mean([d[0] for d in deviations])) if deviations else None,
                  constraint_width_change_lanes=float(np.mean([d[1] for d in deviations])) if deviations else None,
                  shaped_slides=sum(o['object_kind']=='slide' and len(o['points'])>2 for o in objects),
                  sus_roundtrip=sus['round_trip_verified'],export_max_error_ms=sus['max_timing_error_ms'],
                  widths=dict(Counter(g['width'] for g in geometry)),lanes=dict(Counter(g['lane'] for g in geometry)))
        report.append(item);print(json.dumps(item),flush=True)
    from .pattern_layout import OUT as PATTERN_OUT
    destination=PATTERN_OUT if args.layout_style=='patterns' else PHRASE_OUT
    destination.mkdir(parents=True,exist_ok=True)
    from pathlib import Path
    target=Path(args.output) if args.output else destination/'pipeline_verification.json';target.parent.mkdir(parents=True,exist_ok=True)
    dump(target,dict(songs=report,all_passed=all(r['verification']['passed'] and r['exported_verification']['passed'] and r['sus_roundtrip'] for r in report),
        pattern_model=str(Path(args.pattern_model).resolve()) if args.pattern_model else None,
        pattern_sha256=hashlib.sha256(Path(args.pattern_model).read_bytes()).hexdigest() if args.pattern_model else None,
        layout_sha256=hashlib.sha256((LAYOUT_OUT/'layout_model.pth').read_bytes()).hexdigest(),
        protocol='Existing candidate excerpts; no stem filtering in this layout-only regression; all learned geometry and two-finger witnesses checked.',
        limitations='A feasibility witness under the configured lane-speed and release assumptions, not a human playtest.'))


if __name__=='__main__':main()
