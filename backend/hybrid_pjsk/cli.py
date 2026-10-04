"""Generate fixed GenéLive heads, then PJSK structures through adapted NS."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
from autoosu_pjsk.backend import TimingModel,ROOT,SHA256
from mapper_pjsk.sus_export import export_preview
from .model import ObjectSelector,acoustic_features,free_select
from .data import OUT,PRIOR,placement_features,dump
from .decode import to_objects
from .genelive_settings import FIELDS,validate


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--audio');p.add_argument('--analysis');p.add_argument('--features')
    p.add_argument('--realign-grid',type=int,choices=[8,16,32],help='Reprocess cached probabilities onto the SUS grid without audio model inference')
    p.add_argument('--output',required=True);p.add_argument('--title',default='GenéLive + AutoOsu PJSK draft')
    p.add_argument('--selector',default=str(OUT/'object_selector.pth'))
    p.add_argument('--project-manifest',help='OpenSekai manifest: verifies padded audio and reads fillerSec')
    p.add_argument('--audio-shift',type=float,default=0.,help='Project FillerSec in seconds; subtracted only for SUS export')
    p.add_argument('--layout-model',help='Official-chart-trained layout checkpoint')
    p.add_argument('--pattern-model',help='Explicit vocabulary checkpoint; defaults to the verified active vocabulary')
    p.add_argument('--legacy-placeholder',action='store_true',help='Disable learned geometry and two-finger projection')
    p.add_argument('--layout-style',choices=['patterns','phrases','model'],default='patterns',help='Learned motif selection, legacy retrieval or original GRU only')
    for key,label,help_text in FIELDS:
        p.add_argument('--genelive-'+key.replace('_','-'),type=str if key=='method' else float,
                       help=label+'：'+help_text)
    args=p.parse_args()
    overrides={key:getattr(args,'genelive_'+key) for key,_,_ in FIELDS if getattr(args,'genelive_'+key) is not None}
    try:settings=validate(overrides)
    except ValueError as e:p.error(str(e))
    if args.analysis and overrides:p.error('--analysis 复用已有采音；请用 --audio 重新采音以调整 GenéLive 参数')
    if args.realign_grid and not args.analysis:p.error('--realign-grid requires --analysis')
    torch.set_num_threads(2)
    if not args.audio and not args.analysis:raise ValueError('Provide --audio or --analysis')
    output=Path(args.output).resolve()
    if output.with_suffix('.json').exists() or output.with_suffix('.sus').exists():raise FileExistsError('Output exists; choose a new name')
    from engine import decode
    from genelive_backend import analyze_genelive,infer,filter_inaudible_events
    if args.analysis:
        analysis_path=Path(args.analysis).resolve()
    else:
        detection=output.parent/(output.name+'_genelive')
        prior={'checkpoint':str(PRIOR/'model_head_finetuned.pth')}
        if detection.exists():raise FileExistsError('Detection output exists; reuse with --analysis')
        grid_min_time=args.audio_shift
        if args.project_manifest:
            grid_min_time=float(json.loads(Path(args.project_manifest).read_text(encoding='utf-8-sig'))['fillerSec'])
        print('GenéLive settings: '+json.dumps(settings,ensure_ascii=False),flush=True)
        analyze_genelive(args.audio,detection,start=0,duration=0,condition=settings['condition'],threshold=settings['threshold'],
            bpm=settings['bpm'],phase=settings['phase'],
            checkpoint_path=prior['checkpoint'],feature_path=detection/'features.npy',
            postprocess_config=dict(method=settings['method'],min_distance=settings['min_distance'],
                                   shift_ms=settings['shift_ms'],snap_ms=settings['snap_ms'],
                                   snap_division=settings['snap_division'],grid_min_time=grid_min_time,
                                   basis='User-configurable GenéLive settings; defaults from development calibration'),
            progress=lambda message,percent:print(f'GenéLive {percent}% {message}',flush=True))
        analysis_path=detection/'analysis.json'
    original_features=analysis_path.parent/'features.npy'
    data=json.loads(analysis_path.read_text(encoding='utf-8'))
    print('Using GenéLive mixed-audio candidates; stem separation and stem-based filtering disabled',flush=True)
    source=Path(args.audio or data['audio']).resolve()
    audio_shift=args.audio_shift
    if args.project_manifest:
        manifest_path=Path(args.project_manifest).resolve()
        manifest=json.loads(manifest_path.read_text(encoding='utf-8-sig'))
        project_audio=manifest_path.parent/manifest['audioFileName']
        if hashlib.sha256(project_audio.read_bytes()).digest()!=hashlib.sha256(source.read_bytes()).digest():
            raise ValueError('Analysis audio differs from OpenSekai project audio; cannot infer a safe origin')
        audio_shift=float(manifest['fillerSec'])
    if not np.isfinite(audio_shift) or audio_shift<0:raise ValueError('Invalid project audio origin')
    division=args.realign_grid or int(data.get('postprocessing',{}).get('snap_division',0))
    if division and (args.analysis or data['postprocessing']['grid_origin']!=audio_shift):
        from genelive_backend import events_from_probabilities
        post=data['postprocessing'];parameters=data['parameters']
        probabilities=analysis_path.parent/'frame_probabilities.npy'
        data['events']=events_from_probabilities(np.load(probabilities),max(0,float(data['start'])-5),
            float(data['start']),float(data['start'])+float(data['duration']),float(data['estimated_bpm']),audio_shift,
            parameters['threshold'],post.get('min_distance',4),post.get('method','upstream'),
            post.get('shift_ms',0),post.get('snap_ms',0),division,audio_shift)
        data['postprocessing']=dict(post,musical_grid_origin=post.get('musical_grid_origin',post['grid_origin']),
            grid_origin=audio_shift,grid_min_time=audio_shift,snap_division=division,grid_reference='SUS origin')
        data['original_genelive_analysis']=str(analysis_path)
        analysis_path=output.parent/(output.name+'_grid_analysis.json')
        if analysis_path.exists():raise FileExistsError('Grid analysis output exists')
        dump(analysis_path,data)
    full_audio=decode(source)
    candidates,audibility=filter_inaudible_events([e for e in data['events'] if e.get('selected',True)],full_audio)
    from .audio_origin import constrain_candidates
    candidates,origin_adjustments=constrain_candidates(candidates,audio_shift,data.get('postprocessing'))
    print('Project audio origin: '+json.dumps(origin_adjustments,ensure_ascii=False),flush=True)
    times=[float(e['time']) for e in candidates]
    if not times:raise ValueError('No audible GenéLive head instants')
    if times!=sorted(set(times)):raise ValueError('GenéLive analysis must provide distinct chronological head instants')
    start=float(data['start']);duration=float(data['duration']);bpm=float(data['estimated_bpm'])
    features_path=Path(args.features) if args.features else original_features
    if not features_path.exists():
        # Extract contextual features only. Imported onset times stay frozen,
        # regardless of probabilities produced during this feature pass.
        context=max(0,start-5)
        wave=decode(source,context,duration+start-context+5).mean(axis=0)
        model_versions=data.get('model_versions',{})
        checkpoint=model_versions.get('checkpoint')
        origin=(data['estimated_beat_offset']-context)%(240/bpm)
        features_path=output.parent/(output.name+'_genelive_features.npy')
        infer(wave,bpm,origin,condition=data.get('parameters',{}).get('condition',30),
              checkpoint_path=checkpoint,feature_path=features_path)
    placement=placement_features(times,[float(e['score']) for e in candidates],bpm,
        np.load(features_path),max(0,start-5))
    state=torch.load(args.selector,map_location='cpu',weights_only=True)
    loader=TimingModel()
    selector=ObjectSelector(loader.model,state['placement_size']);selector.restore(state);selector.eval()
    full_wave=full_audio.mean(axis=0)
    source_duration=len(full_wave)/44100
    wave=full_wave[round(start*44100):min(len(full_wave),round((start+duration)*44100))]
    specs=loader.features(wave)
    acoustic=acoustic_features(loader,specs,times,start,bpm,data['postprocessing']['grid_origin'])
    print(f'NS: fixed {len(times)} GenéLive head instants; decoding PJSK object sets',flush=True)
    actions=free_select(selector,acoustic,torch.from_numpy(placement))
    objects,events,agreement=to_objects(times,actions,bpm,source_duration,snap_division=division)
    from .expression import apply_expression
    objects,expression=apply_expression(objects,bpm,audio_shift)
    geometry=None;playability=None;layout_version=None
    if not args.legacy_placeholder:
        from .layout_model import load_layout,OUT as LAYOUT_OUT
        from .two_finger import project
        print('Learned geometry + two-finger projection',flush=True)
        layout_path=Path(args.layout_model) if args.layout_model else LAYOUT_OUT/'layout_model.pth'
        layout_version=dict(checkpoint=str(layout_path.resolve()),sha256=hashlib.sha256(layout_path.read_bytes()).hexdigest())
        objects,events,geometry,playability=project(objects,load_layout(layout_path),bpm,audio_shift,
            phrase_style=args.layout_style=='phrases',pattern_style=args.layout_style=='patterns',pattern_model=args.pattern_model)
        agreement['object_counts']=dict(Counter(o['object_kind'] if o['object_kind']=='slide' else o['points'][0]['kind'] for o in objects))
    sus,verification=export_preview(events,bpm,args.title,audio_shift=audio_shift,geometry=geometry)
    if division:
        from .grid_verification import verify_sus_grid
        verification['grid_verification']=verify_sus_grid(sus,bpm,division)
        if not verification['grid_verification']['passed']:raise AssertionError(verification['grid_verification'])
    if playability is not None:
        from .two_finger import verify_exported_witness
        playability['exported_verification']=verify_exported_witness(objects,playability['witness'],sus,bpm,audio_shift)
        if not playability['exported_verification']['passed']:raise AssertionError(playability['exported_verification'])
    if geometry is not None:
        events=[dict(e,**g) for e,g in zip(events,geometry)]
        for record in playability['witness']['paths']:
            obj=objects[record['object']]
            obj['points']=[dict(p,**g) for p,g in zip(obj['points'],record['points'])]
    result=dict(schema='pjsk-layout-v1' if geometry is not None else 'pjsk-position-free-v1',time_reference='audio_absolute',audio=str(source),
                start=start,duration=duration,source_duration=source_duration,events=events,objects=objects,expression=expression,genelive_candidates=candidates,
                onset_agreement=agreement,tempo=data['rhythm'],selector=str(Path(args.selector).resolve()),
                selector_sha256=hashlib.sha256(Path(args.selector).read_bytes()).hexdigest(),
                selection_architecture='AutoOsu transferred NS memory; PJSK factorized heads; GenéLive acoustic context',
                autoosu_source_sha256=SHA256,genelive_analysis=str(analysis_path),
                genelive_model_versions=data.get('model_versions',{}),
                genelive_parameters=data.get('parameters',{}),genelive_postprocessing=data.get('postprocessing',{}),
                positions_generated=geometry is not None,widths_generated=geometry is not None,sus_preview=verification,
                two_finger=playability,layout_model=layout_version,
                onset_policy='genelive_mix',stem_separation_enabled=False,
                imported_salience_selection=data.get('salience_selection'),
                audibility_filter=audibility,audio_origin_adjustments=origin_adjustments,
                warnings=['Experimental PJSK NS transfer; head times preserved after explicit project-origin preprocessing.',
                          'Slide path controls may be transferred from training-chart motifs; hidden controls add no new acoustic judgments.',
                          'Geometry learned from official charts and projected to conservative two-finger motion constraints.' if geometry is not None else 'Legacy placeholder geometry.',
                          'Infeasible/out-of-audio slides become explicit tap fallbacks, without deleting heads.',
                          'No Trace or hidden endpoints in training excerpts; unsupported types disabled.'])
    dump(output.with_suffix('.json'),result)
    output.with_suffix('.sus').write_text(sus,encoding='utf-8')
    torch.save(actions,output.with_suffix('.actions.pth'))
    print(json.dumps(dict(objects=agreement['object_counts'],heads=agreement['input_head_instants'],
                         structure_repairs=len(agreement['repairs']),sus=verification),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
