"""Conservative non-crossing two-finger projection and independent witness checks.

The witness uses lane units, piecewise-linear hold paths, 18 lanes/second
maximum movement, and 25/60 ms tap/flick release intervals. It demonstrates
feasibility under these explicit assumptions, not universal human comfort.
"""
from collections import defaultdict
from copy import deepcopy
import itertools
import numpy as np
from .layout_features import INTERVALS,features
from autoosu_pjsk.objects import unpack_objects
from mapper_pjsk.schema import KINDS


def flatten(objects):
    rows=[]
    for i,o in enumerate(objects):
        for j,p in enumerate(o['points']):rows.append((i,j,p))
    return sorted(rows,key=lambda r:(r[2]['time'],KINDS.index(r[2]['kind']),r[0]))


def visual_overlaps(paths):
    """Find positive-area horizontal collisions over continuous linear paths.

    Singles occupy their judgment instant; holds occupy their entire span.
    Shared edges are allowed. Splitting at all knots and edge-crossing roots
    catches crossings between knots, including disjoint segment endpoints.
    """
    def edges(points,time):
        clocks=[p['time'] for p in points]
        left=float(np.interp(time,clocks,[p['lane'] for p in points]))
        right=float(np.interp(time,clocks,[p['lane']+p['width'] for p in points]))
        return left,right
    collisions=[]
    for n,a in enumerate(paths):
        pa=a['points']
        for b in paths[n+1:]:
            pb=b['points'];start=max(pa[0]['time'],pb[0]['time']);end=min(pa[-1]['time'],pb[-1]['time'])
            if start>end+1e-9:continue
            knots=sorted({start,end,*[p['time'] for p in pa+pb if start<p['time']<end]})
            samples=list(knots)
            for t0,t1 in zip(knots,knots[1:]):
                al,ar=edges(pa,t0);bl,br=edges(pb,t0)
                al1,ar1=edges(pa,t1);bl1,br1=edges(pb,t1)
                cuts=[0.,1.]
                for v0,v1 in ((ar-bl,ar1-bl1),(br-al,br1-al1)):
                    if abs(v1-v0)>1e-12:
                        root=-v0/(v1-v0)
                        if 0<root<1:cuts.append(root)
                cuts.sort()
                samples.extend(t0+(t1-t0)*(x+y)/2 for x,y in zip(cuts,cuts[1:]))
            for time in samples:
                al,ar=edges(pa,time);bl,br=edges(pb,time)
                if min(ar,br)-max(al,bl)>1e-7:
                    collisions.append(dict(objects=[a['object'],b['object']],time=float(time)))
                    break
    return collisions


def project(objects,model,bpm,offset=0.,max_speed=18.,tap_release=.025,flick_release=.060,phrase_style=True,pattern_style=False,pattern_model=None):
    style=None
    if pattern_style:
        from .pattern_layout import apply_patterns
        objects,style=apply_patterns(objects,bpm,offset,model_path=pattern_model)
    elif phrase_style:
        from .phrase_layout import apply_phrases
        objects,style=apply_phrases(objects,bpm,offset)
    objects=deepcopy(objects);raw=flatten(objects)
    logits=model.predict(features([p for _,_,p in raw],bpm,offset))
    logits=logits.numpy() if hasattr(logits,'numpy') else np.asarray(logits)
    preferences={(i,j):logits[n] for n,(i,j,_) in enumerate(raw)}
    intervals=np.asarray(INTERVALS);lanes=intervals[:,0];widths=intervals[:,1];centers=lanes+widths/2
    costs={}
    for i,j,p in raw:
        cost=-preferences[(i,j)].astype(np.float64)
        if '_preferred_interval' in p:
            target_lane,target_width=p['_preferred_interval']
            # The previous GRU collapses toward width 3. Phrase widths are
            # coordinated roles, so retain their signal instead of allowing
            # the single-key classifier to overwrite most width-2 targets.
            cost=.35*cost+1.5*np.abs(centers-target_lane-target_width/2)+3*np.abs(widths-target_width)
        costs[(i,j)]=cost
    repairs=[];initial_heads={o['start'] for o in objects}
    # Reserve enough room for SUS rounding of both ends of a transition.
    rounding_margin=2*60/bpm/480;planning_speed=max_speed*.97
    tail_guard=max(.08,min(.16,60/bpm*.25));tail_rects=[];reserved_lanes={}
    groups=defaultdict(list)
    for i,o in enumerate(objects):groups[o['start']].append(i)
    if any(len(g)>2 for g in groups.values()):raise ValueError('More than two simultaneous objects; retrain/limit count upstream')

    def trajectory(index,hand,state,local_reservations=None):
        o=objects[index];lo,hi=(0.,6.) if hand==0 else (6.,12.)
        previous_time,previous_x,_,_=state;out=[];cost=0.
        for j,p in enumerate(o['points']):
            dt=p['time']-previous_time
            if dt<0:return None
            if j:
                # A short hold segment can lose more than the 3% speed
                # reserve when its endpoints round to SUS ticks. Bound the
                # motion by BOTH clocks, without moving acoustic timestamps.
                # SUS stores BPM to 6 decimals; parsed seconds round to 6
                # decimals. Reserve 2 us for endpoint/parser rounding.
                tick=round((p['time']-offset)*bpm/60*480)
                previous_tick=round((previous_time-offset)*bpm/60*480)
                exported_dt=(tick-previous_tick)*60/round(bpm,6)/480
                dt=min(dt,max(0.,exported_dt-2e-6))
            reach=planning_speed*dt
            direction=p['direction'];swipe=-.35 if direction=='left' else .35 if direction=='right' else 0.
            lower=np.maximum(lanes+.15,max(lo+.15-min(0,swipe),previous_x-reach))
            upper=np.minimum(lanes+widths-.15,min(hi-.15-max(0,swipe),previous_x+reach))
            valid=lower<=upper
            # Concurrent objects use disjoint visual halves, not merely
            # disjoint finger contacts. Every edge stays in its half even
            # between knots. Isolated objects retain full-field widths.
            if crowded[index]:valid&=(lanes>=lo)&(lanes+widths<=hi)
            tick=round((p['time']-offset)*bpm/60*480)
            forbidden=reserved_lanes.get(tick,set())|((local_reservations or {}).get(tick,set()))
            if forbidden:valid&=~np.isin(lanes,list(forbidden))
            if o['object_kind']=='single':
                for t in tail_rects:
                    if -rounding_margin<=p['time']-t['time']<tail_guard+rounding_margin:
                        valid&=np.minimum(lanes+widths,t['lane']+t['width'])<=np.maximum(lanes,t['lane'])
            if not valid.any():return None
            xs=np.minimum(np.maximum(centers,lower),upper)
            scores=np.where(valid,costs[(index,j)]+.08*np.abs(xs-previous_x),np.inf)
            token=int(scores.argmin());x=float(xs[token])
            out.append(dict(time=p['time'],lane=int(lanes[token]),width=int(widths[token]),x=x,hand=hand,direction=direction))
            cost+=float(scores[token]);previous_time=p['time'];previous_x=x
        last=o['points'][-1];release=flick_release if last['direction']!='none' else tap_release
        swipe=-.35 if last['direction']=='left' else .35 if last['direction']=='right' else 0.
        state=(last['time']+release+rounding_margin,previous_x+swipe,index,o['object_kind']=='slide')
        return cost,out,state

    # Restart after converting a blocking hold, so all previous trajectories
    # are recomputed. A converted hold keeps its audio-supported head; its
    # unsupported generated tail is removed rather than shortened arbitrarily.
    for attempt in range(len(objects)+1):
        spans=[(o['points'][0]['time'],o['points'][-1]['time']) for o in objects]
        crowded=[any(i!=k and max(a,c)<=min(b,d)+rounding_margin
                     for k,(c,d) in enumerate(spans)) for i,(a,b) in enumerate(spans)]
        states=[(-10.,3.,None,False),(-10.,9.,None,False)];paths={};restart=False;tail_rects=[];reserved_lanes={}
        for time,indices in sorted(groups.items()):
            tail_rects[:]=[t for t in tail_rects if t['time']>=time-tail_guard-rounding_margin]
            solutions=[]
            for hands in itertools.permutations(range(2),len(indices)):
                choices=[]
                for index,hand in zip(indices,hands):
                    temporary=defaultdict(set)
                    for choice in choices:
                        for p in choice[2][1]:temporary[round((p['time']-offset)*bpm/60*480)].add(p['lane'])
                    item=trajectory(index,hand,states[hand],temporary)
                    if item is None:break
                    choices.append((index,hand,item))
                if len(choices)==len(indices):solutions.append((sum(c[2][0] for c in choices),choices))
            if not solutions:
                blocking=[s[2] for s in states if s[3] and s[0]>time and s[2] is not None]
                blocking+=list({t['object'] for t in tail_rects if -rounding_margin<=time-t['time']<tail_guard+rounding_margin})
                if not blocking:raise ValueError(f'Two-finger motion cannot satisfy the onset at {time:.6f}s')
                victim=min(blocking,key=lambda i:objects[i]['end'])
                o=objects[victim];end=o['end'];point=dict(o['points'][0],kind='tap',direction='none',trace=False)
                objects[victim]=dict(object_kind='single',start=o['start'],points=[point])
                repairs.append(dict(object=victim,time=o['start'],removed_tail=end,reason='two-finger occupancy/release conflict',from_kind='slide',to_kind='tap'))
                restart=True;break
            _,chosen=min(solutions,key=lambda s:s[0])
            for index,hand,item in chosen:
                paths[index]=item[1];states[hand]=item[2]
                for p in item[1]:reserved_lanes.setdefault(round((p['time']-offset)*bpm/60*480),set()).add(p['lane'])
                if objects[index]['object_kind']=='slide':tail_rects.append(dict(item[1][-1],object=index))
        if not restart:break
    else:raise AssertionError('Constraint projection did not converge')
    if {o['start'] for o in objects}!=initial_heads:raise AssertionError('Projection changed onset instants')
    events=unpack_objects(objects);ordered=flatten(objects)
    # unpack_objects order is kind/chain, not raw object id. Match by multiset
    # with queues so a chord containing identical types retains both widths.
    queues=defaultdict(list)
    chain=0
    for i,o in enumerate(objects):
        identity=chain if o['object_kind']=='slide' else None
        if identity is not None:chain+=1
        for j,p in enumerate(o['points']):queues[(p['time'],p['kind'],identity)].append(paths[i][j])
    geometry=[queues[(e['time'],e['kind'],e['chain'])].pop(0) for e in events]
    witness=dict(max_speed_lanes_per_second=max_speed,tap_release_seconds=tap_release,flick_release_seconds=flick_release,
                 tail_exclusion_seconds=tail_guard,
                 timing_bpm=bpm,audio_shift=offset,
                 hold_motion_clock='minimum of original interval and SUS tick interval, with parser rounding reserve',
                 policy='two fingers in separate halves; linear hold paths; bounded speed; explicit release/swipe margins',
                 paths=[dict(object=i,kind=objects[i]['object_kind'],points=paths[i]) for i in range(len(objects))])
    check=verify_witness(objects,witness)
    if not check['passed']:raise AssertionError(check)
    return objects,events,geometry,dict(repairs=repairs,witness=witness,verification=check,phrase_style=style,
        head_instants_preserved=True,raw_model_predictions=len(logits),limitations=['Theoretical feasibility under stated motion/release assumptions.',
        'Concurrent objects use disjoint visual halves; isolated rectangles may span the whole field.',
        'Source path breakpoints generate hidden geometry controls; no new acoustic judgments or easing.'])


def verify_exported_witness(objects,witness,text,bpm,audio_shift):
    """Verify the actual quantized SUS clock and intervals, not only JSON."""
    from mapper_pjsk.schema import parse_position_free
    parsed=parse_position_free(text,audio_shift,include_geometry=True)['events']
    def key(time,kind,lane,width,direction):
        return round((time-audio_shift)*bpm/60*480),kind,lane,width,direction
    queues=defaultdict(list)
    for e in parsed:queues[key(e['time'],e['kind'],e['lane'],e['width'],e['direction'])].append(e['time'])
    adjusted=deepcopy(objects);proof=deepcopy(witness)
    for record in proof['paths']:
        o=adjusted[record['object']]
        for p,g in zip(o['points'],record['points']):
            lookup=key(p['time'],p['kind'],g['lane'],g['width'],p['direction'])
            if not queues[lookup]:return dict(passed=False,errors=['Missing/mismatched exported object point'])
            time=queues[lookup].pop(0);p['time']=time;g['time']=time
        o['start']=o['points'][0]['time']
        if o['object_kind']=='slide':o['end']=o['points'][-1]['time']
    if any(queues.values()):return dict(passed=False,errors=['Unexpected exported point'])
    return verify_witness(adjusted,proof)


def verify_witness(objects,witness):
    """Check a given schedule independently; does not call model or projector."""
    errors=[];tasks=[[],[]];speed=witness['max_speed_lanes_per_second']
    if len(witness['paths'])!=len(objects):errors.append('missing objects')
    ids=[r['object'] for r in witness['paths']]
    if sorted(ids)!=list(range(len(objects))):return dict(passed=False,errors=['invalid or duplicate object identities'])
    for record in witness['paths']:
        index=record['object'];o=objects[index];path=record['points']
        if len(path)!=len(o['points']):errors.append(f'object {index}: missing points');continue
        hand=path[0]['hand']
        if hand not in (0,1):errors.append(f'object {index}: invalid hand');continue
        low,high=(0,6) if hand==0 else (6,12)
        for p,ref in zip(path,o['points']):
            if p['hand']!=hand or abs(p['time']-ref['time'])>1e-6:errors.append(f'object {index}: clock/hand mismatch')
            if not (isinstance(p['lane'],int) and isinstance(p['width'],int) and 1<=p['width'] and
                    0<=p['lane'] and p['lane']+p['width']<=12 and low<=p['x']<=high and p['lane']<=p['x']<=p['lane']+p['width']):errors.append(f'object {index}: outside note interval')
        # Linear interpolation of both note edges and contact positions stays
        # within the moving rectangle if it is inside at both endpoints.
        for a,b in zip(path,path[1:]):
            if abs(b['x']-a['x'])>speed*(b['time']-a['time'])+1e-6:errors.append(f'object {index}: hold movement too fast')
        flick=path[-1]['direction']!='none';release=witness['flick_release_seconds'] if flick else witness['tap_release_seconds']
        swipe=-.35 if path[-1]['direction']=='left' else .35 if path[-1]['direction']=='right' else 0.
        end_x=path[-1]['x']+swipe
        if not low<=end_x<=high:errors.append(f'object {index}: swipe outside half')
        if abs(swipe)>speed*release+1e-6:errors.append(f'object {index}: swipe speed')
        tasks[hand].append((path[0]['time'],path[-1]['time']+release,path[0]['x'],end_x,index))
    maximum=0.
    encoded=set()
    for r in witness['paths']:
        for p in r['points']:
            if 'timing_bpm' in witness:
                key=(round((p['time']-witness['audio_shift'])*witness['timing_bpm']/60*480),p['lane'])
                if key in encoded:errors.append('Ambiguous SUS overlay at same tick and left edge')
                encoded.add(key)
    for a in witness['paths']:
        if objects[a['object']]['object_kind']!='slide':continue
        tail=a['points'][-1]
        for b in witness['paths']:
            if objects[b['object']]['object_kind']!='single':continue
            head=b['points'][0];dt=head['time']-tail['time']
            if -1e-7<=dt<witness.get('tail_exclusion_seconds',0)-1e-7 and min(head['lane']+head['width'],tail['lane']+tail['width'])>max(head['lane'],tail['lane']):
                errors.append(f'tail visual buffer: objects {a["object"]}/{b["object"]}')
    for hand in range(2):
        ordered=sorted(tasks[hand])
        for a,b in zip(ordered,ordered[1:]):
            dt=b[0]-a[1]
            if dt<-1e-7:errors.append(f'finger {hand}: overlapping objects {a[4]}/{b[4]}');continue
            if abs(b[2]-a[3])>speed*max(dt,0)+1e-6:errors.append(f'finger {hand}: transition too fast')
            if dt>0:maximum=max(maximum,abs(b[2]-a[3])/dt)
    collisions=visual_overlaps(witness['paths'])
    errors.extend(f'visual overlap: objects {c["objects"]} at {c["time"]:.6f}s' for c in collisions)
    return dict(passed=not errors,errors=errors,objects=len(objects),finger_count=2,max_transition_speed=maximum,
                occupancy_checked=True,hold_paths_checked=True,release_and_swipe_checked=True,tail_visual_buffer_checked=True,
                visual_overlap_checked=True,visual_overlap_count=len(collisions),visual_overlaps=collisions)
