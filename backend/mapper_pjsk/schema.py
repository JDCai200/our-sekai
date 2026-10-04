"""SUS -> position-free events, preserving chords and slide identity.

Overlay semantics follow sonolus-pjsekai-engine/lib/src/sus/convert.ts.
Source positions are used only to join SUS modifiers, then discarded.
"""
import re
import math
from bisect import bisect_right
from collections import defaultdict
from charts import parse_sus

KINDS=['tap','flick','trace','slide_start','slide_start_hidden','slide_tick',
       'slide_attach','slide_hidden','slide_end','slide_end_hidden']
DIRECTIONS=['none','up','left','right']
STARTS={'slide_start','slide_start_hidden'}
ENDS={'slide_end','slide_end_hidden'}

def parse_position_free(text,audio_shift=0.,include_geometry=False):
    if '#MEASUREBS' in text:raise ValueError('Measure base changes are not supported in this prototype')
    timing=parse_sus(text,audio_shift);lengths={0:4.};rows=[]
    for line in text.splitlines():
        m=re.match(r'#(\d{3})([0-9a-z]{2,3}):\s*(\S+)',line.strip(),re.I)
        if not m:continue
        bar,ch,data=int(m[1]),m[2].lower(),m[3]
        if ch=='02':lengths[bar]=float(data)
        elif ch[0] in '1359':rows.append((bar,ch,data))
    starts=[0.];current=4.
    for bar in range(max([r[0] for r in rows],default=0)+1):
        current=lengths.get(bar,current);starts.append(starts[-1]+current)
    singles={};dirs={};streams=defaultdict(list);decorations=0;geometry={}
    for bar,ch,data in rows:
        if len(data)%2:raise ValueError('Odd SUS row length')
        lane=int(ch[1],36)
        if not 2<=lane<=13:continue
        for n in range(len(data)//2):
            value=data[2*n:2*n+2]
            if value=='00':continue
            beat=round(starts[bar]+(starts[bar+1]-starts[bar])*n/(len(data)//2),8)
            typ=int(value[0],36);key=(beat,lane)
            if ch[0] in '13':geometry[key]=dict(lane=lane-2,width=int(value[1],36))
            if ch[0]=='1':singles[key]=typ
            elif ch[0]=='5':dirs[key]=typ
            elif ch[0]=='3':
                if len(ch)!=3:raise ValueError('Slide channel must include identity')
                streams[ch[2]].append((beat,lane,typ))
            elif ch[0]=='9':decorations+=1
    tempo=timing['tempos'];beats=[t['beat'] for t in tempo]
    def seconds(beat):
        i=bisect_right(beats,beat)-1;t=tempo[i]
        return round(t['time']+(beat-t['beat'])*60/t['bpm'],6)
    def direction(key):return {1:'up',3:'left',4:'right'}.get(dirs.get(key),'none')
    prevented=set();chains=[]
    for channel,points in sorted(streams.items()):
        active=None
        for beat,lane,typ in sorted(set(points)):
            key=(beat,lane);overlay=singles.get(key,0)
            if typ==1:
                if active is not None:raise ValueError(f'Overlapping starts in SUS channel {channel}')
                active=[];chains.append(active)
                critical=overlay in (2,6,8)
            elif active is None:raise ValueError(f'Orphan slide point in channel {channel}')
            if typ not in (1,2,3,5):raise ValueError(f'Unsupported SUS slide point {typ}')
            prevented.add(key)
            if typ==5 and overlay==3:continue
            kind={1:'slide_start',2:'slide_end',3:'slide_tick',5:'slide_hidden'}[typ]
            if typ in (1,2) and overlay in (7,8):kind+='_hidden'
            if typ==3 and overlay==3:kind='slide_attach'
            active.append(dict(time=seconds(beat),kind=kind,critical=critical or overlay in (2,6,8),
                               direction=direction(key) if typ==2 and overlay not in (7,8) else 'none',
                               trace=overlay in (5,6),chain=len(chains)-1,
                               **(geometry[key] if include_geometry else {})))
            if typ==2:active=None
        if active is not None:raise ValueError(f'Unclosed slide in channel {channel}')
    events=[e for c in chains for e in c]
    for key,typ in sorted(singles.items()):
        if key in prevented or typ not in (1,2,5,6):continue
        d=direction(key);kind='trace' if typ in (5,6) else 'flick' if d!='none' else 'tap'
        events.append(dict(time=seconds(key[0]),kind=kind,critical=typ in (2,6),direction=d,
                           trace=typ in (5,6),chain=None,
                           **(geometry[key] if include_geometry else {})))
    events.sort(key=lambda e:(e['time'],KINDS.index(e['kind']),e['critical'],e['direction'],e['chain'] if e['chain'] is not None else -1))
    validate(events,allow_geometry=include_geometry)
    return dict(schema='pjsk-layout-v1' if include_geometry else 'pjsk-position-free-v1',time_reference='audio_absolute',events=events,tempos=tempo,
                ignored_decoration_points=decorations,positions_generated=False,widths_generated=False,
                audio_shift=audio_shift,geometry_included=include_geometry,warnings=['Guide decorations and easing are excluded; hidden points remain structural, not acoustic onsets.'])

def validate(events,require_closed=True,allow_geometry=False):
    active=set();last=-float('inf')
    for e in events:
        if not math.isfinite(e['time']):raise ValueError('Non-finite event time')
        if not allow_geometry and any(k in e for k in ('lane','position','width','size','x')):raise ValueError('Position leaked into event schema')
        if allow_geometry and (not isinstance(e.get('lane'),int) or not isinstance(e.get('width'),int) or
                               not 0<=e['lane']<12 or not 1<=e['width']<=12-e['lane']):raise ValueError('Invalid note interval')
        if e['time']<last:raise ValueError('Events are not chronological')
        last=e['time'];kind=e['kind'];chain=e['chain']
        if kind not in KINDS or e['direction'] not in DIRECTIONS:raise ValueError('Unknown event attribute')
        if kind in STARTS:
            if chain is None or chain in active:raise ValueError('Invalid slide start')
            active.add(chain)
        elif kind.startswith('slide_'):
            if chain not in active:raise ValueError('Slide references inactive chain')
            if kind in ENDS:active.remove(chain)
        elif chain is not None:raise ValueError('Single note has slide reference')
        if e['direction']!='none' and kind not in ('flick','trace','slide_end'):raise ValueError('Invalid flick direction')
        if kind=='flick' and e['direction']=='none':raise ValueError('Flick must have direction')
        if kind in ('tap','flick') and e['trace']:raise ValueError('Tap/flick cannot also be trace')
        if kind=='trace' and not e['trace']:raise ValueError('Trace note must carry trace attribute')
    if require_closed and active:raise ValueError('Unclosed slides')
    return active

def complete_window(events,start,duration):
    """Keep whole chains only; never turn a crop boundary into a musical note."""
    end=start+duration;chains=defaultdict(list)
    for e in events:
        if e['chain'] is not None:chains[e['chain']].append(e)
    allowed={c for c,es in chains.items() if es[0]['time']>=start and es[-1]['time']<end}
    chosen=[dict(e) for e in events if start<=e['time']<end and (e['chain'] is None or e['chain'] in allowed)]
    ids={c:n for n,c in enumerate(sorted(allowed,key=lambda c:(chains[c][0]['time'],c)))}
    for e in chosen:
        e['time']=round(e['time']-start,6)
        if e['chain'] is not None:e['chain']=ids[e['chain']]
    validate(chosen)
    return chosen

class Codec:
    """Factorized 10ms time/type/critical/trace/direction/reference grammar."""
    def __init__(self,duration=4.,max_chains=16):
        self.duration=duration;self.max_chains=max_chains
        self.names=['PAD','BOS','EOS']+[f'TIME_{i}' for i in range(round(duration*100)+1)]
        self.groups={}
        for group,values in [('kind',KINDS),('critical',[False,True]),('trace',[False,True]),
                             ('direction',DIRECTIONS),('chain',[None]+list(range(max_chains)))]:
            self.groups[group]={v:len(self.names)+n for n,v in enumerate(values)}
            self.names.extend(f'{group}:{v}' for v in values)
        self.time_end=3+round(duration*100)+1
    def encode(self,events):
        validate(events);tokens=[1]
        for e in events:
            t=round(e['time']*100)
            if not 0<=t<=round(self.duration*100):raise ValueError('Event outside token time range')
            tokens.append(3+t)
            for group in self.groups:tokens.append(self.groups[group][e[group]])
        return tokens+[2]
    def decode(self,tokens):
        if tokens[0]!=1 or tokens[-1]!=2:raise ValueError('Missing BOS/EOS')
        body=tokens[1:-1]
        if len(body)%6:raise ValueError('Incomplete event')
        inverse={g:{v:k for k,v in ids.items()} for g,ids in self.groups.items()};events=[]
        for n in range(0,len(body),6):
            t=body[n]
            if not 3<=t<self.time_end:raise ValueError('Invalid time token')
            e={'time':(t-3)/100}
            for j,g in enumerate(self.groups,1):
                if body[n+j] not in inverse[g]:raise ValueError('Invalid attribute token')
                e[g]=inverse[g][body[n+j]]
            events.append(e)
        validate(events);return events
