"""Auditable, resumable MASTER-only chart collection; no audio downloads."""
import argparse,hashlib,json,random,time
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
import requests
from mapper_pjsk.schema import parse_position_free
from .phrase_layout import read_objects,groups_of
from .pattern_layout import source_sequence,occupation_at
from .data import ROOT,PRIOR

OUT=ROOT/'diagnostics/pjsk_patterns_v2'
SEED=731


def save(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,ensure_ascii=False,allow_nan=False),encoding='utf-8')


def chunks_from_chart(text,song):
    chart=parse_position_free(text,0.,True);objects=read_objects(chart);groups=groups_of(objects)
    def tempo(t):return max((v for v in chart['tempos'] if v['time']<=t+1e-6),key=lambda v:v['time'],default=chart['tempos'][0])
    def beat(t):
        v=tempo(t);return v['beat']+(t-v['time'])*v['bpm']/60
    chunks=[];rejected=Counter();seen=Counter()
    for a in range(0,len(groups),8):
        part=groups[a:a+8]
        if len(part)!=8:rejected['short_tail']+=1;continue
        if any(len(ids)+sum(o['start']<t-1e-6 and o.get('end',o['start'])>t+1e-6 for o in objects)>2 for t,ids in part):
            rejected['more_than_two_contacts']+=1;continue
        t0=part[0][0];bpm=tempo(t0)['bpm'];seq=source_sequence(objects,part,bpm)
        for g,(t,ids) in zip(seq,part):
            g['beat']=beat(t)-beat(t0)
            for entry,index in zip(g['objects'],ids):entry['span']=beat(objects[index].get('end',t))-beat(t)
        signature=json.dumps([[round(g['beat'],3),[(o['kind'],round(o['span'],3),[(round(p['u'],3),p['lane'],p['width']) for p in o['path']]) for o in g['objects']]] for g in seq],separators=(',',':'))
        if seen[signature]>=2:rejected['within_song_repetition_cap']+=1;continue
        seen[signature]+=1
        chunks.append(dict(song=song['id'],source_start=t0,phase=beat(t0),sequence=seq,
                           occupation=occupation_at(objects,t0,bpm),difficulty=song['difficulty']))
    return chunks,dict(rejected)


def collect(count=300,workers=4):
    charts=OUT/'corpus/charts';charts.mkdir(parents=True,exist_ok=True)
    metadata={}
    for name in ('musics','musicDifficulties'):
        path=OUT/'corpus'/f'{name}.json'
        url=f'https://raw.githubusercontent.com/Sekai-World/sekai-master-db-diff/main/{name}.json'
        if not path.exists():
            response=requests.get(url,timeout=30);response.raise_for_status();path.write_bytes(response.content)
        metadata[name]=json.loads(path.read_text(encoding='utf-8'))
    songs={s['id']:s for s in metadata['musics']}
    levels={d['musicId']:d['playLevel'] for d in metadata['musicDifficulties'] if d['musicDifficulty']=='master'}
    old={s['id'] for s in json.loads((PRIOR/'manifest.json').read_text(encoding='utf-8'))['rows']}
    buckets=defaultdict(list)
    for i,level in levels.items():
        if i in songs and 24<=level<=36 and i not in old:buckets[level].append(i)
    rng=random.Random(SEED)
    for values in buckets.values():rng.shuffle(values)
    # Round-robin levels gives rare difficulties a place before filling common
    # levels. Known development songs are always training, never blind tests.
    order=sorted(old & set(levels))
    while any(buckets.values()):
        for level in sorted(buckets):
            if buckets[level]:order.append(buckets[level].pop())
    records=[];all_chunks={};failures=[]
    def fetch(i):
        path=charts/f'{i:04d}_master.sus';urls=[f'https://assets.unipjsk.com/startapp/music/music_score/{i:04d}_01/master',f'https://storage.sekai.best/sekai-jp-assets/music/music_score/{i:04d}_01/master.txt']
        source=path.with_suffix('.source.json')
        try:
            if not path.exists():
                errors=[]
                for url in urls:
                    try:
                        response=requests.get(url,timeout=20);response.raise_for_status()
                        if '#BPM' not in response.text:raise ValueError('Not a SUS chart')
                        path.write_bytes(response.content);save(source,dict(url=url));break
                    except Exception as exc:errors.append(str(exc))
                else:raise ValueError('; '.join(errors))
            song=dict(id=i,title=songs[i]['title'],difficulty=levels[i],path=str(path.resolve()),
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source=json.loads(source.read_text(encoding='utf-8')))
            chunks,rejected=chunks_from_chart(path.read_text(encoding='utf-8-sig'),song)
            if len(chunks)<10:raise ValueError(f'Only {len(chunks)} eligible eight-head motifs')
            song.update(chunks=len(chunks),rejected=rejected);return song,chunks
        except Exception as exc:return dict(id=i,error=str(exc)),None
    start=time.monotonic()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for a in range(0,len(order),32):
            if len(records)>=count:break
            for song,chunks in pool.map(fetch,order[a:a+32]):
                if chunks is None:failures.append(song)
                elif len(records)<count:records.append(song);all_chunks[song['id']]=chunks
            print(json.dumps(dict(collected=len(records),target=count,failures=len(failures),seconds=round(time.monotonic()-start,1)),ensure_ascii=False),flush=True)
    if len(records)<count:raise RuntimeError(f'Only {len(records)} eligible charts; target {count}')
    by_level=defaultdict(list)
    for song in records:
        if song['id'] in old:song['split']='train'
        else:by_level[song['difficulty']].append(song)
    for level,rows in sorted(by_level.items()):
        random.Random(SEED+level).shuffle(rows);nv=max(1,round(len(rows)*.15));nt=max(1,round(len(rows)*.15))
        for n,song in enumerate(rows):song['split']='test' if n<nt else 'validation' if n<nt+nv else 'train'
    splits={name:[c for s in records if s['split']==name for c in all_chunks[s['id']]] for name in ('train','validation','test')}
    for name,chunks in splits.items():save(OUT/'corpus'/f'{name}_chunks.json',chunks)
    manifest=dict(seed=SEED,rows=records,failures=failures,known_development_ids=sorted(old),
                  splits={n:dict(songs=sum(s['split']==n for s in records),chunks=len(c)) for n,c in splits.items()},
                  policy='MASTER, difficulty-balanced collection; full charts, nonoverlapping eight-head windows, <=2 contacts, at most two exact motif copies within each song; song-disjoint splits; known development songs forced to train.',
                  metadata_sources={n:dict(url=f'https://raw.githubusercontent.com/Sekai-World/sekai-master-db-diff/main/{n}.json',sha256=hashlib.sha256((OUT/'corpus'/f'{n}.json').read_bytes()).hexdigest()) for n in metadata},
                  seconds=time.monotonic()-start)
    save(OUT/'corpus/manifest.json',manifest)
    print(json.dumps(manifest['splits']),flush=True)
    return manifest


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--songs',type=int,default=300);p.add_argument('--workers',type=int,default=4)
    p.add_argument('--output',type=Path)
    args=p.parse_args()
    if args.output:OUT=args.output.resolve()
    collect(args.songs,args.workers)
