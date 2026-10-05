"""OpenSekai community package export without a Unity installation.

The generated SUS subset uses 480 ticks/beat, 4/4, constant BPM and channels
1/3/5. Read the exported SUS, rather than the pre-export floating point chart.
Schema and enum values follow the community source retained in Assets/.
"""
from collections import defaultdict
from fractions import Fraction
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import zipfile


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def score_from_sus(text, music_id=0):
    if not re.search(r'^#00002:\s*4\s*$', text, re.M):
        raise ValueError('生成谱面必须是固定 4/4 时间轴')
    bpm_match = re.search(r'^#BPM01:\s*([\d.]+)\s*$', text, re.M)
    if not bpm_match or not re.search(r'^#00008:\s*01\s*$', text, re.M):
        raise ValueError('缺少生成谱面的初始 BPM')
    bpm = float(bpm_match[1])
    nodes = {}
    streams = defaultdict(list)
    rows = []
    for line in text.splitlines():
        match = re.fullmatch(r'#(\d{3})([135][0-9a-z]{1,2}):\s*([0-9a-z]+)', line, re.I)
        if match:
            rows.append((int(match[1]), match[2].lower(), match[3].lower()))
        elif re.match(r'#\d{3}(02|08):', line) and line not in ('#00002: 4', '#00008: 01'):
            raise ValueError('此导出器仅处理工具生成的固定 BPM / 4/4 SUS')
    for bar, channel, payload in sorted(rows, key=lambda row: (row[1][0], row[0], row[1])):
        if len(payload) % 2:
            raise ValueError('无效 SUS 数据长度')
        lane = int(channel[1], 36) - 2
        if not 0 <= lane < 12:
            raise ValueError('SUS 音符超出 12 轨')
        count = len(payload) // 2
        for index in range(count):
            value = payload[index*2:index*2+2]
            if value == '00':
                continue
            tick = Fraction(bar*1920) + Fraction(index*1920, count)
            if tick.denominator != 1:
                raise ValueError('SUS 音符无法精确映射到 480 tick 时间轴')
            tick = int(tick)
            typ, width = int(value[0], 36), int(value[1], 36)
            key = tick, lane
            if channel[0] != '5' and not 1 <= width <= 12-lane:
                raise ValueError('无效 SUS 音符宽度')
            if key not in nodes:
                nodes[key] = dict(ticks=tick, laneStart=lane, laneEnd=lane+width-1,
                                  category=0, type=0, speedRatio=1., noteLineType=0,
                                  direction=0, isSkip=False, isDecoration=False,
                                  guideColor=None, previousConnectionId=-1, nextConnectionId=-1)
            node = nodes[key]
            if channel[0] == '1':
                categories = {1:0, 2:0, 3:14, 5:4, 6:4, 7:5, 8:5}
                if typ not in categories:
                    raise ValueError('不支持的生成单点类型')
                node['category'] = categories[typ]
                node['type'] = int(typ in (2, 6, 8))
                node['isSkip'] = typ == 3
            elif channel[0] == '3':
                if len(channel) != 3 or typ not in (1, 2, 3, 5):
                    raise ValueError('无效生成长条通道')
                old_category = node['category']
                if typ == 1:
                    node['category'] = 6 if old_category == 4 else 7 if old_category == 5 else 1
                elif typ == 2:
                    # A release tail must carry a tail category, not the default
                    # Normal(0). The editor's NoteGroupUtility.IsLongEndNote only
                    # accepts Long/Flick/Friction/FrictionHide/Friction* for a
                    # chain end; category 0 renders as a bare tap and is flagged
                    # as an illegal note. Friction/Hidden overlays keep their
                    # categories; otherwise the tail matches the Long head's
                    # colour (OpenSekai AddLongEndNoteVariations).
                    node['category'] = old_category if old_category in (4, 5) else 1
                elif typ in (3, 5):
                    node['category'] = 2 if typ == 3 else 13
                streams[channel[2]].append((tick, lane, typ))
            else:
                if typ in (1, 3, 4):
                    node['category'] = 8 if node['category'] == 4 else 3
                    node['direction'] = {1:0, 3:1, 4:2}[typ]
                elif typ in (2, 5):
                    node['noteLineType'] = 2 if typ == 2 else 1
                else:
                    raise ValueError('不支持的生成方向类型')
    base_types = {0:1, 1:2, 2:5, 3:3, 4:11, 5:12, 6:8, 7:9, 8:4, 13:6, 14:1}
    notes = sorted(nodes.values(), key=lambda node: (node['ticks'], node['laneStart']))
    for index, node in enumerate(notes, 5):
        node['id'] = index
        node['noteBaseType'] = base_types[node['category']]
    for points in streams.values():
        previous = None
        for tick, lane, typ in sorted(points):
            node = nodes[tick, lane]
            if typ == 1:
                if previous is not None:
                    raise ValueError('长条通道重复起点')
                previous = node
            else:
                if previous is None:
                    raise ValueError('长条没有起点')
                previous['nextConnectionId'] = node['id']
                node['previousConnectionId'] = previous['id']
                previous = None if typ == 2 else node
        if previous is not None:
            raise ValueError('长条未闭合')
    if not notes:
        raise ValueError('谱面没有音符')
    return dict(VersionCode=1, MusicScoreEventDataList=[
        dict(id=1,eventType=0,ticks=0,changeValue=bpm),
        dict(id=2,eventType=3,ticks=0,changeValue='4/4'),
        dict(id=3,eventType=1,ticks=0,changeValue=1.),
        dict(id=4,eventType=2,ticks=0,changeValue=1.)], EventArray=[], NoteList=notes,
        MusicScoreTicksMax=max(node['ticks'] for node in notes), MusicId=music_id, FullComboDataHash=None)


def music_id(identifier):
    # Matches CustomMusicScoreEntry.CreateStableMusicId (see upstream source).
    value = 2166136261
    raw = (identifier or 'custom').lower().encode('utf-16-le')
    for index in range(0, len(raw), 2):
        value = ((value ^ int.from_bytes(raw[index:index+2], 'little')) * 16777619) & 0xffffffff
    value &= 0x7fffffff
    return -value if value else -1


def safe_asset(root, name):
    if not name:
        return None
    root = Path(root).resolve()
    path = (root / name).resolve()
    if root not in path.parents:
        raise ValueError('歌曲包路径超出歌曲目录')
    return path


def finalize_project(stage, original_manifest=None):
    stage = Path(stage)
    chart = load_json(stage / 'chart.json')
    check = chart['two_finger']['exported_verification']
    if not (check.get('passed') and check.get('visual_overlap_checked') and check.get('visual_overlap_count') == 0):
        raise ValueError('实际谱面可游玩核验失败')
    manifest = load_json(stage / 'manifest.json')
    if original_manifest:
        original_path = Path(original_manifest)
        original = load_json(original_path)
        for key in ('title','scoreTitle','composer','lyricist','arranger','singer','collaborationLabel','description','userName'):
            if key in original:
                manifest[key] = original[key]
        for key, basename in (('jacketFileName','jacket'), ('videoFileName','video')):
            source = safe_asset(original_path.parent, original.get(key))
            if source and source.is_file():
                manifest[key] = basename + source.suffix
                shutil.copy2(source, stage / manifest[key])
    score = score_from_sus((stage / 'chart.sus').read_text(encoding='utf-8-sig'), music_id(manifest['id']))
    if len(score['NoteList']) != len(chart['events']):
        raise ValueError('OpenSekai 数据转换丢失或增加了音符')
    write_json(stage / 'score.json', score)
    manifest['scoreFileName'] = 'score.json'
    write_json(stage / 'manifest.json', manifest)
    return manifest


def publish(stage, songs, original_manifest=None):
    stage, songs = Path(stage), Path(songs)
    manifest = finalize_project(stage, original_manifest)
    songs.mkdir(parents=True, exist_ok=True)
    title = re.sub(r'[<>:"/\\|?*\x00-\x1f\s]+', '_', manifest['title']).strip(' ._')[:80] or 'Song'
    destination = songs / (title + '_' + manifest['id'])
    if destination.exists():
        raise FileExistsError('歌曲目录已存在')
    # Expose only player assets. Inference logs and original private paths remain in jobs.
    with tempfile.TemporaryDirectory(prefix='.publish-', dir=songs) as temporary:
        pending = Path(temporary) / destination.name
        pending.mkdir()
        for name in ('manifest.json','score.json','chart.sus',manifest['audioFileName'],
                     manifest.get('jacketFileName'),manifest.get('videoFileName')):
            source = safe_asset(stage, name)
            if source and source.is_file():
                shutil.copy2(source, pending / Path(name).name)
        os.replace(pending, destination)
    return destination


def export_zip(project, destination):
    project, destination = Path(project), Path(destination)
    manifest = load_json(project / 'manifest.json')
    names = ['manifest.json','score.json','chart.sus',manifest['audioFileName'],
             manifest.get('jacketFileName'),manifest.get('videoFileName')]
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.package-', suffix='.zip', dir=destination.parent)
    os.close(fd)
    try:
        with zipfile.ZipFile(temporary,'w',zipfile.ZIP_DEFLATED) as archive:
            for name in names:
                source = safe_asset(project, name)
                if source and source.is_file():
                    archive.write(source, Path(name).name)
        os.replace(temporary, destination)
    finally:
        if Path(temporary).exists():
            Path(temporary).unlink()


def extract_existing(package, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(package) as archive:
        if sum(info.file_size for info in archive.infolist()) > 4*1024**3:
            raise ValueError('歌曲包解压尺寸超过 4 GB')
        for info in archive.infolist():
            name = info.filename.replace('\\','/')
            if name.startswith('/') or ':' in name or '..' in Path(name).parts:
                raise ValueError('歌曲包包含越界路径')
            target = (destination / name).resolve()
            if destination not in target.parents:
                raise ValueError('歌曲包包含越界路径')
            if info.is_dir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.open(info) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
    manifests = list(destination.rglob('manifest.json'))
    if len(manifests) != 1:
        raise ValueError('歌曲包应包含且仅包含一张歌曲配置')
    manifest = manifests[0]
    data = load_json(manifest)
    audio = safe_asset(manifest.parent, data['audioFileName'])
    if not audio or not audio.is_file():
        raise ValueError('歌曲包缺少音频')
    return manifest, audio
