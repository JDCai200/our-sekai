from pathlib import Path
import importlib
import sys
root = Path(SPECPATH).parent
if (root/'.build-tools/pefile.py').is_file():
    sys.path.insert(0,str(root/'.build-tools'))
    import pefile
    importlib.reload(pefile)
analysis = Analysis([str(root/'companion/windows_app.py')],
    pathex=[str(root),str(root/'backend')],
    datas=[], binaries=[], hiddenimports=[],
    excludes=['torch','torchaudio','numpy','scipy','sklearn','librosa','matplotlib','IPython'],
    noarchive=False)
pyz = PYZ(analysis.pure)
exe = EXE(pyz,analysis.scripts,[],exclude_binaries=True,name='OurSekai',console=False,upx=False)
collection = COLLECT(exe,analysis.binaries,analysis.datas,name='OurSekai',upx=False)
