# PyInstaller Windows worker. Build with the backend's configured Python.
from pathlib import Path
import importlib
import sys
from PyInstaller.utils.hooks import collect_data_files

root = Path(SPECPATH)
# Build-only override when an existing developer environment has the slow
# pefile 2024.8.26 release. Clean installations pin this in setup_backend.ps1.
build_tools = root.parent / '.build-tools'
if (build_tools / 'pefile.py').is_file():
    sys.path.insert(0, str(build_tools))
    import pefile
    importlib.reload(pefile)
model_paths = [
    'diagnostics/genelive_finetune_conservative/model_head_finetuned.pth',
    'diagnostics/genelive_autoosu/object_selector.pth',
    'diagnostics/pjsk_layout/layout_model.pth',
    'diagnostics/pjsk_patterns/patterns.joblib',
    'diagnostics/pjsk_patterns/active_model.json',
    'diagnostics/pjsk_patterns_v2/expression.json',
    'diagnostics/pjsk_patterns_v2/slide_paths.json',
    'diagnostics/pjsk_patterns_v2/patterns.joblib',
    'models/autoosu/checkpoints/osu_model_v2.pt',
    'models/autoosu/models.py',
    'model-lock.json',
]
datas = [(str(root / path), str(Path(path).parent)) for path in model_paths]
datas += [(str(root / 'models/genelive/notes_generator'), 'models/genelive/notes_generator')]
datas += [(str(root / 'models/genelive/LICENSE'), 'licenses/Genelive')]
datas += collect_data_files('librosa')
binaries = [(str(path), 'bin') for path in (root / 'bin').glob('*') if path.suffix in ('.exe', '.dll')]
datas += [(str(root / 'bin/LICENSE'), 'licenses/FFmpeg')]
analysis = Analysis([str(root / 'worker.py')], pathex=[str(root), str(root / 'models/genelive')],
    binaries=binaries, datas=datas,
    hiddenimports=['hybrid_pjsk.cli', 'notes_generator.models.onsets', 'notes_generator.models.beats',
                   'notes_generator.constants', 'sklearn.ensemble._forest', 'sklearn.tree._tree'],
    excludes=['tkinter', 'IPython', 'notebook', 'pytest', 'matplotlib', 'tensorflow', 'cv2',
              'pandas', 'pyarrow', 'openpyxl', 'pygame', 'h5py', 'bokeh', 'plotly', 'seaborn',
              'statsmodels', 'xarray', 'numexpr', 'tables', 'transformers', 'datasets',
              'accelerate', 'huggingface_hub', 'sphinx'],
    noarchive=False)
pyz = PYZ(analysis.pure)
exe = EXE(pyz, analysis.scripts, [], exclude_binaries=True, name='OurSekai.Worker', console=True, upx=False)
collection = COLLECT(exe, analysis.binaries, analysis.datas, name='AutoChart', upx=False)
