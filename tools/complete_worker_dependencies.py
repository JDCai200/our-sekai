"""Supply modules missed by older PyInstaller hooks for pinned NumPy/SciPy.

Copies from the build interpreter into the onedir package, never a user path.
These modules are imported dynamically by compiled extensions.
"""
import argparse
import importlib.metadata
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('package', type=Path)
    args = parser.parse_args()
    import numpy
    import scipy
    import sklearn
    internal = args.package.resolve() / '_internal'
    if not (args.package / 'OurSekai.Worker.exe').is_file():
        raise ValueError('Worker package does not exist')
    numpy_root = Path(numpy.__file__).resolve().parent
    scipy_root = Path(scipy.__file__).resolve().parent
    sources = [(numpy_root / '_core/_exceptions.py', Path('numpy/_core/_exceptions.py'))]
    extensions = list(scipy_root.glob('_cyutility*.pyd'))
    if len(extensions) != 1:
        raise ValueError('Expected one SciPy _cyutility Windows extension')
    sources += [(extensions[0], Path('scipy') / extensions[0].name)]
    sklearn_root = Path(sklearn.__file__).resolve().parent
    sklearn_extensions = list(sklearn_root.glob('_cyutility*.pyd'))
    if len(sklearn_extensions) != 1:
        raise ValueError('Expected one scikit-learn _cyutility Windows extension')
    sources += [(sklearn_extensions[0], Path('sklearn') / sklearn_extensions[0].name)]
    # Cython extension-to-extension imports are invisible to Python bytecode hooks.
    sources += [(path, Path('sklearn') / path.relative_to(sklearn_root))
                for path in sklearn_root.rglob('*.pyd')]
    # array_api_compat imports subpackages by their string names at runtime.
    sources += [(path, Path('sklearn') / path.relative_to(sklearn_root))
                for path in (sklearn_root / 'externals').rglob('*.py')]
    for source, relative in sources:
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = internal / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    for name in ('torch', 'torchaudio', 'numpy', 'scipy', 'librosa', 'scikit-learn',
                 'numba', 'llvmlite', 'soundfile', 'soxr', 'joblib', 'lazy-loader'):
        distribution = importlib.metadata.distribution(name)
        for relative in distribution.files or ():
            if not (relative.name.lower().startswith(('license', 'copying', 'notice'))
                    or 'licenses' in relative.parts):
                continue
            source = Path(distribution.locate_file(relative))
            if source.is_file():
                destination = internal / 'licenses' / name / Path(str(relative))
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
    print('NumPy/SciPy/scikit-learn dynamic import dependencies included')


if __name__ == '__main__':
    main()
