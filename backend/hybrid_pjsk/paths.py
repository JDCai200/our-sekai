"""Runtime paths and JSON writer, without training framework imports."""
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
PRIOR = ROOT/'diagnostics/genelive_finetune_conservative'


def dump(path, data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
