"""Export neural inference stages for an Android ONNX runtime implementation.

This is an export/parity tool, not a finished Android generation engine.
The two-pass 640-frame GenéLive recurrent clock is preserved in parity tests.
"""
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
sys.path.insert(0, str(ROOT / 'backend/models/genelive'))
# Optional build-only dependencies; never added to the existing tool's venv.
sys.path.insert(0, str(ROOT / '.build-tools'))
import numpy as np
import torch
from torch import nn
import onnxruntime as ort
from notes_generator.models.onsets import SimpleOnsets
from notes_generator.constants import ConvStackType, NMELS
from autoosu_pjsk.backend import TimingModel
from hybrid_pjsk.model import ObjectSelector, empty_action
from hybrid_pjsk.layout_model import load_layout, INTERVALS
from hybrid_pjsk.pattern_layout import load_patterns

OUT = ROOT / 'artifacts/portable-models'

class OnsetStack(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, mel, condition, beats):
        return torch.cat((self.model.onset_stack(mel), condition, beats), -1)

class RecurrentChunk(nn.Module):
    def __init__(self, rnn): super().__init__(); self.rnn = rnn
    def forward(self, features, hidden, cell):
        output, (hidden, cell) = self.rnn(features, (hidden, cell))
        return output, hidden, cell

class Acoustic(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, specs, indices, phases, numbers, conditions):
        conv = self.model.stack(specs).permute(0, 2, 1, 3).flatten(2, 3)[0]
        return torch.cat((conv[indices], self.model.beat_phase_emb(phases),
                          self.model.beat_num_emb(numbers), self.model.difficulty_proj(conditions)), -1)

class SelectorStep(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, acoustic, placement, count, kind, critical, direction, span, hidden):
        previous = dict(count=count, kind=kind, critical=critical, direction=direction, span=span)
        logits, hidden = self.model(acoustic, placement, previous, hidden)
        return tuple(logits[k] for k in ('count','kind','critical','direction','tail_direction','span')) + (hidden,)

class LayoutStep(nn.Module):
    def __init__(self, model): super().__init__(); self.model = model
    def forward(self, event, previous, hidden): return self.model(event, previous, hidden)

def export(name, model, arguments, inputs, outputs, dynamic=None):
    path = OUT / (name + '.onnx')
    model.eval()
    with torch.inference_mode():
        expected = model(*arguments)
    torch.onnx.export(model, arguments, str(path), input_names=inputs, output_names=outputs,
                      opset_version=17, dynamic_axes=dynamic or {}, do_constant_folding=True)
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    actual = session.run(None, {key: value.detach().numpy() for key, value in zip(inputs, arguments)})
    if isinstance(expected, torch.Tensor): expected = (expected,)
    errors = []
    for a, b in zip(actual, expected):
        target = b.detach().numpy()
        np.testing.assert_allclose(a, target, atol=3e-5, rtol=2e-4)
        errors.append(float(np.max(np.abs(a - target))))
    return dict(file=path.name, sha256=hashlib.sha256(path.read_bytes()).hexdigest(), max_absolute_errors=errors)

def plain(value):
    if isinstance(value, np.ndarray): return value.tolist()
    if isinstance(value, np.generic): return value.item()
    if isinstance(value, dict): return {str(k):plain(v) for k,v in value.items()}
    if isinstance(value, (tuple,list)): return [plain(v) for v in value]
    return value

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    onset = SimpleOnsets(NMELS, 1, enable_condition=True, enable_beats=True,
                        conv_stack_type=ConvStackType.v7, num_layers=2, onset_weight=64, dropout=.5)
    onset.load_state_dict(torch.load(ROOT/'backend/diagnostics/genelive_finetune_conservative/model_head_finetuned.pth',
                                   map_location='cpu', weights_only=True))
    onset.eval()
    mel = torch.randn(1, 96, 229)
    condition, beats = torch.full((1,96,1),40.), torch.zeros(1,96,1)
    rows = [export('genelive_stack', OnsetStack(onset), (mel,condition,beats),
        ['mel','condition','beats'], ['features'], {k:{1:'frames'} for k in ('mel','condition','beats','features')})]
    features = OnsetStack(onset)(mel,condition,beats).detach()
    hidden, cell = torch.zeros(4,1,384), torch.zeros(4,1,384)
    rows.append(export('genelive_recurrent', RecurrentChunk(onset.onset_sequence.rnn), (features,hidden,cell),
        ['features','hidden','cell'], ['context','next_hidden','next_cell'], {'features':{1:'frames'},'context':{1:'frames'}}))
    context = onset.onset_sequence(features).detach()
    rows.append(export('genelive_head', onset.onset_linear, (context,), ['context'], ['probabilities'],
        {'context':{1:'frames'},'probabilities':{1:'frames'}}))
    # Validate the actual two-pass runtime behavior on more than one recurrent chunk.
    session = ort.InferenceSession(str(OUT/'genelive_recurrent.onnx'), providers=['CPUExecutionProvider'])
    multi = torch.randn(1, 701, 770)
    with torch.inference_mode(): reference = onset.onset_sequence(multi).numpy()
    result = np.zeros((1,701,768), np.float32)
    positions = list(range(0,701,640))
    for reverse in (False, True):
        h = np.zeros((4,1,384),np.float32); c = h.copy()
        for start in (reversed(positions) if reverse else positions):
            end = min(start+640,701)
            out,h,c = session.run(None, {'features':multi[:,start:end].numpy(),'hidden':h,'cell':c})
            if reverse: result[:,start:end,384:] = out[:,:,384:]
            else: result[:,start:end] = out
    np.testing.assert_allclose(result,reference,atol=3e-5,rtol=2e-4)
    timing = TimingModel()
    specs = torch.randn(1,3,120,80)
    indices = torch.tensor([10,30,80]); phases=torch.tensor([0,24,48]); numbers=torch.tensor([0,1,3]); conditions=torch.full((3,1),3.)
    rows.append(export('acoustic', Acoustic(timing.model), (specs,indices,phases,numbers,conditions),
        ['specs','indices','phases','numbers','conditions'], ['acoustic'],
        {'specs':{2:'audio_frames'}, **{k:{0:'heads'} for k in ('indices','phases','numbers','conditions','acoustic')}}))
    state = torch.load(ROOT/'backend/diagnostics/genelive_autoosu/object_selector.pth',map_location='cpu',weights_only=True)
    selector = ObjectSelector(timing.model,state['placement_size']); selector.restore(state); selector.eval()
    previous = empty_action()
    rows.append(export('selector_step', SelectorStep(selector),
        (torch.randn(1,384),torch.randn(1,state['placement_size']),previous['count'],previous['kind'],previous['critical'],
         previous['direction'],previous['span'],torch.zeros(2,1,256)),
        ['acoustic','placement','previous_count','previous_kind','previous_critical','previous_direction','previous_span','hidden'],
        ['count_logits','kind_logits','critical_logits','direction_logits','tail_direction_logits','span','next_hidden']))
    layout = load_layout(ROOT/'backend/diagnostics/pjsk_layout/layout_model.pth')
    input_size = layout.memory.input_size - 24
    rows.append(export('layout_step', LayoutStep(layout), (torch.randn(1,input_size),torch.tensor([len(INTERVALS)]),torch.zeros(2,1,96)),
        ['event','previous_interval','hidden'],['interval_logits','next_hidden']))
    bundle = load_patterns()
    forest = bundle['selector']
    trees = [dict(children_left=t.tree_.children_left,children_right=t.tree_.children_right,
                  feature=t.tree_.feature,threshold=t.tree_.threshold,value=t.tree_.value) for t in forest.estimators_]
    pattern = {key:value for key,value in bundle.items() if key not in ('selector','_checkpoint_path','scaler','clusterer')}
    pattern['scaler'] = dict(mean=bundle['scaler'].mean_, scale=bundle['scaler'].scale_)
    pattern['cluster_centers'] = bundle['clusterer'].cluster_centers_
    pattern['forest'] = dict(classes=forest.classes_,trees=trees)
    (OUT/'patterns.json').write_text(json.dumps(plain(pattern),ensure_ascii=False,allow_nan=False),encoding='utf-8')
    report = dict(schema=1, neural_stages=rows, recurrent_chunk_frames=640,
        multi_chunk_max_absolute_error=float(np.max(np.abs(result-reference))),
        selector_allowed_counts=selector.allowed_counts, selector_allowed_types=selector.allowed_types,
        intervals=INTERVALS, limits=['Export/parity validated on CPU; Android runtime integration and audio DSP parity remain required.'])
    (OUT/'parity-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__ == '__main__': main()
