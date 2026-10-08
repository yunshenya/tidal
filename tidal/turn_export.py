"""Export a frozen turn candidate to ONNX; scores remain shadow-only.

python -m tidal.turn_export audio | timing
The exporter never fits or selects weights. It uses the validation-frozen champion.
"""
import argparse
import json
from pathlib import Path
import time

import joblib
import numpy as np
import onnxruntime as ort
import torch
from torch import nn

from tidal.audio_fe import FRAME_READY, STEP
from tidal.turn_data import (CONTEXT, TIMING_NAMES, PROSODY_NAMES, audio_data, candor_data)
from tidal.turn_optimize import (ROOT, STORE, VAL_GROUPS, model_for, inference,
                                 verify_selection, digest)


class FrozenBundle(nn.Module):
    def __init__(self, freeze):
        super().__init__()
        self.task=freeze['task']; self.kind=freeze['champion']
        if self.kind=='lr':
            lr=joblib.load(ROOT/freeze['lr']['path'])
            self.register_buffer('f_mu', torch.tensor(lr['scaler'].mean_, dtype=torch.float32))
            self.register_buffer('f_sd', torch.tensor(lr['scaler'].scale_, dtype=torch.float32))
            self.linear=nn.Linear(len(lr['scaler'].mean_), 1)
            with torch.no_grad():
                self.linear.weight.copy_(torch.tensor(lr['model'].coef_, dtype=torch.float32))
                self.linear.bias.copy_(torch.tensor(lr['model'].intercept_, dtype=torch.float32))
            params=lr['calibration']
        else:
            cks=[torch.load(ROOT/p, weights_only=False) for p in freeze['selected_neural']['checkpoints']]
            self.models=nn.ModuleList([model_for(ck) for ck in cks])
            for i, ck in enumerate(cks):
                self.register_buffer(f'f_mu_{i}', torch.from_numpy(ck['f_mu']))
                self.register_buffer(f'f_sd_{i}', torch.from_numpy(ck['f_sd']))
                if self.task=='audio':
                    self.register_buffer(f'mu_{i}', torch.from_numpy(ck['mu']))
                    self.register_buffer(f'sd_{i}', torch.from_numpy(ck['sd']))
            params=freeze['selected_neural']['calibration']
        self.register_buffer('scale', torch.tensor(params['scale'], dtype=torch.float32))
        self.register_buffer('bias', torch.tensor(params['bias'], dtype=torch.float32))

    def forward(self, window, features):
        if self.kind=='lr':
            z=self.linear((features-self.f_mu)/self.f_sd)[:, 0]
        else:
            if self.kind=='cmvn':
                centered=window-window.mean(1, keepdim=True)
                window=centered/torch.sqrt((centered**2).mean(1, keepdim=True)).clamp_min(.5)
            predictions=[]
            for i, model in enumerate(self.models):
                f=(features-getattr(self, f'f_mu_{i}'))/getattr(self, f'f_sd_{i}')
                if self.task=='audio':
                    w=(window-getattr(self, f'mu_{i}'))/getattr(self, f'sd_{i}')
                    raw=model(w, f)
                else:
                    raw=model(f)
                predictions.append(torch.sigmoid(raw))
            p=torch.stack(predictions).mean(0).clamp(1e-6, 1.-1e-6)
            z=torch.log(p)-torch.log1p(-p)
        return torch.sigmoid(self.scale*z+self.bias)


def export(task):
    torch.set_num_threads(2)
    freeze=json.loads((STORE/f'{task}_selection.json').read_text())
    verify_selection(freeze, freeze['signature'])
    report_path=ROOT/'reports'/f'turn_v2_{task}.json'
    report=json.loads(report_path.read_text())
    if report['signature']!=freeze['signature']:
        raise ValueError('Report and frozen selection differ')
    bundle=FrozenBundle(freeze).eval()
    # Export checks use development data; the exporter never touches test labels.
    data=audio_data(VAL_GROUPS) if task=='audio' else candor_data('val')
    n=min(64, len(data.y)); features=data.features[:n]
    window=data.windows[:n] if task=='audio' else np.zeros((n, CONTEXT, 160), np.float32)
    onnx_path=STORE/f'{task}_champion.onnx'
    torch.onnx.export(bundle, (torch.from_numpy(window[:1]), torch.from_numpy(features[:1])), onnx_path,
                      input_names=['window', 'features'], output_names=['probability'], opset_version=17,
                      dynamic_axes={'window': {0:'batch'}, 'features': {0:'batch'}, 'probability': {0:'batch'}},
                      dynamo=False)
    options=ort.SessionOptions(); options.intra_op_num_threads=2; options.inter_op_num_threads=1
    session=ort.InferenceSession(str(onnx_path), options, providers=['CPUExecutionProvider'])
    input_names=[x.name for x in session.get_inputs()]
    feed={k: (window if k=='window' else features) for k in input_names}
    native_neural, native_lr=inference(data, freeze)
    native=(native_lr if freeze['champion']=='lr' else native_neural)[:n]
    with torch.no_grad():
        eager=bundle(torch.from_numpy(window), torch.from_numpy(features)).numpy()
    actual=session.run(None, feed)[0]
    errors=dict(onnx_vs_eager=float(np.max(np.abs(actual-eager))),
                bundle_vs_training=float(np.max(np.abs(eager-native))))
    if not np.isfinite(actual).all() or max(errors.values())>1e-4:
        raise RuntimeError(f'Export parity failed: {errors}')
    single={key:value[:1] for key,value in feed.items()}
    for _ in range(10): session.run(None, single)
    timings=[]
    for _ in range(100):
        start=time.perf_counter(); session.run(None, single)
        timings.append((time.perf_counter()-start)*1000)
    metadata=dict(schema_version=1, task=task, champion=freeze['champion'], signature=freeze['signature'],
                  selection_manifest_sha256=digest(STORE/f'{task}_selection.json'),
                  model_file=onnx_path.name, model_sha256=digest(onnx_path), input_names=input_names,
                  feature_names=list(TIMING_NAMES+(PROSODY_NAMES if task=='audio' else ())),
                  context=CONTEXT, frame_ready=FRAME_READY, step=STEP, logmel_cache_dtype='float16',
                  channel_order=['other','self'], threshold=.75, research_only=True,
                  shadow_only=True, automatic_action_enabled=False,
                  neural_passed_gate=report['neural_passed_gate'],
                  reference_boundaries_required=True, parity=errors,
                  cpu_ms=dict(p50=float(np.median(timings)),p95=float(np.quantile(timings,.95))),
                  parameters=int(sum(p.numel() for p in bundle.parameters())),
                  limitation='Weak behavior labels; reference speech boundaries; not assistant appropriateness.')
    (STORE/f'{task}_champion.json').write_text(json.dumps(metadata, indent=2, allow_nan=False))
    report['export']=metadata
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(dict(task=task, phase='exported', metadata=metadata)), flush=True)
    return metadata


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('task',choices=('audio','timing'))
    export(p.parse_args().task)
