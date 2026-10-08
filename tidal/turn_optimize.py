"""Bounded, resumable unattended turn-prediction optimization.

python -m tidal.turn_optimize all
python -m tidal.turn_optimize timing | audio

Candidates and splits are fixed in reports/audio_turn_v2_protocol.md. Test data
is loaded only after an immutable selection manifest is written. No live action.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import joblib
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn

from tidal.audio_fe import AudioEncoder, FRAME_READY
from tidal.audio_train import prep
from tidal.audio_turn import scores, decisions
from tidal.metrics import boot, ci
from tidal.public_data.manifest import DATA, DATASETS
from tidal.turn_data import (CONTEXT, TIMING_NAMES, PROSODY_NAMES, audio_data, candor_data,
                             assert_disjoint, window_cmvn)

ROOT = Path(__file__).resolve().parents[1]
STORE = ROOT/'models'/'turn_v2'
REPORTS = ROOT/'reports'
SEEDS = (0, 1, 2)
EPOCHS = 30
TRAIN_GROUPS = {'zh': ('A1012', 'A1091'), 'en': ('Group0006', 'Group0030')}
VAL_GROUPS = {'zh': ('A1102',), 'en': ('Group0046',)}
TEST_GROUPS = {'en': ('Group0078',)}


def emit(**values):
    print(json.dumps(values), flush=True)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def signature(task):
    files = ('turn_optimize.py', 'turn_data.py', 'audio_fe.py', 'audio_train.py', 'audio_turn.py', 'audio_spec.py', 'turn_features.py')
    sources = {name: digest(Path(__file__).parent/name) for name in files}
    revisions = {name: DATASETS[name]['revision'] for name in ('magicdata_ms', 'magicdata_en', 'candor_tt')}
    spec = dict(task=task, sources=sources, revisions=revisions, seeds=SEEDS, epochs=EPOCHS,
                train=TRAIN_GROUPS, val=VAL_GROUPS, test=TEST_GROUPS)
    return hashlib.sha256(json.dumps(spec, sort_keys=True).encode()).hexdigest(), spec


class TimingNet(nn.Module):
    def __init__(self, n_features=12):
        super().__init__()
        self.body = nn.Sequential(nn.Linear(n_features, 16), nn.GELU(), nn.Linear(16, 16),
                                  nn.GELU(), nn.Linear(16, 1))
    def forward(self, features):
        return self.body(features)[:, 0]


class WindowNet(nn.Module):
    def __init__(self, n_features=20):
        super().__init__()
        self.encoder = AudioEncoder(d=48)
        self.head = nn.Sequential(nn.Linear(48+n_features, 24), nn.GELU(), nn.Linear(24, 1))
    def forward(self, window, features):
        outputs, _ = self.encoder(window)
        return self.head(torch.cat([outputs['h'][:, -1], features], 1))[:, 0]


def calibration(logits, y):
    """Monotonic affine logit calibration, fitted only to validation observations."""
    logits, y = np.asarray(logits, float), np.asarray(y, float)
    def objective(v):
        z = np.exp(v[0])*logits+v[1]
        return np.mean(np.logaddexp(0., z)-y*z)+1e-4*np.dot(v, v)
    fit = minimize(objective, np.zeros(2), method='L-BFGS-B', bounds=((-3., 3.), (-5., 5.)))
    if not fit.success or not np.isfinite(fit.fun):
        raise RuntimeError(f'Calibration failed: {fit.message}')
    return dict(scale=float(np.exp(fit.x[0])), bias=float(fit.x[1]))


def apply_calibration(logits, params):
    return expit(params['scale']*np.asarray(logits)+params['bias'])


def logit(p):
    p = np.clip(p, 1e-6, 1.-1e-6)
    return np.log(p)-np.log1p(-p)


def batches(indices, size):
    for start in range(0, len(indices), size):
        yield indices[start:start+size]


def prepare_input(data, ck):
    f = torch.from_numpy(((data.features-ck['f_mu'])/ck['f_sd']).astype(np.float32))
    if ck['kind']=='timing':
        return (f,)
    w = window_cmvn(data.windows) if ck['kind']=='cmvn' else data.windows
    w = torch.from_numpy(((w-ck['mu'])/ck['sd']).astype(np.float32))
    return w, f


def model_for(ck):
    model = TimingNet(len(ck['f_mu'])) if ck['kind']=='timing' else WindowNet(len(ck['f_mu']))
    model.load_state_dict(ck['state'])
    return model.eval()


@torch.no_grad()
def predict_checkpoint(data, ck):
    model, inputs = model_for(ck), prepare_input(data, ck)
    logits=[]
    for idx in batches(np.arange(len(data.y)), 128):
        logits.append(model(*(x[idx] for x in inputs)))
    return torch.cat(logits).numpy()


def train_seed(task, kind, train, val, seed, sig):
    path = STORE/f'{task}_{kind}_{seed}.pt'
    if path.exists():
        ck = torch.load(path, weights_only=False)
        if ck.get('signature')==sig:
            emit(task=task, kind=kind, seed=seed, phase='resume')
            return ck, path
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    ck = dict(kind=kind, f_mu=train.features.mean(0), f_sd=np.maximum(train.features.std(0), 1e-4),
              signature=sig, seed=seed, research_only=True, context=CONTEXT)
    if kind!='timing':
        w = window_cmvn(train.windows) if kind=='cmvn' else train.windows
        ck.update(mu=w.mean((0, 1)), sd=np.maximum(w.std((0, 1)), 1e-4))
    model = TimingNet(train.features.shape[1]) if kind=='timing' else WindowNet(train.features.shape[1])
    inputs, vi = prepare_input(train, ck), prepare_input(val, ck)
    y, vy = torch.from_numpy(train.y), torch.from_numpy(val.y)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=.01)
    best, best_epoch, state = float('inf'), -1, None
    history=[]; size=256 if kind=='timing' else 32
    for epoch in range(EPOCHS):
        start=time.monotonic(); model.train(); losses=[]
        for idx in batches(rng.permutation(len(y)), size):
            loss=nn.functional.binary_cross_entropy_with_logits(model(*(x[idx] for x in inputs)), y[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training loss')
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            predictions=torch.cat([model(*(x[idx] for x in vi)) for idx in batches(np.arange(len(vy)), 128)])
            value=float(nn.functional.binary_cross_entropy_with_logits(predictions, vy))
        if not np.isfinite(value):
            raise FloatingPointError('Nonfinite validation loss')
        record=dict(epoch=epoch, train_bce=float(np.mean(losses)), val_bce=value, seconds=time.monotonic()-start)
        history.append(record); emit(task=task, kind=kind, seed=seed, **record)
        if value<best-1e-4:
            best, best_epoch= value, epoch
            state={key: tensor.detach().clone() for key, tensor in model.state_dict().items()}
        elif epoch-best_epoch>=6:
            break
    ck.update(state=state, best_epoch=best_epoch, best_val_bce=best, history=history)
    torch.save(ck, path)
    return ck, path


def fit_candidates(task, train, val, sig, spec):
    scaler=StandardScaler().fit(train.features)
    tx, vx=scaler.transform(train.features), scaler.transform(val.features)
    lr_search=[]; best=None
    for c in (.01, .1, 1., 10.):
        model=LogisticRegression(C=c, max_iter=2000).fit(tx, train.y)
        vl=model.decision_function(vx)
        score=float(np.mean(np.logaddexp(0., vl)-val.y*vl))
        lr_search.append(dict(C=c, val_bce=score))
        if best is None or score<best[0]:
            best=score, model, vl
    lr_cal=calibration(best[2], val.y)
    lr_val=apply_calibration(best[2], lr_cal)
    lr_path=STORE/f'{task}_lr.joblib'
    joblib.dump(dict(model=best[1], scaler=scaler, calibration=lr_cal, signature=sig), lr_path)
    kinds=('timing',) if task=='timing' else ('raw', 'cmvn')
    neural=[]
    for kind in kinds:
        values=[]; paths=[]; runs=[]
        for seed in SEEDS:
            ck, path=train_seed(task, kind, train, val, seed, sig)
            values.append(expit(predict_checkpoint(val, ck))); paths.append(str(path.relative_to(ROOT)))
            runs.append(dict(seed=seed, best_epoch=ck['best_epoch'], val_bce=ck['best_val_bce']))
        vl=logit(np.mean(values, axis=0)); params=calibration(vl, val.y)
        p=apply_calibration(vl, params)
        neural.append(dict(kind=kind, checkpoints=paths, calibration=params,
                           val=scores(val.y, p), runs=runs))
    chosen=min(neural, key=lambda r:r['val']['bce'])
    lr_metrics=scores(val.y, lr_val)
    champion=chosen['kind'] if chosen['val']['bce']<lr_metrics['bce'] else 'lr'
    freeze=dict(signature=sig, specification=spec, task=task, selected_neural=chosen,
                champion=champion, lr=dict(path=str(lr_path.relative_to(ROOT)), C=best[1].C,
                                          calibration=lr_cal, val=lr_metrics, search=lr_search),
                candidates=neural, n_train=len(train.y), n_val=len(val.y),
                train_conversations=len({r['conv'] for r in train.rows}),
                val_conversations=len({r['conv'] for r in val.rows}),
                train_prior=float(train.y.mean()))
    artifacts=[lr_path]+[ROOT/p for item in neural for p in item['checkpoints']]
    freeze['artifact_sha256']={str(p.relative_to(ROOT)):digest(p) for p in artifacts}
    path=STORE/f'{task}_selection.json'
    path.write_text(json.dumps(freeze, indent=2, allow_nan=False))
    emit(task=task, phase='selection_frozen', champion=champion,
         selected_neural=chosen['kind'], manifest_sha256=digest(path))
    return freeze


def verify_selection(freeze, sig):
    if freeze['signature']!=sig:
        raise ValueError('Selection signature changed')
    for path, sha in freeze['artifact_sha256'].items():
        if digest(ROOT/path)!=sha:
            raise ValueError(f'Frozen artifact changed: {path}')


def inference(data, freeze):
    neural=freeze['selected_neural']; predictions=[]
    for path in neural['checkpoints']:
        ck=torch.load(ROOT/path, weights_only=False)
        predictions.append(expit(predict_checkpoint(data, ck)))
    p=apply_calibration(logit(np.mean(predictions, axis=0)), neural['calibration'])
    lr=joblib.load(ROOT/freeze['lr']['path'])
    q=apply_calibration(lr['model'].decision_function(lr['scaler'].transform(data.features)), lr['calibration'])
    return p, q


def evaluate(task, test, freeze):
    p, q=inference(test, freeze); y=test.y
    blocks=np.array([r['conv'] for r in test.rows])
    def delta(idx):
        if not 0<y[idx].sum()<len(idx):
            return np.nan, np.nan
        return (roc_auc_score(y[idx], p[idx])-roc_auc_score(y[idx], q[idx]),
                np.mean((q[idx]-y[idx])**2)-np.mean((p[idx]-y[idx])**2))
    samples=boot(delta, blocks, 1000, 0)
    intervals=dict(roc_delta=ci(samples[:, 0]), brier_improvement=ci(samples[:, 1]))
    passed=bool(intervals['roc_delta'][0]>0 and intervals['brier_improvement'][0]>0)
    report=dict(task=task, signature=freeze['signature'], n_test=len(y),
                test_conversations=len(set(blocks)), positive_rate=float(y.mean()),
                n_train=freeze['n_train'], n_val=freeze['n_val'],
                train_conversations=freeze['train_conversations'], val_conversations=freeze['val_conversations'],
                selected_neural=freeze['selected_neural']['kind'], champion=freeze['champion'],
                neural=scores(y, p), lr=scores(y, q), prior=scores(y, np.full(len(y), freeze['train_prior'])),
                paired_ci=intervals, neural_passed_gate=passed, automatic_action_enabled=False,
                decisions=dict(neural=decisions(y, p, .75), lr=decisions(y, q, .75)),
                validation=dict(neural=freeze['selected_neural']['val'], lr=freeze['lr']['val']),
                selection_manifest_sha256=digest(STORE/f'{task}_selection.json'))
    for group in sorted({r.get('group', '') for r in test.rows}):
        if group:
            report.setdefault('test_groups', []).append(group)
    predictions=dict(rows=test.rows, y=y.tolist(), neural=p.tolist(), lr=q.tolist())
    out=DATA/'proc'/f'turn_v2_{task}_predictions.json'; out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(predictions, allow_nan=False))
    return report


def ensure_data():
    missing=[]
    for name in ('magicdata_ms', 'magicdata_en', 'candor_tt'):
        path=DATA/name/'_done.json'
        done=json.loads(path.read_text()) if path.exists() else {}
        if done.get('revision')!=DATASETS[name]['revision'] or done.get('ok', -1)!=done.get('files', -2):
            missing.append(name)
    if missing:
        from tidal.public_data.download import main
        main(missing, workers=4)
    prep()
    prep(DATA/'magicdata_en', DATA/'proc'/'audio_en')


def run(task):
    torch.set_num_threads(2); STORE.mkdir(parents=True, exist_ok=True)
    sig, spec=signature(task); report_path=REPORTS/f'turn_v2_{task}.json'
    if report_path.exists():
        prior=json.loads(report_path.read_text())
        if prior.get('signature')==sig:
            emit(task=task, phase='completed_resume')
            return prior
    if task=='timing':
        train, val=candor_data('train'), candor_data('val')
    else:
        train, val=audio_data(TRAIN_GROUPS), audio_data(VAL_GROUPS)
    assert_disjoint(train, val, speakers=task=='audio')
    emit(task=task, phase='development_ready', n_train=len(train.y), n_val=len(val.y))
    selection=STORE/f'{task}_selection.json'
    freeze=json.loads(selection.read_text()) if selection.exists() else {}
    if freeze.get('signature')!=sig:
        freeze=fit_candidates(task, train, val, sig, spec)
    verify_selection(freeze, sig)
    # The first load/target construction of the test split happens after freezing.
    test=candor_data('test') if task=='timing' else audio_data(TEST_GROUPS)
    assert_disjoint(train, val, test, speakers=task=='audio')
    report=evaluate(task, test, freeze)
    verify_selection(freeze, sig)
    REPORTS.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, allow_nan=False))
    emit(phase='test_completed', **report)
    return report


def main():
    p=argparse.ArgumentParser()
    p.add_argument('task', choices=('all', 'timing', 'audio'))
    a=p.parse_args(); ensure_data()
    for task in (('timing', 'audio') if a.task=='all' else (a.task,)):
        run(task)


if __name__=='__main__':
    main()
