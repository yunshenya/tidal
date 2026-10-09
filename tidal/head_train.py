"""Bounded task-head training with validation-only selection and frozen test evaluation."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

from tidal.task_heads import VERSION, CLASSES, PAIR_NAMES, OVERLAP_NAMES, MAX_CONTEXT, HASH_DIM

SEEDS = (0, 1, 2)
EPOCHS = 20
PATIENCE = 4
TEMPERATURES = (.5, .75, 1., 1.5, 2., 3.)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TaskNet(nn.Module):
    def __init__(self, n_features, n_classes, kind='mlp'):
        super().__init__()
        self.body = (nn.Linear(n_features, n_classes) if kind == 'linear' else
                     nn.Sequential(nn.Linear(n_features, 64 if kind == 'mlp64' else 32), nn.GELU(), nn.Linear(64 if kind == 'mlp64' else 32, n_classes)))
    def forward(self, x): return self.body(x)


def loss_vector(logits, y, mask=None):
    if mask is None:
        return nn.functional.cross_entropy(logits, y.long(), reduction='none')
    if y.shape != mask.shape or not bool((y & mask).any(-1).all()) or bool((y & ~mask).any()):
        raise ValueError('Each pointer needs at least one visible gold target')
    logits = logits.squeeze(-1).masked_fill(~mask, -1e9)
    return torch.logsumexp(logits, -1)-torch.logsumexp(logits.masked_fill(~y, -1e9), -1)


def probabilities(logits, temperature=1., mask=None):
    z = np.asarray(logits, np.float64)/temperature
    if mask is not None: z = np.where(mask, z, -1e9)
    z -= z.max(-1, keepdims=True); p = np.exp(z); p /= p.sum(-1, keepdims=True)
    return p


def per_row(p, y):
    if y.ndim == 2:
        success = y[np.arange(len(y)), p.argmax(-1)]
        nll = -np.log(np.maximum((p*y).sum(-1), 1e-12))
    else:
        success = p.argmax(-1) == y
        nll = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12))
    return success.astype(float), nll


def summary(p, y):
    correct, nll = per_row(p, y)
    result = dict(n=len(y), accuracy=float(correct.mean()), nll=float(nll.mean()))
    if y.ndim == 2:
        null_pred = p.argmax(-1) == 0; null_gold = y[:, 0]
        result.update(null_precision=float((null_pred & null_gold).sum()/max(1, null_pred.sum())),
                      null_recall=float((null_pred & null_gold).sum()/max(1, null_gold.sum())),
                      visible_target_rate=float(y[:, 1:].any(-1).mean()),
                      null_target_rate=float(null_gold.mean()))
    else:
        result.update(macro_f1=float(f1_score(y, p.argmax(-1), labels=np.arange(p.shape[1]), average='macro', zero_division=0)),
                      balanced_accuracy=float(balanced_accuracy_score(y, p.argmax(-1))))
        if p.shape[1] == 2 and len(np.unique(y)) == 2:
            result.update(roc_auc=float(roc_auc_score(y, p[:, 1])), brier=float(np.mean((p[:, 1]-y)**2)))
    return result


def block_delta(champion, baseline, y, groups):
    a, an = per_row(champion, y); b, bn = per_row(baseline, y)
    unique = sorted(set(groups)); totals = np.array([[sum(groups==g), (a-b)[groups==g].sum(), (bn-an)[groups==g].sum()] for g in unique])
    rng = np.random.default_rng(601); deltas = []
    for _ in range(2000):
        sample = totals[rng.integers(0, len(totals), len(totals))].sum(0)
        deltas.append(sample[1:]/sample[0])
    q = np.quantile(deltas, [.025, .975], axis=0)
    return dict(blocks=len(unique), accuracy_improvement_ci=q[:, 0].tolist(), nll_improvement_ci=q[:, 1].tolist())


def disjoint(*datasets):
    for i, d in enumerate(datasets):
        for other in datasets[i+1:]:
            if set(d['groups']) & set(other['groups']): raise ValueError('Conversation/date leakage')
            if 'speakers' in d and set(d['speakers']) & set(other.get('speakers', ())): raise ValueError('Speaker leakage')


def train_model(kind, train, val, mu, sd, n_classes, seed, pointer):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    x = torch.from_numpy((train['x']-mu)/sd); v = torch.from_numpy((val['x']-mu)/sd)
    y = torch.from_numpy(train['y']); vy = torch.from_numpy(val['y'])
    mask = torch.from_numpy(train['mask']) if pointer else None
    vm = torch.from_numpy(val['mask']) if pointer else None
    model = TaskNet(x.shape[-1], n_classes, kind)
    opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.01)
    best = (float('inf'), None, -1); history = []
    for epoch in range(EPOCHS):
        model.train(); order = rng.permutation(len(x)); losses = []
        for start in range(0, len(x), 256):
            ids = order[start:start+256]
            loss = loss_vector(model(x[ids]), y[ids], None if mask is None else mask[ids]).mean()
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            vl = float(loss_vector(model(v), vy, vm).mean())
        history.append(dict(epoch=epoch, train=float(np.mean(losses)), val=vl))
        if vl < best[0]-1e-4: best = (vl, {k: t.detach().clone() for k, t in model.state_dict().items()}, epoch)
        elif epoch-best[2] >= PATIENCE: break
    if best[1] is None: raise RuntimeError('No finite trained model')
    model.load_state_dict(best[1]); model.eval()
    print(json.dumps(dict(kind=kind, seed=seed, best_epoch=best[2], val=best[0])), flush=True)
    return model, history


class ExportedTask(nn.Module):
    def __init__(self, models, mu, sd, temperature, pointer):
        super().__init__(); self.models = nn.ModuleList(models); self.pointer = pointer; self.temperature = temperature
        self.register_buffer('mu', torch.from_numpy(mu)); self.register_buffer('sd', torch.from_numpy(sd))
    def forward(self, features, mask=None):
        x = (features-self.mu)/self.sd
        logits = torch.stack([m(x) for m in self.models]).mean(0)/self.temperature
        if self.pointer: logits = logits.squeeze(-1).masked_fill(~mask, -1e9)
        return torch.softmax(logits, -1)


def fit_task(task, loader, directory, signature):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    train, val = loader('train'), loader('val'); disjoint(train, val)
    pointer = task == 'reply_to'; n_classes = 1 if pointer else len(CLASSES[task])
    if not pointer:
        for d in (train, val):
            if len(d['y']) < 20 or set(d['y']) != set(range(n_classes)):
                return dict(task=task, status='data_insufficient', train_n=len(train['y']), val_n=len(val['y']))
    flat = train['x'][train['mask']] if pointer else train['x']
    mu = flat.mean(0).astype(np.float32); sd = flat.std(0).clip(.05).astype(np.float32)
    candidates = {}; calibration = {}; selection = {}
    for kind in ('linear', 'mlp'):
        models = []; histories = []
        for seed in SEEDS:
            model, history = train_model(kind, train, val, mu, sd, n_classes, seed, pointer)
            path = directory/f'{task}_{kind}_{seed}.pt'
            torch.save(dict(state=model.state_dict(), kind=kind, n_features=len(mu), n_classes=n_classes,
                            mu=mu, sd=sd, signature=signature, history=history), path)
            models.append(model); histories.append(history)
        with torch.no_grad():
            z = torch.stack([m(torch.from_numpy((val['x']-mu)/sd)) for m in models]).mean(0).numpy()
        if pointer: z = z.squeeze(-1)
        losses = [(float(per_row(probabilities(z, t, val.get('mask')), val['y'])[1].mean()), t) for t in TEMPERATURES]
        vl, temperature = min(losses); candidates[kind] = models; calibration[kind] = temperature; selection[kind] = vl
    champion = min(selection, key=selection.get)
    freeze = dict(task=task, signature=signature, champion=champion, val_nll=selection, temperature=calibration,
                  weights={p.name: digest(p) for p in directory.glob(f'{task}_*.pt')}, selection_basis='validation_only')
    freeze_path = directory/f'{task}_selection.json'; freeze_path.write_text(json.dumps(freeze, indent=2)+'\n')
    print(json.dumps(dict(task=task, phase='frozen', champion=champion)), flush=True)
    # Test is first constructed only after selection and weight hashes are frozen.
    test = loader('test'); disjoint(train, val, test)
    if not pointer and (len(test['y']) < 20 or set(test['y']) != set(range(n_classes))):
        return dict(task=task, status='test_data_insufficient', train_n=len(train['y']), val_n=len(val['y']), test_n=len(test['y']))
    predictions = {}
    for kind, models in candidates.items():
        with torch.no_grad():
            z = torch.stack([m(torch.from_numpy((test['x']-mu)/sd)) for m in models]).mean(0).numpy()
        if pointer: z = z.squeeze(-1)
        predictions[kind] = probabilities(z, calibration[kind], test.get('mask'))
    p = predictions[champion]; baselines = {}
    if pointer:
        for name in ('null', 'latest', 'mention'):
            choices = np.zeros(len(p), int)
            if name != 'null': choices = test['mask'].sum(-1)-1
            if name == 'mention':
                for i, row in enumerate(test['x']):
                    mentions = np.flatnonzero((row[:, 3] > .5) & test['mask'][i])
                    if len(mentions): choices[i] = mentions[-1]
            bp = test['mask'].astype(float)*1e-4; bp[np.arange(len(p)), choices] += 1.; bp /= bp.sum(-1, keepdims=True)
            baselines[name] = bp
    else:
        prior = np.bincount(train['y'], minlength=n_classes)/len(train['y'])
        baselines['prior'] = np.repeat(prior[None], len(p), axis=0)
    baselines['linear'] = predictions['linear']
    deltas = {name: block_delta(p, bp, test['y'], test['groups']) for name, bp in baselines.items()}
    passed = champion == 'mlp' and all(v['blocks'] >= 5 and v['accuracy_improvement_ci'][0] > 0 and v['nll_improvement_ci'][0] > 0 for v in deltas.values())
    report = dict(task=task, status='trained_evaluated', signature=signature, champion=champion,
                  train_n=len(train['y']), val_n=len(val['y']), test_n=len(test['y']),
                  split_blocks={s: len(set(d['groups'])) for s, d in [('train', train), ('val', val), ('test', test)]},
                  counts={s: d.get('counts', {}) for s, d in [('train', train), ('val', val), ('test', test)]},
                  neural=summary(predictions['mlp'], test['y']), selected=summary(p, test['y']),
                  baselines={name: summary(bp, test['y']) for name, bp in baselines.items()},
                  deltas=deltas, neural_passed_gate=passed, shadow_only=True, automatic_action_enabled=False,
                  selection_manifest_sha256=digest(freeze_path))
    # Local-only raw predictions for reproducibility.
    np.savez(directory/f'{task}_predictions.npz', probabilities=p, y=test['y'], groups=test['groups'].astype(str))
    exported = ExportedTask(candidates[champion], mu, sd, calibration[champion], pointer).eval()
    sample = torch.from_numpy(test['x'][:4]); args = (sample, torch.from_numpy(test['mask'][:4])) if pointer else (sample,)
    model_path = directory/f'{task}.onnx'; names = ['features', 'mask'] if pointer else ['features']
    torch.onnx.export(exported, args, str(model_path), input_names=names, output_names=['probabilities'],
                      dynamic_axes={n: {0: 'batch'} for n in names+['probabilities']}, opset_version=17, dynamo=False)
    import onnxruntime as ort
    session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
    feed = {name: value.numpy() for name, value in zip(names, args)}
    with torch.no_grad(): expected = exported(*args).numpy()
    got = session.run(None, feed)[0]; error = float(np.max(np.abs(got-expected)))
    if error > 1e-5: raise RuntimeError('ONNX parity failed')
    meta = dict(schema_version=VERSION, task=task, classes=list(CLASSES.get(task, ())),
                feature_names=list(PAIR_NAMES if pointer else OVERLAP_NAMES if task == 'overlap_outcome' else ()),
                feature_contract='task_heads_v1', max_context=MAX_CONTEXT, hash_dim=HASH_DIM,
                model_file=model_path.name, model_sha256=digest(model_path), signature=signature,
                selection_manifest_sha256=digest(freeze_path), neural_passed_gate=passed,
                shadow_only=True, automatic_action_enabled=False, onnx_parity_max_error=error,
                draft_conditioned=pointer, weak_behavior_proxy=task == 'overlap_outcome', research_only=task != 'reply_to',
                parameters=sum(sum(t.numel() for t in m.parameters()) for m in candidates[champion]))
    (directory/f'{task}_bundle.json').write_text(json.dumps(meta, indent=2)+'\n'); report['export'] = meta
    return report
