"""Causal event-level shift experiment. Run: python -m tidal.audio_turn.

Decision opportunities use reference segment ends (not a deployable VAD).
A speaker-pair-disjoint validation fold selects epochs; test never selects weights.
Audio windows end at the latest fully available stacked frame, not its start.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from torch import nn
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, log_loss, brier_score_loss
from tidal.audio_fe import AudioEncoder, STEP, STACK, FRAME_READY
from tidal.audio_train import SRC, CACHE, CACHE_VERSION, load_conv, persp, shift_events


def available_index(t):
    """Last feature whose entire PCM window is observable at wall-clock t."""
    return int(np.floor((t - FRAME_READY + 1e-10) / STEP))

class TurnModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = AudioEncoder(d=48)
        self.head = nn.Sequential(nn.Linear(49, 24), nn.GELU(), nn.Linear(24, 1))
    def forward(self, x, duration):
        out, _ = self.encoder(x)
        return self.head(torch.cat([out['h'][:, -1], duration[:, None]], 1))[:, 0]

def dataset(context=150):
    rows, windows = [], []
    for path in sorted(CACHE.glob('*.npz')):
        c = path.stem
        with np.load(path) as z:
            if "cache_version" not in z or int(z["cache_version"]) != CACHE_VERSION:
                raise ValueError(f"Stale cache {path}; rerun tidal.audio_train prep")
        speakers = sorted(p.stem.rsplit("_",1)[-1] for p in (SRC/"TXT").glob(f"{c}_0_*.txt"))
        if len(speakers) != 2:
            raise ValueError(f"Cannot verify two speaker identities for {c}")
        cv = load_conv(c)
        for ch in (0, 1):
            x, _, _, _ = persp(cv, ch)
            observed_until = (len(x)-1)*STEP + FRAME_READY
            for t, y, dur in shift_events(cv['S'], ch, observed_until):
                k = available_index(t)
                if k < context - 1 or k >= len(x):
                    continue
                w = x[k-context+1:k+1]
                # Prosody baseline uses the same observable acoustic window.
                a = w.reshape(context, STACK, 2, -1).mean(1)
                f = np.r_[a[-10:].mean((0,2)), a[-50:-10].mean((0,2)),
                          a[-10:].std((0,2)), a.mean((0,2)), np.log1p(dur)]
                rows.append(dict(conv=c, pair=c.split('_')[0], ch=ch, t=t,
                                 y=y, speakers=speakers, duration=np.log1p(dur), f=f.tolist()))
                windows.append(w.astype(np.float32))
    if not rows:
        raise ValueError('No cached conversations; run tidal.audio_train prep first')
    return rows, np.stack(windows)

def scores(y, p):
    return dict(roc_auc=float(roc_auc_score(y,p)), pr_auc=float(average_precision_score(y,p)),
                bce=float(log_loss(y,p,labels=[0,1])), brier=float(brier_score_loss(y,p)))

def select_threshold(y, p, false_take_cost=3.):
    """Validation-only threshold; WAIT is a candidate, with utility zero."""
    thresholds = np.r_[np.inf, np.unique(p)[::-1]]
    utility = [np.sum(y[p >= t]) - false_take_cost*np.sum(1-y[p >= t]) for t in thresholds]
    return float(thresholds[int(np.argmax(utility))])

def decisions(y, p, threshold):
    take = p >= threshold
    tp = int(np.sum(take & (y == 1))); fp = int(np.sum(take & (y == 0)))
    return dict(threshold=None if not np.isfinite(threshold) else threshold,
                take_count=int(take.sum()),wrong_take_count=fp,
                precision=tp / max(1,int(take.sum())),recall=tp / max(1,int(y.sum())),
                utility_per_event=(tp-3*fp)/len(y))

def experiment(epochs=30, seeds=(0,1,2)):
    if epochs < 1 or not seeds:
        raise ValueError("epochs and seeds must be nonempty")
    manifest = json.loads((SRC/"_done.json").read_text())
    if manifest["ok"] != manifest["files"]:
        raise ValueError("Incomplete source dataset")
    torch.set_num_threads(2)
    rows, X = dataset(); y = np.array([r['y'] for r in rows],np.float32)
    d = np.array([r['duration'] for r in rows],np.float32)
    F = np.array([r['f'] for r in rows]); pairs = sorted({r['pair'] for r in rows})
    if len(pairs) < 3:
        raise ValueError('Need at least three distinct speaker pairs')
    predictions=[]; folds=[]
    for i, test in enumerate(pairs):
        val = pairs[(i+1)%len(pairs)]
        tr = np.array([r['pair'] not in (test,val) for r in rows])
        va = np.array([r['pair']==val for r in rows]); te = np.array([r['pair']==test for r in rows])
        identities=[{s for j in np.flatnonzero(mask) for s in rows[j]["speakers"]} for mask in (tr,va,te)]
        if any(identities[a] & identities[b] for a,b in ((0,1),(0,2),(1,2))):
            raise ValueError("Speaker overlap across train/validation/test")
        mu=X[tr].mean((0,1)); sd=X[tr].std((0,1))+1e-4
        Z=torch.from_numpy((X-mu)/sd); D=torch.from_numpy(d); Y=torch.from_numpy(y)
        baseline=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000))
        baseline.fit(F[tr],y[tr]); lp=baseline.predict_proba(F[te])[:,1]
        out=[]; validation=[]; runs=[]
        for seed in seeds:
            torch.manual_seed(seed); rng=np.random.default_rng(seed); m=TurnModel()
            opt=torch.optim.AdamW(m.parameters(),lr=1e-3,weight_decay=.01)
            best=float('inf'); best_state=None; best_ep=-1
            for ep in range(epochs):
                m.train()
                for idx in np.array_split(rng.permutation(np.flatnonzero(tr)),max(1,int(np.ceil(tr.sum()/32)))):
                    loss=nn.functional.binary_cross_entropy_with_logits(m(Z[idx],D[idx]),Y[idx])
                    opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1);opt.step()
                m.eval()
                with torch.no_grad():
                    p=torch.cat([m(Z[idx],D[idx]) for idx in np.array_split(np.flatnonzero(va),max(1,int(np.ceil(va.sum()/64))))])
                    vl=float(nn.functional.binary_cross_entropy_with_logits(p,Y[va]))
                print(json.dumps(dict(test=test,val=val,seed=seed,epoch=ep,val_bce=vl)),flush=True)
                if vl < best-1e-4:
                    best=vl;best_ep=ep;best_state={k:v.detach().clone() for k,v in m.state_dict().items()}
                elif ep-best_ep>=6:
                    break
            m.load_state_dict(best_state);m.eval()
            with torch.no_grad():
                p=torch.cat([torch.sigmoid(m(Z[idx],D[idx])) for idx in np.array_split(np.flatnonzero(te),max(1,int(np.ceil(te.sum()/64))))]).numpy()
            out.append(p)
            with torch.no_grad():
                validation.append(torch.cat([torch.sigmoid(m(Z[idx],D[idx])) for idx in np.array_split(np.flatnonzero(va),max(1,int(np.ceil(va.sum()/64))))]).numpy())
            runs.append(dict(seed=seed,best_epoch=best_ep,val_bce=best))
            Path('models').mkdir(exist_ok=True)
            torch.save(dict(state=best_state,mu=mu,sd=sd,context=150,frame_ready=FRAME_READY,
                            test_pair=test,val_pair=val,seed=seed,best_epoch=best_ep,
                            research_only=True,evaluation_only=True),f'models/audio_turn_{test}_{seed}.pt')
        p=np.mean(out,axis=0)
        mt=select_threshold(y[va],np.mean(validation,axis=0))
        lt=select_threshold(y[va],baseline.predict_proba(F[va])[:,1])
        fold=dict(test=test,val=val,n_train=int(tr.sum()),n_val=int(va.sum()),n_test=int(te.sum()),
                  model=scores(y[te],p),lr=scores(y[te],lp),runs=runs,
                  prior=scores(y[te],np.full(te.sum(),float(y[tr].mean()))),
                  decisions=dict(model=decisions(y[te],p,mt),lr=decisions(y[te],lp,lt)))
        folds.append(fold);print(json.dumps(fold),flush=True)
        for j,pm,pl in zip(np.flatnonzero(te),p,lp):
            predictions.append(dict(conv=rows[j]['conv'],fold=test,y=float(y[j]),model=float(pm),lr=float(pl)))
    report=dict(protocol='Reference segment-end +200ms; 3s causal audio; speaker-pair-disjoint train/val/test; seeds averaged',
                limitation='Behavior prediction with oracle segment boundaries, not annotated appropriateness or deployed VAD.',
                seeds=list(seeds),folds=folds)
    from tidal.metrics import boot, ci, strat_auc
    yy=np.array([r['y'] for r in predictions]); pm=np.array([r['model'] for r in predictions])
    pl=np.array([r['lr'] for r in predictions]); ff=np.array([r['fold'] for r in predictions])
    blocks=np.array([r['conv'] for r in predictions])
    delta=boot(lambda idx: (strat_auc(yy[idx],pm[idx],ff[idx])-strat_auc(yy[idx],pl[idx],ff[idx]),
                           np.mean((pl[idx]-yy[idx])**2)-np.mean((pm[idx]-yy[idx])**2)),blocks,1000,0)
    report['aggregate']=dict(n=len(yy),positive_rate=float(yy.mean()),
        model_roc_auc=strat_auc(yy,pm,ff),lr_roc_auc=strat_auc(yy,pl,ff),
        model_pr_auc=strat_auc(yy,pm,ff,'pr'),lr_pr_auc=strat_auc(yy,pl,ff,'pr'),
        delta_roc_ci=ci(delta[:,0]),brier_improvement_ci=ci(delta[:,1]))
    # Require both ranking and calibrated probability improvement before adoption.
    report['adopt_neural']=bool(report['aggregate']['delta_roc_ci'][0]>0 and
                              report['aggregate']['brier_improvement_ci'][0]>0)
    report['dataset_revision']=manifest
    report['parameters']=sum(p.numel() for p in m.parameters())
    Path('reports').mkdir(exist_ok=True)
    Path('reports/audio_turn_optimization.json').write_text(json.dumps(report,indent=2))
    Path('data/public/proc/audio_turn_events.json').write_text(json.dumps(predictions))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=30);p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
    a=p.parse_args();experiment(a.epochs,a.seeds)
