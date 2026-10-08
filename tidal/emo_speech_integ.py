"""Phase 4A (speech) integration test: do speech-emotion features of the OTHER speaker help the audio turn-taking
decisions (shift/hold at the other's segment end; backchannel ticks)? Public audio only (MagicData multi-stream zh).
Design (pre-registered 14:40, reports/phase4_progress.md): the phase-3 fold models' out-of-fold scores (p_shift / p_bc)
+ prosody/timing feats -> stacked logistic regression, with vs without 10 speech-emotion features (8 probs + valence/
arousal of the speech head, causal mean over the last 1 s of the other channel, self channel silent as in training).
Outer loop = the 3 speaker-pair folds (stacker fit on the other 2 folds' events, scored on the held-out fold).
Selection = mean log-loss on internal validation convs (every 5th conv of the training folds; LR fit on the rest).
Caveat: the speech-emotion head was distilled partly on MagicData segments (teacher emotion tags only, no turn labels).
-> reports/emotion_speech_integration_phase4.json (public data, public-safe numbers)."""
import json, numpy as np, torch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss
from tidal.audio_fe import AudioEncoder, stack_frames
from tidal.emotion_speech import SpeechEmoHead
from tidal.audio_train import CACHE
from tidal.public_data.manifest import DATA
from tidal.metrics import boot, ci, strat_auc

def emo_tracks(convs):
    ck = torch.load("models/audio_fe_oto.pt", weights_only=False); enc = AudioEncoder(); enc.load_state_dict(ck["state"]); enc.eval(); mu, sd = ck["mu"], ck["sd"]
    heads = []
    for s in range(3):
        z = torch.load(f"models/emo_speech_s{s}.pt", weights_only=False); h = SpeechEmoHead(); h.load_state_dict(z["state"]); heads.append((h.eval(), z["T"]))
    out = {}
    for c in convs:
        z = np.load(CACHE / f"{c}.npz")
        for o in (0, 1):
            mo = z[f"m{o}"].astype(np.float32); ms = np.full_like(mo, np.log(1e-6)); x = ((stack_frames(mo, ms) - mu) / sd).astype(np.float32)
            with torch.no_grad():
                hh = enc(torch.from_numpy(x[None]))[0]["h"]; inp = torch.cat([hh, torch.from_numpy(x[None])], -1); P, A = [], []
                for h, T in heads:
                    lo, va, _ = h(inp); P.append(torch.softmax(lo[0] / T, -1).numpy()); A.append(va[0].numpy())
            f = np.concatenate([np.mean(P, 0), np.mean(A, 0)], 1); cs = np.r_[np.zeros((1, f.shape[1])), np.cumsum(f, 0)]
            k = np.arange(len(f)); a = np.maximum(0, k - 49); out[(c, o)] = ((cs[k + 1] - cs[a]) / (k + 1 - a)[:, None]).astype(np.float32)
    return out

def main(n_boot=1000):
    R = json.load(open(DATA / "proc" / "audio_cv_events.json")); convs = sorted({r["conv"] for r in R}); E = emo_tracks(convs)
    folds = sorted({r["fold"] for r in R}); res = dict(note=__doc__.split("\n")[1].strip(), folds=folds)
    for kind, sc in (("shift", "pshift"), ("bc", "pbc")):
        rr = [r for r in R if r["kind"] == kind]; y = np.array([r["y"] for r in rr]); fold = np.array([r["fold"] for r in rr]); cv_ = np.array([r["conv"] for r in rr])
        Xb = np.array([[r[sc]] + r["f"] for r in rr]); Xe = np.array([E[(r["conv"], 1 - r["ch"])][min(r["k"], len(E[(r["conv"], 1 - r["ch"])]) - 1)] for r in rr])
        Xf = {"base": Xb, "emo": np.concatenate([Xb, Xe], 1)}; val_ll = {k: [] for k in Xf}; P = {k: np.zeros(len(y)) for k in Xf}
        for f in folds:
            tr = fold != f; te = fold == f; trc = sorted(set(cv_[tr])); vc = set(trc[::5]); iv = tr & np.isin(cv_, list(vc)); it = tr & ~iv
            for k, X in Xf.items():
                lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)); lr.fit(X[it], y[it]); val_ll[k].append(float(log_loss(y[iv], lr.predict_proba(X[iv])[:, 1], labels=[0, 1])))
                lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0)); lr.fit(X[tr], y[tr]); P[k][te] = lr.predict_proba(X[te])[:, 1]
        out = dict(n=int(len(y)), pos_rate=float(y.mean()), val_logloss={k: float(np.mean(v)) for k, v in val_ll.items()}, val_logloss_per_fold=val_ll)
        out["adopted_by_rule"] = bool(out["val_logloss"]["emo"] < out["val_logloss"]["base"])
        for k in Xf: out[k] = dict(pr_auc=strat_auc(y, P[k], fold, "pr"), roc_auc=strat_auc(y, P[k], fold, "roc"))
        out["model_score_only"] = dict(pr_auc=strat_auc(y, Xb[:, 0], fold, "pr"), roc_auc=strat_auc(y, Xb[:, 0], fold, "roc"))
        d = boot(lambda ix: (strat_auc(y[ix], P["emo"][ix], fold[ix], "pr") - strat_auc(y[ix], P["base"][ix], fold[ix], "pr"),
                             strat_auc(y[ix], P["emo"][ix], fold[ix]) - strat_auc(y[ix], P["base"][ix], fold[ix])), cv_, n_boot, 0)
        out["emo_minus_base_test"] = dict(d_pr_auc=float(np.nanmean(d[:, 0])), d_pr_auc_ci=ci(d[:, 0]), d_roc_auc=float(np.nanmean(d[:, 1])), d_roc_auc_ci=ci(d[:, 1]))
        res[kind] = out; print(kind, json.dumps({k: v for k, v in out.items() if k != "val_logloss_per_fold"}), flush=True)
    json.dump(res, open("reports/emotion_speech_integration_phase4.json", "w"), indent=1)

if __name__ == "__main__": main()
