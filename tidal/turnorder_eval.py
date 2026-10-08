"""Turn-order-only evaluation for short AI-host dialogue snippets (no timing inside snippets): source "ai_stream" in
data/proc/p3 (built only if a private snippet corpus is present; nothing here is published but the code).
Tasks on held-out snippets (split ai_test), all binary:
  respond      after a human turn, does the AI host take the next turn?           model score: P(speak)
  continue     after an AI turn, does the AI host also take the next turn?         model score: 1 - P(end of turn)
  next_is_ai   union of both (who speaks next: AI host vs a human)
Baselines: prior; logistic regression on turn-order features (current role, previous role, position, participants,
partner kind) fitted on ai_train. Block bootstrap over snippets. -> reports/turnorder_phase3.json (private)"""
import json, os, sys, numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from tidal import vap as VP, metrics as M

def task_rows(meta, split):
    m = meta[(meta.source == "ai_stream")].copy(); m["row"] = m.index
    g = m.groupby("conv"); m["next_role"] = g.role.shift(-1); m["k"] = g.cumcount(); m["prev_role"] = g.role.shift(1).fillna("none")
    gapn = g.ts.shift(-1) - m.ts; m = m[m.next_role.notna() & (gapn < 100) & (m.split == split)]
    m["y"] = (m.next_role == "self").astype(int); return m

def feats(m, npart):
    return np.c_[(m.role == "self"), (m.prev_role == "self"), (m.prev_role == "none"), np.log1p(m.k), np.log1p(npart),
                 (m.partner_kind == "creator"), (m.partner_kind == "guest"), (m.partner_kind == "chat")].astype(float)

def main(tags, boot=1000):
    D = VP.load("p3"); meta = D["meta"]
    z = np.load("data/proc/p3.npz"); Graw = z["G"]; npart = np.where(Graw[:, 11] > 0, np.expm1(Graw[:, 10] * 4.0), 2.0)
    tr = task_rows(meta, "ai_train"); te = task_rows(meta, "ai_test")
    lr = LogisticRegression(max_iter=1000).fit(feats(tr, npart[tr.row]), tr.y)
    sc = {"prior": np.full(len(te), tr.y.mean()) + 1e-9 * np.random.default_rng(0).random(len(te)), "lr_turn_order": lr.predict_proba(feats(te, npart[te.row]))[:, 1]}
    ai_mask = (meta.source == "ai_stream").to_numpy()
    for t in tags:
        if not os.path.exists(f"models/{t}.pt"): continue
        lg, _ = VP.predict(t, D, te.row.to_numpy(), conv_mask=ai_mask)
        a = lg["y_act"]; pspeak = np.exp(a[:, 0]) / np.exp(a).sum(1); peot = 1 / (1 + np.exp(-lg["y_eot"][:, 0]))
        sc[t] = np.where(te.role == "self", 1 - peot, pspeak)
    out = dict(n_snippets_test=int(te.conv.nunique()), n_snippets_train=int(tr.conv.nunique()), tasks={})
    for task, m in (("respond", te.role == "other"), ("continue", te.role == "self"), ("next_is_ai", te.role.notna())):
        m = m.to_numpy(); y = te.y.to_numpy()[m]; bl = te.conv.to_numpy()[m]
        R = dict(n=int(m.sum()), pos_rate=float(y.mean()))
        for n, s in sc.items():
            R[n] = dict(pr_auc=float(average_precision_score(y, s[m])) if 0 < y.sum() < len(y) else None,
                        roc_auc=float(roc_auc_score(y, s[m])) if 0 < y.sum() < len(y) else None)
        for n in sc:
            if n in ("prior", "lr_turn_order"): continue
            for ref in ("lr_turn_order", "prior"):
                d = M.boot(lambda idx: (roc_auc_score(y[idx], sc[n][m][idx]) - roc_auc_score(y[idx], sc[ref][m][idx])) if 0 < y[idx].sum() < len(idx) else np.nan, bl, boot, 0)
                R[n][f"d_roc_auc_vs_{ref}"] = dict(mean=float(np.nanmean(d)), ci=M.ci(d.ravel()))
        out["tasks"][task] = R
    json.dump(out, open("reports/turnorder_phase3.json", "w"), indent=1); print(json.dumps(out, indent=1))

if __name__ == "__main__": main(sys.argv[1:] or ["P2a_s1", "P3mix_s0", "P3ai_s0"])
