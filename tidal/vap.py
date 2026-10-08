"""Phase 2: VAP-style future-event projection as the main self-supervised objective, then multi-task fine-tuning.

Stage 1 (pretrain): causal GRU over scenario-general event inputs (tidal/features_g.py), trained ONLY on the projection
          targets (tidal/vap_targets.py) -- uses every event, with or without recovered text.
Stage 2 (fine-tune): initialise from stage 1; train the 6 downstream heads + the projection head as an auxiliary loss.
          addressed-to-bot / speak-wait-silent heads get synthetic weight 0 (phase 1 showed synthetic data hurts them).
usage: python -m tidal.vap TAG --data real,llm[,live,1on1] [--no-pretrain] [--seed S] [--val real|syn]"""
import argparse, json, os, time, numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as Fn
from tidal.model import TurnModel, n_params, HEAD_DIMS
from tidal.dataset import HEADS, HOLDOUT_GROUPS
from tidal.seqdata import CTX, train_windows, eval_windows
from tidal import features_g as FG, vap_targets as VT
torch.set_num_threads(int(os.environ.get("TIDAL_THREADS", 1)))

SRC = {"real": ("real", ["train"]), "llm": ("synth", ["synth"]), "live": ("syn_live", ["syn_train"]), "1on1": ("syn_1on1", ["syn_train"]),
       # phase 3 (dataset p3 only): public streams + optional private AI-streamer turn-order set
       "tg": ("pub_tg", ["pub_train"]), "irc": ("pub_irc", ["pub_train"]), "twitch": ("pub_twitch", ["pub_train"]),
       "candor": ("pub_candor", ["pub_train"]), "aishell4": ("pub_aishell4", ["pub_train"]), "danmaku": ("pub_danmaku", ["pub_train"]), "ai": ("ai_stream", ["ai_train"])}
SYNTH_SOURCES = {"synth", "syn_live", "syn_1on1"}
NO_SYNTH_HEADS = {"y_addr", "y_act"}

class VAPModel(TurnModel):
    def __init__(self, n_feat, **kw):
        super().__init__(n_feat, False, **kw)
        out = self.body.hidden_size if self.kind == "gru" else self.fproj.out_features
        self.vap = nn.Sequential(nn.Linear(out, 64), nn.GELU(), nn.Linear(64, VT.NV))
    def forward(self, feats, valid, text=None):
        h = self.drop(self.inorm(self.fproj(feats))) * valid.unsqueeze(-1).to(feats.dtype)
        h, _ = self.body(h)
        o = {k: m(h) for k, m in self.heads.items()}; o["vap"] = self.vap(h); return o
    def step(self, x, state=None):
        """Streaming: one event [B, F] -> (outputs at this event, new GRU state). O(1) per event."""
        h = self.inorm(self.fproj(x)).unsqueeze(1)
        h, state = self.body(h, state); h = h[:, 0]
        o = {k: m(h) for k, m in self.heads.items()}; o["vap"] = self.vap(h); return o, state

def load(dataset=None):
    dataset = dataset or os.environ.get("TIDAL_DATASET", "p2")
    z = np.load(f"data/proc/{dataset}.npz"); meta = pd.read_parquet(f"data/proc/{dataset}_meta.parquet")
    tr = ((meta.source == "real") & (meta.split == "train")).to_numpy()
    mu = z["G"][tr, :FG.NB].mean(0); sd = z["G"][tr, :FG.NB].std(0) + 1e-6
    W = z["W"] if "W" in z.files else np.ones(len(meta), np.float32)
    return dict(X=FG.normalize(z["G"], mu, sd), Y=z["Y"], V=z["V"], meta=meta, mu=mu, sd=sd, W=W)

def conv_rows(meta, mask):
    idx = np.flatnonzero(mask); conv = meta.conv.to_numpy()[idx]
    st = np.r_[0, np.flatnonzero(conv[1:] != conv[:-1]) + 1, len(idx)]
    return [idx[a:b] for a, b in zip(st[:-1], st[1:])]

def batch(D, wins, loss_from=None, rng=None, Ymask=None, Vmask=None):
    B = len(wins); T = max(len(w) for w in wins); nf = D["X"].shape[1]
    F = np.zeros((B, T, nf), np.float32); Y = np.full((B, T, len(HEADS)), np.nan, np.float32)
    V = np.full((B, T, VT.NV), np.nan, np.float32); valid = np.zeros((B, T), bool)
    for i, w in enumerate(wins):
        o = T - len(w); F[i, o:] = D["X"][w]; valid[i, o:] = True
        Y[i, o:] = (Ymask if Ymask is not None else D["Y"])[w]; V[i, o:] = (Vmask if Vmask is not None else D["V"])[w]
        if loss_from is not None: Y[i, :o + loss_from[i]] = np.nan; V[i, :o + loss_from[i]] = np.nan
    if rng is not None: F = FG.drop_optional(F, rng)
    return F, Y, V, valid

def vap_loss(o, V):
    m = ~torch.isnan(V)
    if m.sum() == 0: return torch.zeros(())
    return Fn.binary_cross_entropy_with_logits(o[m], V[m])

def head_loss(out, Yb, wsyn):
    tot = 0.0; parts = {}
    for hi, h in enumerate(HEADS):
        y = Yb[..., hi]; m = ~torch.isnan(y)
        if m.sum() == 0: continue
        o = out[h][m]
        l = Fn.binary_cross_entropy_with_logits(o.squeeze(-1), y[m], reduction="none") if HEAD_DIMS[h] == 1 else Fn.cross_entropy(o, y[m].long(), reduction="none")
        w = wsyn[h][m]; l = (l * w).sum() / w.sum().clamp_min(1e-6)
        parts[h] = float(l.detach()); tot = tot + l
    return tot, parts

def evaluate_loss(model, D, wins, rows, Ymask=None):
    model.eval(); tot = {}; cnt = {}; vl = []; vn = []
    with torch.no_grad():
        for b in range(0, len(wins), 256):
            F, Y, V, valid = batch(D, wins[b:b + 256], Ymask=Ymask)
            o = model(torch.from_numpy(F), torch.from_numpy(valid))
            last = {k: v[:, -1:] for k, v in o.items()}
            _, parts = head_loss(last, torch.from_numpy(Y[:, -1:]), {h: torch.ones(Y.shape[0], 1) for h in HEADS})
            n = (~np.isnan(Y[:, -1])).sum(0)
            for hi, h in enumerate(HEADS):
                if h in parts: tot[h] = tot.get(h, 0) + parts[h] * n[hi]; cnt[h] = cnt.get(h, 0) + n[hi]
            Vt = torch.from_numpy(V[:, -1]); m = ~torch.isnan(Vt)
            if m.sum(): vl.append(float(Fn.binary_cross_entropy_with_logits(last["vap"][:, 0][m], Vt[m], reduction="sum"))); vn.append(int(m.sum()))
    parts = {h: tot[h] / cnt[h] for h in tot}
    return float(sum(parts.values())), parts, (sum(vl) / max(1, sum(vn)))

def run(tag, data, pretrain=True, seed=0, val="real", synth_w=0.5, aux=0.5, epochs=60, patience=8, bs=32, log=print,
        dataset=None, init=None, pub_w=0.5, pre_epochs=None):
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    D = load(dataset); meta = D["meta"]; src = meta.source.to_numpy(); split = meta.split.to_numpy()
    train_mask = np.zeros(len(meta), bool); is_syn = np.isin(src, list(SYNTH_SOURCES))
    for k in data:
        s, sp = SRC[k]; train_mask |= (src == s) & np.isin(split, sp)
    if "real" in data:   # whole real conversations are windowed (context), but loss only on train rows; holdout groups excluded
        ctx = (src == "real") & ~meta.conv.isin(HOLDOUT_GROUPS).to_numpy()
    else: ctx = np.zeros(len(meta), bool)
    convs = conv_rows(meta, ctx | train_mask)
    W = train_windows(convs)
    Ytr = D["Y"].copy(); Ytr[~train_mask] = np.nan; Vtr = D["V"].copy(); Vtr[~train_mask] = np.nan
    is_pub = np.char.startswith(src.astype(str), "pub_")
    wsyn = np.where(is_syn, synth_w, np.where(is_pub, pub_w, 1.0)).astype(np.float32) * D["W"]   # W: per-row reliability
    syn_row = is_syn
    if val == "real":
        vrows = np.flatnonzero((src == "real") & (split == "val") & ~np.all(np.isnan(D["Y"]), 1))
        vconv = conv_rows(meta, src == "real")
    elif val == "pub":
        vm = np.isin(src, [SRC[k][0] for k in data]) & (split == "pub_val")
        vrows = np.flatnonzero(vm & ~np.all(np.isnan(D["Y"]), 1)); vrows = vrows[np.random.default_rng(0).permutation(len(vrows))[:30000]]
        vrows.sort(); vconv = conv_rows(meta, vm)
    else:
        vm = np.isin(src, [SRC[k][0] for k in data]) & (split == "syn_val")
        vrows = np.flatnonzero(vm & ~np.all(np.isnan(D["Y"]), 1)); vconv = conv_rows(meta, vm)
    VW = eval_windows(vconv, vrows)
    model = VAPModel(D["X"].shape[1]); t_start = time.time()
    if init:                                          # phase 3: start from a model pretrained on public data
        model.load_state_dict(torch.load(f"models/{init}.pt", weights_only=False)["state"]); pretrain = False
    log(f"[{tag}] params={n_params(model)} data={data} pretrain={pretrain} windows={len(W)} val_rows={len(vrows)}")
    def fit(stage, lr):
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2); best = (1e9, None, -1); hist = []
        for ep in range(pre_epochs if (stage == "pre" and pre_epochs) else epochs):
            model.train(); perm = rng.permutation(len(W)); t0 = time.time(); tl = []
            for b in range(0, len(W), bs):
                ch = [W[k] for k in perm[b:b + bs]]
                F, Y, V, valid = batch(D, [c[0] for c in ch], [c[1] for c in ch], rng, Ymask=Ytr, Vmask=Vtr)
                o = model(torch.from_numpy(F), torch.from_numpy(valid))
                lv = vap_loss(o["vap"], torch.from_numpy(V))
                if stage == "pre": loss = lv
                else:
                    ws = np.zeros(F.shape[:2], np.float32)
                    for i, c in enumerate(ch): ws[i, F.shape[1] - len(c[0]):] = wsyn[c[0]]
                    sy = np.zeros(F.shape[:2], bool)
                    for i, c in enumerate(ch): sy[i, F.shape[1] - len(c[0]):] = syn_row[c[0]]
                    wt = {h: torch.from_numpy(np.where(sy, 0.0, ws).astype(np.float32) if h in NO_SYNTH_HEADS else ws) for h in HEADS}
                    lh, _ = head_loss(o, torch.from_numpy(Y), wt); loss = lh + aux * lv
                if not torch.is_tensor(loss) or not loss.requires_grad: continue
                opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl.append(float(loss))
            hl, parts, vv = evaluate_loss(model, D, VW, vrows)
            crit = vv if stage == "pre" else hl
            hist.append(dict(stage=stage, ep=ep, train=float(np.mean(tl)) if tl else None, val_heads=hl, val_vap=vv, sec=round(time.time() - t0, 1), **{f"val_{k}": v for k, v in parts.items()}))
            log(json.dumps(hist[-1]))
            if crit < best[0] - 1e-4: best = (crit, {k: v.clone() for k, v in model.state_dict().items()}, ep)
            elif ep - best[2] >= patience: break
        model.load_state_dict(best[1]); return best, hist
    hist = []; pre_state = None
    if pretrain:
        b, h = fit("pre", 1e-3); hist += h; pre_state = b[1]; pre_seconds = round(time.time() - t_start, 1)
    b, h = fit("ft", 5e-4 if (pretrain or init) else 1e-3); hist += h
    os.makedirs("models", exist_ok=True)
    torch.save(dict(state=b[1], pre_state=pre_state, pre_seconds=pre_seconds if pretrain else 0, arch="vap", kind="gru", n_feat=D["X"].shape[1], feats=FG.FEAT_G, mu=D["mu"], sd=D["sd"], data=data,
                    pretrain=pretrain, init=init, dataset=dataset or os.environ.get("TIDAL_DATASET", "p2"), pub_w=pub_w, seed=seed, best_val=b[0], best_ep=b[2], hist=hist, params=n_params(model),
                    train_seconds=round(time.time() - t_start, 1), threads=torch.get_num_threads()), f"models/{tag}.pt")
    log(f"[{tag}] best_val={b[0]:.4f} ep={b[2]} train_seconds={time.time() - t_start:.0f}")

def load_model(tag, stage="ft"):
    ck = torch.load(f"models/{tag}.pt", weights_only=False)
    m = VAPModel(ck["n_feat"]); m.load_state_dict(ck["state"] if stage == "ft" else ck["pre_state"]); m.eval(); return m, ck

def predict(tag, D, rows, conv_mask=None, stage="ft"):
    """logits at each target row (causal window of <=64 events of its conversation)."""
    m, ck = load_model(tag, stage); meta = D["meta"]
    convs = conv_rows(meta, conv_mask if conv_mask is not None else np.ones(len(meta), bool))
    W = eval_windows(convs, rows); out = {k: [] for k in HEADS + ["vap"]}
    with torch.no_grad():
        for b in range(0, len(W), 512):
            F, _, _, valid = batch(D, W[b:b + 512])
            o = m(torch.from_numpy(F), torch.from_numpy(valid))
            for k in out: out[k].append(o[k][:, -1].numpy())
    return {k: np.concatenate(v) for k, v in out.items()}, ck

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("tag"); ap.add_argument("--data", default="real,llm")
    ap.add_argument("--no-pretrain", action="store_true"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--val", default="real")
    ap.add_argument("--dataset"); ap.add_argument("--init"); ap.add_argument("--pub-w", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--pre-epochs", type=int); ap.add_argument("--patience", type=int, default=8)
    a = ap.parse_args()
    run(a.tag, a.data.split(","), not a.no_pretrain, a.seed, a.val, epochs=a.epochs, patience=a.patience, dataset=a.dataset,
        init=a.init, pub_w=a.pub_w, pre_epochs=a.pre_epochs)
