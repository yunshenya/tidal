"""Train the temporal model. usage: python -m tidal.train REGIME(T|TS) DATA(real|real+synth) KIND(gru|tf) [synth_w] [tag]"""
import sys, json, time, os, numpy as np, torch, torch.nn.functional as Fn
from tidal.seqdata import load, conv_index, train_windows, eval_windows, batchify
from tidal.model import TurnModel, n_params, HEAD_DIMS
from tidal.dataset import HEADS, HOLDOUT_GROUPS
torch.set_num_threads(int(os.environ.get("TIDAL_THREADS", 4)))
def masked_loss(out, Yb, w_pos=None):
    tot = 0.0; parts = {}
    for hi, h in enumerate(HEADS):
        y = Yb[..., hi]; m = ~torch.isnan(y)
        if m.sum() == 0: continue
        o = out[h][m]
        if HEAD_DIMS[h] == 1: l = Fn.binary_cross_entropy_with_logits(o.squeeze(-1), y[m], reduction="none")
        else: l = Fn.cross_entropy(o, y[m].long(), reduction="none")
        wt = w_pos[m] if w_pos is not None else torch.ones_like(l)
        l = (l * wt).sum() / wt.sum().clamp_min(1e-6)
        parts[h] = float(l.detach()); tot = tot + l
    return tot, parts
def run(regime, data, kind, synth_w=0.5, tag=None, epochs=60, patience=8, seed=0, bs=32):
    torch.manual_seed(seed); np.random.seed(seed)
    tag = tag or f"{regime}_{data}_{kind}"
    D = load(regime); use_text = regime == "TS"; meta = D["meta"]
    src = meta.source.to_numpy(); split = meta.split.to_numpy()
    convs = conv_index(D)
    real_train = [c for c in convs if src[c[0]] == "real" and meta.conv.iat[c[0]] not in HOLDOUT_GROUPS]
    syn = [c for c in convs if src[c[0]] == "synth"] if data == "real+synth" else []
    W = [(w, lf, 1.0) for w, lf in train_windows(real_train)] + [(w, lf, synth_w) for w, lf in train_windows(syn)]
    # loss only on real-train positions (+synth): mask val/test/embargo labels inside real windows
    trainable = (split == "train") | (split == "synth")
    val_rows = np.flatnonzero(D["keep"] & (src == "real") & (split == "val") & ~np.all(np.isnan(D["Y"]), 1))
    VW = eval_windows(convs, val_rows)
    model = TurnModel(D["X"].shape[1], use_text, kind=kind)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    print(f"[{tag}] params={n_params(model)} train_windows={len(W)} (real {len(real_train)} convs, synth {len(syn)} convs) val_targets={len(val_rows)}", flush=True)
    Yall = D["Y"].copy(); Yall[~trainable] = np.nan; Dtr = dict(D, Y=Yall)
    best = (1e9, None, -1); hist = []
    for ep in range(epochs):
        model.train(); perm = np.random.permutation(len(W)); t0 = time.time(); tl = []
        for b in range(0, len(W), bs):
            ch = [W[k] for k in perm[b:b + bs]]
            F, E, Yb, valid = batchify(Dtr, [c[0] for c in ch], use_text, [c[1] for c in ch])
            wpos = torch.tensor(np.repeat(np.array([c[2] for c in ch], np.float32)[:, None], F.shape[1], 1))
            out = model(torch.from_numpy(F), torch.from_numpy(valid), torch.from_numpy(E) if use_text else None)
            loss, _ = masked_loss(out, torch.from_numpy(Yb), wpos)
            if not torch.is_tensor(loss): continue
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tl.append(float(loss))
        vl, parts = evaluate_loss(model, D, VW, val_rows, use_text)
        hist.append(dict(ep=ep, train=float(np.mean(tl)), val=vl, **{f"val_{k}": v for k, v in parts.items()}, sec=round(time.time() - t0, 1)))
        print(json.dumps(hist[-1]), flush=True)
        if vl < best[0] - 1e-4: best = (vl, {k: v.clone() for k, v in model.state_dict().items()}, ep)
        elif ep - best[2] >= patience: break
    model.load_state_dict(best[1]); os.makedirs("models", exist_ok=True)
    torch.save(dict(state=best[1], regime=regime, kind=kind, n_feat=D["X"].shape[1], use_text=use_text, mu=D["mu"], sd=D["sd"],
                    best_ep=best[2], hist=hist, data=data, synth_w=synth_w, params=n_params(model)), f"models/{tag}.pt")
    print(f"[{tag}] best_ep={best[2]} val={best[0]:.4f}", flush=True)
def evaluate_loss(model, D, VW, rows, use_text, bs=256):
    model.eval(); tot = {}; cnt = {}
    with torch.no_grad():
        for b in range(0, len(VW), bs):
            F, E, Yb, valid = batchify(D, VW[b:b + bs], use_text)
            out = model(torch.from_numpy(F), torch.from_numpy(valid), torch.from_numpy(E) if use_text else None)
            last = {k: v[:, -1:] for k, v in out.items()}
            _, parts = masked_loss(last, torch.from_numpy(Yb[:, -1:]))
            n = (~np.isnan(Yb[:, -1])).sum(0)
            for hi, h in enumerate(HEADS):
                if h in parts: tot[h] = tot.get(h, 0) + parts[h] * n[hi]; cnt[h] = cnt.get(h, 0) + n[hi]
    parts = {h: tot[h] / cnt[h] for h in tot}
    return float(sum(parts.values())), parts
if __name__ == "__main__":
    a = sys.argv[1:]
    run(a[0], a[1], a[2], float(a[3]) if len(a) > 3 else 0.5, a[4] if len(a) > 4 else None, seed=int(a[5]) if len(a) > 5 else 0)
