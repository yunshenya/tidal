"""Joint-train y_addr, y_interrupt, y_topic on cached frozen body states.

Default cache is data/proc/p5_h.npy (frozen P4emo_mamba3_siso_s0, the pre-registered run;
heads models/P5hd_s*.pt). The shadow winner's cache is p5_h_m2.npy (frozen
P4emo_m3_ablate_m2_s0; heads models/P5hd_m2_s*.pt), same rows, protocol and seeds.
Adoption rules: reports/phase5_prereg.md. Test rows are not read until the epoch is chosen.
usage: python -m tidal.phase5_train [all|seed S|probe] [FEATS TAG]
"""
import json, os
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

PROC = "data/proc"
HEADS = ("y_addr", "y_interrupt", "y_topic")


from tidal.model import BinHead   # same module/keys as before; lives in model.py so shadow code need not import sklearn


def _bce_np(logits, y):
    return float(torch.nn.functional.binary_cross_entropy_with_logits(
        torch.as_tensor(logits, dtype=torch.float32), torch.as_tensor(y, dtype=torch.float32)))


def _split_masks(df):
    sp = df.split.to_numpy()
    real = df.source.to_numpy() == "real"
    pub_addr = (~real) & (df.livestream.to_numpy() == 0) & df.y_addr.notna().to_numpy()
    return dict(
        addr_real_tr=real & (sp == "train") & df.y_addr.notna().to_numpy(),
        addr_real_va=real & (sp == "val") & df.y_addr.notna().to_numpy(),
        addr_real_te=real & np.isin(sp, ["test_time", "test_group"]) & df.y_addr.notna().to_numpy(),
        addr_pub_tr=pub_addr & (sp == "pub_train"),
        addr_pub_va=pub_addr & (sp == "pub_val"),
        addr_pub_te=pub_addr & (sp == "pub_test"),
        int_tr=(sp == "pub_train") & df.y_interrupt.notna().to_numpy(),
        int_va=(sp == "pub_val") & df.y_interrupt.notna().to_numpy(),
        int_te=(sp == "pub_test") & df.y_interrupt.notna().to_numpy(),
        top_tr=(sp == "pub_train") & df.y_topic.notna().to_numpy(),
        top_va=(sp == "pub_val") & df.y_topic.notna().to_numpy(),
        top_te=(sp == "pub_test") & df.y_topic.notna().to_numpy(),
    )


def _take(h, y, mask, rng, n, replace=False):
    ix = np.flatnonzero(mask)
    if len(ix) == 0:
        return None
    ch = rng.choice(ix, n, replace=replace or len(ix) < n)
    return h[ch], y[ch]


def train_seed(seed, epochs=40, patience=8, bs=64, lr=5e-4, feats="p5_h", tag="P5hd"):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    df = pd.read_parquet(PROC + "/p5_rows.parquet")
    H = np.load(f"{PROC}/{feats}.npy", mmap_mode="r")
    Y = {k: df[k].to_numpy(np.float32) for k in HEADS}
    M = _split_masks(df)
    # training rows per head (addr = real train + public message-value train)
    # y_addr is the mean of the real addressed-to-bot BCE and the public message-value BCE,
    # so the larger public set cannot drown the deployment labels (same idea as the head-level mean).
    pools = {
        "y_interrupt": M["int_tr"],
        "y_topic": M["top_tr"],
    }
    addr_parts = [M["addr_real_tr"], M["addr_pub_tr"]]
    for k, m in pools.items():
        print(f"seed {seed} train {k} {int(m.sum())}", flush=True)
    device = "cpu"
    heads = {k: BinHead().to(device) for k in HEADS}
    opt = torch.optim.AdamW([p for net in heads.values() for p in net.parameters()], lr=lr, weight_decay=1e-2)
    # val index sets (fixed)
    val_sets = {
        "addr_real": M["addr_real_va"],
        "addr_pub": M["addr_pub_va"],
        "interrupt": M["int_va"],
        "topic": M["top_va"],
    }
    best = (1e9, None, -1)
    hist = []
    biggest = max([int(m.sum()) for m in pools.values()] + [int(m.sum()) for m in addr_parts])
    steps = max(1, biggest // bs)
    for ep in range(epochs):
        for net in heads.values():
            net.train()
        losses = []
        for _ in range(steps):
            loss = 0.0
            nterm = 0
            opt.zero_grad()
            for k, mask in pools.items():
                got = _take(H, Y[k], mask, rng, bs, replace=True)
                if got is None:
                    continue
                hb, yb = got
                logit = heads[k](torch.from_numpy(np.asarray(hb, np.float32)))
                loss = loss + torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.from_numpy(yb))
                nterm += 1
            apart = []
            for mask in addr_parts:
                got = _take(H, Y["y_addr"], mask, rng, bs, replace=True)
                if got is None:
                    continue
                hb, yb = got
                logit = heads["y_addr"](torch.from_numpy(np.asarray(hb, np.float32)))
                apart.append(torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.from_numpy(yb)))
            if apart:
                loss = loss + sum(apart) / len(apart)
                nterm += 1
            if nterm == 0:
                continue
            loss = loss / nterm
            loss.backward()
            nn.utils.clip_grad_norm_([p for net in heads.values() for p in net.parameters()], 1.0)
            opt.step()
            losses.append(float(loss.detach()))
        metrics = eval_heads(heads, H, Y, val_sets)
        crit = float(np.mean(list(metrics.values())))
        rec = dict(ep=ep, train=float(np.mean(losses)) if losses else None, crit=crit, **metrics)
        hist.append(rec)
        print(json.dumps(rec), flush=True)
        if crit < best[0] - 1e-4:
            best = (crit, {k: {kk: vv.detach().cpu().clone() for kk, vv in net.state_dict().items()} for k, net in heads.items()}, ep)
        elif ep - best[2] >= patience:
            break
    for k, net in heads.items():
        net.load_state_dict(best[1][k])
    test_sets = {
        "addr_real": M["addr_real_te"],
        "addr_pub": M["addr_pub_te"],
        "interrupt": M["int_te"],
        "topic": M["top_te"],
    }
    test = eval_heads(heads, H, Y, test_sets)
    # per-split real test, still report-only
    detail = {}
    for name, mask in (("addr_real_test_time", (df.source.to_numpy()=="real") & (df.split.to_numpy()=="test_time") & df.y_addr.notna().to_numpy()),
                       ("addr_real_test_group", (df.source.to_numpy()=="real") & (df.split.to_numpy()=="test_group") & df.y_addr.notna().to_numpy())):
        detail[name] = _one(heads["y_addr"], H, Y["y_addr"], mask)
    ck = dict(seed=seed, best_ep=best[2], best_crit=best[0], hist=hist, test=test, test_detail=detail, feats=feats,
              val=hist[best[2]], state={k: v for k, v in best[1].items()})
    torch.save(ck, f"models/{tag}_s{seed}.pt")
    print(f"seed {seed} best_ep {best[2]} crit {best[0]:.4f} test {json.dumps(test)}", flush=True)
    return ck


@torch.no_grad()
def _one(net, H, y, mask):
    net.eval()
    ix = np.flatnonzero(mask)
    if len(ix) == 0:
        return None
    logits = []
    for s in range(0, len(ix), 4096):
        sl = ix[s:s + 4096]
        logits.append(net(torch.from_numpy(np.asarray(H[sl], np.float32))).numpy())
    lg = np.concatenate(logits)
    return dict(n=int(len(ix)), bce=_bce_np(lg, y[ix]), pos=int(y[ix].sum()))


@torch.no_grad()
def eval_heads(heads, H, Y, sets):
    spec = {"addr_real": "y_addr", "addr_pub": "y_addr", "interrupt": "y_interrupt", "topic": "y_topic"}
    out = {}
    for name, mask in sets.items():
        r = _one(heads[spec[name]], H, Y[spec[name]], mask)
        if r:
            out[name] = r["bce"]
    return out


def probe_bce(H, y, tr, va):
    xtr, ytr = np.asarray(H[tr], np.float32), y[tr]
    xva, yva = np.asarray(H[va], np.float32), y[va]
    if len(np.unique(ytr)) < 2 or len(yva) == 0:
        return None
    clf = LogisticRegression(max_iter=400, C=1.0)
    clf.fit(xtr, ytr)
    p = np.clip(clf.predict_proba(xva)[:, 1], 1e-6, 1 - 1e-6)
    # match BCE-with-logits of a probability via log_loss (natural log)
    return float(log_loss(yva, p, labels=[0, 1]))


def prior_bce(y, tr, va):
    p = float(np.clip(y[tr].mean(), 1e-6, 1 - 1e-6))
    yv = y[va]
    return float(-(yv * np.log(p) + (1 - yv) * np.log(1 - p)).mean())


def interrupt_probe(feats="p5_h"):
    """Pre-registered bar for y_interrupt on a given cache: linear probe + train prior, public val only."""
    df = pd.read_parquet(PROC + "/p5_rows.parquet", columns=["source", "split", "y_addr", "y_interrupt", "y_topic", "livestream"])
    H = np.load(f"{PROC}/{feats}.npy", mmap_mode="r")
    M = _split_masks(df)
    y_i = df.y_interrupt.to_numpy(np.float32)
    out = dict(feats=feats, interrupt_probe_val_bce=probe_bce(H, y_i, M["int_tr"], M["int_va"]),
               interrupt_prior_val_bce=prior_bce(y_i, M["int_tr"], M["int_va"]),
               n_train=int(M["int_tr"].sum()), n_val=int(M["int_va"].sum()))
    print(json.dumps(out), flush=True)
    return out


def baselines():
    """mamba3_siso + text-emotion without the new head / without the new y_addr target."""
    df = pd.read_parquet(PROC + "/p5_rows.parquet", columns=["source", "split", "y_addr", "y_interrupt", "y_topic", "livestream", "p3_index"])
    H = np.load(PROC + "/p5_h.npy", mmap_mode="r")
    M = _split_masks(df)
    out = {}
    y_i = df.y_interrupt.to_numpy(np.float32)
    y_t = df.y_topic.to_numpy(np.float32)
    out["interrupt_probe_val_bce"] = probe_bce(H, y_i, M["int_tr"], M["int_va"])
    out["interrupt_prior_val_bce"] = prior_bce(y_i, M["int_tr"], M["int_va"])
    out["topic_probe_val_bce"] = probe_bce(H, y_t, M["top_tr"], M["top_va"])
    out["topic_prior_val_bce"] = prior_bce(y_t, M["top_tr"], M["top_va"])
    # original y_addr head on the three phase-4 checkpoints, private real val only
    from tidal import vap
    D = vap.load("p3", "emo")
    meta = D["meta"]
    vrows = np.flatnonzero((meta.source.to_numpy() == "real") & (meta.split.to_numpy() == "val"))
    y = D["Y"][vrows, 2]
    m = ~np.isnan(y)
    vrows, y = vrows[m], y[m]
    bces = []
    for s in range(3):
        logits, _ck = vap.predict(f"P4emo_mamba3_siso_s{s}", D, vrows, conv_mask=(meta.source.to_numpy() == "real"))
        bces.append(_bce_np(logits["y_addr"].reshape(-1), y))
    out["addr_baseline_real_val_bce_seeds"] = bces
    out["addr_baseline_real_val_bce_mean"] = float(np.mean(bces))
    out["addr_baseline_real_val_n"] = int(len(y))
    json.dump(out, open("reports/phase5_baselines.json", "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)
    return out


def livestream_scores():
    df = pd.read_parquet(PROC + "/p5_rows.parquet", columns=["source", "livestream"])
    H = np.load(PROC + "/p5_h.npy", mmap_mode="r")
    live = df.livestream.to_numpy() == 1
    out = {}
    for s in range(3):
        ck = torch.load(f"models/P5hd_s{s}.pt", map_location="cpu", weights_only=False)
        net = BinHead(); net.load_state_dict(ck["state"]["y_addr"]); net.eval()
        for src in ("pub_twitch", "pub_artemis", "pub_danmaku"):
            ix = np.flatnonzero(live & (df.source.to_numpy() == src))
            if len(ix) == 0:
                continue
            lg = []
            with torch.no_grad():
                for a in range(0, len(ix), 4096):
                    sl = ix[a:a + 4096]
                    lg.append(net(torch.from_numpy(np.asarray(H[sl], np.float32))).numpy())
            p = 1 / (1 + np.exp(-np.concatenate(lg)))
            rec = out.setdefault(src, [])
            rec.append(dict(n=int(len(p)), mean=float(p.mean()), p50=float(np.percentile(p, 50)),
                            p95=float(np.percentile(p, 95)), frac_gt_05=float((p > 0.5).mean())))
    json.dump(out, open("reports/phase5_livestream_scores.json", "w"), indent=1)
    print(json.dumps(out, indent=1), flush=True)


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "all"
    if cmd == "baselines":
        baselines()
    elif cmd == "live":
        livestream_scores()
    elif cmd == "probe":
        interrupt_probe(*(sys.argv[2:3] or ["p5_h"]))
    elif cmd == "seed":
        train_seed(int(sys.argv[2]), **(dict(feats=sys.argv[3], tag=sys.argv[4]) if len(sys.argv) > 4 else {}))
    else:
        kw = dict(feats=sys.argv[2], tag=sys.argv[3]) if len(sys.argv) > 3 else {}
        for s in (0, 1, 2):
            if os.path.exists(f"models/{kw.get('tag', 'P5hd')}_s{s}.pt"):
                print("skip", s, flush=True)
                continue
            train_seed(s, **kw)
