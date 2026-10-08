"""Forward barge-in rows on frozen m3_ablate_m2 states. Rules: reports/phase5_barge_prereg.md.

Builds data/proc/p5_barge_rows.parquet and data/proc/p5_barge_x.npy (132-d). Trains only the head.
usage: python -m tidal.phase5_barge build | probe | seed S
"""
import glob, json, os
import numpy as np, pandas as pd, torch
from tidal.phase5_labels import (
    BARGE_H_SPEECH, BARGE_SELF_CAP, barge_self_speaker, forward_barge_speech, forward_barge_text,
    nonspeech_span,
)
from tidal.phase5_data import DATA

PROC = "data/proc"


def _candor():
    rng = np.random.default_rng(5)
    c = pd.read_parquet(DATA / "candor_tt" / "data" / "train-00000-of-00001.parquet",
                        columns=["conversation_id", "channel", "offset", "duration"])
    convs = sorted(c.conversation_id.unique())
    convs = list(rng.choice(convs, min(700, len(convs)), replace=False))
    sub = c[c.conversation_id.isin(set(convs))]
    for i, (_cid, g) in enumerate(sub.groupby("conversation_id")):
        g = g.assign(start=g.offset.astype(float), end=(g.offset + g.duration).astype(float)).sort_values("end", kind="stable")
        segs = list(zip(g.start.to_numpy(), g.end.to_numpy(), g.channel.astype(str).to_numpy()))
        yield f"cd:{i}", segs


def _aishell():
    i_kept = 0
    for p in sorted(glob.glob(str(DATA / "aishell4_seg" / "**" / "*.rttm"), recursive=True)):
        rows = [l.split() for l in open(p) if l.startswith("SPEAKER")]
        if len(rows) < 30:
            continue
        segs = [(float(r[3]), float(r[3]) + float(r[4]), r[7]) for r in rows]
        order = sorted(range(len(segs)), key=lambda k: (segs[k][1], segs[k][0], k))
        yield f"a4:{i_kept}", [segs[k] for k in order]
        i_kept += 1


def _magic_oto():
    convs = sorted({os.path.basename(p).rsplit("_0_", 1)[0] for p in glob.glob(str(DATA / "magicdata_ms" / "TXT" / "*.txt"))})
    pat = __import__("re").compile(r"\[([0-9.]+),([0-9.]+)\]\s+(\S+)\s+\S+\s*(.*)$")
    for conv in convs:
        segs = []
        for pth in sorted(glob.glob(str(DATA / "magicdata_ms" / "TXT" / f"{conv}_0_*.txt"))):
            for line in open(pth, encoding="utf-8"):
                m = pat.match(line.strip().lstrip("\ufeff"))
                if not m or nonspeech_span(m.group(3), m.group(4)):
                    continue
                segs.append((float(m.group(1)), float(m.group(2)), m.group(3)))
        if len(segs) >= 4:
            yield f"md:{conv}", segs
    for p in sorted(glob.glob(str(DATA / "proc" / "audio_oto" / "*.npz"))):
        sid = os.path.basename(p)[:-4]
        z = np.load(p, allow_pickle=False)
        segs = [(float(a), float(b), f"c{ch}") for ch, key in ((0, "s0"), (1, "s1")) for a, b, *_r in json.loads(str(z[key]))]
        if len(segs) >= 4:
            yield f"oto:{sid}", segs


def _timing(t, floor_start, last_event_ts, last_self_ts, point):
    since_self = BARGE_SELF_CAP if last_self_ts is None else min(BARGE_SELF_CAP, t - last_self_ts)
    return (np.log1p(t - floor_start), np.log1p(max(0.0, t - last_event_ts)), np.log1p(max(0.0, since_self)), float(point))


def build():
    rows = pd.read_parquet(PROC + "/p5_rows.parquet", columns=["source", "conv", "ts", "role", "split"])
    rows = rows.reset_index().rename(columns={"index": "h_index"})
    split_of = rows.groupby("conv", sort=False).split.first().to_dict()
    src_of = rows.groupby("conv", sort=False).source.first().to_dict()
    # per conv: event ts, h_index, role, sorted by ts
    ev = {c: g.sort_values("ts", kind="stable") for c, g in rows.groupby("conv", sort=False)}
    recs = []

    def add_speech(conv, segs):
        if conv not in ev:
            return
        g = ev[conv]
        ts = g.ts.to_numpy()
        idx = g.h_index.to_numpy()
        role = g.role.to_numpy()
        self_sp = barge_self_speaker(segs)
        rec_end = max(e for _s, e, _sp in segs)
        for t, y, floor_start in forward_barge_speech(segs, self_sp, rec_end=rec_end):
            if y != y:
                continue
            k = np.searchsorted(ts, t, side="right") - 1
            if k < 0:
                continue
            last_self = ts[:k + 1][role[:k + 1] == "self"]
            recs.append((conv, src_of[conv], split_of[conv], t, y, *_timing(
                t, floor_start, ts[k], None if len(last_self) == 0 else float(last_self[-1]), 0), int(idx[k])))

    print("speech candor", flush=True)
    for conv, segs in _candor():
        add_speech(conv, segs)
    print("speech aishell", len(recs), flush=True)
    for conv, segs in _aishell():
        add_speech(conv, segs)
    print("speech magic/oto", len(recs), flush=True)
    for conv, segs in _magic_oto():
        add_speech(conv, segs)
    print("text tg", len(recs), flush=True)
    tg = rows[rows.source == "pub_tg"]
    for conv, g in tg.groupby("conv", sort=False):
        g = g.sort_values("ts", kind="stable")
        ts = g.ts.to_numpy(); idx = g.h_index.to_numpy(); role = g.role.to_numpy()
        for i, y in forward_barge_text(ts, role):
            if y != y:
                continue
            last_self = ts[:i + 1][role[:i + 1] == "self"]
            # last event at or before t includes this message
            recs.append((conv, "pub_tg", split_of[conv], float(ts[i]), y, *_timing(
                float(ts[i]), float(ts[i]), float(ts[i]), None if len(last_self) == 0 else float(last_self[-1]), 1), int(idx[i])))
    print("decisions", len(recs), flush=True)
    out = pd.DataFrame(recs, columns=["conv", "source", "split", "t", "y", "f_hold", "f_since_event", "f_since_self", "f_point", "h_index"])
    out.to_parquet(PROC + "/p5_barge_rows.parquet", index=False)
    H = np.load(PROC + "/p5_h_m2.npy", mmap_mode="r")
    X = np.empty((len(out), 132), np.float32)
    h = np.asarray(H[out.h_index.to_numpy()], np.float32)
    X[:, :128] = h
    X[:, 128:] = out[["f_hold", "f_since_event", "f_since_self", "f_point"]].to_numpy(np.float32)
    np.save(PROC + "/p5_barge_x.npy", X)
    y = out.y.to_numpy()
    sp = out.split.to_numpy()
    print(json.dumps({s: dict(n=int((sp == s).sum()), pos=float(y[sp == s].mean()) if (sp == s).any() else None) for s in ("pub_train", "pub_val", "pub_test")}), flush=True)


def _masks(df):
    sp = df.split.to_numpy(); y = df.y.notna().to_numpy()
    return sp == "pub_train", sp == "pub_val", sp == "pub_test"


def probe():
    from tidal.phase5_train import probe_bce, prior_bce
    df = pd.read_parquet(PROC + "/p5_barge_rows.parquet", columns=["split", "y"])
    X = np.load(PROC + "/p5_barge_x.npy", mmap_mode="r")
    tr, va, _te = _masks(df)
    y = df.y.to_numpy(np.float32)
    out = dict(probe_val_bce=probe_bce(X, y, tr, va), prior_val_bce=prior_bce(y, tr, va),
               n_train=int(tr.sum()), n_val=int(va.sum()), n_test=int(_masks(df)[2].sum()),
               pos_train=float(y[tr].mean()), pos_val=float(y[va].mean()))
    print(json.dumps(out), flush=True)
    return out


def train_seed(seed, epochs=40, patience=8, bs=64, lr=5e-4):
    from tidal.model import BinHead
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    df = pd.read_parquet(PROC + "/p5_barge_rows.parquet", columns=["split", "y"])
    X = np.load(PROC + "/p5_barge_x.npy", mmap_mode="r")
    y = df.y.to_numpy(np.float32)
    tr, va, te = _masks(df)
    tr_ix, va_ix, te_ix = np.flatnonzero(tr), np.flatnonzero(va), np.flatnonzero(te)
    net = BinHead(d=132)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
    best = (1e9, None, -1)
    hist = []
    steps = max(1, len(tr_ix) // bs)

    def bce_at(ix):
        net.eval(); lg = []
        with torch.no_grad():
            for a in range(0, len(ix), 8192):
                sl = ix[a:a + 8192]
                lg.append(net(torch.from_numpy(np.asarray(X[sl], np.float32))).numpy())
        return float(torch.nn.functional.binary_cross_entropy_with_logits(torch.from_numpy(np.concatenate(lg)), torch.from_numpy(y[ix])))

    for ep in range(epochs):
        net.train(); losses = []
        for _ in range(steps):
            ch = rng.choice(tr_ix, bs, replace=True)
            logit = net(torch.from_numpy(np.asarray(X[ch], np.float32)))
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, torch.from_numpy(y[ch]))
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0); opt.step()
            losses.append(float(loss.detach()))
        vb = bce_at(va_ix)
        hist.append(dict(ep=ep, train=float(np.mean(losses)), val=vb))
        print(json.dumps(hist[-1]), flush=True)
        if vb < best[0] - 1e-4:
            best = (vb, {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}, ep)
        elif ep - best[2] >= patience:
            break
    net.load_state_dict(best[1])
    test = bce_at(te_ix)
    ck = dict(seed=seed, feats="p5_barge_m2", best_ep=best[2], val_bce=best[0], test_bce=test, hist=hist,
              state=best[1])
    torch.save(ck, f"models/P5barge_s{seed}.pt")
    print(f"seed {seed} best_ep {best[2]} val {best[0]:.4f} test {test:.4f}", flush=True)
    return ck


if __name__ == "__main__":
    import sys
    cmd = sys.argv[1]
    if cmd == "build":
        build()
    elif cmd == "probe":
        probe()
    elif cmd == "seed":
        train_seed(int(sys.argv[2]))
