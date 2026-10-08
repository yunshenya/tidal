"""Phase 4C (shadow only): multi-timescale online adaptation of a frozen event encoder ("continuum memory").

Inspired by Nested Learning / Hope's Continuum Memory System (Behrouz et al., NeurIPS 2025, arXiv:2512.24695) and
Titans' surprise-driven test-time memory (Behrouz et al., arXiv:2501.00663). Levels, fastest -> slowest:
  fast    the backbone's recurrent / KV state (per event, unchanged)
  medium  per-session fast weights: low-rank residual adapter on the encoder output h, updated online after every
          matured target with Titans-style momentum + surprise-scaled step + forgetting (weight decay toward 0)
  slow    per-group adapter (same form), updated every C_slow matured targets with the accumulated gradient
          (CMS eq. 71: chunked update), persists across that group's sessions
  global  per-deployment (one "streamer"/persona) adapter, updated every C_glob targets, tiny step
  frozen  base model weights (all heads included)
h' = h + sum_levels B_l (A_l h) + b_l ; heads(h') -> logits. B_l, b_l start at 0, so every level starts as the static model.
Learning signal: the self-supervised VAP targets only (no labels). Strictly causal: target bin k of event i is used only
once the stream clock has passed ts_i + hi_k (bin end); each (event, bin) is used exactly once.
No live decisions: this module only replays logged streams and reports metrics."""
import heapq, json, math, os, sys, time, numpy as np, pandas as pd, torch, torch.nn.functional as F
from tidal import vap_targets as VT
BIN_END = np.repeat(np.array([hi for _, hi in VT.BINS], float)[None], len(VT.CHANNELS), 0).reshape(-1)  # [NV] channel-major

class Level:
    def __init__(self, d, rank, lr, mom, forget, every, surprise=True, seed=0):
        g = torch.Generator().manual_seed(seed)
        self.A = torch.randn(rank, d, generator=g) / math.sqrt(d)   # fixed random down-projection (key map)
        self.B = torch.zeros(d, rank); self.b = torch.zeros(d)
        self.SB = torch.zeros_like(self.B); self.Sb = torch.zeros_like(self.b)
        self.gB = torch.zeros_like(self.B); self.gb = torch.zeros_like(self.b); self.n = 0
        self.lr, self.mom, self.forget, self.every, self.surprise = lr, mom, forget, every, surprise
    def delta(self, h): return (h @ self.A.T) @ self.B.T + self.b
    def accumulate(self, gB, gb, w):
        self.gB += w * gB; self.gb += w * gb; self.n += 1
        if self.n >= self.every:
            # Titans: S_t = mom*S_{t-1} - lr*grad ; M_t = (1-forget)*M_{t-1} + S_t   (grad = chunk mean, CMS eq. 71)
            self.SB = self.mom * self.SB - self.lr * self.gB / self.n; self.Sb = self.mom * self.Sb - self.lr * self.gb / self.n
            self.B = (1 - self.forget) * self.B + self.SB; self.b = (1 - self.forget) * self.b + self.Sb
            self.gB.zero_(); self.gb.zero_(); self.n = 0
    def state(self): return [t.clone() for t in (self.B, self.b, self.SB, self.Sb)]
    def load(self, s): self.B, self.b, self.SB, self.Sb = [t.clone() for t in s]; self.gB.zero_(); self.gb.zero_(); self.n = 0

CFG_DEFAULT = dict(rank=8, med=dict(lr=3e-3, mom=0.9, forget=0.01, every=1), slow=dict(lr=1e-3, mom=0.9, forget=0.0, every=32),
                   glob=dict(lr=3e-4, mom=0.9, forget=0.0, every=256), surprise=True, levels=("med", "slow", "glob"), session_gap=1800.0)

class Continuum:
    def __init__(self, model, cfg):
        self.m = model; self.cfg = cfg; d = model.vap[0].in_features; self.d = d
        mk = lambda name, seed: Level(d, cfg["rank"], seed=seed, surprise=cfg["surprise"], **cfg[name])
        self.glob = mk("glob", 3); self.slow_by_group = {}; self.med_by_group = {}; self.mk = mk; self.ema = None
        for p in model.parameters(): p.requires_grad_(False)
    def active(self, group):
        lv = []
        if "glob" in self.cfg["levels"]: lv.append(self.glob)
        if "slow" in self.cfg["levels"]:
            if group not in self.slow_by_group: self.slow_by_group[group] = self.mk("slow", 2)
            lv.append(self.slow_by_group[group])
        if "med" in self.cfg["levels"]: lv.append(self.med_by_group[group] if group in self.med_by_group else self.med)
        return lv
    def new_session(self, group=None):
        self.med = self.mk("med", 1)
        if group is not None: self.med_by_group[group] = self.med
    def heads(self, h, lv):
        hp = h + sum(l.delta(h) for l in lv) if lv else h
        o = {k: m(hp) for k, m in self.m.heads.items()}; o["vap"] = self.m.vap(hp); return o
    def update(self, h, v, lv):
        """h [n, d] for matured events, v [n, NV] target with NaN except the newly matured bins."""
        params = []
        for l in lv: l.B.requires_grad_(True); l.b.requires_grad_(True); params += [l.B, l.b]
        hp = h + sum(l.delta(h) for l in lv); lo = self.m.vap(hp); mk = ~torch.isnan(v)
        per = F.binary_cross_entropy_with_logits(lo[mk], v[mk], reduction="none")
        loss = per.mean(); grads = torch.autograd.grad(loss, params)
        for l in lv: l.B.requires_grad_(False); l.b.requires_grad_(False)
        s = float(loss.detach()); self.ema = s if self.ema is None else 0.99 * self.ema + 0.01 * s
        w = min(3.0, s / max(self.ema, 1e-6)) if self.cfg["surprise"] else 1.0     # surprise-scaled step (Titans)
        for i, l in enumerate(lv): l.accumulate(grads[2 * i].detach(), grads[2 * i + 1].detach(), w)
        return s

def replay(cont, H, V, ts, conv, rows_eval, log_every=None):
    """Replay in global timestamp order; return outputs/costs in the original row order.

    Each group has its own session adapter. Matured bins update the originating session,
    group and shared deployment adapter before the next prediction, even if that group
    has gone quiet. No group's future targets can influence another group's past.
    """
    n = len(H); out_vap = torch.zeros(n, VT.NV)
    Ht = torch.from_numpy(H); Vt = torch.from_numpy(V); cost = np.zeros(n)
    head_out = {k: torch.zeros(n, m[-1].out_features) for k, m in cont.m.heads.items()}
    order = np.argsort(ts, kind="stable")
    pending = []; row_state = {}; sessions = {}; last_ts = {}
    nb = len(VT.BINS)
    for i in order:
        t0 = time.perf_counter(); now = ts[i]
        # Apply updates at their actual maturity times, rather than batching by the
        # next arriving event. Another group's traffic must not change local cadence.
        while pending and pending[0][0] <= now:
            due = pending[0][0]; matured = {}
            while pending and pending[0][0] == due:
                _, r, k = heapq.heappop(pending)
                session, lv, remaining = row_state[r]
                batch = matured.setdefault(session, (lv, {}))[1]
                cols = np.arange(k, VT.NV, nb)
                batch.setdefault(r, []).extend(cols[np.isfinite(V[r, cols])].tolist())
                remaining -= 1
                if remaining: row_state[r] = (session, lv, remaining)
                else: del row_state[r]
            for lv, batch in matured.values():
                idx = list(batch); vv = torch.full((len(idx), VT.NV), float("nan"))
                for j, r in enumerate(idx): vv[j, batch[r]] = Vt[r, batch[r]]
                cont.update(Ht[idx], vv, lv)
        group = conv[i]
        if group not in last_ts or now - last_ts[group] > cont.cfg["session_gap"]:
            cont.new_session(group); sessions[group] = sessions.get(group, -1) + 1
        last_ts[group] = now; lv = cont.active(group)
        with torch.no_grad():
            o = cont.heads(Ht[i:i + 1], lv); out_vap[i] = o["vap"][0]
            for k in head_out: head_out[k][i] = o[k][0]
        bins = [k for k in range(nb) if np.isfinite(V[i, k::nb]).any()]
        if bins and lv:
            row_state[i] = ((group, sessions[group]), lv, len(bins))
            for k in bins: heapq.heappush(pending, (now + VT.BINS[k][1], i, k))
        cost[i] = time.perf_counter() - t0
    return out_vap.numpy(), {k: v.numpy() for k, v in head_out.items()}, cost


def encode_rows(model, D, rows, conv_mask):
    """frozen base: encoder output h at each row (causal <=64-event window, same as evaluation in phase 3)."""
    from tidal.vap import conv_rows, batch
    from tidal.seqdata import eval_windows
    convs = conv_rows(D["meta"], conv_mask); W = eval_windows(convs, rows); out = []
    with torch.no_grad():
        for b in range(0, len(W), 512):
            Fm, _, _, valid = batch(D, W[b:b + 512]); out.append(model(torch.from_numpy(Fm), torch.from_numpy(valid))["h"][:, -1].numpy())
    return np.concatenate(out)

# ----------------------------------------------------------------------------------------------------------------------
GRID = [dict(levels=lv, surprise=sp, med=dict(lr=lr, mom=0.9, forget=0.01, every=1))
        for lv in (("med",), ("med", "slow"), ("med", "slow", "glob")) for sp in (True, False) for lr in (1e-3, 3e-3, 1e-2)]

def _cfg(over):
    c = json.loads(json.dumps(CFG_DEFAULT)); c.update({k: v for k, v in over.items() if k != "med"})
    if "med" in over: c["med"] = over["med"]
    c["levels"] = tuple(c["levels"]); return c

def _rowloss(vl, V):
    m = ~np.isnan(V); l = np.where(m, np.logaddexp(0, vl) - np.nan_to_num(V) * vl, 0.0); return l.sum(1) / np.maximum(m.sum(1), 1), m.any(1)

def _bdelta(la, lb, blocks, n_boot=1000):
    from tidal.metrics import ci
    ub, inv = np.unique(blocks, return_inverse=True); rng = np.random.default_rng(0)
    sa, sb, n = np.bincount(inv, la), np.bincount(inv, lb), np.bincount(inv); out = []
    for _ in range(n_boot):
        k = np.bincount(rng.integers(0, len(ub), len(ub)), minlength=len(ub)); out.append(((k * sa).sum() - (k * sb).sum()) / (k * n).sum())
    return dict(adapted=float(la.mean()), static=float(lb.mean()), d=float(la.mean() - lb.mean()), ci=ci(np.array(out)), n=int(len(la)))

def stream_set(D, mask, max_events=None, seed=0):
    """all rows of the selected conversations, in (conv, ts) order (the meta table is already sorted that way)."""
    meta = D["meta"]; rows = np.flatnonzero(mask)
    if max_events and len(rows) > max_events:
        convs = meta.conv.to_numpy()[rows]; u = np.random.default_rng(seed).permutation(np.unique(convs)); keep = []; tot = 0
        cnt = pd.Series(convs).value_counts()
        for c in u:
            if tot >= max_events: break
            keep.append(c); tot += cnt[c]
        rows = rows[np.isin(convs, keep)]
    return rows


def _test(model, cfg, R, mode, split, conv, ts, D, boot):
    from tidal.eval3 import probs, prim
    from tidal.dataset import HEADS
    from tidal.model import HEAD_DIMS
    from tidal.metrics import paired_delta, best_threshold
    rows, H, V, sv, sh = R["test"]; c = Continuum(model, _cfg(cfg)); av, ah, cost = replay(c, H, V, ts[rows], conv[rows], None)
    la, ok = _rowloss(av, V); ls, _ = _rowloss(sv, V)
    blocks = np.array([f"{a}:{int(b // 3600)}" for a, b in zip(conv[rows], ts[rows])])
    pos = np.zeros(len(rows), int); starts = np.r_[0, np.flatnonzero(conv[rows][1:] != conv[rows][:-1]) + 1]
    for a, b in zip(starts, list(starts[1:]) + [len(rows)]): pos[a:b] = np.arange(b - a)
    T = dict(selected=cfg, cost_ms=dict(p50=float(np.percentile(cost, 50) * 1e3), p95=float(np.percentile(cost, 95) * 1e3), mean=float(cost.mean() * 1e3)))
    if mode == "real":
        ev = {"unseen_groups(test_group)": split[rows] == "test_group", "seen_groups(test_time)": split[rows] == "test_time"}
    else:
        ev = {"unseen_streams(pub_test)": np.ones(len(rows), bool)}
    for nm, em in ev.items():
        s = em & ok; T[nm] = _bdelta(la[s], ls[s], blocks[s], boot)
        for N in (16, 64):
            s2 = s & (pos < N); T[f"{nm}_first{N}"] = _bdelta(la[s2], ls[s2], blocks[s2], boot) if s2.sum() > 20 else None
    # drift / stability: Δ by position bucket in long streams + adapter norm
    edges = [0, 64, 256, 1024, 4096, 16384, 10 ** 9]; T["by_position"] = []
    evr = np.isin(split[rows], ["test_time", "test_group"]) if mode == "real" else np.ones(len(rows), bool)
    for a_, b_ in zip(edges[:-1], edges[1:]):
        s = ok & evr & (pos >= a_) & (pos < b_)
        if s.sum() > 50: T["by_position"].append(dict(pos=[a_, b_], **_bdelta(la[s], ls[s], blocks[s], 300)))
    T["adapter_norms"] = dict(glob=float(c.glob.B.norm()), slow_max=float(max([l.B.norm() for l in c.slow_by_group.values()], default=torch.zeros(())).item() if c.slow_by_group else 0.0))
    # forgetting: re-score the evaluation rows with the FINAL adapter states (after the whole replay), no updates
    if mode == "real":
        fin = []
        with torch.no_grad():
            Ht = torch.from_numpy(H)
            for i in range(len(rows)):
                lv = []
                if "glob" in cfg["levels"]: lv.append(c.glob)
                if "slow" in cfg["levels"] and conv[rows][i] in c.slow_by_group: lv.append(c.slow_by_group[conv[rows][i]])
                fin.append(c.heads(Ht[i:i + 1], lv)["vap"][0].numpy())
        lf, _ = _rowloss(np.array(fin), V); s = ok & (split[rows] == "test_time")
        T["forgetting_seen_final_state_vs_static"] = _bdelta(lf[s], ls[s], blocks[s], boot)
        # heads on real held-out rows (labels never used for adaptation)
        Y = D["Y"][rows]; T["heads"] = {}
        for hi, h in enumerate(HEADS):
            b = HEAD_DIMS[h] == 1; y = Y[:, hi]; lab = ~np.isnan(y)
            pa = probs({h: ah[h]}, h); pb = probs({h: sh[h]}, h); va = lab & (split[rows] == "val")
            for spn in ("test_time", "test_group"):
                te = lab & (split[rows] == spn)
                if te.sum() < 5 or (b and not (0 < y[te].sum() < te.sum())): continue
                thr_a = best_threshold(y[va], pa[va]) if b else .5; thr_b = best_threshold(y[va], pb[va]) if b else .5
                T["heads"][f"{h}|{spn}"] = dict(adapted=prim(y[te], pa[te], b), static=prim(y[te], pb[te], b),
                                                delta=paired_delta(y[te], pa[te], pb[te], thr_a, thr_b, blocks[te], "binary" if b else "multi", n_boot=boot))
    return T

def run_eval(mode="real", base_tags=None, boot=1000, out_path=None, log=print):
    from tidal import vap as VP
    from tidal.eval3 import probs, prim
    from tidal.dataset import HEADS
    from tidal.model import HEAD_DIMS
    from tidal.metrics import paired_delta, best_threshold
    torch.set_num_threads(int(os.environ.get("TIDAL_THREADS", 2)))
    D = VP.load("p3"); meta = D["meta"]; src = meta.source.to_numpy(); split = meta.split.to_numpy(); conv = meta.conv.to_numpy(); ts = meta.ts.to_numpy(float)
    if mode == "real":
        allm = src == "real"; sets = {"val": allm, "test": allm}
    else:
        pubs = ["pub_tg", "pub_irc", "pub_twitch", "pub_candor", "pub_aishell4"]
        sets = {"val": np.isin(src, pubs) & (split == "pub_val"), "test": np.isin(src, pubs) & (split == "pub_test")}
    res = dict(mode=mode, base=base_tags, grid=[], note="shadow replay only; VAP loss per event predicted before any of its targets matured")
    for tag in base_tags:
        model, ck = VP.load_model(tag)
        R = {}
        for sname, sm in sets.items():
            rows = stream_set(D, sm, max_events=None if mode == "real" else 60000, seed=0)
            H = encode_rows(model, D, rows, sm); V = D["V"][rows]
            static = Continuum(model, _cfg(dict(levels=()))); sv, sh, _ = replay(static, H, V, ts[rows], conv[rows], None)
            R[sname] = (rows, H, V, sv, sh)
        # ---- pre-registered selection on the validation streams: lowest mean VAP loss (val rows only for real)
        rows, H, V, sv, _ = R["val"]; vm = (split[rows] == "val") if mode == "real" else np.ones(len(rows), bool)
        ls, ok = _rowloss(sv, V); best = None
        for g in GRID:
            c = Continuum(model, _cfg(g)); av, _, cost = replay(c, H, V, ts[rows], conv[rows], None); la, _ = _rowloss(av, V)
            s = vm & ok; d = float(la[s].mean() - ls[s].mean()); res["grid"].append(dict(tag=tag, cfg=g, val_d_vap=d)); log(f"[{tag}] grid {g} val Δvap={d:+.5f}")
            if best is None or d < best[0]: best = (d, g)
        full = min([g for g in res["grid"] if g["tag"] == tag and len(g["cfg"]["levels"]) == 3], key=lambda g: g["val_d_vap"])["cfg"]
        TT = {}
        for cname, cfg in (("selected", best[1]), ("full_cms", full)):
            log(f"[{tag}] {cname} {cfg}")
            TT[cname] = _test(model, cfg, R, mode, split, conv, ts, D, boot)
        T = TT
        res[tag] = T
        for cname in T: log(cname + " " + json.dumps({k: v for k, v in T[cname].items() if k not in ("by_position", "heads")}, default=float)[:1500])
    json.dump(res, open(out_path or f"reports/continuum_{mode}_phase4.json", "w"), indent=1, default=float)
    return res

if __name__ == "__main__":
    mode = sys.argv[1]; tags = sys.argv[2].split(",")
    run_eval(mode, tags, out_path=sys.argv[3] if len(sys.argv) > 3 else None)
