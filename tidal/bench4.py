"""Phase 4B: streaming cost of each backbone (1 CPU thread): per-event step latency p50/p95, state memory, params,
ONNX export of the O(1) step (state passed as flat tensors) and its size + onnxruntime parity/latency.
-> reports/bench_phase4.json (public-safe: random weights / architecture only)."""
import io, json, os, time, numpy as np, torch, torch.nn as nn
from tidal.vap import VAPModel
torch.set_num_threads(1)
KINDS = ["gru", "tx_kv", "mamba3", "mamba3_siso", "m3_ablate_m2"]

def flat(st):
    if torch.is_tensor(st): return [st]
    if isinstance(st, dict): return [t for k in sorted(st) for t in flat(st[k])]
    if isinstance(st, (list, tuple)): return [t for s in st for t in flat(s)]
    return []

class StepWrap(nn.Module):
    """fixed-structure step for ONNX: (x, *state tensors) -> (vap logits, eot logit, *new state)."""
    def __init__(self, m, template): super().__init__(); self.m = m; self.template = template
    def unflat(self, ts):
        it = iter(ts)
        def rec(t):
            if torch.is_tensor(t): return next(it)
            if isinstance(t, dict): return {k: rec(t[k]) for k in sorted(t)}
            if isinstance(t, (list, tuple)): return type(t)(rec(s) for s in t)
            return t
        return rec(self.template)
    def forward(self, x, *state):
        o, st = self.m.step(x, self.unflat(list(state))); return (o["vap"], o["y_eot"], *flat(st))

class TxKVStep(nn.Module):
    """fixed-shape streaming step for the RoPE+KV transformer (ONNX-exportable): ring buffer of the last `window`
    post-RoPE keys/values (shift-left by one per event) + validity mask + absolute position t. Same math as
    CausalTransformer.step (RoPE scores depend only on relative position)."""
    def __init__(self, m): super().__init__(); self.m = m; self.b = m.body
    def forward(self, x, t, mask, *kv):
        import math
        h = self.m.inorm(self.m.fproj(x)).unsqueeze(1); new = []; B = x.shape[0]
        m2 = torch.cat([mask[:, 1:], torch.ones_like(mask[:, :1])], 1)
        for i, (at, ff, a, b) in enumerate(zip(self.b.attn, self.b.ff, self.b.n1, self.b.n2)):
            q, k, v = at.split(a(h)); q, k = at.rope(q, t), at.rope(k, t)
            K = torch.cat([kv[2 * i][:, :, 1:], k], 2); V = torch.cat([kv[2 * i + 1][:, :, 1:], v], 2)
            sc = q @ K.transpose(-1, -2) / math.sqrt(at.dh) + (1.0 - m2)[:, None, None, :] * -1e9
            h = h + at.o((torch.softmax(sc, -1) @ V).transpose(1, 2).reshape(B, 1, -1)); h = h + ff(b(h)); new += [K, V]
        h = self.b.nf(h)[:, 0]; return (self.m.vap(h), self.m.heads["y_eot"](h), t + 1, m2, *new)

def tx_kv_onnx(m, x):
    import onnxruntime as ort
    W = m.body.window; at = m.body.attn[0]; z = torch.zeros(1, at.h, W, at.dh)
    st = (torch.zeros(1), torch.zeros(1, W), *([z] * (2 * len(m.body.attn)))); w = TxKVStep(m).eval(); errs = []; ref_st = None
    with torch.no_grad():
        for i in range(150):                                 # parity vs eager KV-cache step, beyond the window length
            o = w(x[i], *st); st = o[2:]; r, ref_st = m.step(x[i], ref_st); errs.append(float((o[0] - r["vap"]).abs().max()))
    f = "/tmp/bench4_tx_kv.onnx"; ins = (x[0], *st)
    torch.onnx.export(w, ins, f, opset_version=17, dynamo=False, input_names=["x", "t", "mask"] + [f"kv{i}" for i in range(len(st) - 2)])
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; s = ort.InferenceSession(f, so, providers=["CPUExecutionProvider"])
    feed = {n.name: a.numpy() for n, a in zip(s.get_inputs(), ins)}
    with torch.no_grad(): ref = w(*ins)
    out = s.run(None, feed); err = float(max(np.abs(a - b.detach().numpy()).max() for a, b in zip(out, ref))); t = []
    for _ in range(500):
        t0 = time.perf_counter(); s.run(None, feed); t.append((time.perf_counter() - t0) * 1e3)
    return dict(kind="step(ring-buffer KV, fixed shape)", bytes=os.path.getsize(f), max_abs_err=err, ring_vs_eager_max_err=max(errs),
                ort_p50_ms=round(float(np.percentile(t, 50)), 3), ort_p95_ms=round(float(np.percentile(t, 95)), 3))

def bench(kind, n_feat=60, steps=2000):
    torch.manual_seed(0); m = VAPModel(n_feat, kind=kind).eval(); x = torch.randn(steps, 1, n_feat); st = None; lat = []
    with torch.no_grad():
        for i in range(steps):
            t0 = time.perf_counter(); o, st = m.step(x[i], st); lat.append((time.perf_counter() - t0) * 1e3)
    lat = np.array(lat[100:]); sb = sum(t.numel() * t.element_size() for t in flat(st if kind != "tx_kv" else st["kv"]))
    r = dict(params=sum(p.numel() for p in m.parameters()), step_p50_ms=round(float(np.percentile(lat, 50)), 3), step_p95_ms=round(float(np.percentile(lat, 95)), 3),
             state_bytes=int(sb), param_bytes_fp32=int(sum(p.numel() * 4 for p in m.parameters())))
    # window forward (how training/eval computes one 64-event window), for reference
    with torch.no_grad():
        xx = torch.randn(1, 64, n_feat); v = torch.ones(1, 64, dtype=torch.bool); m(xx, v); t0 = time.perf_counter()
        for _ in range(50): m(xx, v)
    r["window64_forward_ms"] = round((time.perf_counter() - t0) / 50 * 1e3, 3)
    try:
        if kind == "tx_kv":
            r["onnx"] = tx_kv_onnx(m, x); return r
        with torch.no_grad(): o, st0 = m.step(x[0], None)
        for i in range(1, 70): o, st0 = m.step(x[i], st0)
        w = StepWrap(m, st0).eval(); ins = tuple(t.detach() for t in (x[0], *flat(st0))); f = f"/tmp/bench4_{kind}.onnx"
        torch.onnx.export(w, ins, f, opset_version=17, dynamo=False, input_names=["x"] + [f"s{i}" for i in range(len(ins) - 1)])
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; s = ort.InferenceSession(f, so, providers=["CPUExecutionProvider"])
        feed = {n.name: a.numpy() for n, a in zip(s.get_inputs(), ins)}
        with torch.no_grad(): ref = w(*ins)
        out = s.run(None, feed); err = float(max((np.abs(a - b.detach().numpy()).max() for a, b in zip(out, ref) if a.size), default=0.0))
        t = []
        for _ in range(500):
            t0 = time.perf_counter(); s.run(None, feed); t.append((time.perf_counter() - t0) * 1e3)
        r["onnx"] = dict(kind="step", bytes=os.path.getsize(f), max_abs_err=err, ort_p50_ms=round(float(np.percentile(t, 50)), 3), ort_p95_ms=round(float(np.percentile(t, 95)), 3))
    except Exception as e:
        r["onnx_step_error"] = str(e)[:200]
        try:
            class Win(nn.Module):
                def __init__(s_, m): super().__init__(); s_.m = m
                def forward(s_, x, v): o = s_.m(x, v); return o["vap"][:, -1], o["y_eot"][:, -1]
            f = f"/tmp/bench4_{kind}_win.onnx"; torch.onnx.export(Win(m).eval(), (torch.randn(1, 64, n_feat), torch.ones(1, 64, dtype=torch.bool)), f, opset_version=17, dynamo=False)
            r["onnx"] = dict(kind="window64", bytes=os.path.getsize(f))
        except Exception as e2: r["onnx_window_error"] = str(e2)[:200]
    return r

if __name__ == "__main__":
    import sys
    nf = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    res = {k: bench(k, nf) for k in KINDS}; res["note"] = "1 CPU thread, torch eager fp32, batch 1, random weights; box is shared with training jobs (load noted)"
    res["loadavg"] = os.getloadavg(); json.dump(res, open("reports/bench_phase4.json", "w"), indent=1); print(json.dumps(res, indent=1))
