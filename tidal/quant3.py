"""int8 dynamic quantization (ONNX Runtime) of a VAP model: accuracy delta (per-head primary metric, same rows) and
single-thread latency delta, fp32 ONNX vs int8 ONNX. Models stay local (never committed).
usage: python -m tidal.quant3 TAG [rows: real|pub]  -> reports/quant_phase3_<TAG>_<rows>.json"""
import sys, os, time, json, numpy as np, torch, onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from tidal.dataset import HEADS
from tidal.model import HEAD_DIMS
from tidal.seqdata import CTX, eval_windows
from tidal import vap as VP, metrics as M
from tidal.eval3 import prim
OUT = [h.replace("y_", "p_") for h in HEADS]

class Wrapped(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, feats, valid):
        o = self.m(feats, valid)
        return tuple(torch.sigmoid(o[h][:, -1])[:, 0] if HEAD_DIMS[h] == 1 else torch.softmax(o[h][:, -1], -1) for h in HEADS)

def export(tag, n_feat):
    m, ck = VP.load_model(tag); W = Wrapped(m).eval(); p32 = f"models/{tag}.fp32.onnx"; p8 = f"models/{tag}.int8.onnx"
    torch.onnx.export(W, (torch.zeros(2, CTX, n_feat), torch.ones(2, CTX, dtype=torch.bool)), p32, input_names=["feats", "valid"],
                      output_names=OUT, opset_version=17, dynamic_axes={n: {0: "batch"} for n in ["feats", "valid"] + OUT}, dynamo=False)
    quantize_dynamic(p32, p8, weight_type=QuantType.QInt8)
    return W, p32, p8, ck

def main(tag, which="real", n=4000, boot=500):
    D = VP.load("p3"); meta = D["meta"]; Y = D["Y"]; split = meta.split.to_numpy()
    if which == "real":
        mask = (meta.source == "real").to_numpy(); sel = mask & np.isin(split, ["test_time", "test_group"])
        blk = lambda r: np.array([f"{c}:{int(t // 3600)}" for c, t in zip(meta.conv.to_numpy()[r], meta.ts.to_numpy()[r])])
    else:
        mask = meta.source.isin(["pub_tg", "pub_irc", "pub_twitch", "pub_candor", "pub_aishell4"]).to_numpy(); sel = mask & (split == "pub_test")
        blk = lambda r: np.array([f"{c}:{int(t // 300)}" for c, t in zip(meta.conv.to_numpy()[r], meta.ts.to_numpy()[r])])
    rows = np.flatnonzero(sel & ~np.all(np.isnan(Y), 1))
    if len(rows) > n: rows = np.sort(np.random.default_rng(0).choice(rows, n, replace=False))
    W, p32, p8, ck = export(tag, D["X"].shape[1])
    win = eval_windows(VP.conv_rows(meta, mask), rows); F, _, _, V = VP.batch(D, win)
    pad = CTX - F.shape[1]; F = np.pad(F, ((0, 0), (pad, 0), (0, 0))); V = np.pad(V, ((0, 0), (pad, 0)))
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    S = {k: ort.InferenceSession(p, so, providers=["CPUExecutionProvider"]) for k, p in (("fp32", p32), ("int8", p8))}
    P = {k: [np.concatenate(x) for x in zip(*[s.run(None, {"feats": F[b:b + 256], "valid": V[b:b + 256]}) for b in range(0, len(F), 256)])] for k, s in S.items()}
    res = dict(tag=tag, rows=which, n=len(rows), bytes={k: os.path.getsize(p) for k, p in (("fp32", p32), ("int8", p8))}, heads={}, latency_ms={})
    B = blk(rows)
    for hi, h in enumerate(HEADS):
        b = HEAD_DIMS[h] == 1; y = Y[rows, hi]; lab = ~np.isnan(y)
        if lab.sum() < 20 or (b and not 0 < y[lab].sum() < lab.sum()): continue
        a32, a8 = P["fp32"][hi][lab], P["int8"][hi][lab]
        d = M.paired_delta(y[lab], a8, a32, .5, .5, B[lab], "binary" if b else "multi", n_boot=boot)
        res["heads"][h] = dict(n=int(lab.sum()), fp32=prim(y[lab], a32, b), int8=prim(y[lab], a8, b), int8_minus_fp32=d,
                               max_abs_prob_diff=float(np.abs(a32 - a8).max()),
                               decision_agreement=float(((a32 > .5) == (a8 > .5)).mean()) if b else float((a32.argmax(1) == a8.argmax(1)).mean()))
    x1 = {"feats": F[:1], "valid": V[:1]}
    for k, s in S.items():
        for _ in range(50): s.run(None, x1)
        lat = []
        for i in range(1000):
            x = {"feats": F[i % len(F):i % len(F) + 1], "valid": V[i % len(F):i % len(F) + 1]}; t0 = time.perf_counter(); s.run(None, x); lat.append((time.perf_counter() - t0) * 1000)
        res["latency_ms"][k] = dict(p50=float(np.percentile(lat, 50)), p95=float(np.percentile(lat, 95)), note="1 thread, batch 1, full 64-event window")
    json.dump(res, open(f"reports/quant_phase3_{tag}_{which}.json", "w"), indent=1, default=float); print(json.dumps(res, indent=1, default=float))

if __name__ == "__main__": main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "real")
