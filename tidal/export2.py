"""Export a phase-2 VAP model to ONNX (temperature-calibrated head probabilities + 20 projection probabilities at the
newest position) and verify parity with PyTorch on real windows (local only; the exported weights are never committed).
usage: python -m tidal.export2 TAG"""
import sys, os, json, numpy as np, torch, onnxruntime as ort
from tidal.model import HEAD_DIMS
from tidal.dataset import HEADS
from tidal import vap as VP
from tidal.seqdata import CTX
from tidal.export import require_parity
OUT = [h.replace("y_", "p_") for h in HEADS] + ["p_vap"]
class Wrapped(torch.nn.Module):
    def __init__(self, m, temps):
        super().__init__(); self.m = m; self.t = {h: float(temps.get(h, 1.0)) for h in HEADS}
    def forward(self, feats, valid):
        o = self.m(feats, valid); outs = []
        for h in HEADS:
            z = o[h][:, -1] / self.t[h]; outs.append(torch.sigmoid(z)[:, 0] if HEAD_DIMS[h] == 1 else torch.softmax(z, -1))
        return tuple(outs) + (torch.sigmoid(o["vap"][:, -1]),)
def main(tag):
    m, ck = VP.load_model(tag); temps = json.load(open("reports/eval_phase2.json"))["temps"]["p2:" + tag]
    W = Wrapped(m, temps).eval(); path = f"models/{tag}.onnx"
    F = torch.zeros(2, CTX, ck["n_feat"]); V = torch.ones(2, CTX, dtype=torch.bool)
    torch.onnx.export(W, (F, V), path, input_names=["feats", "valid"], output_names=OUT, opset_version=17,
                      dynamic_axes={n: {0: "batch"} for n in ["feats", "valid"] + OUT}, dynamo=False)
    D = VP.load(); meta = D["meta"]; real = (meta.source == "real").to_numpy()
    rows = np.random.default_rng(0).choice(np.flatnonzero(real), 1000, replace=False)
    from tidal.seqdata import eval_windows
    W_ = eval_windows(VP.conv_rows(meta, real), rows); Fb, _, _, Vb = VP.batch(D, W_)
    pad = CTX - Fb.shape[1]; Fb = np.pad(Fb, ((0, 0), (pad, 0), (0, 0))); Vb = np.pad(Vb, ((0, 0), (pad, 0)))
    with torch.no_grad(): tt = [x.numpy() for x in W(torch.from_numpy(Fb), torch.from_numpy(Vb))]
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    oo = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"]).run(None, {"feats": Fb, "valid": Vb})
    diffs = {n: float(np.abs(a - b).max()) for n, a, b in zip(OUT, tt, oo)}
    require_parity(diffs)
    agree = {n: float((a.argmax(-1) == b.argmax(-1)).mean()) if a.ndim > 1 and n != "p_vap" else float(((a > .5) == (b > .5)).mean()) for n, a, b in zip(OUT, tt, oo)}
    # left-padding invariance: the same window with and without padding must give the same output (streaming-safe)
    short = [w for w in W_ if len(w) < CTX][:200]
    if short:
        Fs, _, _, Vs = VP.batch(D, short)
        with torch.no_grad(): a = W(torch.from_numpy(Fs), torch.from_numpy(Vs))[-1].numpy()
        ps = CTX - Fs.shape[1]; b = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"]).run(None, {"feats": np.pad(Fs, ((0, 0), (ps, 0), (0, 0))), "valid": np.pad(Vs, ((0, 0), (ps, 0)))})[-1]
        pad_diff = float(np.abs(a - b).max())
        require_parity({"padding": pad_diff})
    else: pad_diff = None
    res = dict(tag=tag, bytes=os.path.getsize(path), params=int(ck["params"]), max_abs_diff=diffs, decision_agreement=agree, n=len(rows), padding_invariance_max_diff=pad_diff)
    json.dump(res, open(f"reports/onnx_parity_{tag}.json", "w"), indent=1); print(json.dumps(res, indent=1))
if __name__ == "__main__": main(sys.argv[1])
