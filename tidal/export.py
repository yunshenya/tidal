"""Export a trained model to ONNX (calibrated probabilities at the newest position) and verify parity with PyTorch.
usage: python -m tidal.export TAG"""
import sys, json, numpy as np, torch, onnxruntime as ort
from tidal.model import TurnModel, HEAD_DIMS
from tidal.dataset import HEADS
from tidal.seqdata import load, conv_index, eval_windows, batchify
def require_parity(diffs, atol=1e-4):
    bad = {k: v for k, v in diffs.items() if not np.isfinite(v) or v > atol}
    if bad: raise ValueError(f"ONNX parity failed (tolerance {atol}): {bad}")

class Wrapped(torch.nn.Module):
    def __init__(self, m, temps, use_text):
        super().__init__(); self.m = m; self.use_text = use_text
        self.t = {h: float(temps.get(h, 1.0)) for h in HEADS}
    def forward(self, feats, valid, text=None):
        o = self.m(feats, valid, text if self.use_text else None); outs = []
        for h in HEADS:
            z = o[h][:, -1] / self.t[h]
            outs.append(torch.sigmoid(z) if HEAD_DIMS[h] == 1 else torch.softmax(z, -1))
        return tuple(outs)
def main(tag, regime_eval_json=None):
    ck = torch.load(f"models/{tag}.pt", weights_only=False)
    m = TurnModel(ck["n_feat"], ck["use_text"], kind=ck["kind"]); m.load_state_dict(ck["state"]); m.eval()
    regime = ck["regime"]
    temps = json.load(open(f"reports/eval_{regime}.json"))["models"].get(tag, {}).get("temps", {})
    W = Wrapped(m, temps, ck["use_text"]).eval()
    F = torch.zeros(1, 64, ck["n_feat"]); V = torch.ones(1, 64, dtype=torch.bool); E = torch.zeros(1, 64, 512)
    args = (F, V, E) if ck["use_text"] else (F, V)
    names = ["feats", "valid"] + (["text"] if ck["use_text"] else [])
    path = f"models/{tag}.onnx"
    torch.onnx.export(W, args, path, input_names=names, output_names=[h.replace("y_", "p_") for h in HEADS], opset_version=17,
                      dynamic_axes={n: {0: "batch"} for n in names} | {h.replace("y_", "p_"): {0: "batch"} for h in HEADS}, dynamo=False)
    # parity on real windows
    D = load(regime); rows = np.flatnonzero(D["keep"] & (D["meta"].source == "real").to_numpy())
    rng = np.random.default_rng(0); rows = rng.choice(rows, 500, replace=False)
    wins = eval_windows(conv_index(D), rows)
    Fb, Eb, _, Vb = batchify(D, wins, ck["use_text"])
    if Fb.shape[1] < 64:   # pad to 64 for the fixed-shape check
        pad = 64 - Fb.shape[1]; Fb = np.pad(Fb, ((0, 0), (pad, 0), (0, 0))); Vb = np.pad(Vb, ((0, 0), (pad, 0)))
        if Eb is not None: Eb = np.pad(Eb, ((0, 0), (pad, 0), (0, 0)))
    with torch.no_grad():
        tt = W(torch.from_numpy(Fb), torch.from_numpy(Vb), torch.from_numpy(Eb) if ck["use_text"] else None)
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    feed = {"feats": Fb, "valid": Vb}
    if ck["use_text"]: feed["text"] = Eb
    oo = s.run(None, feed)
    diffs = {h: float(np.abs(a.numpy() - b).max()) for h, a, b in zip(HEADS, tt, oo)}
    require_parity(diffs)
    agree = {h: float((a.numpy().argmax(-1) == b.argmax(-1)).mean()) if HEAD_DIMS[h] > 1 else float(((a.numpy() > .5) == (b > .5)).mean()) for h, a, b in zip(HEADS, tt, oo)}
    res = dict(tag=tag, onnx=path, bytes=__import__("os").path.getsize(path), max_abs_diff=diffs, decision_agreement=agree, n=len(rows))
    json.dump(res, open(f"reports/onnx_parity_{tag}.json", "w"), indent=1); print(json.dumps(res, indent=1))
if __name__ == "__main__":
    main(sys.argv[1])
