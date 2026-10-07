"""One-time setup for shadow mode (needs the phase-1 artifacts: data/proc, models/*.pt|onnx, reports/eval_*.json).
  - refits the phase-1 baselines and saves them (verified identical to phase-1 predictions)
  - freezes normalization stats, val-chosen thresholds and the val-selected strongest baseline per head
usage: python -m tidal.shadow.setup"""
import json, numpy as np, torch
from tidal.config import ROOT
from tidal.dataset import HEADS
from tidal.shadow import store as S
from tidal.shadow.infer import MODELS, FROZEN
from tidal import baselines

def main():
    S.STATE.mkdir(parents=True, exist_ok=True)
    cfg = dict(models=MODELS, norm={}, thresholds={}, strongest_baseline={}, version="phase1")
    z = np.load(ROOT / "data/proc/dataset.npz"); X, eidx = z["X"], z["eidx"]
    emb = np.load(ROOT / "data/proc/emb.npy").astype(np.float32)
    for r, tag in MODELS.items():
        ck = torch.load(ROOT / "models" / f"{tag}.pt", weights_only=False)
        cfg["norm"][r] = dict(mu=np.asarray(ck["mu"]).tolist(), sd=np.asarray(ck["sd"]).tolist())
        path = S.STATE / f"baselines_{r}.joblib"
        baselines.main(r, save_models=str(path), write_preds=False)
        b = __import__("joblib").load(path); old = dict(np.load(ROOT / f"data/proc/baseline_preds_{r}.npz"))
        E = np.where(eidx[:, None] >= 0, emb[np.clip(eidx, 0, None)], 0.0) if r == "TS" else None
        new = baselines.predict_bundle(b, X, E)
        diff = max(float(np.abs(new[k] - old[k]).max()) for k in old)
        assert diff < 1e-4, f"baseline refit mismatch {r}: {diff}"
        ev = json.load(open(ROOT / f"reports/eval_{r}.json"))
        cfg["thresholds"][r] = {}; cfg["strongest_baseline"][r] = {}
        for h in HEADS:
            hr = ev["heads"][h]; cfg["strongest_baseline"][r][h] = hr["strongest_baseline"]
            ss = hr["systems"]["test_time"]
            cfg["thresholds"][r][h] = {n: m["threshold"] for n, m in ss.items() if "threshold" in m and (not n.startswith("model:") or n == "model:" + tag)}
        cfg.setdefault("baseline_refit_max_abs_diff", {})[r] = diff
    FROZEN.write_text(json.dumps(cfg, indent=1)); FROZEN.chmod(0o600)
    print(json.dumps({k: v for k, v in cfg.items() if k != "norm"}, indent=1))

if __name__ == "__main__":
    main()
