"""Set up the phase-2 VAP model as an ADDITIONAL shadow system (phase 1 keeps running unchanged).
Freezes: model tag, normalisation stats, val-chosen thresholds, the val-fitted abstention threshold for y_act.
Needs models/TAG.pt + models/TAG.onnx (tidal.export2) and reports/eval_phase2.json (tidal.evaluate2).
usage: python -m tidal.shadow.setup2 TAG"""
import sys, json, numpy as np, torch
from tidal.config import ROOT
from tidal.shadow import store as S
from tidal.shadow.infer import P2_FROZEN

def main(tag):
    ck = torch.load(ROOT / "models" / f"{tag}.pt", weights_only=False)
    ev = json.load(open(ROOT / "reports/eval_phase2.json")); name = "p2:" + tag
    assert (ROOT / "models" / f"{tag}.onnx").exists(), "export the ONNX model first (python -m tidal.export2 TAG)"
    thr = {h: hr["thr"][name] for h, hr in ev["heads"].items() if name in hr.get("thr", {})}
    tau = ev["heads"]["y_act"]["abstention"]["test_time"][name]["tau"]          # fitted on val (20% least confident)
    cfg = dict(tag=tag, version="phase2", feats=ck["feats"], mu=np.asarray(ck["mu"]).tolist(), sd=np.asarray(ck["sd"]).tolist(),
               thresholds=thr, abstain_tau={"y_act": tau}, temps=ev["temps"][name],
               note="participant count / modality are not supplied in shadow mode (zeros + known-flag 0); training dropped these optional blocks with p=0.5, effect measured in reports/phase2_extra.json (participants_unknown)")
    P2_FROZEN.write_text(json.dumps(cfg, indent=1)); P2_FROZEN.chmod(0o600)
    print(json.dumps({k: v for k, v in cfg.items() if k not in ("mu", "sd")}, indent=1))

if __name__ == "__main__": main(sys.argv[1])
