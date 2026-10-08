"""Frozen CANDOR timing model on previously exposed Chinese development data.

This is an exploratory transfer diagnostic, never a new independent test. No
fitting, recalibration, threshold selection, or checkpoint selection happens here.
"""
import json
import numpy as np
import torch

from tidal.audio_turn import scores, decisions
from tidal.turn_data import TurnData, audio_data
from tidal.turn_optimize import ROOT, STORE, inference, verify_selection, digest


def diagnostic():
    torch.set_num_threads(2)
    freeze=json.loads((STORE/'timing_selection.json').read_text())
    verify_selection(freeze,freeze['signature'])
    source=audio_data({'zh':('A1012','A1091','A1102')})
    data=TurnData(source.rows,source.features[:,:12],source.y)
    p,q=inference(data,freeze)
    report=dict(protocol='Exploratory zero-shot Chinese diagnostic; data was exposed in earlier audio experiments.',
                independent_test=False,model_fitted_on_chinese=False,selection_or_calibration_changed=False,
                selection_sha256=digest(STORE/'timing_selection.json'),n=len(data.y),
                positive_rate=float(data.y.mean()),neural=scores(data.y,p),lr=scores(data.y,q),
                decisions=dict(neural=decisions(data.y,p,.75),lr=decisions(data.y,q,.75)),groups={})
    for group in sorted({r['group'] for r in data.rows}):
        mask=np.array([r['group']==group for r in data.rows])
        report['groups'][group]=dict(n=int(mask.sum()),neural=scores(data.y[mask],p[mask]),lr=scores(data.y[mask],q[mask]))
    verify_selection(freeze,freeze['signature'])
    (ROOT/'reports'/'turn_v2_zh_transfer.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps(report),flush=True)
    return report


if __name__=='__main__':
    diagnostic()
