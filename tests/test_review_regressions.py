"""Regression coverage for exports and causal replay."""
import numpy as np
import pytest
import torch


def test_bootstrap_ignores_single_class_draws():
    from tidal.metrics import paired_delta
    y=np.array([0,0,1,1]); p=np.array([.1,.2,.8,.9])
    r=paired_delta(y,p,p,.5,.5,np.array(['a','a','b','b']),n_boot=200)
    assert r['p_le0'] == 1.
    assert 0 < r['n_boot_valid'] < 200
    one=paired_delta(np.zeros(4),p,p,.5,.5,np.array(['a','a','b','b']),n_boot=20)
    assert np.isnan(one['p_le0']) and one['n_boot_valid'] == 0


def test_onnx_gru_padding_parity(tmp_path):
    import onnxruntime as ort
    from tidal.model import TurnModel
    from tidal.export import Wrapped as W1
    torch.manual_seed(3)
    w=W1(TurnModel(17,True).eval(),{},True)
    w.eval(); x=torch.randn(2,64,17); v=torch.ones(2,64,dtype=torch.bool); text=torch.randn(2,64,512)
    path=tmp_path/'p1.onnx'; names=['feats','valid','text']
    args=(x,v,text)
    torch.onnx.export(w,args,str(path),input_names=names,opset_version=17,dynamo=False,
                      dynamic_axes={k:{0:'batch'} for k in names})
    so=ort.SessionOptions(); so.intra_op_num_threads=1
    sess=ort.InferenceSession(str(path),so,providers=['CPUExecutionProvider'])
    for lengths in ([64],[1,5,64],[0,2]):
        B=len(lengths); xb=x[:1].expand(B,-1,-1).clone(); tb=text[:1].expand(B,-1,-1).clone()
        vb=torch.arange(64)[None]>=64-torch.tensor(lengths)[:,None]
        inp=(xb,vb,tb)
        with torch.no_grad(): ref=w(*inp)
        out=sess.run(None,{k:t.numpy() for k,t in zip(names,inp)})
        for a,b in zip(ref,out): assert np.allclose(a.numpy(),b,atol=1e-5,rtol=1e-5)


def test_export_rejects_bad_parity():
    from tidal.export import require_parity
    require_parity({'ok':1e-6})
    for bad in (1e-2,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='parity'): require_parity({'bad':bad})
