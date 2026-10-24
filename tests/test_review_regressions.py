"""Regression coverage for split alignment, exports and causal replay."""
import numpy as np
import pandas as pd
import pytest
import torch


def test_public_only_dataset_build(tmp_path, monkeypatch):
    from tidal import dataset3 as D
    from tidal.dataset import HEADS
    monkeypatch.chdir(tmp_path)
    proc = tmp_path / 'data' / 'proc'; proc.mkdir(parents=True)
    pub = tmp_path / 'public' / 'proc'; pub.mkdir(parents=True)
    monkeypatch.setattr(D, 'DATA', pub.parent)
    meta = dict(source='real', conv='real', conv_type='group', ts=0., role='other', split='train',
                scenario='real_chat', speaker_known=True, msg_id='real:0', responds_to=None, initiate=0, has_text=False)
    pd.DataFrame([meta]).to_parquet(proc / 'p2_meta.parquet')
    np.savez(proc / 'p2.npz', G=np.zeros((1,17),np.float32), Y=np.full((1,6),np.nan,np.float32), V=np.full((1,20),np.nan,np.float32))
    frame = pd.DataFrame([dict(meta, source='pub_tg', conv='tg', msg_id='tg:0', split='pub_train',
                               speaker='p0', text=None, modality='text', n_participants=2., **{h:np.nan for h in HEADS})])
    frame.to_parquet(pub / 'pub_tg.parquet')
    D.build()
    z = np.load(proc / 'p3.npz')
    assert z['W'].tolist() == [1.,1.]
    assert z['G'].shape == (2,17)


def test_bootstrap_ignores_single_class_draws():
    from tidal.metrics import paired_delta
    y=np.array([0,0,1,1]); p=np.array([.1,.2,.8,.9])
    r=paired_delta(y,p,p,.5,.5,np.array(['a','a','b','b']),n_boot=200)
    assert r['p_le0'] == 1.
    assert 0 < r['n_boot_valid'] < 200
    one=paired_delta(np.zeros(4),p,p,.5,.5,np.array(['a','a','b','b']),n_boot=20)
    assert np.isnan(one['p_le0']) and one['n_boot_valid'] == 0


def test_continuum_cross_group_future_causality():
    from tidal.vap import VAPModel
    from tidal.continuum import Continuum, replay, _cfg
    torch.manual_seed(7); m=VAPModel(17).eval()
    H=np.ones((4,192),np.float32); V=np.ones((4,20),np.float32)
    ts=np.array([1000.,1200.,0.,200.]); conv=np.array(['a','a','b','b'])
    cfg=_cfg(dict(levels=('med','slow','glob'),glob=dict(lr=.1,mom=0,forget=0,every=1)))
    a,_,_=replay(Continuum(m,cfg),H,V,ts,conv,None)
    V2=V.copy(); V2[:2]=0
    b,_,_=replay(Continuum(m,cfg),H,V2,ts,conv,None)
    assert np.array_equal(a[2:],b[2:])
    # Row grouping must not affect a shared deployment's time-ordered results.
    order=np.argsort(ts)
    sorted_out,_,_=replay(Continuum(m,cfg),H[order],V[order],ts[order],conv[order],None)
    assert np.allclose(a[order],sorted_out,atol=1e-6)


def test_continuum_sessions_remain_independent():
    from tidal.vap import VAPModel
    from tidal.continuum import Continuum,replay,_cfg
    torch.manual_seed(8); m=VAPModel(17).eval(); rng=np.random.default_rng(4)
    H=rng.normal(size=(6,192)).astype(np.float32); V=np.ones((6,20),np.float32)
    ts=np.array([0.,4.,8.,1.,5.,9.]); conv=np.array(['a']*3+['b']*3)
    cfg=_cfg(dict(levels=('med',),surprise=False))
    both,_,_=replay(Continuum(m,cfg),H,V,ts,conv,None)
    for sl in (slice(0,3),slice(3,6)):
        alone,_,_=replay(Continuum(m,cfg),H[sl],V[sl],ts[sl],conv[sl],None)
        assert np.allclose(both[sl],alone,atol=1e-6)


@pytest.mark.parametrize('phase',[1,2])
def test_onnx_gru_padding_parity(tmp_path,phase):
    import onnxruntime as ort
    from tidal.model import TurnModel
    from tidal.vap import VAPModel
    from tidal.export import Wrapped as W1
    from tidal.export2 import Wrapped as W2
    torch.manual_seed(3)
    w=W1(TurnModel(17,True).eval(),{},True) if phase==1 else W2(VAPModel(17).eval(),{})
    w.eval(); x=torch.randn(2,64,17); v=torch.ones(2,64,dtype=torch.bool); text=torch.randn(2,64,512)
    path=tmp_path/f'p{phase}.onnx'; names=['feats','valid']+(['text'] if phase==1 else [])
    args=(x,v,text) if phase==1 else (x,v)
    torch.onnx.export(w,args,str(path),input_names=names,opset_version=17,dynamo=False,
                      dynamic_axes={k:{0:'batch'} for k in names})
    so=ort.SessionOptions(); so.intra_op_num_threads=1
    sess=ort.InferenceSession(str(path),so,providers=['CPUExecutionProvider'])
    for lengths in ([64],[1,5,64],[0,2]):
        B=len(lengths); xb=x[:1].expand(B,-1,-1).clone(); tb=text[:1].expand(B,-1,-1).clone()
        vb=torch.arange(64)[None]>=64-torch.tensor(lengths)[:,None]
        inp=(xb,vb,tb) if phase==1 else (xb,vb)
        with torch.no_grad(): ref=w(*inp)
        out=sess.run(None,{k:t.numpy() for k,t in zip(names,inp)})
        for a,b in zip(ref,out): assert np.allclose(a.numpy(),b,atol=1e-5,rtol=1e-5)


def test_export_rejects_bad_parity():
    from tidal.export import require_parity
    require_parity({'ok':1e-6})
    for bad in (1e-2,float('nan'),float('inf')):
        with pytest.raises(ValueError,match='parity'): require_parity({'bad':bad})
