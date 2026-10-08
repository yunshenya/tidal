"""Regression coverage for data loss, split alignment, exports and causal replay."""
import json
import sqlite3
import numpy as np
import pandas as pd
import pytest
import torch
from tidal.shadow import run, store as S


def shadow_env(tmp_path, monkeypatch):
    monkeypatch.setattr(S, 'STATE', tmp_path / 'state')
    monkeypatch.setattr(S, 'DB', tmp_path / 'state' / 'shadow.db')
    monkeypatch.delenv('TIDAL_SHADOW_P4', raising=False)
    monkeypatch.setattr(run, 'predict', lambda *args: None)
    p = tmp_path / 'events.jsonl'
    monkeypatch.setenv('TIDAL_SHADOW_JSONL', str(p))
    return p


def test_failed_ingest_rolls_back_cursor_and_events(tmp_path, monkeypatch):
    p = shadow_env(tmp_path, monkeypatch)
    ev = [dict(msg_id=str(i), conv='c', ts=10+i, bot_action='silent') for i in range(2)]
    p.write_text(''.join(json.dumps(e)+'\n' for e in ev))
    original = run.upsert_events
    def fail_after_one(c, frame, now):
        original(c, frame.iloc[:1], now)
        raise RuntimeError('injected ingest failure')
    monkeypatch.setattr(run, 'upsert_events', fail_after_one)
    assert run.main(['--source', 'jsonl']) == 1
    with S.connect() as c:
        assert S.get_meta(c, 'jsonl_offset', 0) == 0
        assert c.execute('select count(*) from events').fetchone()[0] == 0
        assert c.execute('select status from runs').fetchone()[0] == 'error'
    monkeypatch.setattr(run, 'upsert_events', original)
    assert run.main(['--source', 'jsonl']) == 0
    with S.connect() as c:
        assert S.get_meta(c, 'jsonl_offset') == p.stat().st_size
        assert c.execute('select count(*) from events').fetchone()[0] == 2


def test_future_events_are_stored_but_excluded_from_labels(tmp_path, monkeypatch):
    p = shadow_env(tmp_path, monkeypatch)
    monkeypatch.setattr(run.time, 'time', lambda: 10000.)
    ev = [dict(msg_id='old', conv='c', ts=100., speaker='a', bot_action='silent'),
          dict(msg_id='future', conv='c', ts=20000., speaker='b', bot_action='speak')]
    p.write_text(''.join(json.dumps(e)+'\n' for e in ev))
    assert run.main(['--source', 'jsonl']) == 0
    with S.connect() as c:
        assert c.execute('select count(*) from events').fetchone()[0] == 2
        # An unseen future event must not supply the old event's recheck label.
        assert c.execute('select y_recheck from labels').fetchone()[0] == 6
        assert c.execute('select count(*) from labels').fetchone()[0] == 1
        conv = c.execute('select conv from events limit 1').fetchone()[0]
        assert len(run.load_frame(c, [conv], 0., 10000.)) == 1
    monkeypatch.setattr(run.time, 'time', lambda: 25000.)
    assert run.main(['--source', 'jsonl']) == 0
    with S.connect() as c:
        assert c.execute('select count(*) from events').fetchone()[0] == 2
        assert c.execute('select count(*) from labels').fetchone()[0] == 2


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


def test_aishell_labels_share_converter_ids_and_tie_order(tmp_path, monkeypatch):
    from tidal import phase5_data as P5
    from tidal.public_data import convert as C
    d = tmp_path / 'aishell4_seg'; d.mkdir()
    def line(st, du, sp): return f'SPEAKER x 1 {st} {du} <NA> <NA> {sp} <NA> <NA>\n'
    (d / '00_short.rttm').write_text(line(0,1,'A'))
    segments = [(1.,1.,'B'),(0.,2.,'A')] + [(float(i*3),1.,'A' if i%2 else 'B') for i in range(2,30)]
    (d / '01_kept.rttm').write_text(''.join(line(*s) for s in segments))
    monkeypatch.setattr(P5,'DATA',tmp_path); monkeypatch.setattr(C,'DATA',tmp_path)
    monkeypatch.setattr(C,'finish',lambda frames,*args,**kwargs:pd.concat(frames))
    frame = C.aishell4(); labels = P5.aishell_labels()
    assert set(frame.msg_id) == set(labels)
    triples = [(s,s+du,sp) for s,du,sp in segments]
    order = sorted(range(len(triples)),key=lambda k:(triples[k][1],triples[k][0],k))
    yi,_ = P5.speech_interrupt([triples[i] for i in order])
    assert np.allclose([labels[m][0] for m in frame.msg_id],yi,equal_nan=True)


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
