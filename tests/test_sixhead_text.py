"""Regression checks for full-stream text training and replay unknown labels."""
import csv
import numpy as np
import pandas as pd
import torch


def test_private_unknown_is_not_a_negative_or_a_bot_message(tmp_path):
    from scripts.build_bootstrap_sixhead import private_rows
    fields = ['id','occurred_at','event_type','is_private','mentioned_bot','reply_to_bot','scope_key']
    p = tmp_path / 'replays.csv'
    with p.open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for i, et, private, mention, reply in [(0,'message_received','','',''),
                (1,'idle_tick','f','f','f'), (2,'message_received','f','t',''),
                (3,'message_received','f','f','f')]:
            w.writerow(dict(id=i, occurred_at='2026-10-01T00:00:00+08:00', event_type=et,
                            is_private=private, mentioned_bot=mention, reply_to_bot=reply, scope_key='c'))
    d = private_rows(p, b'fictional-test-salt')
    assert len(d) == 3 and set(d.role) == {'other'}
    assert d.conv_type.tolist() == ['unknown', 'group', 'group']
    assert np.isnan(d.addr_src.iat[0]) and d.addr_src.iloc[1:].tolist() == [1., 0.]


def test_full_stream_text_keeps_non_text_context(tmp_path, monkeypatch):
    from tidal.seqdata import load, conv_index, eval_windows, batchify
    monkeypatch.chdir(tmp_path); p=tmp_path/'data/proc'; p.mkdir(parents=True)
    np.savez(p/'dataset.npz', X=np.zeros((3,31),np.float32), Y=np.full((3,6),np.nan,np.float32),
             eidx=np.array([0,-1,1]))
    np.save(p/'emb.npy', np.ones((2,512),np.float16))
    pd.DataFrame(dict(conv=['c']*3, source=['real']*3, split=['train']*3)).to_parquet(p/'meta.parquet')
    d=load('TSA'); assert d['keep'].all()
    windows=eval_windows(conv_index(d), [2]); assert windows[0].tolist() == [0,1,2]
    _,e,_,valid=batchify(d, windows, True)
    assert valid.all() and np.all(e[0,1] == 0) and np.all(e[0,2] == 1)
    assert load('TS')['keep'].tolist() == [True,False,True]


def test_missing_text_projection_does_not_inject_bias():
    from tidal.model import TurnModel
    torch.manual_seed(7)
    m=TurnModel(11,True,mask_text=True).eval()
    feats=torch.randn(1,3,11); valid=torch.ones(1,3,dtype=torch.bool)
    text=torch.zeros(1,3,512)
    with torch.inference_mode():
        a=m(feats,valid,text)
        # With no text, masking must make the entire text projection irrelevant.
        m.tproj[1].bias.fill_(100.)
        b=m(feats,valid,text)
    for h in a: torch.testing.assert_close(a[h],b[h])


def test_demo_uses_requested_gaps_and_isolates_conversations():
    from scripts.demo_sixhead import parse_line, score
    from tidal.model import TurnModel
    a=parse_line('{"text":"hi","gap_s":2}',0,10.)
    assert a['ts'] == 12.
    b=dict(a, msg_id='next',ts=13.,role='self',speaker='BOT')
    outsider=dict(a,conv='different',ts=12.5,text='unrelated')
    ck=dict(regime='T',use_text=False,mu=np.zeros(11),sd=np.ones(11))
    model=TurnModel(11,False).eval()
    assert score([a,b],model,ck) == score([a,outsider,b],model,ck)
