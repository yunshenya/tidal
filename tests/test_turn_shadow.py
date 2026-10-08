import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch
from torch import nn

from tidal.audio_spec import SR, FRAME_READY, STEP, numpy_logmel
from tidal.audio_fe import logmel
from tidal.turn_features import TIMING_NAMES, PROSODY_NAMES, CONTEXT
from tidal.turn_shadow import AudioTurnShadow, TimingTurnShadow, TurnSession, candidate_history, digest


@pytest.fixture(scope='module')
def bundles(tmp_path_factory):
    folder=tmp_path_factory.mktemp('turn_bundles');result={}
    for task in ('audio','timing'):
        names=list(TIMING_NAMES+(PROSODY_NAMES if task=='audio' else ()))
        class Classifier(nn.Module):
            def forward(self, features):
                return torch.sigmoid(features.sum(1)*.02)
        model=folder/f'{task}.onnx'
        torch.onnx.export(Classifier(),torch.zeros(1,len(names)),model,
                          input_names=['features'],output_names=['probability'],
                          dynamic_axes={'features':{0:'batch'},'probability':{0:'batch'}},
                          opset_version=17,dynamo=False)
        meta=dict(schema_version=1,task=task,shadow_only=True,research_only=True,
                  automatic_action_enabled=False,model_file=model.name,model_sha256=digest(model),
                  feature_names=names,input_names=['features'],context=CONTEXT,
                  frame_ready=FRAME_READY,step=STEP,channel_order=['other','self'],
                  neural_passed_gate=False)
        path=folder/f'{task}.json';path.write_text(json.dumps(meta));result[task]=path
    return result


def test_runtime_does_not_import_training_dependencies():
    root=Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable,'-c',"import tidal.turn_shadow,sys; assert not {'torch','sklearn','pandas'} & set(sys.modules)"],cwd=root,check=True)


def test_numpy_audio_frontend_matches_training_geometry():
    rng=np.random.default_rng(0)
    for signal in (np.zeros(SR,np.float32),rng.normal(0,.05,SR).astype(np.float32)):
        assert np.allclose(numpy_logmel(signal),logmel(signal),atol=1e-5)
    assert numpy_logmel(np.zeros(399,np.float32)).shape==(0,40)


def test_audio_scores_ignore_chunk_boundaries_and_future(bundles):
    rng=np.random.default_rng(1)
    o=rng.normal(0,.05,int(SR*3.3)).astype(np.float32)
    s=rng.normal(0,.01,len(o)).astype(np.float32)
    histories=[[(.5,3.,'statement')],[(.1,.4,'answer')]]
    a,b=AudioTurnShadow(bundles['audio']),AudioTurnShadow(bundles['audio'])
    a.push(o,s)
    for k in range(0,len(o),701):b.push(o[k:k+701],s[k:k+701])
    pa,pb=a.score(histories,3.2),b.score(histories,3.2)
    assert pa and pb and pa['shadow_only']
    assert abs(pa['p_take']-pb['p_take'])<1e-4
    a.push(np.full(SR//2,.8,np.float32),np.zeros(SR//2,np.float32))
    future=[histories[0]+[(3.4,4.,'future')],histories[1]+[(3.5,4.,'later')]]
    assert a.score(future,3.2)['p_take']==pa['p_take']
    assert len(a.frames)<=CONTEXT+50


def test_incomplete_context_and_off_protocol_are_not_scored(bundles):
    m=AudioTurnShadow(bundles['audio'])
    m.push(np.zeros(SR*2,np.float32),np.zeros(SR*2,np.float32))
    assert m.score([[(.1,1.8,'long phrase')],[]],2.) is None
    assert m.score([[(.1,3.,'long phrase')],[]],3.2) is None
    t=TimingTurnShadow(bundles['timing'])
    assert t.score([[(0.,1.,'long phrase')],[]],1.2)
    assert t.score([[(0.,1.,'long phrase')],[]],1.3) is None


def test_bad_pcm_does_not_partially_advance_stream(bundles):
    m=AudioTurnShadow(bundles['audio'])
    with pytest.raises(ValueError):m.push(np.zeros(100,np.float32),np.array([np.nan],np.float32))
    assert m.received==[0,0] and all(len(b)==0 for b in m.buffer)
    with pytest.raises(ValueError,match='five seconds'):
        m.push(np.zeros(SR*5+1,np.float32),np.empty(0,np.float32))
    assert m.received==[0,0]


def test_checksum_and_feature_schema_are_checked(bundles,tmp_path):
    meta=json.loads(bundles['timing'].read_text())
    meta['model_file']=str(bundles['timing'].parent/meta['model_file'])
    path=tmp_path/'broken.json'
    meta['model_sha256']='wrong';path.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='checksum'):
        TurnSession(path)
    meta['model_sha256']=digest(meta['model_file']);meta['feature_names']=meta['feature_names'][::-1]
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError,match='feature schema'):
        TurnSession(path)


def test_candidate_cannot_use_unfinished_speech():
    assert candidate_history([[(0.,1.,'statement')],[(1.1,1.3,'overlap')]],1.2) is None
    assert candidate_history([[(0.,1.,'statement'),(.9,1.3,'overlap')],[]],1.2) is None
    assert candidate_history([[(0.,1.,'statement')],[]],1.2)==1.
