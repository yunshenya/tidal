"""Verify real frozen bundles without fitting or inspecting test labels."""
import argparse
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np

from tidal.audio_spec import SR, STEP, FRAME_READY, stack_frames, numpy_logmel
from tidal.turn_features import CONTEXT, timing_features, prosody_features
from tidal.turn_shadow import AudioTurnShadow, TimingTurnShadow

ROOT=Path(__file__).resolve().parents[1]
STORE=ROOT/'models'/'turn_v2'


def verify_runtime():
    rng=np.random.default_rng(11)
    other=rng.normal(0,.05,SR*4).astype(np.float32)
    own=rng.normal(0,.01,SR*4).astype(np.float32)
    other[3*SR:]=0.;own[SR//2:]=0.
    segments=[[(.5,3.,'long statement')],[(.2,.4,'brief speech')]]
    decision=3.2
    stream=AudioTurnShadow(STORE/'audio_champion.json')
    stop=int(SR*3.3)
    for k in range(0,stop,SR//10):
        stream.push(other[k:min(k+SR//10,stop)],own[k:min(k+SR//10,stop)])
    online=stream.score(segments,decision)
    if online is None:
        raise AssertionError('Complete causal context did not produce a shadow score')
    mel=[numpy_logmel(pcm[:stop]).astype(np.float16).astype(np.float32) for pcm in (other,own)]
    x=stack_frames(*mel)
    k=int(np.floor((decision-FRAME_READY+1e-10)/STEP)); window=x[k-CONTEXT+1:k+1]
    features=np.r_[timing_features(segments,1,decision,2.5),prosody_features(window)]
    batch=stream.model.predict(features,window)
    uneven=AudioTurnShadow(STORE/'audio_champion.json')
    pos=0; sizes=(1,7,401,913,1600,123)
    while pos<stop:
        size=sizes[pos%len(sizes)];end=min(pos+size,stop)
        uneven.push(other[pos:end],own[pos:end]);pos=end
    split=uneven.score(segments,decision)
    if split is None:
        raise AssertionError('Partitioned complete context missing')
    error=max(abs(online['p_take']-batch['p_take']),abs(online['p_take']-split['p_take']))
    if error>1e-4:
        raise AssertionError(f'PCM chunk parity failed: {error}')
    # PCM and reference events after decision must not change its shadow prediction.
    extra=other[stop:].copy();extra[:]=.8
    stream.push(extra,own[stop:])
    future=[segments[0]+[(3.5,4.,'future words')],segments[1]+[(3.4,4.,'later')]]
    after=stream.score(future,decision)
    if after is None or abs(after['p_take']-online['p_take'])>1e-6:
        raise AssertionError('Future PCM/events affected a past score')
    if len(stream.frames)>CONTEXT+50:
        raise AssertionError('Unbounded acoustic context')
    timing=TimingTurnShadow(STORE/'timing_champion.json')
    p=timing.score(segments,decision)
    if p is None or not np.isfinite(p['p_take']):
        raise AssertionError('Timing bundle did not score an eligible opportunity')
    if timing.score(segments,decision+.1) is not None:
        raise AssertionError('Off-protocol timing opportunity was scored')
    return dict(max_chunk_probability_error=error,future_invariance=True,bounded_context=True,
                shadow_only=True,automatic_action_enabled=False)


def benchmark(task):
    path=STORE/f'{task}_champion.json'
    segments=[[],[]]
    for i in range(100):
        segments[i%2].append((i*3.,i*3.+.9,'past speech'))
    segments[0].append((300.5,303.,'current statement'))
    decision=303.2
    if task=='audio':
        model=AudioTurnShadow(path);model.reset(start_time=300.)
        rng=np.random.default_rng(0)
        pcm=rng.normal(0,.05,int(SR*3.3)).astype(np.float32); pcm[3*SR:]=0.
        for k in range(0,len(pcm),SR//10):
            model.push(pcm[k:k+SR//10],np.zeros_like(pcm[k:k+SR//10]))
    else:
        model=TimingTurnShadow(path)
    for _ in range(10):model.score(segments,decision)
    times=[]
    for _ in range(100):
        start=time.perf_counter();model.score(segments,decision);times.append((time.perf_counter()-start)*1000)
    combined=[]
    if task=='audio':
        chunk=np.zeros(SR//10,np.float32)
        for i in range(100):
            current=300.+(len(pcm)+(i+1)*len(chunk))/SR
            start=time.perf_counter()
            model.push(chunk,chunk)
            segments[0].append((current-.7,current-.2,'current statement'))
            output=model.score(segments,current)
            if output is None:
                raise AssertionError('Benchmark tick did not score an eligible candidate')
            combined.append((time.perf_counter()-start)*1000)
    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_mb=rss/(1024**2 if sys.platform=='darwin' else 1024)
    return dict(task=task,p50_ms=float(np.median(times)),p95_ms=float(np.quantile(times,.95)),
                combined_tick_p95_ms=float(np.quantile(combined,.95)) if combined else None,
                peak_rss_mb=float(rss_mb),torch_loaded='torch' in sys.modules,
                sklearn_loaded='sklearn' in sys.modules,pandas_loaded='pandas' in sys.modules,
                scope='Fresh process; synthetic 100-segment history; score includes causal statistics and ONNX. Combined audio tick includes 100ms PCM log-mel. Excludes VAD/ASR.')


def verify_all():
    checks=verify_runtime();measurements={}
    for task in ('timing','audio'):
        output=subprocess.check_output([sys.executable,'-m','tidal.turn_verify','--bench',task],text=True,cwd=ROOT)
        result=json.loads(output)
        if result['torch_loaded'] or result['sklearn_loaded'] or result['pandas_loaded']:
            raise AssertionError('Training dependency was imported by the runtime')
        measurements[task]=result
    result=dict(checks=checks,fresh_process=measurements)
    (ROOT/'reports'/'turn_v2_runtime.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    print(json.dumps(result),flush=True)
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--bench',choices=('audio','timing'))
    arg=p.parse_args()
    if arg.bench:print(json.dumps(benchmark(arg.bench)),flush=True)
    else:verify_all()
