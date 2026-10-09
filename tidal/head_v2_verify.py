"""Verify exported v2 features/models and full independent runtime latency."""
import json
import subprocess
import sys

import numpy as np
from tidal.head_v2_runtime import V2Shadow
from tidal.task_heads import HeadEvent


def verify(directory):
    model=V2Shadow(directory)
    if set(model.models)!={'reply_to_v2','overlap_timing'}:raise AssertionError('Missing v2 models')
    h=[HeadEvent(str(i),float(i),f'u{i%3}: question about audio {i}?',f'u{i%3}') for i in range(32)]
    first=model.reply(h,'u1: the audio setting is here',33.,speaker='u2')
    second=model.reply(h+[HeadEvent('future',40.,'future text')],'u1: the audio setting is here',33.,speaker='u2')
    if first!=second:raise AssertionError('Future text changes previous score')
    obs=dict(history=[[(0.,1.)],[(.5,1.5)]],incoming_start=2.,self_start=1.,decision=2.2,
             incoming_active=True,observed_incoming_duration=.2)
    score=model.overlap_timing(**obs)
    if score is None:raise AssertionError('No timing result')
    script='''
import json,time,resource,sys
from tidal.head_v2_runtime import V2Shadow
from tidal.task_heads import HeadEvent
import numpy as np
m=V2Shadow(sys.argv[1]);h=[HeadEvent(str(i),i*.1,'question about audio '+str(i)+'?','u') for i in range(32)];lat=[];tim=[]
obs=dict(history=[[(0.,1.)],[(.5,1.5)]],incoming_start=2.,self_start=1.,decision=2.2,incoming_active=True,observed_incoming_duration=.2)
for i in range(120):
 now=4.+i*.1;ctx=h+[HeadEvent('new'+str(i),now,'new question '+str(i)+'?','v')]
 t=time.perf_counter();m.reply(ctx,'reply about audio '+str(i),now);lat.append((time.perf_counter()-t)*1000)
 t=time.perf_counter();m.overlap_timing(**obs);tim.append((time.perf_counter()-t)*1000)
print(json.dumps(dict(reply_p95_ms=float(np.quantile(lat[20:],.95)),timing_p95_ms=float(np.quantile(tim[20:],.95)),peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024 if sys.platform=='darwin' else 1024),torch_loaded='torch' in sys.modules,pandas_loaded='pandas' in sys.modules,sklearn_loaded='sklearn' in sys.modules)))
'''
    result=subprocess.run([sys.executable,'-c',script,str(directory)],text=True,capture_output=True,check=True)
    stats=json.loads(result.stdout)
    if any(stats[k] for k in ('torch_loaded','pandas_loaded','sklearn_loaded')):raise AssertionError('Training dependency leaked into runtime')
    return dict(future_text_invariance=True,runtime=stats,scope='Independent v2 features and ONNX, 32 synthetic reply candidates with changing draft/message; excludes VAD/ASR, generation and controller.')
