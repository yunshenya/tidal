"""Verify actual bundles, causal chunking, future invariance and independent runtime cost."""
import json
from pathlib import Path
import resource
import subprocess
import sys
import time

import numpy as np
from tidal.task_heads import HeadEvent, TaskHeadShadow, OverlapStream


def verify(directory):
    model = TaskHeadShadow(directory)
    if not {'reply_to', 'overlap_outcome'} <= set(model.models): raise AssertionError('Required trained bundles missing')
    context = [HeadEvent(f'e{i}', i*.1, f'user{i%3}: question about audio {i}?', f'user{i%3}') for i in range(32)]
    first = model.reply(context, 'user1: the audio setting is here', 4.)
    second = model.reply(context+[HeadEvent('future', 10., 'future answer')], 'user1: the audio setting is here', 4.)
    if first != second: raise AssertionError('Future text changes prior reply score')
    rng = np.random.default_rng(600); a = rng.normal(0, .01, 32000).astype(np.float32); b = rng.normal(0, .02, 32000).astype(np.float32)
    scores = []
    for sizes in ([32000], [1600]*20, [700, 321, 7000, 1111, 9000, 13868]):
        stream = OverlapStream(model); pos = 0
        for size in sizes:
            stream.push(a[pos:pos+size], b[pos:pos+size]); pos += size
        scores.append(stream.score(2., 1.8, .2, True))
    if any(s is None for s in scores): raise AssertionError('Actual overlap bundle not scoreable')
    p = [np.array(list(s['probabilities'].values())) for s in scores]
    error = max(float(np.max(np.abs(v-p[0]))) for v in p)
    if error > 1e-6: raise AssertionError('PCM chunk invariance failed')
    # Frame history retains the previous decision when up to 0.5s of future PCM arrives.
    stream.push(rng.normal(0, .4, 8000).astype(np.float32), rng.normal(0, .4, 8000).astype(np.float32))
    future = stream.score(2., 1.8, .2, True)
    if future != scores[-1]: raise AssertionError('Future PCM changes a past score')
    if len(stream.frames) > 100 or max(map(len, stream.buffers)) > 400: raise AssertionError('Unbounded audio buffer')
    script = '''
import json,resource,sys,time
from tidal.task_heads import HeadEvent,TaskHeadShadow
m=TaskHeadShadow(sys.argv[1]); h=[HeadEvent(str(i), i*.1, 'question about audio?', 'u') for i in range(32)]
latency=[]
for i in range(120):
 now=4.+i*.1;context=h+[HeadEvent('new'+str(i),now,'new question '+str(i)+'?', 'v')]
 t=time.perf_counter();m.reply(context,'reply about audio '+str(i),now);latency.append((time.perf_counter()-t)*1000)
import numpy as np
print(json.dumps(dict(reply_full_p95_ms=float(np.quantile(latency[20:],.95)),peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024*1024 if sys.platform=='darwin' else 1024),torch_loaded='torch' in sys.modules,pandas_loaded='pandas' in sys.modules,sklearn_loaded='sklearn' in sys.modules)))
'''
    result = subprocess.run([sys.executable, '-c', script, str(directory)], text=True, capture_output=True, check=True)
    stats = json.loads(result.stdout)
    if any(stats[k] for k in ('torch_loaded', 'pandas_loaded', 'sklearn_loaded')): raise AssertionError('Heavy training dependencies imported')
    return dict(max_chunk_probability_error=error, future_text_invariance=True, future_pcm_invariance=True,
                bounded_audio_buffer=True, runtime=stats, benchmark_scope='32 synthetic candidates with a new message and draft each iteration; feature construction and ONNX only; no VAD/ASR/content generation')
