"""Opt-in ONNX shadow scores; no action selection or TTS side effects.

AudioTurnShadow.push(other_pcm, self_pcm) retains a bounded causal feature ring.
score(segments, decision_time) uses reference/VAD segment ends supplied by caller.
Training used reference boundaries; VAD accuracy is not established by this API.
The model uses an exact bounded context, not an unbounded recurrent state.
"""
from collections import deque
import json
import hashlib
from pathlib import Path

import numpy as np
import onnxruntime as ort

from tidal.audio_spec import SR, HOP, WIN, STACK, STEP, FRAME_READY, numpy_logmel, stack_frames
from tidal.turn_features import CONTEXT, TIMING_NAMES, PROSODY_NAMES, short_feedback, timing_features, prosody_features


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class TurnSession:
    def __init__(self, metadata_path):
        metadata_path=Path(metadata_path)
        self.metadata=json.loads(metadata_path.read_text())
        if self.metadata.get('schema_version')!=1 or self.metadata.get('shadow_only') is not True:
            raise ValueError('Expected a versioned shadow-only turn bundle')
        task = self.metadata.get('task')
        names = list(TIMING_NAMES + (PROSODY_NAMES if task=='audio' else ()))
        if task not in ('audio','timing') or self.metadata.get('feature_names') != names:
            raise ValueError('Turn feature schema mismatch')
        if self.metadata.get('context') != CONTEXT or self.metadata.get('channel_order') != ['other','self']:
            raise ValueError('Turn context schema mismatch')
        if self.metadata.get('frame_ready') != FRAME_READY or self.metadata.get('step') != STEP:
            raise ValueError('Turn frame clock mismatch')
        model=metadata_path.parent/self.metadata['model_file']
        if digest(model)!=self.metadata['model_sha256']:
            raise ValueError('Turn model checksum mismatch')
        if self.metadata.get('research_only') is not True or self.metadata.get('automatic_action_enabled') is not False:
            raise ValueError('Invalid research shadow metadata')
        if type(self.metadata.get('neural_passed_gate')) is not bool:
            raise ValueError('Invalid turn gate metadata')
        options=ort.SessionOptions();options.intra_op_num_threads=2;options.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(model),options,providers=['CPUExecutionProvider'])
        if [i.name for i in self.session.get_inputs()]!=self.metadata['input_names']:
            raise ValueError('ONNX input schema mismatch')

    def predict(self, features, window=None):
        features=np.asarray(features,np.float32).reshape(1,-1)
        if features.shape[1]!=len(self.metadata['feature_names']) or not np.isfinite(features).all():
            raise ValueError('Invalid turn features')
        feed={'features':features}
        if 'window' in self.metadata['input_names']:
            window=np.asarray(window,np.float32)
            if window.shape!=(CONTEXT,160) or not np.isfinite(window).all():
                raise ValueError('Expected a finite complete audio context')
            feed['window']=window[None]
        output=self.session.run(None,feed)[0]
        if output.shape != (1,):
            raise ValueError('Unexpected turn output shape')
        p=float(output[0])
        if not np.isfinite(p) or not 0<=p<=1:
            raise ValueError('Nonfinite turn score')
        return dict(p_take=p, shadow_only=True,
                    neural_passed_gate=bool(self.metadata['neural_passed_gate']))


def candidate_history(segments, decision_time):
    """Validate the exact +200ms opportunity, using only observed history."""
    if len(segments)!=2 or not np.isfinite(decision_time):
        raise ValueError('Expected other/self segment histories and a finite time')
    clean=[]
    for channel in segments:
        for start,end,text in channel:
            if not np.isfinite(start) or not np.isfinite(end) or start<0 or end<=start:
                raise ValueError('Invalid speech segment')
        clean.append([s for s in channel if s[0]<=decision_time])
    other=[s for s in clean[0] if s[1]<=decision_time and not short_feedback(s)]
    if not other:
        return None
    latest=max(other,key=lambda s:s[1]); end=latest[1]
    if abs(decision_time-(end+.2))>1e-6:
        return None
    if any(s[1]>end and s[0]<=decision_time for channel in clean for s in channel):
        return None
    return latest[1]-latest[0]


class TimingTurnShadow:
    def __init__(self, metadata_path):
        self.model=TurnSession(metadata_path)
        if self.model.metadata['task']!='timing':
            raise ValueError('Expected a timing turn bundle')
    def score(self, segments, decision_time):
        duration=candidate_history(segments,decision_time)
        if duration is None:
            return None
        features=timing_features(segments,1,decision_time,duration)
        return self.model.predict(features)


class AudioTurnShadow:
    def __init__(self, metadata_path):
        self.model=TurnSession(metadata_path)
        if self.model.metadata['task']!='audio':
            raise ValueError('Expected an audio turn bundle')
        self.reset()
    def reset(self, start_time=0.):
        if not np.isfinite(start_time):
            raise ValueError('Invalid stream start time')
        self.start_time=float(start_time);self.next_step=0
        self.buffer=[np.empty(0,np.float32),np.empty(0,np.float32)]
        self.received=[0,0]
        # Allow a decision to be scored up to one second after PCM arrival.
        self.frames=deque(maxlen=CONTEXT+50)
    def push(self, other_pcm, self_pcm):
        incoming=[np.asarray(pcm,np.float32) for pcm in (other_pcm,self_pcm)]
        for pcm in incoming:
            if pcm.ndim!=1 or not np.isfinite(pcm).all():
                raise ValueError('PCM must be finite mono arrays')
        lengths=[self.received[i]+len(pcm) for i,pcm in enumerate(incoming)]
        if abs(lengths[0]-lengths[1]) > SR*5:
            raise ValueError('Duplex channels differ by more than five seconds')
        for i,pcm in enumerate(incoming):
            self.received[i]+=len(pcm)
            self.buffer[i]=np.concatenate([self.buffer[i],pcm])
        n=min(map(len,self.buffer))
        frames=max(0,(n-WIN)//HOP+1)//STACK*STACK
        if not frames:
            return False
        end=(frames-1)*HOP+WIN
        mels=[numpy_logmel(b[:end]).astype(np.float16).astype(np.float32) for b in self.buffer]
        x=stack_frames(*mels)
        for row in x:
            self.frames.append((self.next_step,row.copy()));self.next_step+=1
        self.buffer=[b[frames*HOP:] for b in self.buffer]
        return len(self.frames)>=CONTEXT
    def score(self, segments, decision_time):
        duration=candidate_history(segments,decision_time)
        if duration is None:
            return None
        if decision_time>self.start_time+min(self.received)/SR:
            return None
        rows=[(k,x) for k,x in self.frames
              if self.start_time+k*STEP+FRAME_READY<=decision_time+1e-10]
        if len(rows)<CONTEXT:
            return None
        selected=rows[-CONTEXT:]
        if selected[-1][0]-selected[0][0]!=CONTEXT-1:
            return None
        window=np.stack([x for _,x in selected])
        features=np.r_[timing_features(segments,1,decision_time,duration),prosody_features(window)]
        return self.model.predict(features,window)
