"""Optional v2 reply/timing heads, ONNX only; no control action selection."""
import hashlib
import json
from pathlib import Path

import numpy as np
from tidal.task_heads import MAX_CONTEXT, context_at
from tidal.head_v2_features import ReplyFeatures, REPLY_NAMES, TIMING_NAMES, timing_features


class V2Shadow:
    def __init__(self, directory):
        import onnxruntime as ort
        directory=Path(directory)
        if not directory.is_dir(): raise FileNotFoundError(directory)
        self.models={};self.metadata={};self.reply_features=None
        for task,features in [('reply_to_v2',REPLY_NAMES),('overlap_timing',TIMING_NAMES)]:
            path=directory/f'{task}_bundle.json'
            if not path.is_file(): continue
            meta=json.loads(path.read_text())
            if (meta.get('schema_version')!=2 or meta.get('task')!=task or meta.get('feature_names')!=list(features)
                or meta.get('feature_contract')!='task_heads_v2' or meta.get('shadow_only') is not True
                or meta.get('automatic_action_enabled') is not False or meta.get('max_context')!=MAX_CONTEXT):
                raise ValueError('Invalid v2 feature contract')
            expected_classes=[] if task=='reply_to_v2' else ['backchannel','floor_claim']
            if meta.get('classes')!=expected_classes:raise ValueError('Invalid class schema')
            name=meta['model_file']
            if Path(name).name!=name: raise ValueError('Invalid model file')
            model=directory/name
            if hashlib.sha256(model.read_bytes()).hexdigest()!=meta['model_sha256']:raise ValueError('Model checksum mismatch')
            so=ort.SessionOptions();so.intra_op_num_threads=2;so.inter_op_num_threads=1
            sess=ort.InferenceSession(str(model),sess_options=so,providers=['CPUExecutionProvider'])
            names=['features','mask'] if task=='reply_to_v2' else ['features']
            if [i.name for i in sess.get_inputs()]!=names:raise ValueError('Input schema mismatch')
            if task=='reply_to_v2':
                idf_name=meta['idf_file']
                if Path(idf_name).name!=idf_name:raise ValueError('Invalid IDF path')
                idf=directory/idf_name
                if hashlib.sha256(idf.read_bytes()).hexdigest()!=meta['idf_sha256']:raise ValueError('IDF checksum mismatch')
                self.reply_features=ReplyFeatures(np.load(idf,allow_pickle=False))
            self.models[task]=sess;self.metadata[task]=meta

    def _predict(self,task,x,mask=None):
        if task not in self.models:return None
        x=np.asarray(x,np.float32)
        if not np.isfinite(x).all():raise ValueError('Invalid features')
        feed={'features':x[None]}
        if mask is not None:feed['mask']=np.asarray(mask,bool)[None]
        p=self.models[task].run(None,feed)[0][0]
        n=MAX_CONTEXT+1 if mask is not None else 2
        if p.shape!=(n,) or not np.isfinite(p).all() or np.any(p<0) or not np.isclose(p.sum(),1.,atol=1e-5):raise ValueError('Invalid model probabilities')
        if mask is not None and np.any(p[~mask]>1e-6):raise ValueError('Mask violation')
        return p

    def reply(self,context,draft,now,speaker=''):
        if draft is None:return None
        if not isinstance(draft,str):raise ValueError('Expected draft text')
        if not draft.strip() or self.reply_features is None:return None
        x,mask,ids=self.reply_features.build(context,draft,now,speaker)
        p=self._predict('reply_to_v2',x,mask)
        if p is None:return None
        return dict(target=ids[int(p.argmax())],probabilities={str(k) if k is not None else '__null__':float(p[i]) for i,k in enumerate(ids)},
                    shadow_only=True,draft_conditioned=True,feature_version=2)

    def overlap_timing(self,**observation):
        p=self._predict('overlap_timing',timing_features(**observation))
        return None if p is None else dict(probabilities=dict(zip(('backchannel','floor_claim'),map(float,p))),shadow_only=True,weak_behavior_proxy=True)
