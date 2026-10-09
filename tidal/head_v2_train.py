"""Fixed v2 search. Fit on train/val, freeze, then construct external test data."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from tidal.head_train import train_model, probabilities, summary, block_delta, disjoint, digest, ExportedTask
from tidal.head_v2_data import _irc_rows, channel2_v2, magic_overlap
from tidal.head_v2_features import REPLY_NAMES,TIMING_NAMES,ReplyFeatures
from tidal.task_heads import MAX_CONTEXT

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'models'/'head_v2';CACHE=ROOT/'data'/'public'/'proc'/'head_v2';REPORT=ROOT/'reports'/'head_v2.json'
CALIBRATION_TEMPERATURES=tuple(np.geomspace(.35,3.5,25).round(6))


def source_signature():
    files=[ROOT/'tidal'/n for n in ('head_v2_features.py','head_v2_data.py','head_v2_runtime.py','head_v2_train.py','head_train.py','head_data.py','task_heads.py')]
    files+=[ROOT/'reports'/'head_v2_protocol.md']
    from tidal.public_data.manifest import DATA
    for directory in ('irc_dis','irc_channel2'):
        files+=list((DATA/directory).rglob('*.parquet'))
    files+=list((DATA/'proc'/'audio_md').glob('*.npz'))+list((DATA/'proc'/'audio_en').glob('*.npz'))
    hashes={str(p.relative_to(ROOT)):digest(p) for p in files}
    return hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest()


def reply_development():
    from tidal.public_data.manifest import DATA
    paths=[ROOT/'tidal'/'head_v2_data.py',ROOT/'tidal'/'head_v2_features.py',ROOT/'tidal'/'task_heads.py',DATA/'irc_dis'/'ubuntu'/'train-00000-of-00001.parquet',DATA/'irc_dis'/'ubuntu'/'validation-00000-of-00001.parquet']
    sig=hashlib.sha256(''.join(digest(p) for p in paths).encode()).hexdigest();CACHE.mkdir(parents=True,exist_ok=True)
    path=CACHE/'reply_development.npz'
    if path.is_file():
        with np.load(path,allow_pickle=False) as z:
            if str(z['signature'])==sig:
                datasets=[{k:z[f'{split}_{k}'].copy() for k in ('x','mask','y','groups')} for split in ('train','val')]
                return *datasets,ReplyFeatures(z['idf'].copy())
    print('Building train-only IDF and reply development features',flush=True)
    train=_irc_rows('train');rf=train.pop('rf');val=_irc_rows('val',rf=rf);val.pop('rf')
    np.savez(path,signature=sig,idf=rf.idf,**{f'{split}_{k}':d[k].astype(str) if k=='groups' else d[k] for split,d in [('train',train),('val',val)] for k in ('x','mask','y','groups')})
    return train,val,rf


def evaluate_model(models,mu,sd,temperature,data,pointer):
    with torch.no_grad():z=torch.stack([m(torch.from_numpy((data['x']-mu)/sd)) for m in models]).mean(0).numpy()
    if pointer:z=z.squeeze(-1)
    return probabilities(z,temperature,data.get('mask'))


def fit(task,train,val,test_loader,sig,rf=None):
    pointer=task=='reply_to_v2';features=REPLY_NAMES if pointer else TIMING_NAMES
    disjoint(train,val)
    flat=train['x'][train['mask']] if pointer else train['x']
    mu=flat.mean(0).astype(np.float32);sd=flat.std(0).clip(.05).astype(np.float32)
    candidates={};temps={};val_loss={}
    kinds=('linear','mlp','mlp64') if pointer else ('linear','mlp')
    for kind in kinds:
        models=[]
        for seed in (0,1,2):
            model,hist=train_model(kind,train,val,mu,sd,1 if pointer else 2,seed,pointer)
            path=OUT/f'{task}_{kind}_{seed}.pt'
            torch.save(dict(state=model.state_dict(),mu=mu,sd=sd,kind=kind,signature=sig,history=hist),path);models.append(model)
        p0=evaluate_model(models,mu,sd,1.,val,pointer)
        # Recover equivalent logits up to a sample-wise constant to calibrate.
        z=np.log(np.maximum(p0,1e-300))
        calibration_grid = CALIBRATION_TEMPERATURES if pointer else (.5,.75,1.,1.5,2.,3.)
        choices=[(summary(probabilities(z,t,val.get('mask')),val['y'])['nll'],t) for t in calibration_grid]
        val_loss[kind],temps[kind]=min(choices);candidates[kind]=models
    champion=min(val_loss,key=val_loss.get)
    if rf is not None:np.save(OUT/'reply_idf.npy',rf.idf)
    manifest=dict(task=task,signature=sig,champion=champion,val_nll=val_loss,temperatures=temps,
                  weights={p.name:digest(p) for p in OUT.glob(f'{task}_*.pt')},idf_sha256=digest(OUT/'reply_idf.npy') if pointer else None)
    frozen=OUT/f'{task}_selection.json';frozen.write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(dict(task=task,phase='frozen',champion=champion)),flush=True)
    # External/new test labels are first parsed here, after freeze.
    test=test_loader();disjoint(train,val,test)
    predictions={kind:evaluate_model(models,mu,sd,temps[kind],test,pointer) for kind,models in candidates.items()}
    p=predictions[champion];baselines={'linear':predictions['linear']}
    if not pointer:
        prior=np.bincount(train['y'],minlength=2)/len(train['y']);baselines['prior']=np.repeat(prior[None],len(p),axis=0)
    else:
        # Fixed v1 is already trained/calibrated; apply its exact first-12 feature contract.
        import onnxruntime as ort
        old=ROOT/'models'/'task_heads'/'reply_to.onnx'
        oldmeta=json.loads((old.parent/'reply_to_bundle.json').read_text())
        if digest(old)!=oldmeta['model_sha256']:raise ValueError('v1 baseline checksum mismatch')
        session=ort.InferenceSession(str(old),providers=['CPUExecutionProvider'])
        baselines['frozen_v1']=session.run(None,{'features':test['x'][...,:12],'mask':test['mask']})[0]
    result=dict(task=task,signature=sig,champion=champion,train_n=len(train['y']),val_n=len(val['y']),test_n=len(test['y']),
                blocks={s:len(set(d['groups'])) for s,d in [('train',train),('val',val),('test',test)]},selected=summary(p,test['y']),
                candidates={kind:summary(v,test['y']) for kind,v in predictions.items()},
                baselines={name:summary(bp,test['y']) for name,bp in baselines.items()},shadow_only=True,automatic_action_enabled=False,
                counts={s:d.get('counts',{}) for s,d in [('train',train),('val',val),('test',test)]},selection_manifest_sha256=digest(frozen))
    if len(set(test['groups']))>=5:
        result['deltas']={name:block_delta(p,bp,test['y'],test['groups']) for name,bp in baselines.items()}
        result['neural_passed_gate']=champion!='linear' and all(d['accuracy_improvement_ci'][0]>0 and d['nll_improvement_ci'][0]>0 for d in result['deltas'].values())
    else:result.update(neural_passed_gate=False,limitation='Single external log; point metrics only, no claim of statistical adoption.')
    np.savez(OUT/f'{task}_predictions.npz',p=p,y=test['y'],groups=test['groups'].astype(str))
    model=ExportedTask(candidates[champion],mu,sd,temps[champion],pointer).eval()
    sample=torch.from_numpy(test['x'][:4]);args=(sample,torch.from_numpy(test['mask'][:4])) if pointer else (sample,)
    names=['features','mask'] if pointer else ['features'];path=OUT/f'{task}.onnx'
    torch.onnx.export(model,args,str(path),input_names=names,output_names=['probabilities'],dynamic_axes={n:{0:'batch'} for n in names+['probabilities']},opset_version=17,dynamo=False)
    import onnxruntime as ort
    sess=ort.InferenceSession(str(path),providers=['CPUExecutionProvider']);got=sess.run(None,{name:a.numpy() for name,a in zip(names,args)})[0]
    with torch.no_grad():error=float(np.max(np.abs(got-model(*args).numpy())))
    if error>1e-5:raise RuntimeError('ONNX parity failed')
    meta=dict(schema_version=2,task=task,feature_names=list(features),feature_contract='task_heads_v2',max_context=MAX_CONTEXT,
              model_file=path.name,model_sha256=digest(path),shadow_only=True,automatic_action_enabled=False,onnx_parity_max_error=error,
              signature=sig,selection_manifest_sha256=digest(frozen),parameters=sum(sum(t.numel() for t in m.parameters()) for m in candidates[champion]),
              research_only=not pointer,classes=[] if pointer else ['backchannel','floor_claim'])
    if pointer:meta.update(idf_file='reply_idf.npy',idf_sha256=digest(OUT/'reply_idf.npy'))
    (OUT/f'{task}_bundle.json').write_text(json.dumps(meta,indent=2)+'\n');result['export']=meta
    if pointer:
        # Previously exposed Ubuntu test is a regression diagnostic, never a new selection gate.
        known=_irc_rows('test',rf=rf)
        kp=evaluate_model(candidates[champion],mu,sd,temps[champion],known,True)
        op=session.run(None,{'features':known['x'][...,:12],'mask':known['mask']})[0]
        result['known_ubuntu_diagnostic']=dict(independent_new_test=False,selected=summary(kp,known['y']),frozen_v1=summary(op,known['y']))
    return result


def run():
    torch.set_num_threads(4)
    from tidal.public_data.download import main as download
    download(['irc_channel2'],workers=1)
    OUT.mkdir(parents=True,exist_ok=True);sig=source_signature()
    old=json.loads(REPORT.read_text()) if REPORT.is_file() else {}
    if old.get('signature')==sig and old.get('artifacts') and all((OUT/name).is_file() and digest(OUT/name)==sha for name,sha in old['artifacts'].items()):
        result=old;print('Resuming validated v2 artifacts',flush=True)
    else:
        tr,va,rf=reply_development()
        reply=fit('reply_to_v2',tr,va,lambda:channel2_v2(rf),sig,rf=rf)
        print('Building MagicData causal overlap development data',flush=True)
        train,val=magic_overlap('train'),magic_overlap('val')
        overlap=fit('overlap_timing',train,val,lambda:magic_overlap('test'),sig)
        result=dict(signature=sig,reply_to=reply,overlap_timing=overlap,shadow_only=True,automatic_action_enabled=False,
                    artifacts={p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.suffix in ('.pt','.npy','.onnx','.json','.npz')})
        REPORT.write_text(json.dumps(result,indent=2)+'\n')
    from tidal.head_v2_verify import verify
    result['runtime_checks']=verify(OUT)
    test=subprocess.run([sys.executable,'-m','pytest','-q'],cwd=ROOT,text=True,capture_output=True)
    (OUT/'pytest.log').write_text(test.stdout+test.stderr)
    if test.returncode:raise RuntimeError('Full tests failed: see local pytest.log')
    result['checks']=dict(pytest_exit_code=0,summary=test.stdout.splitlines()[-1])
    REPORT.write_text(json.dumps(result,indent=2)+'\n')
    from tidal.head_v2_report import markdown
    (ROOT/'reports'/'head_v2.md').write_text(markdown(result))
    print(result['checks']['summary'],flush=True);print(ROOT/'reports'/'head_v2.md',flush=True)
    return result

if __name__=='__main__':run()
