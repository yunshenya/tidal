"""Version-2 task data: train-only weighted reply features and causal MagicData overlap observations."""
import re, hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from tidal.public_data.manifest import DATA
from tidal.task_heads import HeadEvent, MAX_CONTEXT, context_at
from tidal.head_data import annotation_split, _IRC, AUDIO_GROUPS, overlap_points
from tidal.head_v2_features import ReplyFeatures, fit_idf, reference_observation, timing_features

def _irc_rows(split, channel=False, rf=None):
    name='test-00000-of-00001.parquet' if channel else ('validation-00000-of-00001.parquet' if split=='val' else f'{split}-00000-of-00001.parquet')
    sub='channel_two' if channel else 'ubuntu'; df=pd.read_parquet(DATA/('irc_channel2' if channel else 'irc_dis')/sub/name)
    rows=[]; texts=[]
    for r in df.itertuples():
        m=_IRC.match(r.raw)
        if m: texts.append(m.group(4))
    rf=rf or ReplyFeatures(fit_idf(texts))
    groups=[]; xs=[]; masks=[]; ys=[]; counts={'queries':0,'skipped':0,'outside':0}
    for group, frame in df.groupby('date', sort=True) if not channel else [('channel_two',df)]:
        past=[]; last=-1; offset=0
        for r in frame.sort_values('id').itertuples():
            m=_IRC.match(r.raw)
            if not m: counts['skipped']+=1; continue
            hh,mm,nick,text=m.groups(); minute=int(hh)*60+int(mm)
            if minute<last: offset+=86400
            last=minute; now=offset+minute*60.
            links={int(k) for k in r.connections if int(k)<=int(r.id)}
            if links:
                x,mask,ids=rf.build(past,text,now,nick); gold=np.zeros(MAX_CONTEXT+1,bool); visible={int(k):i for i,k in enumerate(ids) if k is not None}
                for k in links:
                    if k==int(r.id) or k not in visible: gold[0]=True
                    else: gold[visible[k]]=True
                xs.append(x);masks.append(mask);ys.append(gold);groups.append(str(group));counts['queries']+=1
                counts['outside']+=int(any(k!=int(r.id) and k not in visible for k in links))
            else: counts['skipped']+=1
            past.append(HeadEvent(str(int(r.id)),now,text,nick,'other'));past=past[-MAX_CONTEXT:]
    return dict(x=np.asarray(xs,np.float32),mask=np.asarray(masks),y=np.asarray(ys),groups=np.asarray(groups,object),counts=counts,rf=rf)

def reply_v2(split):
    train_raw=_irc_rows('train',rf=None); rf=train_raw['rf']
    if split=='train': out=train_raw
    elif split=='val': out=_irc_rows('val',rf=rf)
    else: out=_irc_rows('test',rf=rf)
    out['idf']=rf.idf; return out

def channel2_v2(rf):
    out=_irc_rows('test',channel=True,rf=rf); return out

def _conv_split(cid):
    b=int(hashlib.sha256(str(cid).encode()).hexdigest()[:8],16)%100
    return 'train' if b<70 else 'val' if b<85 else 'test'

def magic_overlap(split):
    """Causal overlap timing from the fixed MagicData/MagicHub group split.

    The corpora provide two synchronized channels and segment boundaries; labels remain
    weak natural outcomes. No future endpoint enters reference_observation features.
    """
    from tidal.audio_train import CACHE_VERSION
    from tidal.audio_spec import FRAME_READY, STEP
    groups={'train':{'magicdata_ms':('A1012','A1091'),'magicdata_en':('Group0006','Group0030')},
            'val':{'magicdata_ms':('A1102',),'magicdata_en':('Group0046',)},
            'test':{'magicdata_en':('Group0078',)}}[split]
    xs=[];ys=[];group_ids=[];speakers=set();counts={'candidates':0,'ambiguous':0}
    for source,names in groups.items():
        root=DATA/source;cache=DATA/'proc'/('audio_md' if source=='magicdata_ms' else 'audio_en')
        for path in sorted(cache.glob('*.npz')):
            conv=path.stem
            if conv.split('_')[0] not in names:continue
            with np.load(path) as z:
                if int(z.get('cache_version',-1))!=CACHE_VERSION:raise ValueError('Stale audio cache')
                seg=[json.loads(str(z['s0'])),json.loads(str(z['s1']))]
                observed=(len(z['m0'])//2-1)*STEP+FRAME_READY
            found=[f'{source}:{p.stem.rsplit("_",1)[-1]}' for p in (root/'TXT').glob(f'{conv}_0_*.txt')]
            if len(found)!=2:raise ValueError('Speaker IDs unavailable')
            speakers.update(found)
            for point in overlap_points(seg,observed):
                counts['candidates']+=1
                if point['label'] is None:counts['ambiguous']+=1;continue
                try: x=timing_features(**reference_observation(seg,point))
                except ValueError:continue
                xs.append(x);ys.append(int(point['label']));group_ids.append(f'{source}:{conv}')
    if not xs:raise ValueError(f'No {split} overlap examples')
    return dict(x=np.asarray(xs,np.float32),y=np.asarray(ys,np.int64),groups=np.asarray(group_ids,object),speakers=sorted(speakers),counts=counts)
