"""Build phase-5 rows: interrupt / message-value / topic labels and the 28-d encoder input.

Nothing here is a lexicon. Text is used only as the frozen bge input to the already-adopted
text-emotion block on the NEW topic dialogues (public p3 rows keep the phase-4 emotion block,
which is zero on public chat). Outputs stay under data/proc/ (gitignored).
"""
import glob, json, os, zipfile
import numpy as np, pandas as pd
from tidal.public_data.manifest import DATA
from tidal.phase5_labels import (
    speech_interrupt, dyad_message_value, text_interjection, reply_message_value,
    domain_change_labels, nonspeech_span, split_hash,
)
from tidal import features, features_g as FG

PROC = "data/proc"
LIVE_SOURCES = {"pub_twitch", "pub_artemis", "pub_danmaku"}


def _rate(y):
    y = np.asarray(y, float)
    m = ~np.isnan(y)
    return dict(n=int(m.sum()), pos=int(np.nansum(y)), rate=round(float(np.nanmean(y)) if m.any() else float("nan"), 4))


def candor_labels():
    """msg_id -> (y_interrupt, y_addr) using offset/duration only. Same conv ids as phase 3."""
    c = pd.read_parquet(DATA / "candor_tt" / "data" / "train-00000-of-00001.parquet",
                        columns=["conversation_id", "channel", "offset", "duration"])
    rng = np.random.default_rng(5)
    convs = sorted(c.conversation_id.unique())
    convs = list(rng.choice(convs, min(700, len(convs)), replace=False))
    sub = c[c.conversation_id.isin(set(convs))]
    out = {}
    for i, (_cid, g) in enumerate(sub.groupby("conversation_id")):
        g = g.assign(start=g.offset.astype(float), end=(g.offset + g.duration).astype(float))
        g = g.sort_values("end", kind="stable")
        segs = list(zip(g.start.to_numpy(), g.end.to_numpy(), g.channel.astype(str).to_numpy()))
        yi, _k = speech_interrupt(segs)
        yv = dyad_message_value(segs)
        for k in range(len(segs)):
            out[f"cd:{i}:{k}"] = (float(yi[k]), float(yv[k]))
    return out


def aishell_labels():
    out = {}
    paths = sorted(glob.glob(str(DATA / "aishell4_seg" / "**" / "*.rttm"), recursive=True))
    i_kept = 0
    for p in paths:
        rows = [l.split() for l in open(p) if l.startswith("SPEAKER")]
        if len(rows) < 30:
            continue
        segs = [(float(r[3]), float(r[3]) + float(r[4]), r[7]) for r in rows]
        # phase-3 convert sorts by end time and numbers msg_id in that order
        order = sorted(range(len(segs)), key=lambda k: (segs[k][1], segs[k][0], k))
        segs_e = [segs[k] for k in order]
        yi, _k = speech_interrupt(segs_e)
        # meetings are not dyads: message-value is defined for dyads only
        for k in range(len(segs_e)):
            out[f"a4:{i_kept}:{k}"] = (float(yi[k]), float("nan"))
        i_kept += 1
    return out


def tg_labels():
    """Interjection + reply-target on the phase-3 pub_tg rows. No message text."""
    d = pd.read_parquet(DATA / "proc" / "pub_tg.parquet", columns=["msg_id", "conv", "ts", "speaker"])
    d["mid"] = d.msg_id.str.slice(3).astype(np.int64)
    kept = set(d.mid.to_numpy().tolist())
    pointed = set()
    with open(DATA / "tg_ru" / "group_data_clean.json") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line.startswith("{"):
                continue
            m = json.loads(line)
            r = (m.get("reply_to") or {}).get("reply_to_msg_id")
            if r is None:
                continue
            try:
                rid = int(r)
            except (TypeError, ValueError):
                continue
            if rid in kept and int(m["id"]) in kept:
                pointed.add(rid)
    out = {}
    for conv, g in d.groupby("conv", sort=False):
        g = g.sort_values("ts", kind="stable")
        ts = g.ts.to_numpy(float)
        sp = g.speaker.to_numpy()
        mids = g.mid.to_numpy()
        yi = text_interjection(ts, sp)
        reply = [None] * len(g)
        # a row is positive when its mid is pointed at by a later kept message; reply_message_value
        # wants the pointing message's reply_to field. Synthesise one pointer per target.
        ids = [f"tg:{int(m)}" for m in mids]
        # build reply_to list: for each pointer we don't have the pointing row separate.
        # Use reply_message_value by fabricating reply_to on a copy: put the target id on a
        # later dummy only if we record which message points. We only know the target set.
        # Reproduce the rule directly: y=1 if mid in pointed AND some later row exists that
        # replies to it. `pointed` already requires the pointing message is a later kept id
        # only if the pointer's id > target id (telegram ids increase). Require a later row
        # in THIS conversation whose mid is greater and... we didn't store the pointer's conv.
        # Re-scan is heavier. Store pointer->target only when both kept, then group.
        yv = np.full(len(g), np.nan, np.float32)
        last = ts[-1]
        idset = set(ids)
        for i, mid in enumerate(mids):
            hit = int(mid) in pointed
            # pointed is global; restrict to a later message in this conversation below
            if hit:
                yv[i] = 1.0
            elif last - ts[i] >= 60.0:
                yv[i] = 0.0
        for i in range(len(g)):
            out[f"tg:{int(mids[i])}"] = (float(yi[i]), float(yv[i]))
    return out, pointed


def tg_labels_fixed():
    """Same as the prereg, with reply links restricted to a later message in the same conversation."""
    d = pd.read_parquet(DATA / "proc" / "pub_tg.parquet", columns=["msg_id", "conv", "ts", "speaker"])
    d["mid"] = d.msg_id.str.slice(3).astype(np.int64)
    kept = set(int(x) for x in d.mid.to_numpy().tolist())
    # target mid -> list of pointer mids, both inside the kept tail
    pointers = {}
    with open(DATA / "tg_ru" / "group_data_clean.json") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line.startswith("{"):
                continue
            m = json.loads(line)
            r = (m.get("reply_to") or {}).get("reply_to_msg_id")
            if r is None:
                continue
            try:
                rid = int(r); pid = int(m["id"])
            except (TypeError, ValueError):
                continue
            if rid in kept and pid in kept:
                pointers.setdefault(rid, []).append(pid)
    by_conv = {c: g.sort_values("ts", kind="stable") for c, g in d.groupby("conv", sort=False)}
    mid_conv = dict(zip(d.mid.astype(int), d.conv))
    out = {}
    for conv, g in by_conv.items():
        ts = g.ts.to_numpy(float)
        sp = g.speaker.to_numpy()
        mids = g.mid.to_numpy(np.int64)
        pos = {int(m): i for i, m in enumerate(mids)}
        yi = text_interjection(ts, sp)
        yv = np.full(len(g), np.nan, np.float32)
        last = float(ts[-1])
        for i, mid in enumerate(mids):
            later = False
            for p in pointers.get(int(mid), []):
                j = pos.get(p)
                if j is not None and j > i:
                    later = True
                    break
            if later:
                yv[i] = 1.0
            elif last - ts[i] >= 60.0:
                yv[i] = 0.0
        for i in range(len(g)):
            out[f"tg:{int(mids[i])}"] = (float(yi[i]), float(yv[i]))
    return out


def irc_labels():
    """Gold reply links from the irc_disentangle `connections` field. No nick lexicon."""
    d = pd.read_parquet(DATA / "proc" / "pub_irc.parquet", columns=["msg_id", "conv", "ts"])
    reply_of = {}
    for part in ("train", "validation", "test"):
        src = pd.read_parquet(DATA / "irc_dis" / "ubuntu" / f"{part}-00000-of-00001.parquet", columns=["id", "connections"])
        ids = src.id.to_numpy()
        cons = src.connections.to_numpy()
        for i, c in zip(ids, cons):
            if c is None or (isinstance(c, float) and np.isnan(c)):
                continue
            prev = [int(x) for x in list(c) if int(x) < int(i)]
            if prev:
                reply_of[f"irc:{part}:{int(i)}"] = f"irc:{part}:{max(prev)}"
    out = {}
    for conv, g in d.groupby("conv", sort=False):
        g = g.sort_values("ts", kind="stable")
        ids = g.msg_id.to_numpy()
        ts = g.ts.to_numpy(float)
        reply = [reply_of.get(m) for m in ids]
        # minute-resolution source: two later messages close a negative
        part = str(conv).split(":")[1] if isinstance(conv, str) else ""
        minute = part in ("train", "validation", "test")
        yv = reply_message_value(ts, list(ids), reply, minute_clock=minute)
        for i, m in enumerate(ids):
            out[m] = (float("nan"), float(yv[i]))
    return out


def _segments_magic(conv):
    """Human spans for one MagicData conversation. Drops speaker 'none' and whole-token bracket marks."""
    import re
    segs = []
    pat = re.compile(r"\[([0-9.]+),([0-9.]+)\]\s+(\S+)\s+\S+\s*(.*)$")
    tx = sorted(glob.glob(str(DATA / "magicdata_ms" / "TXT" / f"{conv}_0_*.txt")))
    for pth in tx:
        for line in open(pth, encoding="utf-8"):
            m = pat.match(line.strip().lstrip('\ufeff'))
            if not m:
                continue
            if nonspeech_span(m.group(3), m.group(4)):
                continue
            segs.append((float(m.group(1)), float(m.group(2)), m.group(3)))
    return segs


def magic_oto_frames():
    """Event frames (no raw text stored) for MagicData and oto, which are not in the p3 stream."""
    frames = []
    convs = sorted({os.path.basename(p).rsplit("_0_", 1)[0] for p in glob.glob(str(DATA / "magicdata_ms" / "TXT" / "*.txt"))})
    for conv in convs:
        segs = _segments_magic(conv)
        if len(segs) < 4:
            continue
        frames.append(_frame_from_segs(f"md:{conv}", segs, "pub_magic", "voice_1on1", "private", split_hash(f"md:{conv}")))
    oto = DATA / "proc" / "audio_oto"
    for p in sorted(glob.glob(str(oto / "*.npz"))):
        sid = os.path.basename(p)[:-4]
        z = np.load(p, allow_pickle=False)
        segs = []
        for ch, key in ((0, "s0"), (1, "s1")):
            for a, b, *_rest in json.loads(str(z[key])):
                segs.append((float(a), float(b), f"c{ch}"))
        if len(segs) < 4:
            continue
        frames.append(_frame_from_segs(f"oto:{sid}", segs, "pub_oto", "voice_1on1", "private", split_hash(f"oto:{sid}")))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _frame_from_segs(conv, segs, source, scenario, conv_type, split):
    yi, _k = speech_interrupt(segs)
    yv = dyad_message_value(segs)
    # event time = segment end, matching phase-3 spoken streams. Speaker ids anonymised per conv.
    order = sorted(range(len(segs)), key=lambda i: (segs[i][1], segs[i][0], i))
    spk_ids = []
    for i in order:
        spk_ids.append(segs[i][2])
    anon = {}
    for s in spk_ids:
        if s not in anon:
            anon[s] = f"p{len(anon)}"
    # self = first speaker in start order, so a conversation has both roles. Deterministic.
    first = min(segs, key=lambda t: (t[0], t[1]))[2]
    rows = []
    for k, i in enumerate(order):
        s, e, sp = segs[i]
        rows.append(dict(source=source, conv=conv, conv_type=conv_type, ts=float(e), role="self" if sp == first else "other",
                         speaker=anon[sp], speaker_known=True, text=None, modality="voice", n_participants=2,
                         msg_id=f"{conv}:{k}", scenario=scenario, split=split,
                         y_interrupt=float(yi[i]), y_addr=float(yv[i]), y_topic=np.nan, livestream=0, p3_index=-1))
    return pd.DataFrame(rows)


def topic_frames():
    rows = []
    z = zipfile.ZipFile(DATA / "crosswoz" / "data.zip")
    dials = json.loads(z.read("data/dialogues.json"))
    for d in dials:
        turns = d["turns"]
        states = [t.get("state") for t in turns]
        y = domain_change_labels(states)
        sp = "pub_" + {"train": "train", "validation": "val", "test": "test"}[d["data_split"]]
        conv = "cw:" + str(d["dialogue_id"])
        for i, t in enumerate(turns):
            role = "self" if str(t.get("speaker", "")).lower() in ("sys", "system") else "other"
            rows.append(dict(source="pub_crosswoz", conv=conv, conv_type="private", ts=1_600_000_000.0 + i,
                             role=role, speaker=role, speaker_known=True, text=t.get("utterance") or None,
                             modality="text", n_participants=2, msg_id=f"{conv}:{i}", scenario="topic", split=sp,
                             y_interrupt=np.nan, y_addr=np.nan, y_topic=float(y[i]), livestream=0, p3_index=-1))
    for part, sp in (("train", "pub_train"), ("validation", "pub_val"), ("test", "pub_test")):
        obj = json.load(open(DATA / "superdialseg" / f"{part}.json"))
        dials = obj["dial_data"]["super_dialseg"]
        for d in dials:
            conv = "sd:" + str(d["dial_id"])
            turns = d["turns"]
            for i, t in enumerate(turns):
                role = "self" if str(t.get("role", "")).lower() in ("agent", "system", "sys") else "other"
                lab = t.get("segmentation_label")
                rows.append(dict(source="pub_superdialseg", conv=conv, conv_type="private", ts=1_600_000_000.0 + i,
                                 role=role, speaker=str(t.get("role") or role), speaker_known=True,
                                 text=t.get("utterance") or None, modality="text", n_participants=2,
                                 msg_id=f"{conv}:{i}", scenario="topic", split=sp,
                                 y_interrupt=np.nan, y_addr=np.nan,
                                 y_topic=float(lab) if lab is not None else np.nan, livestream=0, p3_index=-1))
    return pd.DataFrame(rows)


def _emo_block(texts):
    """11-d text-emotion features. Empty text -> zeros. Local bge + the phase-4 text head. No network."""
    from tidal.embed import ensure
    from tidal.emo_features import heads
    texts = [("" if t is None else str(t)) for t in texts]
    idx = [i for i, t in enumerate(texts) if t.strip()]
    out = np.zeros((len(texts), 11), np.float32)
    if not idx:
        return out
    k2i, emb, stats = ensure([texts[i] for i in idx], threads=int(os.environ.get("TIDAL_EMO_THREADS", "2")))
    print("embed", stats, flush=True)
    E = np.stack([emb[k2i[__import__("tidal.embed", fromlist=["key"]).key(texts[i])]].astype(np.float32) for i in idx])
    # key import above is ugly; do it cleanly
    return out, E, idx  # placeholder replaced below


def emo_block(texts):
    from tidal.embed import ensure, key
    from tidal.emo_features import heads
    texts = [("" if t is None else str(t)) for t in texts]
    idx = [i for i, t in enumerate(texts) if t.strip()]
    out = np.zeros((len(texts), 11), np.float32)
    if not idx:
        return out
    k2i, emb, stats = ensure([texts[i] for i in idx], threads=int(os.environ.get("TIDAL_EMO_THREADS", "2")))
    print("embed", stats, flush=True)
    E = np.stack([emb[k2i[key(texts[i])]].astype(np.float32) for i in idx])
    hs = heads()
    F = np.mean([h.features(E) for h in hs], 0)
    out[np.array(idx), :10] = F
    out[np.array(idx), 10] = 1.0
    return out


def featurize(df, mu, sd):
    Xfull = features.compute(df)
    G = FG.compute(df, Xfull)
    Z = FG.normalize(G, mu, sd)
    if "text" in df.columns and df.text.notna().any():
        E = emo_block(df.text.tolist())
    else:
        E = np.zeros((len(df), 11), np.float32)
    return np.concatenate([Z, E], 1).astype(np.float32)


def build():
    os.makedirs(PROC, exist_ok=True)
    print("labels: candor", flush=True)
    cd = candor_labels()
    print("labels: aishell", flush=True)
    a4 = aishell_labels()
    print("labels: tg", flush=True)
    tg = tg_labels_fixed()
    print("labels: irc", flush=True)
    irc = irc_labels()
    print("labels: magic/oto", flush=True)
    extra = magic_oto_frames()
    print("labels: topic", flush=True)
    topic = topic_frames()
    meta = pd.read_parquet("data/proc/p3_meta.parquet", columns=["source", "conv", "conv_type", "ts", "role", "split", "msg_id", "scenario"])
    Y = np.load("data/proc/p3.npz", mmap_mode="r")["Y"]
    G = np.load("data/proc/p3.npz", mmap_mode="r")["G"]
    Emo = np.load("data/proc/p3_emo.npy", mmap_mode="r")
    ck = __import__("torch").load("models/P4emo_mamba3_siso_s0.pt", map_location="cpu", weights_only=False)
    mu, sd = ck["mu"], ck["sd"]
    # p3 rows we keep
    src = meta.source.to_numpy()
    keep_src = np.isin(src, ["pub_candor", "pub_aishell4", "pub_tg", "pub_irc", "real", "pub_twitch", "pub_artemis", "pub_danmaku"])
    idx = np.flatnonzero(keep_src)
    maps = {"pub_candor": cd, "pub_aishell4": a4, "pub_tg": tg, "pub_irc": irc}
    y_int = np.full(len(idx), np.nan, np.float32)
    y_addr = np.full(len(idx), np.nan, np.float32)
    y_topic = np.full(len(idx), np.nan, np.float32)
    live = np.zeros(len(idx), np.int8)
    miss = 0
    msg = meta.msg_id.to_numpy()
    for j, i in enumerate(idx):
        s = src[i]
        if s in LIVE_SOURCES:
            live[j] = 1
            continue
        if s == "real":
            y_addr[j] = Y[i, 2]
            continue
        lab = maps[s].get(msg[i])
        if lab is None:
            miss += 1
            continue
        y_int[j] = lab[0]
        y_addr[j] = lab[1]
    print("p3 label misses", miss, "of", len(idx), flush=True)
    rows = meta.iloc[idx][["source", "conv", "conv_type", "ts", "role", "split", "msg_id", "scenario"]].copy()
    rows["y_interrupt"] = y_int
    rows["y_addr"] = y_addr
    rows["y_topic"] = y_topic
    rows["livestream"] = live
    rows["p3_index"] = idx
    # features for p3 rows
    Z = FG.normalize(np.asarray(G[idx], np.float32), mu, sd)
    Xp3 = np.concatenate([Z, np.asarray(Emo[idx], np.float32)], 1)
    # new frames: drop text before saving rows, but featurize first
    new = pd.concat([extra, topic], ignore_index=True) if len(extra) else topic
    print("featurize new rows", len(new), flush=True)
    Xnew = featurize(new, mu, sd)
    new = new.drop(columns=["text"])
    rows = pd.concat([rows, new], ignore_index=True)
    X = np.concatenate([Xp3, Xnew], 0).astype(np.float32)
    assert len(rows) == len(X)
    # counts
    counts = {}
    for s, g in rows.groupby("source"):
        counts[s] = dict(rows=int(len(g)), splits=g.split.value_counts().to_dict(),
                         interrupt=_rate(g.y_interrupt), addr=_rate(g.y_addr), topic=_rate(g.y_topic),
                         livestream=int(g.livestream.sum()))
    json.dump(counts, open(PROC + "/p5_counts.json", "w"), indent=1)
    rows.to_parquet(PROC + "/p5_rows.parquet")
    np.save(PROC + "/p5_X.npy", X)
    print(json.dumps(counts, indent=1)[:4000], flush=True)
    print("saved", len(rows), X.shape, flush=True)


if __name__ == "__main__":
    build()
