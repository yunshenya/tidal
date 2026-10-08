"""Convert downloaded public datasets into tidal's unified, identity-free event stream (one parquet per source in
data/public/proc/). Columns follow tidal/dataset2.py: conv, ts, role ('self' | 'other'), speaker, speaker_known, text,
modality, n_participants, msg_id, responds_to, initiate, addr_src, bot_act_src, conv_type, scenario, source, split.

Perspective-taking: public chats have no tidal-like bot, so in each conversation ONE active participant is chosen as
"self" (weighted by activity). Labels are then defined exactly as for the bot: addressed = a message replies to self
(only where gold reply links exist), speak = self's next message replies to / directly follows this message.
For livestreams the streamer's own chat messages are "self" (the host); viewers are "other".
Speakers are re-indexed per conversation (identity-free); raw ids are never written."""
import glob, json, re, sys, numpy as np, pandas as pd
from tidal.public_data.manifest import DATA
from tidal.public_data.fastlabels import compute_labels_fast as compute_labels

PROC = DATA / "proc"; COLS = ["source", "conv", "conv_type", "ts", "role", "speaker", "speaker_known", "text", "modality",
                              "n_participants", "msg_id", "responds_to", "initiate", "addr_src", "bot_act_src", "scenario", "split"]
W_SPEAK = 120.0

def _secs(dt):
    """datetime Series (any unit / tz) -> unix seconds (float), independent of the pandas datetime resolution."""
    dt = pd.to_datetime(dt)
    if getattr(dt.dt, "tz", None) is not None: dt = dt.dt.tz_convert(None)
    return ((dt - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)).astype(float)

def _split(convs, seed, fr=(0.8, 0.1)):
    r = np.random.default_rng(seed); u = dict(zip(convs, r.random(len(convs))))
    return {c: "pub_train" if u[c] < fr[0] else ("pub_val" if u[c] < fr[0] + fr[1] else "pub_test") for c in convs}

def perspective(d, rng, min_msgs=3, have_replies=True, self_speaker=None):
    """d: one conversation sorted by ts with columns speaker (anon), msg_id, reply_to (msg_id or None)."""
    d = d.copy(); vc = d.speaker.value_counts()
    if self_speaker is None:
        cand = vc[vc >= max(min_msgs, int(0.03 * len(d)))]
        self_speaker = rng.choice(cand.index.to_numpy(), p=(cand / cand.sum()).to_numpy()) if len(cand) else None
    d["role"] = np.where(d.speaker == self_speaker, "self", "other")
    selfids = set(d.msg_id[d.role == "self"])
    rep = d.reply_to.where(d.reply_to.notna(), None)
    if have_replies:
        d["addr_src"] = np.where(d.role == "other", rep.isin(selfids).astype(float), np.nan)
        d["responds_to"] = np.where(d.role == "self", rep, None)
    else:
        d["addr_src"] = np.nan; d["responds_to"] = None
    act = np.where(d.role == "other", "silent", None).astype(object)
    answered = set(d.responds_to.dropna())
    ts = d.ts.to_numpy(); role = d.role.to_numpy(); mids = d.msg_id.to_numpy()
    for i in range(len(d)):                                  # self speaks: mark the message it answers, else the last inbound
        if role[i] != "self": continue
        j = i - 1
        if not have_replies or d.responds_to.iat[i] is None:
            while j >= 0 and role[j] == "self": j -= 1
            if j >= 0 and ts[i] - ts[j] <= W_SPEAK: act[j] = "speak"
    for i in range(len(d)):
        if role[i] == "other" and mids[i] in answered: act[i] = "speak"
    d["bot_act_src"] = act
    d["initiate"] = 0
    prev_self = np.r_[False, role[:-1] == "self"]; gap = np.r_[np.inf, np.diff(ts)]
    d.loc[(d.role == "self") & d.responds_to.isna() & ((gap > 300) | prev_self & (gap > 300)), "initiate"] = 1
    spk = {s: f"p{i}" for i, s in enumerate(vc.index)}; d["speaker"] = d.speaker.map(spk)
    return d

def finish(frames, source, scenario, conv_type, seed, splits=None, write=True):
    d = pd.concat(frames, ignore_index=True)
    d["source"] = source; d["scenario"] = scenario; d["conv_type"] = conv_type; d["speaker_known"] = True
    if splits is None: splits = _split(sorted(d.conv.unique()), seed)
    d["split"] = d.conv.map(splits)
    for c in COLS:
        if c not in d: d[c] = None
    d = d[COLS].sort_values(["conv", "ts"], kind="stable").reset_index(drop=True)
    d = compute_labels(d); PROC.mkdir(parents=True, exist_ok=True)
    if write: d.to_parquet(PROC / f"{source}.parquet")
    print(source, len(d), "events", d.conv.nunique(), "convs", d.split.value_counts().to_dict(), flush=True); return d

# ---------------------------------------------------------------- (a) text group chats
def tg_ru(max_events=160_000, seed=1):
    rows = []
    with open(DATA / "tg_ru" / "group_data_clean.json") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if not line.startswith("{"): continue
            m = json.loads(line); s = m.get("sender") or {}; r = m.get("reply_to") or {}
            mt = m.get("type") or "text"
            rows.append((m["id"], m["date_ts"], s.get("id"), r.get("reply_to_msg_id"), "text" if mt == "text" else
                         ("image" if "photo" in mt else ("video" if "video" in mt else "other")), m.get("text") or None))
    d = pd.DataFrame(rows, columns=["mid", "ts", "spk", "rep", "modality", "text"]).sort_values("mid").tail(max_events)
    d["msg_id"] = "tg:" + d.mid.astype(str); d["reply_to"] = np.where(d.rep.notna(), "tg:" + d.rep.astype("Int64").astype(str), None)
    d["speaker"] = d.spk.astype(str); d["ts"] = d.ts.astype(float)
    d["conv"] = "tg:" + pd.to_datetime(d.ts, unit="s").dt.strftime("%Y-%m-%d")
    rng = np.random.default_rng(seed); fr = []
    for c, g in d.groupby("conv", sort=True):
        if len(g) < 20: continue
        g = perspective(g, rng); g["n_participants"] = g.speaker.nunique(); fr.append(g)
    # time split: last 15% of days -> test, previous 10% -> val (pretraining data; test only used for public sanity)
    days = sorted({g.conv.iat[0] for g in fr}); n = len(days)
    sp = {c: ("pub_train" if i < 0.75 * n else ("pub_val" if i < 0.85 * n else "pub_test")) for i, c in enumerate(days)}
    return finish(fr, "pub_tg", "group_text", "group", seed, sp)

def irc_dis(seed=2):
    fr = []; rng = np.random.default_rng(seed)
    for part in ("train", "validation", "test"):
        d = pd.read_parquet(DATA / "irc_dis" / "ubuntu" / f"{part}-00000-of-00001.parquet")
        m = d.raw.str.extract(r"^\[(\d\d):(\d\d)\] <([^>]+)> ?(.*)$")
        d = d.assign(hh=m[0], mm=m[1], nick=m[2], text=m[3]).dropna(subset=["nick"])
        d["minute"] = pd.to_datetime(d.date) + pd.to_timedelta(d.hh.astype(int), "h") + pd.to_timedelta(d.mm.astype(int), "m")
        # minute resolution -> spread messages evenly inside their minute (order preserved)
        k = d.groupby("minute").cumcount(); n = d.groupby("minute").minute.transform("size")
        d["ts"] = _secs(d.minute) + 60.0 * (k + 0.5) / n
        d["msg_id"] = f"irc:{part}:" + d.id.astype(str)
        def rep(row):
            c = [int(x) for x in (row.connections if row.connections is not None else []) if int(x) < row.id]
            return f"irc:{part}:{max(c)}" if c else None
        d["annotated"] = d.connections.map(lambda c: c is not None and len(c) > 0)
        d["reply_to"] = d.apply(rep, axis=1); d["speaker"] = d.nick; d["modality"] = "text"
        d["conv"] = f"irc:{part}:" + d.date.astype(str)
        for c, g in d.groupby("conv", sort=True):
            if len(g) < 50: continue
            have = g.annotated.mean() > 0.5
            g = perspective(g, rng, have_replies=bool(have)); g["n_participants"] = g.speaker.nunique(); fr.append(g)
    sp = {g.conv.iat[0]: {"train": "pub_train", "validation": "pub_val", "test": "pub_test"}[g.conv.iat[0].split(":")[1]] for g in fr}
    return finish(fr, "pub_irc", "group_text", "group", seed, sp)

# ---------------------------------------------------------------- (b) livestream chat
def twitch(n_docs=160, max_msgs=6000, min_msgs=300, seed=3):
    from tidal.optional import require
    pq = require("pyarrow.parquet", "audio", pip_name="pyarrow")
    rng = np.random.default_rng(seed); fr = []; got = 0
    for p in sorted(glob.glob(str(DATA / "twitchchat" / "data" / "*.parquet"))):
        t = pq.read_table(p, columns=["streamer_id", "document_id", "users", "timestamps", "metadata"]).to_pandas()
        for r in t.itertuples():
            n = len(r.users)
            if not (min_msgs <= n <= max_msgs) or r.streamer_id not in set(r.users): continue
            if rng.random() > 0.5: continue
            ts = np.asarray(r.timestamps, float); u = np.asarray(r.users, object)
            k = pd.Series(ts).groupby(ts).cumcount().to_numpy(); cnt = pd.Series(ts).groupby(ts).transform("size").to_numpy()
            g = pd.DataFrame(dict(ts=ts + (k + 0.5) / cnt, speaker=u, modality="text", text=None))
            g["conv"] = f"tw:{got}"; g["msg_id"] = [f"tw:{got}:{i}" for i in range(n)]; g["reply_to"] = None
            g = perspective(g, rng, have_replies=False, self_speaker=r.streamer_id)
            g["n_participants"] = float((r.metadata or {}).get("stream_viewer_count") or np.nan); fr.append(g); got += 1
            if got >= n_docs: break
        if got >= n_docs: break
    return finish(fr, "pub_twitch", "livestream", "group", seed, _split([f"tw:{i}" for i in range(got)], seed, (0.6, 0.1)))

def artemis(window_min=10, max_windows=4, seed=4):
    d = pd.read_parquet(DATA / "artemis" / "data" / "train-00000-of-00001.parquet", columns=["author_id", "published_at", "platform", "message"])
    d["ts"] = _secs(pd.to_datetime(d.published_at, format="ISO8601", utc=True))
    d = d.sort_values("ts"); t0 = d.ts.min(); fr = []; rng = np.random.default_rng(seed)
    d["win"] = ((d.ts - t0) // (window_min * 60)).astype(int)
    keep = {pl: sorted(d[d.platform == pl].win.unique())[::3][:max_windows] for pl in d.platform.unique()}   # spread over the broadcast
    for (pl, w), g in d.groupby(["platform", "win"]):
        if len(g) < 200 or w not in keep[pl]: continue
        g = g.rename(columns={"author_id": "speaker", "message": "text"}).assign(modality="text", reply_to=None)
        g["conv"] = f"ar:{pl}:{w}"; g["msg_id"] = [f"ar:{pl}:{w}:{i}" for i in range(len(g))]
        g = perspective(g, rng, have_replies=False, self_speaker="__no_host__"); g["n_participants"] = np.nan; fr.append(g)
    return finish(fr, "pub_artemis", "livestream", "group", seed, {g.conv.iat[0]: "pub_test" for g in fr})

# ---------------------------------------------------------------- (c) spoken dialogue timing
def candor(n_conv=700, seed=5):
    c = pd.read_parquet(DATA / "candor_tt" / "data" / "train-00000-of-00001.parquet",
                        columns=["conversation_id", "channel", "offset", "duration", "text"])
    rng = np.random.default_rng(seed); convs = sorted(c.conversation_id.unique()); convs = list(rng.choice(convs, min(n_conv, len(convs)), replace=False))
    fr = []
    for i, (cid, g) in enumerate(c[c.conversation_id.isin(convs)].groupby("conversation_id")):
        g = g.assign(ts=g.offset + g.duration, speaker=g.channel, modality="voice", text=g.text.where(g.text.str.len() > 0, None))
        g = g.sort_values("ts"); g["conv"] = f"cd:{i}"; g["msg_id"] = [f"cd:{i}:{k}" for k in range(len(g))]; g["reply_to"] = None
        g = perspective(g, rng, have_replies=False, self_speaker=rng.choice(["L", "R"])); g["n_participants"] = 2; fr.append(g)
    return finish(fr, "pub_candor", "voice_1on1", "private", seed)

def aishell4(seed=6):
    fr = []; rng = np.random.default_rng(seed)
    for i, p in enumerate(sorted(glob.glob(str(DATA / "aishell4_seg" / "**" / "*.rttm"), recursive=True))):
        rows = [l.split() for l in open(p) if l.startswith("SPEAKER")]
        if len(rows) < 30: continue
        g = pd.DataFrame(dict(st=[float(r[3]) for r in rows], du=[float(r[4]) for r in rows], speaker=[r[7] for r in rows]))
        g["ts"] = g.st + g.du; g = g.sort_values("ts").reset_index(drop=True)
        g["conv"] = f"a4:{i}"; g["msg_id"] = [f"a4:{i}:{k}" for k in range(len(g))]; g["reply_to"] = None; g["modality"] = "voice"; g["text"] = None
        g = perspective(g, rng, have_replies=False); g["n_participants"] = g.speaker.nunique(); fr.append(g)
    return finish(fr, "pub_aishell4", "meeting_voice", "group", seed)

def danmaku(n_videos=250, min_msgs=300, max_msgs=5000, seed=7):
    """Bilibili danmaku (DanmakuTPP-Events, research-only): ts = video time of the bullet comment. No user ids and no
    host -> every comment is an anonymous 'other' event (unique speaker); only the self-supervised timing targets (VAP
    future-activity projection) are meaningful, so all supervised head labels are set to NaN. NOTE: VOD danmaku are
    posted by viewers at different wall-clock times and aligned to video time (pseudo-live), not a live room."""
    import zipfile
    rng = np.random.default_rng(seed); cand = []
    for zp in sorted(glob.glob(str(DATA / "danmaku_tpp" / "DanmakuTPP-Events-part-*.zip"))):
        z = zipfile.ZipFile(zp)
        for n in z.namelist():
            if n.endswith(".json"): cand.append((zp, n, z.getinfo(n).file_size))
    cand = [c for c in cand if 25_000 <= c[2] <= 2_500_000]                 # ~300-5000 comments at ~250 B/comment
    order = rng.permutation(len(cand)); fr = []; zips = {}
    for i in order:
        zp, n, _ = cand[i]; z = zips.setdefault(zp, zipfile.ZipFile(zp))
        ev = json.loads(z.read(n))
        if not (min_msgs <= len(ev) <= max_msgs): continue
        k = len(fr); ts = np.array([float(e["time"]) for e in ev]); o = np.argsort(ts, kind="stable")
        g = pd.DataFrame(dict(ts=ts[o], text=[ev[j].get("text") for j in o], modality="text"))
        g["conv"] = f"dm:{k}"; g["msg_id"] = [f"dm:{k}:{j}" for j in range(len(g))]; g["reply_to"] = None
        g["speaker"] = [f"a{j}" for j in range(len(g))]; g["role"] = "other"; g["responds_to"] = None; g["initiate"] = 0
        g["addr_src"] = np.nan; g["bot_act_src"] = None; g["n_participants"] = np.nan; fr.append(g)
        if len(fr) >= n_videos: break
    d = finish(fr, "pub_danmaku", "livestream", "group", seed, _split([f"dm:{i}" for i in range(len(fr))], seed, (0.6, 0.1)), write=False)
    for h in ("y_eot", "y_self", "y_addr", "y_act", "y_recheck", "y_hreply"):
        if h in d: d[h] = np.nan
    d["speaker_known"] = False; d.to_parquet(PROC / "pub_danmaku.parquet"); return d

ALL = {"tg": tg_ru, "irc": irc_dis, "twitch": twitch, "artemis": artemis, "candor": candor, "aishell4": aishell4, "danmaku": danmaku}
if __name__ == "__main__":
    for k in (sys.argv[1:] or list(ALL)): ALL[k]()
