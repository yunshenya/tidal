"""Adapters -> unified event frame (see tidal/labels.py)."""
import json, re, random, hashlib, numpy as np, pandas as pd
from datetime import datetime, timezone, timedelta
from tidal.privacy import scrub

from tidal.config import bot_name_pattern
BOT_NAME_RE = re.compile(bot_name_pattern())

def real_frame(path="data/proc/real_events.parquet", d=None):
    if d is None: d = pd.read_parquet(path)
    name = d.text.fillna("").str.contains(BOT_NAME_RE)
    addr = pd.Series(np.nan, index=d.index)
    has_rep = d.attention_reason.notna()
    explicit = d.attention_reason.isin(["addressed_to_agent", "reply_to_agent"]) | (d.cur_addressed == True) | (d.cur_reply == True)
    addr[has_rep] = 0.0
    addr[has_rep & explicit] = 1.0
    addr[name & (d.role == "other")] = 1.0
    addr[(~has_rep) & d.text.notna() & ~name & ((d.cur_addressed == True) | (d.cur_reply == True))] = 1.0
    act = d.chosen_action.map(lambda a: None if (a is None or (isinstance(a, float) and np.isnan(a))) else ("speak" if a == "speak" else "silent")).astype(object)
    return pd.DataFrame(dict(conv=d.conv, conv_type=d.conv_type, ts=d.ts, role=d.role, speaker=d.speaker,
                             speaker_known=d.speaker_known, text=d.text, addr_src=addr, bot_act_src=act,
                             msg_id=d.msg_id, source="real"))

TIME_HOURS = {"深夜": (0, 3), "早上": (7, 9), "中午": (11, 13), "下午": (14, 17), "晚上": (19, 23)}
def synth_frame(path="data/synth/clean.jsonl"):
    rows = []
    for ci, line in enumerate(open(path)):
        c = json.loads(line)
        rng = random.Random(c["id"])
        h0, h1 = TIME_HOURS.get(c["spec"].get("time"), (8, 23))
        base = datetime(2026, 9, 1, tzinfo=timezone(timedelta(hours=8))) + timedelta(days=rng.randint(0, 25), hours=rng.uniform(h0, h1))
        t = base.timestamp(); conv = "s_" + c["id"]
        msgs = c["messages"]
        for k, m in enumerate(msgs):
            t += float(m["dt"]) if k else 0.0
            bot = m["s"] == "BOT"
            act = None
            if not bot:
                nxt = msgs[k + 1] if k + 1 < len(msgs) else None
                act = "speak" if (nxt is not None and nxt["s"] == "BOT") else "silent"
            addr = np.nan if bot else float(("addr_bot" in m["tags"]) or bool(BOT_NAME_RE.search(m["text"])) or
                                             any(tg.startswith("reply_to:") and _is_bot(msgs, tg) for tg in m["tags"]))
            rows.append(dict(conv=conv, conv_type="group", ts=t, role="self" if bot else "other",
                             speaker="BOT" if bot else f"{conv}:{m['s']}", speaker_known=True, text=scrub(m["text"]),
                             addr_src=addr, bot_act_src=act, msg_id=f"{conv}:{k}", source="synth",
                             tags=",".join(m["tags"])))
    return pd.DataFrame(rows)

def _is_bot(msgs, tag):
    try: j = int(tag.split(":")[1]); return 0 <= j < len(msgs) and msgs[j]["s"] == "BOT"
    except Exception: return False
