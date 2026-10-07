"""Generic JSONL source (for demos/tests and non-DB deployments). One event per line:
  {"msg_id": "...", "conv": "...", "conv_type": "group|private", "ts": 1790000000.0, "role": "self|other",
   "speaker": "...", "text": "...", "addressed": true|false|null, "bot_action": "speak|silent"|null}
Ids are hashed with the local salt on ingest; text is PII-scrubbed. Reads incrementally by byte offset."""
import json, os, pandas as pd
from tidal.privacy import pseudo, scrub
from tidal.shadow import store as S
from tidal.shadow.sources.base import EVENT_COLUMNS, empty_events

class Source:
    name = "jsonl"
    def __init__(self, path=None):
        self.path = path or os.environ.get("TIDAL_SHADOW_JSONL")
    def default_predict_after(self, now):
        return 0.0
    def fetch(self, store, now):
        off = S.get_meta(store, "jsonl_offset", 0); rows = []
        with open(self.path, "rb") as f:
            f.seek(off)
            for line in f:
                if not line.endswith(b"\n"): break           # partial line: pick up next run
                off += len(line); e = json.loads(line)
                if e["ts"] > now: continue
                bot = e.get("role") == "self"
                rows.append(dict(msg_id=pseudo("m", e["msg_id"]), conv=pseudo("c", e["conv"]), conv_type=e.get("conv_type", "group"),
                                 ts=float(e["ts"]), role="self" if bot else "other",
                                 speaker="BOT" if bot else (pseudo("u", e["speaker"]) if e.get("speaker") else None),
                                 speaker_known=bool(bot or e.get("speaker")), text=scrub(e.get("text")),
                                 attention_reason=("addressed_to_agent" if e.get("addressed") else ("none" if e.get("addressed") is False else None)),
                                 chosen_action=e.get("bot_action"), cur_addressed=e.get("addressed"), cur_reply=None))
        S.set_meta(store, "jsonl_offset", off)
        if not rows: return empty_events()
        d = pd.DataFrame(rows)
        for c in EVENT_COLUMNS:
            if c not in d: d[c] = None
        return d[EVENT_COLUMNS]
