"""Parser for the plain-text dialogue-snippet format ("!语料" files) -> tidal's unified, identity-free event stream.

Format (one file, snippets separated by blank lines):
    !语料 2                      optional version line
    # <snippet id>
    @用途 <purpose>               e.g. 离线参考 (offline reference only)
    @背景 <context>               optional, free text
    @角色 <name>                  one line per participant
    @来源 <url> [<free text with timestamps like 1:02:03、1:02:30>]
    Name> utterance               one turn
    @事件 <note>                  e.g. "中间省略若干回合" = some turns skipped here (a gap in the turn order)

Snippets carry turn ORDER only (no timing). Output roles are identity-free: 'self' for any participant listed in
`ai_roles` (the AI host(s)), 'other' for everyone else; speakers become per-snippet indices (s0, s1, ...)."""
import re
from dataclasses import dataclass, field
import pandas as pd

TS_RE = re.compile(r"(?<![\d:])(\d{1,2}(?::\d{2}){1,2})(?![\d:])")

@dataclass
class Snippet:
    sid: str
    purpose: str = ""
    context: str = ""
    roles: list = field(default_factory=list)
    sources: list = field(default_factory=list)          # [(url, [seconds, ...])]
    turns: list = field(default_factory=list)            # [(speaker, text, gap_before: bool)]

def to_seconds(t: str) -> int:
    s = 0
    for p in t.split(":"): s = s * 60 + int(p)
    return s

def parse(text: str) -> list:
    out, cur, gap = [], None, False
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("!"): continue
        if line.startswith("# "):
            cur = Snippet(line[2:].strip()); out.append(cur); gap = False; continue
        if cur is None: continue
        if line.startswith("@"):
            key, _, val = line[1:].partition(" ")
            if key == "用途": cur.purpose = val.strip()
            elif key == "背景": cur.context = (cur.context + " " + val.strip()).strip()
            elif key == "角色": cur.roles.append(val.strip())
            elif key == "来源":
                url, _, rest = val.strip().partition(" ")
                cur.sources.append((url, [to_seconds(t) for t in TS_RE.findall(rest)]))
            elif key == "事件": gap = True
            continue
        m = re.match(r"^([^>]{1,40})>\s?(.*)$", line)
        if m: cur.turns.append((m.group(1).strip(), m.group(2).strip(), gap)); gap = False
    return out

def to_events(snips: list, ai_roles: set, source: str = "corpus") -> pd.DataFrame:
    """Order-only event frame. ts = turn index (1 s apart, +600 s across an omitted-turns gap so time-based features
    see a break). n_participants = listed roles. responds_to: an AI turn directly after another speaker's turn."""
    rows = []
    for sn in snips:
        spk_ix = {}; t = 0.0; prev = None
        npart = max(len(sn.roles), len({s for s, _, _ in sn.turns}))
        for k, (spk, txt, gap) in enumerate(sn.turns):
            if gap: t += 600.0
            sid = spk_ix.setdefault(spk, f"s{len(spk_ix)}")
            role = "self" if spk in ai_roles else "other"
            mid = f"{source}:{sn.sid}:{k}"
            rows.append(dict(source=source, conv=f"{source}:{sn.sid}", conv_type="group" if npart > 2 else "private",
                             ts=t, role=role, speaker=sid, speaker_known=True, text=txt, msg_id=mid,
                             responds_to=prev["msg_id"] if (prev is not None and role == "self" and prev["role"] != "self" and not gap) else None,
                             initiate=int(role == "self" and k == 0), n_participants=npart, modality="voice",
                             gap_before=gap, scenario="ai_stream_snippet"))
            prev = rows[-1]; t += 1.0
    return pd.DataFrame(rows)
