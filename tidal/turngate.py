"""Turngate review windows -> unified causal event rows, with a leakage-safe split.

Split unit is (group/scope id, time block): every row of one group in one block shares a
split, so overlapping review windows from the same conversation never straddle train/test.
Weak rows (auto/pseudo labels) are train-only, excluded by default, and are dropped whenever
their group has any row in val/test. Rows are deduplicated per ``source_key`` (a human label
always wins). Timestamps inside a window are ordinal, so rows carry ``timing_known=False``.
"""
from __future__ import annotations
import hashlib, hmac, json
from datetime import datetime
import numpy as np
import pandas as pd

HEADS = ["y_eot", "y_self", "y_addr", "y_act", "y_recheck", "y_hreply"]
HUMAN = "human_consensus"
AUTO = "assistant_model_v2"
GROUP_FIELDS = ("scope_id", "group_id", "conversation_id", "chat_id")
TIME_FIELDS = ("occurred_at", "created_at", "ts", "timestamp")
ACTION = {"answer": "speak", "ack": "speak", "wait": "wait", "ignore": "silent"}
ACT_CODE = {"speak": 0., "wait": 1., "silent": 2.}
EOT = {"flush_now": 1.0, "hold_for_more": 0.0}


def pseudo(salt: bytes, kind: str, value) -> str:
    return kind + "_" + hmac.new(salt, f"{kind}:{value}".encode(), hashlib.sha256).hexdigest()[:16]


def _field(row, names):
    ctx = row.get("context") or {}
    for n in names:
        for src in (row, ctx):
            v = src.get(n)
            if v not in (None, ""): return v
    return None


def group_of(row) -> str:
    g = _field(row, GROUP_FIELDS)
    if g is None:
        raise ValueError(f"row {row.get('source_key')!r} has no group/scope id ({', '.join(GROUP_FIELDS)}); "
                         "cannot split safely by conversation")
    return str(g)


def epoch(value):
    if value is None: return None
    if isinstance(value, (int, float)): return float(value)
    s = str(value).strip().replace(" ", "T").replace("Z", "+00:00")
    try: return float(s)
    except ValueError: return datetime.fromisoformat(s).timestamp()


def split_of(unit: str, frac=(.7, .15)) -> str:
    u = int(hashlib.sha256(unit.encode()).hexdigest()[:8], 16) / 2**32
    return "train" if u < frac[0] else ("val" if u < frac[0] + frac[1] else "test")


def is_human(row) -> bool:
    return (row.get("label_provenance") or {}).get("source") == HUMAN


def assign(rows, block_days=7.0, embargo_hours=6.0, include_weak=False):
    """Return [(row, split)] after dedupe, group/time-block split and weak-row filtering."""
    block = block_days * 86400; embargo = embargo_hours * 3600
    by_key = {}
    for r in rows:  # dedupe per source_key: human beats weak, first human wins
        k = str(r["source_key"]); old = by_key.get(k)
        if old is None or (is_human(r) and not is_human(old)): by_key[k] = r
    out, eval_groups = [], set()
    for r in by_key.values():
        if not is_human(r): continue
        g = group_of(r); t = epoch(_field(r, TIME_FIELDS))
        b = "all" if t is None else str(int(t // block))
        sp = split_of(f"{g}|{b}")
        # embargo: drop eval rows near a block edge whose window could overlap the neighbour block
        if sp != "train" and t is not None and min(t % block, block - t % block) < embargo: continue
        if sp != "train": eval_groups.add(g)
        out.append((r, sp))
    if include_weak:
        for r in by_key.values():
            if is_human(r) or group_of(r) in eval_groups: continue
            # only model labels from scripts/auto_label_turngate.py; legacy rule labels are refused
            if (r.get("label_provenance") or {}).get("source") != AUTO: continue
            out.append((r, "train"))
    return out


def to_frame(assigned, salt: bytes) -> pd.DataFrame:
    out = []
    for r, split in assigned:
        human = is_human(r); key = str(r["source_key"])
        conv = ("tg_h:" if human else "tg_a:") + pseudo(salt, "key", key)
        source = "group_review_human" if human else "group_review_auto"
        labels = r.get("labels") or {}
        action = ACTION.get(labels.get("response")); y_eot = EOT.get(labels.get("completion"), np.nan)
        turns = list((r.get("context") or {}).get("recent_turns") or []) + [{"role": "other_member", "text": r.get("current_text") or ""}]
        for i, t in enumerate(turns):
            self_ = t.get("role") == "assistant"; cur = i == len(turns) - 1
            spk = "BOT" if self_ else pseudo(salt, "spk", t.get("speaker") or t.get("role") or "member")
            out.append(dict(conv=conv, conv_type="group", ts=float(i), timing_known=False,
                            role="self" if self_ else "other", speaker=spk, speaker_known=True,
                            text=t.get("text") or None, modality="text", msg_id=f"{conv}:{i}", responds_to=None,
                            initiate=0, addr_src=np.nan, bot_act_src=action if cur else None, source=source,
                            split=split, y_eot=y_eot if cur else np.nan, y_self=np.nan, y_addr=np.nan,
                            y_act=ACT_CODE.get(action, np.nan) if cur else np.nan, y_recheck=np.nan, y_hreply=np.nan))
    return pd.DataFrame(out)


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]
