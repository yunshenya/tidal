#!/usr/bin/env python3
"""Convert local turngate review windows into the unified causal event format.

The source contains short group-chat contexts rather than a full raw transcript.
Human-consensus rows are split by source key; lexical weak labels are train-only.
Raw JSONL stays under data/private/ and is gitignored.
"""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

HEADS = ["y_eot", "y_self", "y_addr", "y_act", "y_recheck", "y_hreply"]


def split_key(key):
    u = int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 2**32
    return "train" if u < .7 else ("val" if u < .85 else "test")


def rows(path, force_human=None):
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        r = json.loads(line); ctx = r.get("context") or {}; turns = ctx.get("recent_turns") or []
        human = ((r.get("label_provenance") or {}).get("source") == "human_consensus")
        if force_human is True: human = True
        key = str(r["source_key"]); conv = ("tg_h:" if human else "tg_w:") + key
        current = r.get("current_text") or ""
        all_turns = list(turns) + [{"role": "other_member", "text": current}]
        action = (r.get("labels") or {}).get("response")
        action = {"answer": "speak", "ack": "speak", "wait": "wait", "ignore": "silent"}.get(action)
        completion = (r.get("labels") or {}).get("completion")
        y_eot = {"flush_now": 1.0, "hold_for_more": 0.0}.get(completion, np.nan)
        for i, t in enumerate(all_turns):
            role = "self" if t.get("role") == "assistant" else "other"
            sp = "BOT" if role == "self" else str(t.get("speaker") or t.get("role") or "member")
            is_current = i == len(all_turns) - 1
            out.append(dict(conv=conv, conv_type="group", ts=float(i), role=role, speaker=sp,
                            speaker_known=True, text=t.get("text") or None, modality="text",
                            msg_id=f"{conv}:{i}", responds_to=None, initiate=0,
                            addr_src=np.nan, bot_act_src=action if is_current else None,
                            source="group_review_human" if human else "group_review_pseudo",
                            split=(split_key(key) if human else "train"),
                            y_eot=(y_eot if is_current else np.nan), y_self=np.nan,
                            y_addr=np.nan, y_act=({"speak": 0., "wait": 1., "silent": 2.}.get(action, np.nan) if is_current else np.nan),
                            y_recheck=np.nan, y_hreply=np.nan))
    return pd.DataFrame(out)


def main():
    root = Path(__file__).resolve().parents[1]; src = root / "data/private/turngate"
    files = [src / "review-batch-20260913.jsonl", src / "review-batch-20260913-speakers.jsonl"]
    frames = [rows(p) for p in files if p.exists()]
    if not frames: raise FileNotFoundError(f"no turngate files in {src}")
    d = pd.concat(frames, ignore_index=True)
    out = root / "data/private/processed"; out.mkdir(parents=True, exist_ok=True)
    d.to_parquet(out / "turngate_group.parquet", index=False)
    print(json.dumps({"rows": len(d), "conversations": int(d.conv.nunique()),
                      "samples": int(d[d.role.eq("other")].conv.nunique()),
                      "sources": d.groupby("source").conv.nunique().to_dict(),
                      "splits": d.groupby("split").conv.nunique().to_dict(),
                      "labels": {h: int(np.isfinite(d[h]).sum()) for h in HEADS}}, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
