#!/usr/bin/env python3
"""Prepare locally pulled private PostgreSQL exports for tidal experiments.

Inputs are two local CSV files, ``dialogue_samples.csv`` (id, scope_id, occurred_at,
purpose, context, reply) and ``turn_replays.csv`` (one row per observed chat event).
Outputs stay under ``data/private/`` (which is gitignored). No database access is
performed here; the script is deliberately rerunnable and never mutates the DB.
"""
from __future__ import annotations
import argparse, csv, hashlib, hmac, json, os, re
from datetime import datetime
from pathlib import Path

PII = [
    (re.compile(r"https?://\S+|www\.\S+", re.I), "<URL>"),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "<EMAIL>"),
    (re.compile(r"(?<!\d)\d{17}[\dXx](?!\d)"), "<ID>"),
    (re.compile(r"(?<!\d)(?:\+?86[- ]?)?1[3-9]\d{9}(?!\d)"), "<PHONE>"),
    (re.compile(r"(?<!\d)[1-9]\d{4,11}(?!\d)"), "<NUM>"),
]

def scrub(x):
    if not isinstance(x, str): return x
    for rx, replacement in PII: x = rx.sub(replacement, x)
    return x

def scrub_tree(x):
    if isinstance(x, str): return scrub(x)
    if isinstance(x, list): return [scrub_tree(v) for v in x]
    if isinstance(x, dict): return {k: scrub_tree(v) for k, v in x.items()}
    return x

from tidal.turngate import pseudo  # one HMAC pseudonym function for all private prep scripts

def ts(value):
    value = value.replace(" ", "T")
    if re.search(r"[+-]\d{2}$", value): value += ":00"
    value = re.sub(r"\.(\d{1,6})(?=[+-])", lambda m: "." + m.group(1).ljust(6, "0"), value)
    return datetime.fromisoformat(value).timestamp()

def prepare(src_dir, out, salt):
    out.mkdir(parents=True, exist_ok=True)
    sft_n = 0
    with (src_dir / "dialogue_samples.csv").open("r", encoding="utf-8", newline="") as f, (out / "dialogue_sft.jsonl").open("w", encoding="utf-8") as g:
        for row in csv.DictReader(f):
            try: context = json.loads(row["context"])
            except (TypeError, json.JSONDecodeError): context = row["context"]
            item = {"id": pseudo(salt, "sample", row["id"]),
                    "conversation": pseudo(salt, "conv", row["scope_id"]),
                    "occurred_at": row["occurred_at"], "purpose": row["purpose"],
                    "context": scrub_tree(context), "reply": scrub(row["reply"])}
            g.write(json.dumps(item, ensure_ascii=False) + "\n"); sft_n += 1
    event_n = 0
    with (src_dir / "turn_replays.csv").open("r", encoding="utf-8", newline="") as f, (out / "turn_events.jsonl").open("w", encoding="utf-8") as g:
        for row in csv.DictReader(f):
            decision = row.get("shadow_decision") or ""
            action = row.get("chosen_action") or ""
            if action not in {"speak", "silent"}: action = None
            item = {"conv": pseudo(salt, "conv", row["scope_kind"] + ":" + (row.get("event_id") or row["id"])),
                    "ts": ts(row["occurred_at"]), "role": "other" if row["event_type"] == "message_received" else "self",
                    "speaker": None, "speaker_known": False, "text": None, "modality": "text",
                    "addr_src": 1.0 if row.get("mentioned_bot") == "t" or row.get("reply_to_bot") == "t" else 0.0,
                    "bot_act_src": action, "shadow_decision": decision or None,
                    "msg_chars": int(row["msg_chars"] or 0), "group_activity_1m": int(row["group_activity_1m"] or 0),
                    "has_question": row.get("has_question") == "t", "source": "private_turn_replay"}
            g.write(json.dumps(item, ensure_ascii=False) + "\n"); event_n += 1
    manifest = {"dialogue_samples": sft_n, "turn_events": event_n,
                "privacy": "HMAC conversation/sample IDs; PII patterns scrubbed; raw exports remain local only"}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--input", default="data/private")
    ap.add_argument("--output", default="data/private/processed"); ap.add_argument("--salt-file", default=".secrets/pseudo_salt")
    a = ap.parse_args(); prepare(Path(a.input), Path(a.output), Path(a.salt_file).read_bytes().strip())
