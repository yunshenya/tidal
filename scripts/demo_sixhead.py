#!/usr/bin/env python3
"""Small local demo for the current timing six-head checkpoint.

Examples:
  .venv/bin/python scripts/demo_sixhead.py --input examples/synthetic_stream.jsonl
  .venv/bin/python scripts/demo_sixhead.py --interactive

The demo is deliberately local and does not call an API or write private data.
The checkpoint consumes causal timing/role features; it does not generate a reply.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tidal import features
from tidal.model import TurnModel


HEADS = ("y_eot", "y_self", "y_addr", "y_act", "y_recheck", "y_hreply")
ACT = ("speak", "wait", "silent")
RECHECK_MID = np.array([1.0, 3.5, 7.5, 15.0, 40.0, 180.0, 300.0])


def normalise_event(raw, i, now):
    """Make a compact interactive input conform to the feature extractor schema."""
    e = dict(raw)
    e.setdefault("msg_id", f"demo-{i}")
    e.setdefault("conv", "demo-cli")
    e.setdefault("conv_type", "group")
    e.setdefault("ts", now)
    e.setdefault("role", "other")
    e.setdefault("speaker", None if e["role"] == "self" else "user")
    e.setdefault("text", None)
    e["speaker_known"] = bool(e.get("speaker_known", e.get("speaker") is not None))
    return e


def parse_line(line, i, now):
    line = line.strip()
    if not line:
        return None
    try:
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError("JSON event must be an object")
    except json.JSONDecodeError:
        role = "self" if line.startswith(("我:", "我：", "[self]")) else "other"
        for prefix in ("我:", "我：", "[self]", "他:", "他：", "[other]"):
            if line.startswith(prefix):
                line = line[len(prefix):].strip()
                break
        raw = {"role": role, "text": line, "speaker": None if role == "self" else "user"}
    return normalise_event(raw, i, now)


def score(history, model, ck):
    frame = pd.DataFrame(history).sort_values(["conv", "ts"], kind="stable").reset_index(drop=True)
    x = features.compute(frame)[:, features.cols("T")]
    x = ((x - ck["mu"]) / ck["sd"]).astype(np.float32)
    x = x[-64:]
    with torch.inference_mode():
        out = model(torch.from_numpy(x[None]), torch.ones((1, len(x)), dtype=torch.bool))
    p = {}
    p["y_eot"] = float(torch.sigmoid(out["y_eot"][0, -1, 0]))
    p["y_self"] = float(torch.sigmoid(out["y_self"][0, -1, 0]))
    p["y_addr"] = float(torch.sigmoid(out["y_addr"][0, -1, 0]))
    act = torch.softmax(out["y_act"][0, -1], dim=-1).numpy()
    rc = torch.softmax(out["y_recheck"][0, -1], dim=-1).numpy()
    p["y_act"] = {k: float(v) for k, v in zip(ACT, act)}
    p["y_recheck"] = {f"bucket_{i}": float(v) for i, v in enumerate(rc)}
    p["recheck_after_s"] = float(np.clip(rc @ RECHECK_MID, 2.0, 300.0))
    p["y_hreply"] = float(torch.sigmoid(out["y_hreply"][0, -1, 0]))
    return p


def print_result(event, p):
    action = max(p["y_act"], key=p["y_act"].get)
    text = event.get("text") or "<无文本>"
    print(json.dumps({
        "event": text,
        "role": event["role"],
        "eot": round(p["y_eot"], 3),
        "addressed_to_bot": round(p["y_addr"], 3),
        "action": action,
        "action_prob": round(p["y_act"][action], 3),
        "self_continue": round(p["y_self"], 3),
        "human_reply": round(p["y_hreply"], 3),
        "recheck_after_s": round(p["recheck_after_s"], 1),
    }, ensure_ascii=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/sixhead_interactspeech_magicdata_clean.pt")
    ap.add_argument("--input", help="JSONL file; omit with --interactive")
    ap.add_argument("--interactive", action="store_true")
    args = ap.parse_args()
    if not args.interactive and not args.input:
        ap.error("use --input FILE or --interactive")
    ck = torch.load(Path(args.model), weights_only=False)
    model = TurnModel(ck["n_feat"], ck["use_text"], kind=ck["kind"]).eval()
    model.load_state_dict(ck["state"])
    if ck["use_text"]:
        raise RuntimeError("this demo expects the current timing-only checkpoint (use_text=False)")
    history = []
    now = time.time()
    source = sys.stdin if args.interactive else open(args.input, encoding="utf-8")
    try:
        for i, line in enumerate(source):
            if args.interactive:
                print("event> ", end="", flush=True)
            event = parse_line(line, i, now)
            if event is None:
                continue
            now = max(now + 1.0, float(event["ts"]) + 1.0)
            history.append(event)
            print_result(event, score(history, model, ck))
    finally:
        if not args.interactive:
            source.close()


if __name__ == "__main__":
    main()
