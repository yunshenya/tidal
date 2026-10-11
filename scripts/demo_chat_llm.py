#!/usr/bin/env python3
"""Interactive six-head gate + OpenAI-compatible reply generation.

The entered demo conversation is sent to the configured LLM only when the
six-head gate selects ``speak`` (or when --always-reply is supplied). No raw
private dataset is loaded by this script.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
import torch

from scripts.demo_sixhead import parse_line, score
from tidal.model import TurnModel


def env_first(*names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def context_text(history, limit=12):
    lines = []
    for e in history[-limit:]:
        who = "机器人" if e.get("role") == "self" else str(e.get("speaker") or "群友")
        text = e.get("text") or "<无文本>"
        lines.append(f"{who}: {text}")
    return "\n".join(lines)


def generate(client, base_url, api_key, model, history, decision, temperature, max_tokens):
    prompt = (
        "请根据下面的中文群聊上下文自然回复最后一条消息。\n"
        "只输出准备发送的回复正文，不要解释，不要加引号，不要提及模型或决策。\n"
        "保持简短口语化；如果上下文不够明确，礼貌地追问。\n\n"
        f"六头控制器建议：{decision['action']}。\n"
        f"群聊上下文：\n{context_text(history)}"
    )
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": "你是一个自然、克制的中文群聊助手。"},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    r = client.post("/chat/completions", json=body, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="models/sixhead_tsa_group_auto.pt", help="six-head checkpoint")
    ap.add_argument("--base-url", default=env_first("TIDAL_LLM_BASE_URL", "OPENAI_BASE_URL", "DEEPSEEK_BASE_URL"),
                    help="OpenAI-compatible base URL")
    ap.add_argument("--llm-model", default=env_first("TIDAL_LLM_MODEL", "OPENAI_MODEL", "DEEPSEEK_MODEL",) or "deepseek-chat")
    ap.add_argument("--api-key", default=None, help="API key; prefer TIDAL_LLM_API_KEY/OPENAI_API_KEY/DEEPSEEK_API_KEY")
    ap.add_argument("--always-reply", action="store_true", help="generate even when gate selects wait/silent")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()
    base_url = (args.base_url or "").rstrip("/")
    api_key = args.api_key or env_first("TIDAL_LLM_API_KEY", "OPENAI_API_KEY", "DEEPSEEK_API_KEY")
    if not base_url or not api_key:
        raise SystemExit(
            "缺少大模型配置。请设置 TIDAL_LLM_BASE_URL、TIDAL_LLM_API_KEY、TIDAL_LLM_MODEL，"
            "或使用 --base-url / --api-key。"
        )

    torch.set_num_threads(2)
    ck = torch.load(Path(args.model), map_location="cpu", weights_only=False)
    model = TurnModel(ck["n_feat"], ck["use_text"], kind=ck["kind"], mask_text=ck.get("mask_text", False)).eval()
    model.load_state_dict(ck["state"])
    encoder = None
    if ck["use_text"]:
        from tidal.embed import Encoder
        encoder = Encoder(threads=2)
    embedding_cache = {}
    history = []
    now = time.time()
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    print(f"LLM={args.llm_model} gate={args.model}; 输入 JSON 或普通文本，Ctrl-D 退出。")
    with httpx.Client(base_url=base_url, headers=headers) as client:
        i = 0
        for line in sys.stdin:
            if not line.strip():
                continue
            event = parse_line(line, i, now)
            if event is None:
                continue
            now = float(event["ts"])
            history.append(event)
            p = score(history, model, ck, encoder, embedding_cache)
            action = max(p["y_act"], key=p["y_act"].get)
            decision = {
                "action": action,
                "action_prob": round(p["y_act"][action], 3),
                "eot": round(p["y_eot"], 3),
                "addressed_to_bot": round(p["y_addr"], 3),
            }
            reply = None
            if action == "speak" or args.always_reply:
                try:
                    reply = generate(client, base_url, api_key, args.llm_model, history, decision,
                                     args.temperature, args.max_tokens)
                except Exception as exc:
                    reply = f"[LLM 请求失败: {type(exc).__name__}]"
            result = {"event": event.get("text") or "<无文本>", "decision": decision, "reply": reply}
            print(json.dumps(result, ensure_ascii=False), flush=True)
            if reply and not reply.startswith("[LLM 请求失败"):
                bot_event = dict(event, role="self", speaker=None, text=reply, msg_id=f"llm-{i}", ts=now + 0.001,
                                 speaker_known=False)
                history.append(bot_event)
                now = bot_event["ts"]
            i += 1


if __name__ == "__main__":
    main()
