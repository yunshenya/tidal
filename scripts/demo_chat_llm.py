#!/usr/bin/env python3
"""Interactive six-head gate + OpenAI-compatible reply generation.

The entered demo conversation is sent to the configured LLM only when the
six-head gate selects ``speak`` (or when --always-reply is supplied). No raw
private dataset is loaded by this script.

Privacy: there is no default endpoint or model. A non-local base URL (anything other
than localhost / 127.0.0.0/8 / ::1) is refused unless --allow-external-llm is given,
because everything typed is forwarded to that third party. Never paste real chat.
"""
import ipaddress
from urllib.parse import urlparse
import argparse
import json
import os
import sys
import time
from pathlib import Path

import httpx
import torch

from scripts.demo_sixhead import load_checkpoint, parse_line, score
from tidal.model import TurnModel


def env_first(*names):
    for name in names:
        value = os.environ.get(name)
        if value:
            return value
    return None


def is_local(url):
    host = (urlparse(url).hostname or "").lower()
    if host == "localhost": return True
    try: return ipaddress.ip_address(host).is_loopback
    except ValueError: return False


def context_text(history, limit=12):
    lines = []
    for e in history[-limit:]:
        who = "机器人" if e.get("role") == "self" else str(e.get("speaker") or "群友")
        text = e.get("text") or "<无文本>"
        lines.append(f"{who}: {text}")
    return "\n".join(lines)


def generate(client, model, history, decision, temperature, max_tokens):
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
    ap.add_argument("--base-url", default=env_first("TIDAL_LLM_BASE_URL"), help="OpenAI-compatible base URL (e.g. http://localhost:11434/v1)")
    ap.add_argument("--llm-model", default=env_first("TIDAL_LLM_MODEL"), help="model name; required, no default")
    ap.add_argument("--api-key", default=None, help="API key; prefer TIDAL_LLM_API_KEY (optional for local servers)")
    ap.add_argument("--allow-external-llm", action="store_true", help="permit a non-local endpoint; typed text leaves this machine")
    ap.add_argument("--trust-checkpoint", action="store_true", help="allow full pickle load of the gate checkpoint")
    ap.add_argument("--always-reply", action="store_true", help="generate even when gate selects wait/silent")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()
    base_url = (args.base_url or "").rstrip("/")
    api_key = args.api_key or env_first("TIDAL_LLM_API_KEY")
    if not base_url or not args.llm_model:
        raise SystemExit("缺少大模型配置：请设置 TIDAL_LLM_BASE_URL 和 TIDAL_LLM_MODEL（或 --base-url / --llm-model）。"
                         "推荐本地 Ollama：http://localhost:11434/v1")
    local = is_local(base_url)
    if not local and not args.allow_external_llm:
        raise SystemExit(f"拒绝外部端点 {urlparse(base_url).hostname}：输入内容会发送给第三方。"
                         "确认只输入演示文本后再加 --allow-external-llm。")
    if not local and not api_key:
        raise SystemExit("外部端点需要 TIDAL_LLM_API_KEY 或 --api-key。")
    print(f"警告：触发回复时，最近 12 条输入会发送到 {base_url}"
          f"{'（本机）' if local else '（外部第三方！切勿输入真实聊天）'}。", file=sys.stderr)

    torch.set_num_threads(2)
    ck = load_checkpoint(args.model, args.trust_checkpoint)
    model = TurnModel(ck["n_feat"], ck["use_text"], kind=ck["kind"], mask_text=ck.get("mask_text", False)).eval()
    model.load_state_dict(ck["state"])
    encoder = None
    if ck["use_text"]:
        from tidal.embed import Encoder
        encoder = Encoder(threads=2)
    embedding_cache = {}
    history = []
    now = time.time()
    headers = {"Content-Type": "application/json", **({"Authorization": f"Bearer {api_key}"} if api_key else {})}
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
                    reply = generate(client, args.llm_model, history, decision,
                                     args.temperature, args.max_tokens)
                except httpx.HTTPStatusError as exc:
                    if exc.response.status_code in (400, 401, 403, 404): raise  # configuration error: stop
                    reply = f"[LLM 请求失败: HTTP {exc.response.status_code}]"
                except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
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
