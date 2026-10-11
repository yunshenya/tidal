#!/usr/bin/env python3
"""Create provenance-aware weak labels for pending group-chat turngate rows.

Only rows without a human-consensus label are emitted.  The output is training-only
and keeps ``assistant_auto_v1`` provenance so it cannot be confused with review truth.
"""
import json, re
from pathlib import Path

UNFINISHED = re.compile(r"(?:然后|还有|就是|而且|我跟你说|等下|等等|先|因为|所以|但是|不过|如果|那个|就|和|跟|的)$")
QUESTION = re.compile(r"[?？]|吗[？?]?$|(?:怎么|为什么|为何|谁|哪个|哪一个|什么|啥|能不能|可不可以|可以吗|是否)")
ACK = re.compile(r"^(?:ok|OK|好|好的|嗯|嗯嗯|哦|哦哦|行|可以|收到|哈哈|笑死|666|6+|啊这|确实|对|对的|没错|谢谢|谢了|\+1|👍|😂|😆|🤣|👏)[!！。…~～]*$")
COMMAND = re.compile(r"(?:闭嘴|别说|别回|不用回|不要回|滚|禁言|停一下|安静)")
TERMINAL = re.compile(r"[。！？!?；;…~～]$|[。！？!?；;…~～][\]\)）】》」』]$")

def completion(text):
    s = (text or "").strip()
    if not s or s in {"[image]", "[sticker]"}:
        return "flush_now", 0.55
    if UNFINISHED.search(s) or (len(s) >= 45 and not TERMINAL.search(s) and not s.endswith(("）", ")", "]", "】"))):
        return "hold_for_more", 0.68
    return "flush_now", 0.72

def response(ctx, text):
    s = (text or "").strip()
    addressed = bool(ctx.get("addressed_to_agent") or ctx.get("replies_to_agent"))
    targeting = str(ctx.get("targeting") or "none").lower()
    addressed = addressed or targeting in {"me", "her", "bot", "agent", "assistant"}
    if COMMAND.search(s):
        return "ack", 0.82
    if addressed and QUESTION.search(s):
        return "answer", 0.80
    if addressed and ACK.match(s):
        return "ack", 0.62
    if ctx.get("bot_last_asked_question") and s not in {"[image]", "[sticker]"}:
        return "answer", 0.58
    return "ignore", 0.60 if not addressed else 0.48

def main():
    root = Path(__file__).resolve().parents[1]
    src = root / "data/private/turngate"
    out = src / "review-batch-20260913-auto.jsonl"
    files = [src / "review-batch-20260913.jsonl", src / "review-batch-20260913-speakers.jsonl"]
    n = human = 0
    with out.open("w", encoding="utf-8") as fo:
        for path in files:
            if not path.exists():
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                row = json.loads(line)
                prov = row.get("label_provenance") or {}
                if prov.get("source") == "human_consensus":
                    human += 1
                    continue
                text = row.get("current_text") or ""
                c, cc = completion(text)
                a, ac = response(row.get("context") or {}, text)
                row["labels"] = {"completion": c, "response": a}
                row["label_provenance"] = {
                    "source": "assistant_auto_v1", "agreement": 0.0,
                    "annotator_count": 0, "confidence": {"completion": cc, "response": ac},
                    "method": "deterministic_context_text_rules"
                }
                row["review_status"] = "auto_labeled"
                fo.write(json.dumps(row, ensure_ascii=False) + "\n")
                n += 1
    print(json.dumps({"output": str(out), "auto_rows": n, "skipped_human_rows": human}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
