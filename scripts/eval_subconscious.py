"""Subconscious eval of the shipped SmolLM2 inject path.

Log-probs, greedy checks, and CPU cost. Writes JSON and the markdown report.
Does not train the inject MLP.

    python scripts/eval_subconscious.py
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from tidal.subconscious_eval import _rss_kb, render_report, run_eval


class BGE:
    """CLS top-1 over fact lines. Same pooling as ``scripts/eval_llm_memory.py``.

    No ``Represent this sentence`` prefix. That is the retrieval the explicit
    free-gen numbers used.
    """

    def __init__(self):
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        self.model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").eval()

    @torch.no_grad()
    def __call__(self, texts, query: str) -> str:
        rows = [str(text) for text in texts]
        if not rows:
            raise ValueError("RAG corpus is empty")
        batch = self.tok(rows + [str(query)], padding=True, truncation=True, return_tensors="pt")
        hidden = self.model(**batch).last_hidden_state[:, 0]
        hidden = torch.nn.functional.normalize(hidden, dim=-1)
        scores = hidden[:-1] @ hidden[-1]
        return rows[int(scores.argmax())]


def main() -> None:
    parser = argparse.ArgumentParser(description="subconscious eval of LLMMemory")
    parser.add_argument("--json", default="reports/subconscious_eval.json")
    parser.add_argument("--md", default="reports/subconscious_eval.md")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(os.cpu_count() or 1)
    from tidal.llm_memory import DEFAULT_MODEL, LLMMemory

    print("loading SmolLM2-135M float32", flush=True)
    rss_before = _rss_kb()
    t0 = time.perf_counter()
    mem = LLMMemory(model_name=DEFAULT_MODEL, dtype=torch.float32)
    load_s = time.perf_counter() - t0
    rss_after = _rss_kb()
    print(f"loaded in {load_s:.1f}s layer={mem.layer} hidden={mem.hidden}", flush=True)
    bge = BGE()

    def progress(msg: str) -> None:
        print(msg, flush=True)

    result = run_eval(
        mem,
        bge,
        cost_repeats=args.repeats,
        cost_warmup=args.warmup,
        progress=progress,
    )
    result["load_s"] = float(load_s)
    result["rss_kb_load"] = {"before": rss_before, "after_load": rss_after, "after_eval": _rss_kb()}
    result["cpu_count"] = int(os.cpu_count() or 0)
    text = render_report(result)
    os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
    with open(args.json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, sort_keys=True)
        handle.write("\n")
    with open(args.md, "w", encoding="utf-8") as handle:
        handle.write(text)
    print(f"wrote {args.json} and {args.md}", flush=True)


if __name__ == "__main__":
    main()
