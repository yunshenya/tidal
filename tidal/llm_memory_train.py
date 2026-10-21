"""Fit the residual inject MLP on a frozen causal LM.

Value = answer-span pool at the inject layer (not L2-normalized).
``add = scale(v) + MLP(scale(v))`` with the MLP's last layer starting at zero.
Teacher-forced next-token cross-entropy on the answer, given the prompt only
(the fact text is not in the prompt). A fraction of steps adds the v21 hinge
on a wrong value.

CPU smoke, no download::

    python -m tidal.llm_memory_train --tiny --steps 1 --n-cache 4 --neg-p 0

The original SmolLM run used 1500 steps, batch 8, 2000 cached facts, lr 1e-3,
and a 20% wrong-value hinge. This script's defaults are that step count and
hinge, with a smaller cache so a CPU box can finish a real checkpoint later.
``--model`` defaults to SmolLM2-135M and will download it. ``--tiny`` does not.

Retrieval is the episodic store. This script does not train a second q/k
adapter; contrastive questions for tidal features stay in ``tidal.question_head``.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

_NAMES = ("Ada", "Bea", "Cal", "Dee", "Ezra", "Finn")
_TABLE = {
    "phone": ("555-0100", "555-0199", "555-0134", "555-0118"),
    "city": ("Lisbon", "Milan", "Oslo", "Bern"),
    "color": ("green", "amber", "cobalt", "ivory"),
    "team": ("north", "harbor", "cedar", "lantern"),
}


def sample_facts(n: int, seed: int = 0) -> list:
    """Synthetic ``(fact, prompt, answer)`` rows. ASCII, so the byte tokenizer fits."""
    rng = random.Random(int(seed))
    rels = list(_TABLE)
    facts = []
    for i in range(int(n)):
        name = _NAMES[i % len(_NAMES)]
        rel = rels[i % len(rels)]
        # Rotate the answer with i so a cache of one per name/rel still varies.
        choices = _TABLE[rel]
        answer = choices[(i // max(1, len(_NAMES))) % len(choices)]
        if rng.random() < 0.5:
            answer = rng.choice(choices)
        prompt = f"{name}'s {rel} is "
        facts.append(
            {
                "text": prompt + answer,
                "a": answer,
                "q": prompt,
                "n": name,
                "rel": rel,
            }
        )
    return facts


def _cache(mem, facts: list) -> list:
    rows = []
    for fact in facts:
        value = mem.encode_value(fact["text"], fact["a"])
        rows.append(
            {
                "value": value,
                "prompt": fact["q"],
                "answer": fact["a"],
                "text": fact["text"],
            }
        )
    return rows


def fit_inject(
    mem,
    facts: list,
    *,
    steps: int = 1500,
    bs: int = 8,
    lr: float = 1e-3,
    neg_p: float = 0.2,
    seed: int = 0,
    weight_decay: float = 1e-4,
) -> list:
    """Teacher-forced CE on ``facts``. Returns the logged step records.

    ``facts`` items are mappings with ``text``, ``a`` (answer), and ``q``
    (prompt). The prompt should not contain the answer span.
    """
    import torch

    if steps < 1:
        raise ValueError("steps must be >= 1")
    if bs < 1:
        raise ValueError("bs must be >= 1")
    if not facts:
        raise ValueError("facts is empty")
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    cache = _cache(mem, facts)
    opt = torch.optim.AdamW(mem.inject.parameters(), lr=float(lr), weight_decay=float(weight_decay))
    mem.inject.train()
    log = []
    try:
        for step in range(1, int(steps) + 1):
            losses = []
            for _ in range(int(bs)):
                ex = cache[random.randrange(len(cache))]
                loss = mem.teacher_force_loss(ex["prompt"], ex["answer"], ex["value"])
                if neg_p and random.random() < float(neg_p) and len(cache) > 1:
                    other = ex
                    for _try in range(8):
                        other = cache[random.randrange(len(cache))]
                        if other["answer"] != ex["answer"]:
                            break
                    if other["answer"] != ex["answer"]:
                        loss_neg = mem.teacher_force_loss(ex["prompt"], ex["answer"], other["value"])
                        loss = loss + torch.relu(loss.new_tensor(2.0) - loss_neg) * 0.3
                losses.append(loss)
            loss = torch.stack(losses).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            mem._delta = None
            if step == 1 or step % 100 == 0 or step == int(steps):
                row = {"step": step, "loss": float(loss.detach())}
                log.append(row)
                print(row, flush=True)
    finally:
        mem.inject.eval()
        mem._delta = None
    return log


def main(argv=None) -> None:
    import torch

    from tidal.llm_memory import DEFAULT_MODEL, LLMMemory, build_tiny_lm

    p = argparse.ArgumentParser(description="Fit residual inject on a frozen causal LM (CPU ok).")
    p.add_argument("--model", default=DEFAULT_MODEL, help="Hugging Face causal LM. Ignored with --tiny.")
    p.add_argument("--tiny", action="store_true", help="Random 4-layer Llama and a byte tokenizer. No download.")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--bs", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--neg-p", type=float, default=0.2, dest="neg_p")
    p.add_argument("--n-cache", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layer", type=int, default=None)
    p.add_argument("--hid", type=int, default=None, help="MLP width. Default: the LM hidden size.")
    p.add_argument("--out", default="models/llm_inject.pt")
    args = p.parse_args(argv)

    torch.set_num_threads(max(1, min(8, torch.get_num_threads())))
    if args.tiny:
        model, tok = build_tiny_lm(seed=args.seed)
        mem = LLMMemory(model, tok, layer=args.layer, inject_hidden=args.hid)
    else:
        mem = LLMMemory(model_name=args.model, layer=args.layer, inject_hidden=args.hid)
    facts = sample_facts(args.n_cache, seed=args.seed)
    log = fit_inject(
        mem,
        facts,
        steps=args.steps,
        bs=args.bs,
        lr=args.lr,
        neg_p=args.neg_p,
        seed=args.seed,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "inject": mem.inject.state_dict(),
            "layer": mem.layer,
            "hidden": mem.hidden,
            "log": log,
            "args": vars(args),
        },
        out,
    )
    print("saved", out, flush=True)
    mem.close()


if __name__ == "__main__":
    main()
