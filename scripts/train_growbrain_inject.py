"""Train the stage0 growbrain key center and inject MLP at d_model 192.

Writes ``tidal/llm_key_center_growbrain.npz`` and
``tidal/llm_inject_growbrain_192.pt``. The base checkpoint is read, not
rewritten. A zero last layer is not saved.
"""
from __future__ import annotations

import hashlib
import json
import random
import time
from pathlib import Path

import torch

from tidal.backend import (
    GROWBRAIN_D_MODEL,
    GROWBRAIN_MEM_DIM,
    GROWBRAIN_STAGE0_SHA256,
    GROWBRAIN_VOCAB_SIZE,
    stage0_checkpoint_files,
)
from tidal.llm_memory import (
    GROWBRAIN_CHECKPOINT,
    InjectMLP,
    LLMMemory,
    _growbrain_center_path,
    _growbrain_inject_path,
    save_key_centers,
)
from tidal.llm_memory_data import fact as make_fact
from tidal.llm_memory_data import names
from tidal.llm_memory_train import fit_inject

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _facts(seed: int) -> list:
    """Every train identity once. The prompt is the answer-stripped prefix."""
    rng = random.Random(int(seed))
    rels = ("phone", "birthday", "city", "color", "job", "team")
    rows = []
    for name in names("train"):
        for rel in rels:
            fact = make_fact(rng, "train", n=name, rel=rel)
            answer = fact["a"]
            text = fact["text"]
            if not text.endswith(answer):
                raise RuntimeError("train fact does not end with its answer")
            prompt = text[: -len(answer)]
            if prompt + answer != text:
                raise RuntimeError("train prompt does not rebuild the fact")
            if len(text) > 120:
                raise RuntimeError(f"train fact is longer than the stage0 context: {text!r}")
            rows.append({"text": text, "a": answer, "q": prompt})
    if len(rows) != 144:
        raise RuntimeError(f"expected 144 train facts, got {len(rows)}")
    return rows


def main() -> None:
    if int(GROWBRAIN_D_MODEL) == int(GROWBRAIN_MEM_DIM):
        raise SystemExit("d_model and mem_dim collapsed; the store width must stay d_model")
    found = stage0_checkpoint_files()
    if found is None:
        raise SystemExit("checkpoints/e1_base_rebalance.pt and vocab/charlist.txt are required")
    ckpt, _vocab = found
    digest = _sha256(ckpt)
    if digest != GROWBRAIN_STAGE0_SHA256:
        raise SystemExit(f"checkpoint sha256 {digest} != {GROWBRAIN_STAGE0_SHA256}")
    torch.set_num_threads(4)
    t0 = time.time()
    mem = LLMMemory(dtype=torch.float32)
    try:
        if type(mem.backend).__name__ != "GrowbrainBackend":
            raise SystemExit("default backend is not GrowbrainBackend")
        if mem.model_name != GROWBRAIN_CHECKPOINT:
            raise SystemExit(f"model name {mem.model_name} is not {GROWBRAIN_CHECKPOINT}")
        if int(mem.hidden) != int(GROWBRAIN_D_MODEL) or int(mem.layer) != 2:
            raise SystemExit(f"hidden {mem.hidden} layer {mem.layer} is not stage0")
        if int(mem.backend.mem_dim) == int(mem.hidden):
            raise SystemExit("store width collapsed onto mem_dim")
        if len(mem.tokenizer) != int(GROWBRAIN_VOCAB_SIZE):
            raise SystemExit(f"tokenizer {len(mem.tokenizer)} != {GROWBRAIN_VOCAB_SIZE}")
        mem.inject = InjectMLP(mem.hidden).to(mem._device)
        if int(torch.count_nonzero(mem.inject.net[-1].weight)) != 0:
            raise SystemExit("fresh inject last layer is not zero")
        facts = _facts(0)
        log = fit_inject(mem, facts, steps=1500, bs=8, lr=1e-3, neg_p=0.2, seed=0)
        if int(torch.count_nonzero(mem.inject.net[-1].weight)) == 0:
            raise SystemExit("training left the inject last layer at zero")
        center = _growbrain_center_path()
        inject = _growbrain_inject_path()
        save_key_centers(mem, center)
        mem.save_inject(inject)
        record = {
            "checkpoint_sha256": digest,
            "steps": 1500,
            "bs": 8,
            "lr": 1e-3,
            "neg_p": 0.2,
            "n_facts": len(facts),
            "seed": 0,
            "hidden": int(mem.hidden),
            "layer": int(mem.layer),
            "mem_dim": int(mem.backend.mem_dim),
            "vocab": len(mem.tokenizer),
            "key_center": str(center.relative_to(ROOT)),
            "inject": str(inject.relative_to(ROOT)),
            "key_center_sha256": _sha256(center),
            "inject_sha256": _sha256(inject),
            "elapsed_s": time.time() - t0,
            "log": log,
        }
        out = ROOT / "reports" / "growbrain_stage0_train.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print(json.dumps({k: record[k] for k in record if k != "log"}, sort_keys=True), flush=True)
    finally:
        mem.close()


if __name__ == "__main__":
    main()
