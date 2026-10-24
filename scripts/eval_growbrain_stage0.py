"""Score protocol ``growbrain-stage0-v1``. Fit, then held, once.

No constant is changed. Explicit phone and birthday strings are reported
and are not an input to Cohen's ``d_z``.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import torch

from tidal.backend import GROWBRAIN_STAGE0_SHA256, stage0_checkpoint_files
from tidal.growbrain_stage0_data import FIT, HELD, PROTOCOL, bank_sha256
from tidal.llm_memory import (
    CUE_FAN_S,
    CUE_HALF_LIFE,
    CUE_INTERFACE_VERSION,
    CUE_LAMBDA,
    CUE_THETA,
    LLMMemory,
    RIF_SPAN,
    _growbrain_center_path,
    _growbrain_inject_path,
)
from tidal.subconscious_measure_data import cohens_dz, subconscious_score, wilson_interval

ROOT = Path(__file__).resolve().parents[1]
IMPLICIT = ("tone", "choice", "preference")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _ids(tokenizer, text: str) -> list:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    return [int(i) for i in ids]


def _mean_logp(mem, prompt: str, continuation: str, *, live: bool) -> float:
    prompt_ids = _ids(mem.tokenizer, prompt)
    full_ids = _ids(mem.tokenizer, prompt + continuation)
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise RuntimeError(f"continuation retokenized the prompt: {prompt!r} + {continuation!r}")
    cont = full_ids[len(prompt_ids) :]
    if not cont or not prompt_ids:
        raise RuntimeError("continuation produced no predictable token")
    ids = torch.tensor([full_ids], dtype=torch.long, device=mem._device)
    mem._cue_live = bool(live)
    mem._train_delta = None
    try:
        with torch.no_grad():
            logits = mem.model(ids)
            if isinstance(logits, tuple):
                logits = logits[0]
            pos = [len(prompt_ids) - 1 + i for i in range(len(cont))]
            pred = logits[0, pos].float()
            logp = torch.log_softmax(pred, dim=-1)
            target = torch.tensor(cont, dtype=torch.long, device=logp.device)
            taken = logp.gather(1, target.unsqueeze(1)).squeeze(1)
            vals = [float(x) for x in taken.detach().cpu().tolist()]
    finally:
        mem._cue_live = False
    return float(sum(vals) / len(vals))


def _margin(mem, item, *, live: bool) -> float:
    good = _mean_logp(mem, item["prompt"], item["good"], live=live)
    bad = _mean_logp(mem, item["prompt"], item["bad"], live=live)
    return good - bad


def _score_item(mem, item) -> dict:
    mem.store.clear()
    written = mem.remember(item["text"], item["answer"], name=item["name"], rel=item["rel"])
    rank = mem._rank(item["prompt"])
    top = rank.get("top")
    row = {
        "id": item["id"],
        "family": item["family"],
        "written": bool(written),
        "maxcos": rank.get("maxcos"),
        "retrieval_hit": int(top == (item["name"], item["rel"])),
    }
    if item["family"] in IMPLICIT:
        on = _margin(mem, item, live=True)
        off = _margin(mem, item, live=False)
        row.update(
            {
                "margin_on": on,
                "margin_off": off,
                "delta": on - off,
                "good": item["good"],
                "bad": item["bad"],
            }
        )
        return row
    spelled = mem.generate(item["prompt"], max_new_tokens=24)
    row.update(
        {
            "gold": item["answer"],
            "spelled": spelled,
            "exact": int(spelled == item["answer"]),
        }
    )
    return row


def _family(rows, family: str) -> dict:
    picked = [row for row in rows if row["family"] == family]
    diffs = [row["delta"] for row in picked]
    positives = sum(1 for value in diffs if value > 0.0)
    low, high = wilson_interval(positives, len(picked))
    misses = [
        {
            "id": row["id"],
            "delta": row["delta"],
            "margin_on": row["margin_on"],
            "margin_off": row["margin_off"],
            "maxcos": row["maxcos"],
            "retrieval_hit": row["retrieval_hit"],
            "good": row["good"],
            "bad": row["bad"],
        }
        for row in picked
        if row["delta"] <= 0.0
    ]
    return {
        "n": len(picked),
        "d_z": cohens_dz(diffs),
        "mean_delta": sum(diffs) / len(diffs) if diffs else None,
        "delta_positive": {"k": positives, "n": len(picked), "wilson95": [low, high]},
        "retrieval_hits": sum(row["retrieval_hit"] for row in picked),
        "misses": misses,
    }


def _explicit(rows, family: str) -> dict:
    picked = [row for row in rows if row["family"] == family]
    hits = sum(row["exact"] for row in picked)
    low, high = wilson_interval(hits, len(picked))
    return {
        "n": len(picked),
        "exact": {"k": hits, "n": len(picked), "wilson95": [low, high]},
        "retrieval_hits": sum(row["retrieval_hit"] for row in picked),
        "rows": [
            {
                "id": row["id"],
                "gold": row["gold"],
                "spelled": row["spelled"],
                "exact": row["exact"],
                "maxcos": row["maxcos"],
                "retrieval_hit": row["retrieval_hit"],
            }
            for row in picked
        ],
        "misses": [
            {"id": row["id"], "gold": row["gold"], "spelled": row["spelled"]}
            for row in picked
            if row["exact"] != 1
        ],
    }


def _report(mem, items, *, split: str, elapsed: float, weight_sha: str) -> dict:
    rows = []
    for item in items:
        row = _score_item(mem, item)
        rows.append(row)
        print({"id": row["id"], "family": row["family"]}, flush=True)
    families = {name: _family(rows, name) for name in IMPLICIT}
    score = subconscious_score({name: families[name]["d_z"] for name in IMPLICIT})
    return {
        "protocol": PROTOCOL,
        "split": split,
        "bank_sha256": bank_sha256(),
        "checkpoint_sha256": GROWBRAIN_STAGE0_SHA256,
        "weight_sha256": weight_sha,
        "interface_version": int(CUE_INTERFACE_VERSION),
        "hidden": int(mem.hidden),
        "layer": int(mem.layer),
        "mem_dim": int(mem.backend.mem_dim),
        "theta": float(CUE_THETA),
        "half_life": float(CUE_HALF_LIFE),
        "lambda": float(CUE_LAMBDA),
        "fan_s": float(CUE_FAN_S),
        "rif_span": float(RIF_SPAN),
        "families": families,
        "subconscious_score": score,
        "phone": _explicit(rows, "phone"),
        "birthday": _explicit(rows, "birthday"),
        "elapsed_s": elapsed,
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    found = stage0_checkpoint_files()
    if found is None:
        raise SystemExit("checkpoint and vocab are required")
    ckpt, _vocab = found
    if _sha256(ckpt) != GROWBRAIN_STAGE0_SHA256:
        raise SystemExit("checkpoint sha256 does not match the stage0 pin")
    center = _growbrain_center_path()
    inject = _growbrain_inject_path()
    if not center.is_file() or not inject.is_file():
        raise SystemExit("train the 192-d key center and inject MLP before scoring")
    weight_sha = _sha256(center) + ":" + _sha256(inject)
    torch.set_num_threads(4)
    mem = LLMMemory(dtype=torch.float32)
    try:
        if int(mem.hidden) != 192 or int(mem.backend.mem_dim) != 64 or int(mem.layer) != 2:
            raise SystemExit("loaded model is not the stage0 pin")
        if int(torch.count_nonzero(mem.inject.net[-1].weight)) == 0:
            raise SystemExit("inject last layer is still zero")
        t0 = time.time()
        fit = _report(mem, FIT, split="fit", elapsed=0.0, weight_sha=weight_sha)
        fit["elapsed_s"] = time.time() - t0
        if _sha256(center) + ":" + _sha256(inject) != weight_sha:
            raise SystemExit("fit scoring changed the weights")
        _write(ROOT / "reports" / "growbrain_stage0_fit.json", fit)
        t1 = time.time()
        held = _report(mem, HELD, split="held", elapsed=0.0, weight_sha=weight_sha)
        held["elapsed_s"] = time.time() - t1
        if _sha256(center) + ":" + _sha256(inject) != weight_sha:
            raise SystemExit("held scoring changed the weights")
        _write(ROOT / "reports" / "growbrain_stage0_held.json", held)
        print(
            json.dumps(
                {
                    "fit_score": fit["subconscious_score"],
                    "held_score": held["subconscious_score"],
                    "fit_phone": fit["phone"]["exact"],
                    "held_phone": held["phone"]["exact"],
                    "fit_birthday": fit["birthday"]["exact"],
                    "held_birthday": held["birthday"]["exact"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    finally:
        mem.close()


if __name__ == "__main__":
    main()
