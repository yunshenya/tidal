"""Fit, then held, for the soft-mixture gate.

``fit`` does not score held probes. ``held`` refuses to run until the fit
report's selected k and temperature are ``MIX_K`` and ``MIX_TEMP``, and it
refuses to start again once a complete held report exists.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FIT_PATH = ROOT / "reports" / "subconscious_softmix_fit.json"
FIT_PARTIAL = ROOT / "reports" / "subconscious_softmix_fit_partial.json"
HELD_PATH = ROOT / "reports" / "subconscious_softmix_held.json"
HELD_PARTIAL = ROOT / "reports" / "subconscious_softmix_held_partial.json"


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _load(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _print(message: str) -> None:
    print(message, flush=True)


def _fit(mem) -> None:
    from tidal.subconscious_softmix_score import dump_json, score_fit

    existing = _load(FIT_PATH)
    if existing and existing.get("complete"):
        _print(f"fit already scored selected={existing.get('selected')}")
        return
    seed = _load(FIT_PARTIAL)
    started = time.perf_counter()

    def save(payload):
        dump_json(FIT_PARTIAL, payload)

    report = score_fit(mem, progress=_print, save=save, seed=seed)
    report["git_sha"] = _git_sha()
    report["elapsed_s"] = time.perf_counter() - started
    report["model"] = "HuggingFaceTB/SmolLM2-135M"
    dump_json(FIT_PATH, report)
    if FIT_PARTIAL.exists():
        FIT_PARTIAL.unlink()
    selected = report.get("selected")
    _print(f"fit complete selected={selected}")
    for row in report["cells"]:
        _print(
            f"k={row['k']} T={row['temperature']:.2f} pass={row['pass']} "
            f"soft={row['soft_dz']} top1={row['top1_dz']} part={row['participation_mean']}"
        )


def _held(mem) -> None:
    from tidal.llm_memory import MIX_K, MIX_TEMP
    from tidal.subconscious_softmix_score import dump_json, held_allowed, score_held

    existing = _load(HELD_PATH)
    if existing and existing.get("complete"):
        raise SystemExit("held report already exists; it is not scored again")
    fit_report = _load(FIT_PATH)
    allowed, reason = held_allowed(fit_report, MIX_K, MIX_TEMP, False)
    if not allowed:
        raise SystemExit(reason)
    seed = _load(HELD_PARTIAL)
    started = time.perf_counter()

    def save(payload):
        dump_json(HELD_PARTIAL, payload)

    report = score_held(mem, fit_report, progress=_print, save=save, seed=seed)
    report["git_sha"] = _git_sha()
    report["elapsed_s"] = time.perf_counter() - started
    report["model"] = "HuggingFaceTB/SmolLM2-135M"
    dump_json(HELD_PATH, report)
    if HELD_PARTIAL.exists():
        HELD_PARTIAL.unlink()
    _print(f"held complete pass={report['pass']} soft={report['implicit']['soft_dz']} top1={report['implicit']['top1_dz']}")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("fit", "held"):
        raise SystemExit("usage: eval_subconscious_softmix.py fit|held")
    import torch

    from tidal.llm_memory import LLMMemory

    torch.set_num_threads(os.cpu_count() or 1)
    mem = LLMMemory(dtype=torch.float32)
    try:
        if sys.argv[1] == "fit":
            _fit(mem)
        else:
            _held(mem)
    finally:
        mem.close()


if __name__ == "__main__":
    main()
