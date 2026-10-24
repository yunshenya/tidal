"""Fit, then held, for same-relation surface repeats.

``fit`` does not load the language model and does not score held probes.
``held`` refuses to run until ``REPEAT_JACCARD`` is the fit selection, and
it refuses to start again once a complete held report exists.
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

FIT_PATH = ROOT / "reports" / "subconscious_interfere_fit.json"
HELD_PATH = ROOT / "reports" / "subconscious_interfere_held.json"
HELD_PARTIAL = ROOT / "reports" / "subconscious_interfere_held_partial.json"


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _load(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _print(message: str) -> None:
    print(message, flush=True)


def _fit() -> None:
    from tidal.subconscious_interfere_score import dump_json, score_fit

    existing = _load(FIT_PATH)
    if existing and existing.get("complete"):
        _print(f"fit already scored selected={existing.get('selected')}")
        return
    started = time.perf_counter()
    report = score_fit()
    report["git_sha"] = _git_sha()
    report["elapsed_s"] = time.perf_counter() - started
    dump_json(FIT_PATH, report)
    _print(f"fit complete selected={report.get('selected')}")
    for row in report["cells"]:
        _print(f"threshold={row['threshold']} pass={row['pass']}")


def _held() -> None:
    import torch

    import tidal.episodic as episodic
    from tidal.llm_memory import LLMMemory
    from tidal.subconscious_interfere_score import dump_json, held_allowed, score_held

    existing = _load(HELD_PATH)
    if existing and existing.get("complete"):
        raise SystemExit("held report already exists; it is not scored again")
    fit_report = _load(FIT_PATH)
    allowed, reason = held_allowed(fit_report, episodic.REPEAT_JACCARD, False)
    if not allowed:
        raise SystemExit(reason)
    torch.set_num_threads(os.cpu_count() or 1)
    mem = LLMMemory(dtype=torch.float32)
    try:
        seed = _load(HELD_PARTIAL)
        started = time.perf_counter()

        def save(payload):
            payload = dict(payload)
            payload["bank_sha256"] = fit_report["bank_sha256"]
            payload["measure_bank"] = fit_report["measure_bank"]
            dump_json(HELD_PARTIAL, payload)

        report = score_held(mem, fit_report, progress=_print, save=save, seed=seed)
        report["git_sha"] = _git_sha()
        report["elapsed_s"] = time.perf_counter() - started
        report["model"] = "HuggingFaceTB/SmolLM2-135M"
        dump_json(HELD_PATH, report)
        if HELD_PARTIAL.exists():
            HELD_PARTIAL.unlink()
        _print(
            f"held complete pass={report['pass']} "
            f"soft={report['implicit']['soft_dz']} ge4={report['quartets']['ge4']}"
        )
    finally:
        mem.close()


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("fit", "held"):
        raise SystemExit("usage: eval_subconscious_interfere.py fit|held")
    if sys.argv[1] == "fit":
        _fit()
    else:
        _held()


if __name__ == "__main__":
    main()
