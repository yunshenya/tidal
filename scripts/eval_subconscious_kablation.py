"""Geometry, fit, then held for the K ablation.

``geometry`` encodes keys and does not score a continuation. ``fit`` scores
the fit split only. ``held`` refuses a second complete report and refuses to
run until fit and geometry match this bank. Production ``MIX_K`` and
``MIX_TEMP`` are not written.
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

GEOM_PATH = ROOT / "reports" / "subconscious_kablation_geometry.json"
FIT_PATH = ROOT / "reports" / "subconscious_kablation_fit.json"
FIT_PARTIAL = ROOT / "reports" / "subconscious_kablation_fit_partial.json"
HELD_PATH = ROOT / "reports" / "subconscious_kablation_held.json"
HELD_PARTIAL = ROOT / "reports" / "subconscious_kablation_held_partial.json"


def _git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()


def _load(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _print(message: str) -> None:
    print(message, flush=True)


def _geometry() -> None:
    import torch

    from tidal.llm_memory import LLMMemory
    from tidal.subconscious_kablation_score import collect_geometry, dump_json

    existing = _load(GEOM_PATH)
    if existing and existing.get("complete") and not existing.get("margins_scored"):
        _print(
            f"geometry already recorded separation_ok={existing.get('separation_ok')} "
            f"argmax={existing.get('argmax_hits')}"
        )
        return
    torch.set_num_threads(os.cpu_count() or 1)
    mem = LLMMemory(dtype=torch.float32)
    try:
        started = time.perf_counter()
        report = collect_geometry(mem, progress=_print)
        report["git_sha"] = _git_sha()
        report["elapsed_s"] = time.perf_counter() - started
        report["model"] = "HuggingFaceTB/SmolLM2-135M"
        dump_json(GEOM_PATH, report)
        _print(
            f"geometry complete separation_ok={report['separation_ok']} "
            f"prefix_ok={report['prefix_ok']} argmax={report['argmax_hits']} "
            f"gap_min={report['gap']['min']}"
        )
    finally:
        mem.close()


def _score(kind: str) -> None:
    import torch

    from tidal.llm_memory import LLMMemory
    from tidal.subconscious_kablation_score import dump_json, held_allowed, score_fit, score_held

    final_path = FIT_PATH if kind == "fit" else HELD_PATH
    partial_path = FIT_PARTIAL if kind == "fit" else HELD_PARTIAL
    existing = _load(final_path)
    if existing and existing.get("complete"):
        if kind == "held":
            raise SystemExit("held report already exists; it is not scored again")
        _print(f"fit already scored n={existing.get('summary', {}).get('n_probes')}")
        return
    geometry = _load(GEOM_PATH)
    fit_report = _load(FIT_PATH)
    if kind == "held":
        allowed, reason = held_allowed(fit_report, geometry, False)
        if not allowed:
            raise SystemExit(reason)
    elif not isinstance(geometry, dict) or not geometry.get("separation_ok") or geometry.get("margins_scored"):
        raise SystemExit("geometry is not a completed no-margin report")
    torch.set_num_threads(os.cpu_count() or 1)
    mem = LLMMemory(dtype=torch.float32)
    try:
        seed = _load(partial_path)
        started = time.perf_counter()

        def save(payload):
            payload = dict(payload)
            payload["bank_sha256"] = geometry["bank_sha256"]
            dump_json(partial_path, payload)

        runner = score_fit if kind == "fit" else score_held
        report = runner(mem, progress=_print, save=save, seed=seed)
        report["git_sha"] = _git_sha()
        report["elapsed_s"] = time.perf_counter() - started
        report["model"] = "HuggingFaceTB/SmolLM2-135M"
        dump_json(final_path, report)
        if partial_path.exists():
            partial_path.unlink()
        _print(f"{kind} complete n={report['summary']['n_probes']} elapsed={report['elapsed_s']}")
    finally:
        mem.close()


def _fit() -> None:
    _score("fit")


def _held() -> None:
    _score("held")


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in ("geometry", "fit", "held"):
        raise SystemExit("usage: eval_subconscious_kablation.py geometry|fit|held")
    if sys.argv[1] == "geometry":
        _geometry()
    elif sys.argv[1] == "fit":
        _fit()
    else:
        _held()


if __name__ == "__main__":
    main()
