"""Score the held-out spelling bank once.

Refuses a second pass. Does not edit the preregistration and does not retune.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from tidal.llm_memory import (
    DEFAULT_GAP_TAU,
    LLMMemory,
    POLARITY_SPAN,
    REPLAY_DECAY,
    REPLAY_EVERY,
    REPLAY_M,
)
from tidal.subconscious_replay_data import held_bank as replay_held_bank
from tidal.subconscious_replay_score import score_direction
from tidal.subconscious_spell_data import (
    DIRECTION_HELD_AVOID_MEAN,
    DIRECTION_HELD_FORBIDDEN_LIFT,
    DIRECTION_HELD_PREFER_MEAN,
    REPLAY_HELD_AVOID_MEAN,
    REPLAY_HELD_FORBIDDEN_LIFT,
    REPLAY_HELD_PREFER_MEAN,
    SPELL_GATES,
    bank_sha256,
    held_bank,
)
from tidal.subconscious_spell_score import gate_rows, passes, score_spell


def _fmt(value, digits=3):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _load_fit():
    path = ROOT / "reports" / "subconscious_spell_fit.json"
    return json.loads(path.read_text(encoding="utf-8"))


def render_report(held, fit, direction):
    fit_result = fit["result"]
    fit_rows = gate_rows(fit_result, SPELL_GATES)
    held_rows = gate_rows(held, SPELL_GATES, direction=direction)
    prefer = direction["prefer"]["summary"]
    avoid = direction["avoid"]["summary"]
    exact = held["exact"]
    cost = held["cost"]
    lines = [
        "# Subconscious exact phone and birthday spelling",
        "",
        "Phone and birthday spelling copies the winning slot's stored answer, "
        "the full number or the full date. It does not spell a trained prefix "
        "and it does not spell a digit readout. It runs only when the prompt is "
        "completing that relation. SmolLM2-135M, "
        f"layer {held['layer']}, hidden {held['hidden']}, float32. "
        f"gap_tau {_fmt(DEFAULT_GAP_TAU, 2)}. "
        "The add is `c * s * gain * (scale(v) + MLP(scale(v)))`, "
        "`gain = 1 + 2 tanh(w · normalize(k))`. "
        f"Replay refreshes the top {int(REPLAY_M)} slot every {int(REPLAY_EVERY)} "
        f"accepted writes. Decay is {REPLAY_DECAY:.1f}. "
        "One query per continuation. No feature flag.",
        "",
        f"Fit bank `{fit_result['bank_sha256']}`. Held-out bank `{held['bank_sha256']}`. "
        "The rule was locked on the fit bank before this held-out bank was scored. "
        "Held-out was scored once.",
        "",
        "## Gates",
        "",
        "| bank | gate | pass | detail |",
        "|---|---|---|---|",
    ]
    for name, ok, detail in fit_rows:
        lines.append(f"| fit | {name} | {'yes' if ok else 'no'} | {detail} |")
    for name, ok, detail in held_rows:
        lines.append(f"| held | {name} | {'yes' if ok else 'no'} | {detail} |")
    prefer_delta = float(prefer["mean_delta"]) - float(REPLAY_HELD_PREFER_MEAN)
    avoid_delta = float(avoid["mean_delta"]) - float(REPLAY_HELD_AVOID_MEAN)
    forbid_delta = float(avoid["mean_stored_lift"]) - float(REPLAY_HELD_FORBIDDEN_LIFT)
    prefer_41 = float(prefer["mean_delta"]) - float(DIRECTION_HELD_PREFER_MEAN)
    avoid_41 = float(avoid["mean_delta"]) - float(DIRECTION_HELD_AVOID_MEAN)
    forbid_41 = float(avoid["mean_stored_lift"]) - float(DIRECTION_HELD_FORBIDDEN_LIFT)
    lines.extend(
        [
            "",
            "## Stored strings",
            "",
            f"- phone question `{exact['phone_question_text']}`",
            f"- stored phone `{exact['phone_stored']}`",
            f"- exact {exact['phone_exact']}, prefix detector {exact['phone_prefix']}",
            f"- birthday question `{exact['birthday_question_text']}`",
            f"- stored birthday `{exact['birthday_stored']}`",
            f"- exact {exact['birthday_exact']}, prefix detector {exact['birthday_prefix']}",
            "",
            "## Prefer and avoid against #44 held",
            "",
            f"- prefer mean {_fmt(prefer['mean_delta'])} "
            f"({prefer['n_delta_pos']}/{prefer['n']} > 0), "
            f"delta {_fmt(prefer_delta)} against {REPLAY_HELD_PREFER_MEAN:.3f}",
            f"- avoid mean {_fmt(avoid['mean_delta'])} "
            f"({avoid['n_delta_pos']}/{avoid['n']} > 0), "
            f"delta {_fmt(avoid_delta)} against {REPLAY_HELD_AVOID_MEAN:.3f}",
            f"- forbidden mean stored lift {_fmt(avoid['mean_stored_lift'])} "
            f"({avoid['n_stored_down']}/{avoid['n']} down), "
            f"delta {_fmt(forbid_delta)} against {REPLAY_HELD_FORBIDDEN_LIFT:.3f}",
            "",
            "## Prefer and avoid against #41 held",
            "",
            f"- prefer delta {_fmt(prefer_41)} against {DIRECTION_HELD_PREFER_MEAN:.3f}",
            f"- avoid delta {_fmt(avoid_41)} against {DIRECTION_HELD_AVOID_MEAN:.3f}",
            f"- forbidden delta {_fmt(forbid_41)} against {DIRECTION_HELD_FORBIDDEN_LIFT:.3f}",
            "",
            "## Cost",
            "",
            f"- new parameters: {cost['parameters_new']}",
            f"- MLP parameters: {cost['mlp_parameters']}",
            f"- polarity parameters: {cost['polarity_parameters']}",
            f"- total parameters: {cost['parameters']}",
            f"- buffers: {cost['buffers']}",
            f"- forwards inside one remember: {cost['remember_forwards']}",
            f"- forwards inside one armed phone question: {cost['query_forwards']}",
            "- The phone question returns the stored string. "
            "There is no second query and no feature flag.",
            "",
            f"Polarity span is {POLARITY_SPAN:.0f}. Held-out pass: "
            + ("yes" if passes(held, SPELL_GATES, direction=direction) else "no")
            + ".",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(os.cpu_count() or 1)
    scored = ROOT / "reports" / "subconscious_spell.json"
    if scored.exists():
        raise SystemExit(f"refusing a second held-out pass: {scored}")
    prereg = (ROOT / "reports" / "subconscious_spell_prereg.md").read_text(encoding="utf-8")
    if "not been scored" not in prereg:
        raise SystemExit("preregistration does not say the held-out bank has not been scored")
    held_sha = bank_sha256("held")
    if held_sha not in prereg:
        raise SystemExit("preregistration does not lock the held-out bank")
    fit = _load_fit()
    if fit["result"]["bank_sha256"] != bank_sha256("fit"):
        raise SystemExit("fit json does not match the fit bank")
    print("loading SmolLM2-135M float32", flush=True)
    mem = LLMMemory(dtype=torch.float32)
    result = score_spell(mem, held_bank(), progress=lambda msg: print(msg, flush=True))
    print("direction", flush=True)
    direction = score_direction(mem, replay_held_bank())
    payload = dict(result)
    payload["direction"] = direction
    scored.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = ROOT / "reports" / "subconscious_spell.md"
    report.write_text(render_report(result, fit, direction), encoding="utf-8")
    print("wrote", scored, flush=True)
    print("wrote", report, flush=True)
    for name, ok, detail in gate_rows(result, SPELL_GATES, direction=direction):
        print(f"  {name}: {'yes' if ok else 'NO'} {detail}", flush=True)
    mem.close()


if __name__ == "__main__":
    main()
