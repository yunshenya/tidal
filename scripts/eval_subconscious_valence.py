"""Score the held-out valence bank once.

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
    VALENCE_SPAN,
    VALENCE_STEP,
)
from tidal.subconscious_replay_data import held_bank as replay_held_bank
from tidal.subconscious_replay_score import score_direction
from tidal.subconscious_valence_data import (
    DIRECTION_HELD_AVOID_MEAN,
    DIRECTION_HELD_FORBIDDEN_LIFT,
    DIRECTION_HELD_PREFER_MEAN,
    REPLAY_AVOID_MEAN_MIN,
    REPLAY_FORBIDDEN_DOWN_MIN,
    REPLAY_HELD_AVOID_MEAN,
    REPLAY_HELD_FORBIDDEN_LIFT,
    REPLAY_HELD_PREFER_MEAN,
    REPLAY_PREFER_MEAN_MIN,
    VALENCE_GATES,
    bank_sha256,
    held_bank,
)
from tidal.subconscious_valence_score import gate_rows, passes, score_bank


def _fmt(value, digits=3):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _fmt_num(value):
    return f"{float(value):g}"


def _load_fit():
    path = ROOT / "reports" / "subconscious_valence_fit.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _regression_rows(direction):
    prefer = direction["prefer"]["summary"]
    avoid = direction["avoid"]["summary"]
    return [
        (
            "replay prefer mean",
            float(prefer["mean_delta"]) >= float(REPLAY_PREFER_MEAN_MIN),
            f"{_fmt(prefer['mean_delta'])} >= {REPLAY_PREFER_MEAN_MIN:.1f} "
            f"({prefer['n_delta_pos']}/{prefer['n']} > 0)",
        ),
        (
            "replay avoid mean",
            float(avoid["mean_delta"]) >= float(REPLAY_AVOID_MEAN_MIN),
            f"{_fmt(avoid['mean_delta'])} >= {REPLAY_AVOID_MEAN_MIN:.1f} "
            f"({avoid['n_delta_pos']}/{avoid['n']} > 0)",
        ),
        (
            "replay forbidden down",
            int(avoid["n_stored_down"]) >= int(REPLAY_FORBIDDEN_DOWN_MIN),
            f"{avoid['n_stored_down']}/{avoid['n']} down, mean stored lift {_fmt(avoid['mean_stored_lift'])}",
        ),
    ]


def _held_pass(result, direction) -> bool:
    return passes(result, VALENCE_GATES, held=True) and all(ok for _name, ok, _detail in _regression_rows(direction))


def render_report(held, fit, direction):
    fit_result = fit["result"]
    held_rows = gate_rows(held, VALENCE_GATES, held=True)
    fit_rows = gate_rows(fit_result, VALENCE_GATES, held=False)
    cost = held["cost"]
    exact = held["exact"]
    prefer = direction["prefer"]["summary"]
    avoid = direction["avoid"]["summary"]
    lines = [
        "# Subconscious experience valence",
        "",
        "Each episodic slot carries a scalar valence. A write's experience field "
        "adds its sign times the step. Replay multiplies a nonzero valence by "
        f"{REPLAY_DECAY:.2f}. The add is `c * s * u * gain * (scale(v) + MLP(scale(v)))`, "
        f"`u = 1 + {_fmt_num(VALENCE_SPAN)} tanh(valence)`, "
        "`gain = 1 + 2 tanh(w · normalize(k))`. "
        f"SmolLM2-135M, layer {held['layer']}, hidden {held['hidden']}, float32. "
        f"gap_tau {_fmt(DEFAULT_GAP_TAU, 2)}. "
        "Phone and birthday spelling copies the winning slot's stored answer, "
        "and only when the prompt is completing that relation. "
        "One query per continuation. The fact text is not scored. "
        "The surprise gate and the retrieval-induced-forgetting gate are not in this model.",
        "",
        "Prefer, avoid, and forbidden stay in scope against the #44 and #46 baselines "
        "(those two held scores match) and against #41. They are not out of scope.",
        "",
        f"Fit bank `{fit_result['bank_sha256']}`. Held-out bank `{held['bank_sha256']}`. "
        "Span and step were chosen on the fit bank before the held-out bank was scored. "
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
    for name, ok, detail in _regression_rows(direction):
        lines.append(f"| held | {name} | {'yes' if ok else 'no'} | {detail} |")
    prefer_44 = float(prefer["mean_delta"]) - float(REPLAY_HELD_PREFER_MEAN)
    avoid_44 = float(avoid["mean_delta"]) - float(REPLAY_HELD_AVOID_MEAN)
    forbid_44 = float(avoid["mean_stored_lift"]) - float(REPLAY_HELD_FORBIDDEN_LIFT)
    prefer_41 = float(prefer["mean_delta"]) - float(DIRECTION_HELD_PREFER_MEAN)
    avoid_41 = float(avoid["mean_delta"]) - float(DIRECTION_HELD_AVOID_MEAN)
    forbid_41 = float(avoid["mean_stored_lift"]) - float(DIRECTION_HELD_FORBIDDEN_LIFT)
    lines.extend(
        [
            "",
            "## Exact spelling",
            "",
            f"- phone question generated {exact['phone_question_text']!r}, "
            f"stored {exact['phone_stored']!r}, exact {exact['phone_exact']}",
            f"- birthday question generated {exact['birthday_question_text']!r}, "
            f"stored {exact['birthday_stored']!r}, exact {exact['birthday_exact']}",
            f"- prefix detector on those questions: phone {exact['phone_prefix']}, "
            f"birthday {exact['birthday_prefix']}. It is not the gate.",
            f"- off-relation takeover phone {float(exact['phone_off_takeover']):.3f}, "
            f"birthday {float(exact['birthday_off_takeover']):.3f}",
            f"- off-relation equal to stored: phone {exact['phone_off_equal']}, "
            f"birthday {exact['birthday_off_equal']}",
            "",
            "## Direction against #44 and #41 held",
            "",
            f"- prefer mean {_fmt(prefer['mean_delta'])} "
            f"({prefer['n_delta_pos']}/{prefer['n']} > 0), "
            f"delta {_fmt(prefer_44)} against #44/#46 {REPLAY_HELD_PREFER_MEAN:.3f}, "
            f"delta {_fmt(prefer_41)} against #41 {DIRECTION_HELD_PREFER_MEAN:.3f}",
            f"- avoid mean {_fmt(avoid['mean_delta'])} "
            f"({avoid['n_delta_pos']}/{avoid['n']} > 0), "
            f"delta {_fmt(avoid_44)} against #44/#46 {REPLAY_HELD_AVOID_MEAN:.3f}, "
            f"delta {_fmt(avoid_41)} against #41 {DIRECTION_HELD_AVOID_MEAN:.3f}",
            f"- forbidden mean stored lift {_fmt(avoid['mean_stored_lift'])} "
            f"({avoid['n_stored_down']}/{avoid['n']} down), "
            f"delta {_fmt(forbid_44)} against #44/#46 {REPLAY_HELD_FORBIDDEN_LIFT:.3f}, "
            f"delta {_fmt(forbid_41)} against #41 {DIRECTION_HELD_FORBIDDEN_LIFT:.3f}",
            "",
            "## Cost",
            "",
            f"- new parameters: {cost['parameters_new']}",
            f"- MLP parameters: {cost['mlp_parameters']}",
            f"- polarity parameters: {cost['polarity_parameters']}",
            f"- total parameters: {cost['parameters']}",
            f"- buffers: {cost['buffers']}",
            f"- forwards inside one remember: {cost['remember_forwards']}",
            f"- forwards inside one armed query: {cost['query_forwards']}",
            "- An experience is a scalar on the slot. It does not add a forward "
            "and there is no feature flag.",
            "",
            f"Polarity span is {POLARITY_SPAN:.0f}. "
            f"Valence span is {_fmt_num(VALENCE_SPAN)} and step is {_fmt_num(VALENCE_STEP)}. "
            "Held-out pass: "
            + ("yes" if _held_pass(held, direction) else "no")
            + ".",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(os.cpu_count() or 1)
    scored = ROOT / "reports" / "subconscious_valence.json"
    if scored.exists():
        raise SystemExit(f"refusing a second held-out pass: {scored}")
    prereg = (ROOT / "reports" / "subconscious_valence_prereg.md").read_text(encoding="utf-8")
    if "not been scored" not in prereg:
        raise SystemExit("preregistration does not say the held-out bank has not been scored")
    if "equals the stored" not in prereg:
        raise SystemExit("preregistration does not lock exact stored spelling")
    held_sha = bank_sha256("held")
    if held_sha not in prereg:
        raise SystemExit("preregistration does not lock the held-out bank")
    if f"valence span: {_fmt_num(VALENCE_SPAN)}" not in prereg:
        raise SystemExit("preregistration span does not match the source constant")
    if f"valence step: {_fmt_num(VALENCE_STEP)}" not in prereg:
        raise SystemExit("preregistration step does not match the source constant")
    fit = _load_fit()
    if fit["result"]["bank_sha256"] != bank_sha256("fit"):
        raise SystemExit("fit json does not match the fit bank")
    if float(fit["chosen"]["span"]) != float(VALENCE_SPAN) or float(fit["chosen"]["step"]) != float(VALENCE_STEP):
        raise SystemExit("fit json rule does not match the source constants")
    print("loading SmolLM2-135M float32", flush=True)
    mem = LLMMemory(dtype=torch.float32)
    bank = held_bank()
    result = score_bank(mem, bank, progress=lambda msg: print(msg, flush=True))
    print("replay held direction", flush=True)
    direction = score_direction(mem, replay_held_bank())
    payload = dict(result)
    payload["replay_direction"] = direction
    scored.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = ROOT / "reports" / "subconscious_valence.md"
    report.write_text(render_report(result, fit, direction), encoding="utf-8")
    print("wrote", scored, flush=True)
    print("wrote", report, flush=True)
    for name, ok, detail in gate_rows(result, VALENCE_GATES, held=True):
        print(f"  {name}: {'yes' if ok else 'NO'} {detail}", flush=True)
    for name, ok, detail in _regression_rows(direction):
        print(f"  {name}: {'yes' if ok else 'NO'} {detail}", flush=True)
    print("held pass", "yes" if _held_pass(result, direction) else "no", flush=True)
    mem.close()


if __name__ == "__main__":
    main()
