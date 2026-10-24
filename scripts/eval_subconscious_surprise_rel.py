"""Score the held-out relative-surprise bank once.

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

import tidal.episodic as episodic
import tidal.llm_memory as llm_memory
from tidal.llm_memory import (
    DEFAULT_GAP_TAU,
    LLMMemory,
    POLARITY_SPAN,
    REPLAY_DECAY,
    VALENCE_SPAN,
    VALENCE_STEP,
)
from tidal.subconscious_circadian_data import CLOCK_GATES, held_bank as clock_held_bank
from tidal.subconscious_circadian_score import gate_rows as clock_gate_rows
from tidal.subconscious_circadian_score import passes as clock_passes
from tidal.subconscious_circadian_score import score_bank as clock_score_bank
from tidal.subconscious_replay_data import held_bank as replay_held_bank
from tidal.subconscious_replay_score import score_direction
from tidal.subconscious_surprise_rel_data import (
    DIRECTION_HELD_AVOID_MEAN,
    DIRECTION_HELD_FORBIDDEN_LIFT,
    DIRECTION_HELD_PREFER_MEAN,
    REPLAY_AVOID_MEAN_MIN,
    REPLAY_FORBIDDEN_DOWN_MIN,
    REPLAY_HELD_AVOID_MEAN,
    REPLAY_HELD_FORBIDDEN_LIFT,
    REPLAY_HELD_PREFER_MEAN,
    REPLAY_PREFER_MEAN_MIN,
    SURPRISE_GATES,
    bank_sha256,
    held_bank,
)
from tidal.subconscious_surprise_rel_score import gate_rows, passes, score_bank
from tidal.subconscious_valence_data import VALENCE_GATES, held_bank as valence_held_bank
from tidal.subconscious_valence_score import gate_rows as valence_gate_rows
from tidal.subconscious_valence_score import passes as valence_passes
from tidal.subconscious_valence_score import score_bank as valence_score_bank


def _fmt(value, digits=3):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _fmt_num(value):
    return f"{float(value):g}"


def _load_fit():
    path = ROOT / "reports" / "subconscious_surprise_rel_fit.json"
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


def _held_pass(result, direction, valence, clock) -> bool:
    return (
        passes(result, SURPRISE_GATES, held=True)
        and all(ok for _name, ok, _detail in _regression_rows(direction))
        and valence_passes(valence, VALENCE_GATES, held=True)
        and clock_passes(clock, CLOCK_GATES, held=True)
    )


def render_report(held, fit, direction, valence, clock):
    fit_result = fit["result"]
    held_rows = gate_rows(held, SURPRISE_GATES, held=True)
    fit_rows = gate_rows(fit_result, SURPRISE_GATES, held=False)
    cost = held["cost"]
    exact = held["exact"]
    gate = held["gate"]
    prefer = direction["prefer"]["summary"]
    avoid = direction["avoid"]["summary"]
    keeps = fit["keeps"]
    lines = [
        "# Subconscious relative surprise",
        "",
        "A fact is written when it is salient and, once 32 accepted surprises "
        "are stored, its answer-token NLL is at least the quantile of that "
        "window. Fewer stored surprises leaves the write accepted. "
        "The window stores numbers only. Rejected attempts stay out of it. "
        f"SmolLM2-135M, layer {held['layer']}, hidden {held['hidden']}, float32. "
        f"gap_tau {_fmt(DEFAULT_GAP_TAU, 2)}. "
        f"Quantile {_fmt_num(llm_memory.SURPRISE_QUANTILE)} over the last "
        f"{int(llm_memory.SURPRISE_WINDOW)} accepted write surprises. "
        f"Valence stays `u = 1 + {_fmt_num(VALENCE_SPAN)} tanh(valence)` with step "
        f"{_fmt_num(VALENCE_STEP)}. "
        f"Clock span stays {_fmt_num(episodic.CLOCK_SPAN)}. "
        "Phone and birthday spelling copies the winning slot's stored answer, "
        "and only when the prompt is completing that relation. "
        "One query per continuation. Replay stays every 32 accepted writes. "
        "The retrieval-induced-forgetting gate is not in this model.",
        "",
        "Prefer, avoid, and forbidden stay in scope against the #48 baselines "
        "(those match #47, #44, and #46) and against #41. They are not out of scope.",
        "",
        f"Train bank `{bank_sha256('train')}`. Fit bank `{fit_result['bank_sha256']}`. "
        f"Held-out bank `{held['bank_sha256']}`. "
        "The quantile was chosen on the fit bank before the held-out bank was scored. "
        "The train bank is the post-warmup keep floor. Held-out was scored once.",
        "",
        "## Gates",
        "",
        "| bank | gate | pass | detail |",
        "|---|---|---|---|",
    ]
    for name, ok, detail in fit_rows:
        lines.append(f"| fit | {name} | {'yes' if ok else 'no'} | {detail} |")
    lines.append(
        f"| train | keeps written | {'yes' if int(keeps['keep_written']) == int(keeps['keep_n']) else 'no'} | "
        f"{keeps['keep_written']}/{keeps['keep_n']} after {keeps['seed_slots']} seed slots |"
    )
    for name, ok, detail in held_rows:
        lines.append(f"| held | {name} | {'yes' if ok else 'no'} | {detail} |")
    for name, ok, detail in _regression_rows(direction):
        lines.append(f"| held | {name} | {'yes' if ok else 'no'} | {detail} |")
    for name, ok, detail in valence_gate_rows(valence, VALENCE_GATES, held=True):
        lines.append(f"| valence held | {name} | {'yes' if ok else 'no'} | {detail} |")
    for name, ok, detail in clock_gate_rows(clock, CLOCK_GATES, held=True):
        lines.append(f"| circadian held | {name} | {'yes' if ok else 'no'} | {detail} |")
    prefer_48 = float(prefer["mean_delta"]) - float(REPLAY_HELD_PREFER_MEAN)
    avoid_48 = float(avoid["mean_delta"]) - float(REPLAY_HELD_AVOID_MEAN)
    forbid_48 = float(avoid["mean_stored_lift"]) - float(REPLAY_HELD_FORBIDDEN_LIFT)
    prefer_41 = float(prefer["mean_delta"]) - float(DIRECTION_HELD_PREFER_MEAN)
    avoid_41 = float(avoid["mean_delta"]) - float(DIRECTION_HELD_AVOID_MEAN)
    forbid_41 = float(avoid["mean_stored_lift"]) - float(DIRECTION_HELD_FORBIDDEN_LIFT)
    lines.extend(
        [
            "",
            "## Surprise contrast",
            "",
            f"- held lows written {gate['low_written']}/{gate['low_n']}, "
            f"mean surprise {_fmt(gate['low_mean_surprise'])}",
            f"- held highs written {gate['high_written']}/{gate['high_n']}, "
            f"mean surprise {_fmt(gate['high_mean_surprise'])}",
            f"- held window bar {_fmt(gate['bar'])} at quantile {_fmt_num(gate['quantile'])}",
            f"- train keeps {keeps['keep_written']}/{keeps['keep_n']}",
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
            "## Direction against #48/#47/#44 and #41 held",
            "",
            f"- prefer mean {_fmt(prefer['mean_delta'])} "
            f"({prefer['n_delta_pos']}/{prefer['n']} > 0), "
            f"delta {_fmt(prefer_48)} against #48/#47/#44/#46 {REPLAY_HELD_PREFER_MEAN:.3f}, "
            f"delta {_fmt(prefer_41)} against #41 {DIRECTION_HELD_PREFER_MEAN:.3f}",
            f"- avoid mean {_fmt(avoid['mean_delta'])} "
            f"({avoid['n_delta_pos']}/{avoid['n']} > 0), "
            f"delta {_fmt(avoid_48)} against #48/#47/#44/#46 {REPLAY_HELD_AVOID_MEAN:.3f}, "
            f"delta {_fmt(avoid_41)} against #41 {DIRECTION_HELD_AVOID_MEAN:.3f}",
            f"- forbidden mean stored lift {_fmt(avoid['mean_stored_lift'])} "
            f"({avoid['n_stored_down']}/{avoid['n']} down), "
            f"delta {_fmt(forbid_48)} against #48/#47/#44/#46 {REPLAY_HELD_FORBIDDEN_LIFT:.3f}, "
            f"delta {_fmt(forbid_41)} against #41 {DIRECTION_HELD_FORBIDDEN_LIFT:.3f}",
            "",
            "## Cost",
            "",
            f"- new parameters: {cost['parameters_new']}",
            f"- MLP parameters: {cost['mlp_parameters']}",
            f"- polarity parameters: {cost['polarity_parameters']}",
            f"- total parameters: {cost['parameters']}",
            f"- buffers: {cost['buffers']}",
            f"- chat forwards: {cost['chat_forwards']}",
            f"- warmup forwards: {cost['warmup_forwards']}",
            f"- below-window forwards: {cost['below_forwards']}, written {cost['below_written']}",
            f"- above-window forwards: {cost['above_forwards']}, written {cost['above_written']}",
            "- The gate reads the value forward's logits. It does not add a third forward "
            "and there is no feature flag.",
            "",
            f"Polarity span is {POLARITY_SPAN:.0f}. "
            f"Surprise quantile is {_fmt_num(llm_memory.SURPRISE_QUANTILE)}. "
            f"Window is {int(llm_memory.SURPRISE_WINDOW)}. "
            f"Valence span is {_fmt_num(VALENCE_SPAN)} and step is {_fmt_num(VALENCE_STEP)}. "
            f"Clock span is {_fmt_num(episodic.CLOCK_SPAN)}. "
            f"Replay decay is {REPLAY_DECAY:.2f}. "
            "Held-out pass: "
            + ("yes" if _held_pass(held, direction, valence, clock) else "no")
            + ".",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(os.cpu_count() or 1)
    scored = ROOT / "reports" / "subconscious_surprise_rel.json"
    if scored.exists():
        raise SystemExit(f"refusing a second held-out pass: {scored}")
    prereg = (ROOT / "reports" / "subconscious_surprise_rel_prereg.md").read_text(encoding="utf-8")
    if "not been scored" not in prereg:
        raise SystemExit("preregistration does not say the held-out bank has not been scored")
    if "equals the stored" not in prereg:
        raise SystemExit("preregistration does not lock exact stored spelling")
    if "quantile of the last 32" not in prereg:
        raise SystemExit("preregistration does not lock the relative window")
    for kind in ("train", "fit", "held"):
        if bank_sha256(kind) not in prereg:
            raise SystemExit(f"preregistration does not lock the {kind} bank")
    if f"surprise quantile: {_fmt_num(llm_memory.SURPRISE_QUANTILE)}" not in prereg:
        raise SystemExit("preregistration quantile does not match the source constant")
    fit = _load_fit()
    if fit["result"]["bank_sha256"] != bank_sha256("fit"):
        raise SystemExit("fit json does not match the fit bank")
    if float(fit["chosen"]["quantile"]) != float(llm_memory.SURPRISE_QUANTILE):
        raise SystemExit("fit json quantile does not match the source constant")
    print("loading SmolLM2-135M float32", flush=True)
    mem = LLMMemory(dtype=torch.float32)
    bank = held_bank()
    result = score_bank(mem, bank, progress=lambda msg: print(msg, flush=True))
    print("replay held direction", flush=True)
    direction = score_direction(mem, replay_held_bank())
    print("valence held floor", flush=True)
    valence = valence_score_bank(mem, valence_held_bank(), progress=lambda msg: print(msg, flush=True))
    print("circadian held floor", flush=True)
    clock = clock_score_bank(mem, clock_held_bank(), progress=lambda msg: print(msg, flush=True))
    payload = dict(result)
    payload["replay_direction"] = direction
    payload["valence_held_pass"] = bool(valence_passes(valence, VALENCE_GATES, held=True))
    payload["circadian_held_pass"] = bool(clock_passes(clock, CLOCK_GATES, held=True))
    payload["valence_held"] = valence
    payload["circadian_held"] = clock
    scored.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = ROOT / "reports" / "subconscious_surprise_rel.md"
    report.write_text(render_report(result, fit, direction, valence, clock), encoding="utf-8")
    print("wrote", scored, flush=True)
    print("wrote", report, flush=True)
    for name, ok, detail in gate_rows(result, SURPRISE_GATES, held=True):
        print(f"  {name}: {'yes' if ok else 'NO'} {detail}", flush=True)
    for name, ok, detail in _regression_rows(direction):
        print(f"  {name}: {'yes' if ok else 'NO'} {detail}", flush=True)
    print("valence held", "yes" if payload["valence_held_pass"] else "no", flush=True)
    print("circadian held", "yes" if payload["circadian_held_pass"] else "no", flush=True)
    print("held pass", "yes" if _held_pass(result, direction, valence, clock) else "no", flush=True)
    mem.close()


if __name__ == "__main__":
    main()
