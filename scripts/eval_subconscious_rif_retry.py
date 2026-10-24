"""Score the held-out retrieval-forgetting bank once.

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
from tidal.subconscious_rif_retry_data import (
    DIRECTION_HELD_AVOID_MEAN,
    DIRECTION_HELD_FORBIDDEN_LIFT,
    DIRECTION_HELD_PREFER_MEAN,
    REPLAY_AVOID_MEAN_MIN,
    REPLAY_FORBIDDEN_DOWN_MIN,
    REPLAY_HELD_AVOID_MEAN,
    REPLAY_HELD_FORBIDDEN_LIFT,
    REPLAY_HELD_PREFER_MEAN,
    HUMAN_EFFECT,
    REPLAY_PREFER_MEAN_MIN,
    RIF_GATES,
    bank_sha256,
    held_bank,
)
from tidal.subconscious_rif_retry_score import gate_rows, passes, score_bank
from tidal.subconscious_surprise_rel_data import SURPRISE_GATES, held_bank as surprise_held_bank
from tidal.subconscious_surprise_rel_score import gate_rows as surprise_gate_rows
from tidal.subconscious_surprise_rel_score import passes as surprise_passes
from tidal.subconscious_surprise_rel_score import score_bank as surprise_score_bank
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
    path = ROOT / "reports" / "subconscious_rif_retry_fit.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _direction_rows(direction):
    prefer = direction["prefer"]["summary"]
    avoid = direction["avoid"]["summary"]
    return [
        (
            "replay prefer count",
            int(prefer["n"]) == 8
            and int(prefer["n_delta_pos"]) >= int(RIF_GATES["prefer_delta_pos_min"])
            and float(prefer["mean_delta"]) >= float(REPLAY_PREFER_MEAN_MIN),
            f"{prefer['n_delta_pos']}/{prefer['n']} > 0, mean {_fmt(prefer['mean_delta'])} "
            f">= {REPLAY_PREFER_MEAN_MIN:.1f}",
        ),
        (
            "replay avoid count",
            int(avoid["n"]) == 8
            and int(avoid["n_delta_pos"]) >= int(RIF_GATES["avoid_delta_pos_min"])
            and float(avoid["mean_delta"]) >= float(REPLAY_AVOID_MEAN_MIN),
            f"{avoid['n_delta_pos']}/{avoid['n']} > 0, mean {_fmt(avoid['mean_delta'])} "
            f">= {REPLAY_AVOID_MEAN_MIN:.1f}",
        ),
        (
            "replay forbidden down",
            int(avoid["n_stored_down"]) >= int(REPLAY_FORBIDDEN_DOWN_MIN),
            f"{avoid['n_stored_down']}/{avoid['n']} down, mean stored lift {_fmt(avoid['mean_stored_lift'])}",
        ),
    ]


def _held_pass(result, direction, valence, clock, surprise) -> bool:
    return (
        passes(result, RIF_GATES, held=True)
        and all(ok for _name, ok, _detail in _direction_rows(direction))
        and valence_passes(valence, VALENCE_GATES, held=True)
        and clock_passes(clock, CLOCK_GATES, held=True)
        and surprise_passes(surprise, SURPRISE_GATES, held=True)
    )


def _clears(ok: bool) -> str:
    return "clears" if ok else "misses"


def _effect_lines(quartets):
    """Map this score onto Rp- < Nrp, Rp+ not weakened, Nrp unaffected."""
    min_drop = float(quartets["competitor_drop_min"])
    mean_drop = float(quartets["competitor_drop_mean"])
    winner = float(quartets["winner_abs_drop_max"])
    outsider = float(quartets["noncompetitor_abs_drop_max"])
    rp_ok = (
        min_drop >= float(RIF_GATES["competitor_drop_min"])
        and mean_drop >= float(RIF_GATES["competitor_mean_drop_min"])
    )
    winner_ok = winner <= float(RIF_GATES["winner_abs_drop_max"])
    outsider_ok = outsider <= float(RIF_GATES["noncompetitor_abs_drop_max"])
    return [
        "## This score against that effect",
        "",
        f"- Rp- competitors versus their strength-1 Nrp controls: min drop {_fmt(min_drop)}, "
        f"mean drop {_fmt(mean_drop)}. Anderson, Bjork, and Bjork (1994) predicted Rp- "
        "below Nrp, by about 11 recall points overall (37.5% versus 48.4%) and by "
        "15.7 points for strong exemplars or 6.3 points for weak ones. This row is that "
        "direction in logprob margin: a partial drop, with the items still stored. "
        f"The pre-registered floors are a minimum of {RIF_GATES['competitor_drop_min']:.2f} "
        f"and a mean of {RIF_GATES['competitor_mean_drop_min']:.2f}. "
        f"This bank {_clears(rp_ok)} them.",
        f"- Rp+ target absolute drop {_fmt(winner)}. Human Rp+ recall was facilitated "
        "(73.6% versus the 48.4% Nrp baseline). This mechanism does not add that "
        "facilitation. The pre-registered gate is that the target is not weakened "
        f"(absolute drop at most {RIF_GATES['winner_abs_drop_max']:.2f}). "
        f"This bank {_clears(winner_ok)} it.",
        f"- Nrp outsider absolute drop {_fmt(outsider)}. A different relation was not "
        "the practiced cue, so it should stay put "
        f"(absolute drop at most {RIF_GATES['noncompetitor_abs_drop_max']:.2f}). "
        f"This bank {_clears(outsider_ok)} it.",
        "The human figures are category-cued recall percentages. These drops are "
        "logprob-margin nats. The comparison is the direction and a partial size, "
        "not a conversion of percent recall into nats. Fuzzy recall is the intended "
        "outcome. The floors were not retuned on this bank.",
    ]


def render_report(held, fit, direction, valence, clock, surprise):
    fit_result = fit["result"]
    held_rows = gate_rows(held, RIF_GATES, held=True)
    fit_rows = gate_rows(fit_result, RIF_GATES, held=False)
    cost = held["cost"]
    quartets = held["quartets"]
    prefer = direction["prefer"]["summary"]
    avoid = direction["avoid"]["summary"]
    lines = [
        "# Subconscious retrieval-induced forgetting retry",
        "",
        "Retrieving a target multiplies the strength of at most three other slots "
        "that share the winner's relation. The closer key is scaled down more. "
        "The winner is left as read. A tie, a larger shared relation, a high-salience "
        "slot, and every other relation stay put. "
        f"SmolLM2-135M, layer {held['layer']}, hidden {held['hidden']}, float32. "
        f"gap_tau {_fmt(DEFAULT_GAP_TAU, 2)}. "
        f"Span {_fmt_num(llm_memory.RIF_SPAN)}. "
        f"Set cap {int(llm_memory.RIF_MAX_SET)}. "
        f"Salience exempt {_fmt_num(llm_memory.RIF_SALIENCE_EXEMPT)}. "
        f"Valence stays `u = 1 + {_fmt_num(VALENCE_SPAN)} tanh(valence)` with step "
        f"{_fmt_num(VALENCE_STEP)}. "
        f"Clock span stays {_fmt_num(episodic.CLOCK_SPAN)}. "
        f"Surprise quantile stays {_fmt_num(llm_memory.SURPRISE_QUANTILE)} "
        f"over the last {int(llm_memory.SURPRISE_WINDOW)} accepted surprises. "
        "Phone and birthday spelling copies the winning slot's stored answer, "
        "and only when the prompt is completing that relation. "
        "One query per continuation. Replay stays every 32 accepted writes.",
        "",
        "Prefer, avoid, and forbidden stay in scope against the #49 baselines "
        "(those match #48, #47, #44, and #46) and against #41. They are not out of scope.",
        "",
        f"Fit bank `{fit_result['bank_sha256']}`. Held-out bank `{held['bank_sha256']}`. "
        "The span was chosen on the fit bank before the held-out bank was scored. "
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
    for name, ok, detail in _direction_rows(direction):
        lines.append(f"| held | {name} | {'yes' if ok else 'no'} | {detail} |")
    for name, ok, detail in valence_gate_rows(valence, VALENCE_GATES, held=True):
        lines.append(f"| valence held | {name} | {'yes' if ok else 'no'} | {detail} |")
    for name, ok, detail in clock_gate_rows(clock, CLOCK_GATES, held=True):
        lines.append(f"| circadian held | {name} | {'yes' if ok else 'no'} | {detail} |")
    for name, ok, detail in surprise_gate_rows(surprise, SURPRISE_GATES, held=True):
        lines.append(f"| surprise held | {name} | {'yes' if ok else 'no'} | {detail} |")
    prefer_49 = float(prefer["mean_delta"]) - float(REPLAY_HELD_PREFER_MEAN)
    avoid_49 = float(avoid["mean_delta"]) - float(REPLAY_HELD_AVOID_MEAN)
    forbid_49 = float(avoid["mean_stored_lift"]) - float(REPLAY_HELD_FORBIDDEN_LIFT)
    prefer_41 = float(prefer["mean_delta"]) - float(DIRECTION_HELD_PREFER_MEAN)
    avoid_41 = float(avoid["mean_delta"]) - float(DIRECTION_HELD_AVOID_MEAN)
    forbid_41 = float(avoid["mean_stored_lift"]) - float(DIRECTION_HELD_FORBIDDEN_LIFT)
    passed = _held_pass(held, direction, valence, clock, surprise)
    lines.extend(
        [
            "",
            "## Competitor contrast",
            "",
            f"- held min drop {_fmt(quartets['competitor_drop_min'])}, "
            f"mean {_fmt(quartets['competitor_drop_mean'])}",
            f"- held winner abs {_fmt(quartets['winner_abs_drop_max'])}, "
            f"outsider abs {_fmt(quartets['noncompetitor_abs_drop_max'])}",
            f"- geometry {'yes' if quartets['geometry_ok'] else 'no'}",
            "",
            "## Direction against #49 and #41 held",
            "",
            f"- prefer mean {_fmt(prefer['mean_delta'])} "
            f"({prefer['n_delta_pos']}/{prefer['n']} > 0), "
            f"delta {_fmt(prefer_49)} against #49 {REPLAY_HELD_PREFER_MEAN:.3f}, "
            f"delta {_fmt(prefer_41)} against #41 {DIRECTION_HELD_PREFER_MEAN:.3f}",
            f"- avoid mean {_fmt(avoid['mean_delta'])} "
            f"({avoid['n_delta_pos']}/{avoid['n']} > 0), "
            f"delta {_fmt(avoid_49)} against #49 {REPLAY_HELD_AVOID_MEAN:.3f}, "
            f"delta {_fmt(avoid_41)} against #41 {DIRECTION_HELD_AVOID_MEAN:.3f}",
            f"- forbidden mean stored lift {_fmt(avoid['mean_stored_lift'])} "
            f"({avoid['n_stored_down']}/{avoid['n']} down), "
            f"delta {_fmt(forbid_49)} against #49 {REPLAY_HELD_FORBIDDEN_LIFT:.3f}, "
            f"delta {_fmt(forbid_41)} against #41 {DIRECTION_HELD_FORBIDDEN_LIFT:.3f}",
            "",
            "## Cost",
            "",
            f"- new parameters: {cost['parameters_new']}",
            f"- MLP parameters: {cost['mlp_parameters']}",
            f"- polarity parameters: {cost['polarity_parameters']}",
            f"- total parameters: {cost['parameters']}",
            f"- buffers: {cost['buffers']}",
            f"- forwards inside one armed query: {cost['rank_forwards']}",
            f"- forgetting record contains text: {cost['rif_has_text']}",
            "- Suppression multiplies stored strengths. It does not add a forward "
            "and there is no feature flag.",
            "",
            f"Polarity span is {POLARITY_SPAN:.0f}. "
            f"Retrieval span is {_fmt_num(llm_memory.RIF_SPAN)}. "
            f"Valence span is {_fmt_num(VALENCE_SPAN)} and step is {_fmt_num(VALENCE_STEP)}. "
            f"Clock span is {_fmt_num(episodic.CLOCK_SPAN)}. "
            f"Replay decay is {REPLAY_DECAY:.2f}. "
            "Held-out pass: "
            + ("yes" if passed else "no")
            + ".",
            "",
            "## Human effect",
            "",
            HUMAN_EFFECT.strip(),
            "",
            *_effect_lines(quartets),
            "",
        ]
    )
    if passed:
        lines.append("All gates and floors passed.")
        lines.append("")
    else:
        lines.append("At least one gate or floor missed. No retune.")
        lines.append("")
    return "\n".join(lines)


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(os.cpu_count() or 1)
    scored = ROOT / "reports" / "subconscious_rif_retry.json"
    if scored.exists():
        raise SystemExit(f"refusing a second held-out pass: {scored}")
    prereg = (ROOT / "reports" / "subconscious_rif_retry_prereg.md").read_text(encoding="utf-8")
    if "not been scored" not in prereg:
        raise SystemExit("preregistration does not say the held-out bank has not been scored")
    if "share the winner's relation" not in prereg:
        raise SystemExit("preregistration does not lock the same-relation cue")
    if "not out of scope" not in prereg:
        raise SystemExit("preregistration does not keep direction in scope")
    for kind in ("fit", "held"):
        if bank_sha256(kind) not in prereg:
            raise SystemExit(f"preregistration does not lock the {kind} bank")
    if f"rif span: {_fmt_num(llm_memory.RIF_SPAN)}" not in prereg:
        raise SystemExit("preregistration span does not match the source constant")
    if "Anderson, Bjork, and Bjork (1994)" not in prereg or "Rp-" not in prereg:
        raise SystemExit("preregistration does not lock the human Rp- effect")
    if HUMAN_EFFECT.strip() not in prereg:
        raise SystemExit("preregistration is missing the human-effect section")
    fit = _load_fit()
    if fit["result"]["bank_sha256"] != bank_sha256("fit"):
        raise SystemExit("fit json does not match the fit bank")
    if float(fit["chosen"]["span"]) != float(llm_memory.RIF_SPAN):
        raise SystemExit("fit json span does not match the source constant")
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
    print("surprise held floor", flush=True)
    surprise = surprise_score_bank(mem, surprise_held_bank(), progress=lambda msg: print(msg, flush=True))
    payload = dict(result)
    payload["replay_direction"] = direction
    payload["valence_held_pass"] = bool(valence_passes(valence, VALENCE_GATES, held=True))
    payload["circadian_held_pass"] = bool(clock_passes(clock, CLOCK_GATES, held=True))
    payload["surprise_held_pass"] = bool(surprise_passes(surprise, SURPRISE_GATES, held=True))
    payload["valence_held"] = valence
    payload["circadian_held"] = clock
    payload["surprise_held"] = surprise
    payload["held_pass"] = bool(_held_pass(result, direction, valence, clock, surprise))
    scored.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = ROOT / "reports" / "subconscious_rif_retry.md"
    report.write_text(
        render_report(result, fit, direction, valence, clock, surprise),
        encoding="utf-8",
    )
    print("wrote", scored, flush=True)
    print("wrote", report, flush=True)
    for name, ok, detail in gate_rows(result, RIF_GATES, held=True):
        print(f"  {name}: {'yes' if ok else 'NO'} {detail}", flush=True)
    for name, ok, detail in _direction_rows(direction):
        print(f"  {name}: {'yes' if ok else 'NO'} {detail}", flush=True)
    print("valence held", "yes" if payload["valence_held_pass"] else "no", flush=True)
    print("circadian held", "yes" if payload["circadian_held_pass"] else "no", flush=True)
    print("surprise held", "yes" if payload["surprise_held_pass"] else "no", flush=True)
    print("held pass", "yes" if payload["held_pass"] else "no", flush=True)
    mem.close()


if __name__ == "__main__":
    main()
