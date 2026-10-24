"""Fit the clock span on the fit bank, then score that bank once.

Does not import the held-out bank and does not score it. Prints the constant
to set when the source does not already match the chosen span, and exits
before the full fit score in that case. There is no train bank.
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
from tidal.llm_memory import (
    LLMMemory,
    REPLAY_DECAY,
    REPLAY_EVERY,
    REPLAY_M,
    VALENCE_SPAN,
    VALENCE_STEP,
)
from tidal.subconscious_circadian_data import (
    CLOCK_GATES,
    DIRECTION_HELD_AVOID_MEAN,
    DIRECTION_HELD_FORBIDDEN_LIFT,
    DIRECTION_HELD_PREFER_MEAN,
    PRIOR_VALENCE_FIT_SHA,
    PRIOR_VALENCE_HELD_SHA,
    REPLAY_AVOID_MEAN_MIN,
    REPLAY_FIT_SHA,
    REPLAY_FORBIDDEN_DOWN_MIN,
    REPLAY_HELD_AVOID_MEAN,
    REPLAY_HELD_FORBIDDEN_LIFT,
    REPLAY_HELD_PREFER_MEAN,
    REPLAY_HELD_SHA,
    REPLAY_PREFER_MEAN_MIN,
    RIF_FIT_SHA,
    RIF_HELD_SHA,
    SPELL_FIT_SHA,
    SPELL_HELD_SHA,
    SURPRISE_FIT_SHA,
    SURPRISE_HELD_SHA,
    VALENCE_FIT_SHA,
    VALENCE_HELD_SHA,
    bank_sha256,
    fit_bank,
)
from tidal.subconscious_circadian_score import gate_rows, passes, score_bank, score_phases

# Smallest span that clears the fit count floors. The tick is not on the grid.
GRID = (0.25, 0.5, 1.0)


def _fmt_num(value):
    return f"{float(value):g}"


def _probe(mem, bank, span):
    episodic.CLOCK_SPAN = float(span)
    phases = score_phases(mem, bank.PHASE, progress=lambda msg: print(msg, flush=True))
    ok = (
        int(phases["n_high_pos"]) >= int(CLOCK_GATES["high_pos_min"])
        and int(phases["n_low_neg"]) >= int(CLOCK_GATES["low_neg_min"])
        and int(phases["n_decay_order"]) >= int(CLOCK_GATES["decay_order_min"])
        and bool(phases["phase0_exact"])
        and int(phases["n_strength"]) == int(phases["n"])
        and int(phases["n_audit"]) == int(phases["n"])
        and bool(phases["content_free"])
    )
    return {
        "span": float(span),
        "ok": bool(ok),
        "high_pos": int(phases["n_high_pos"]),
        "low_neg": int(phases["n_low_neg"]),
        "n": int(phases["n"]),
        "high_mean": phases["mean_high"],
        "low_mean": phases["mean_low"],
        "decay_order": int(phases["n_decay_order"]),
        "decay_gap": phases["mean_decay_gap"],
        "strength": int(phases["n_strength"]),
        "audit": int(phases["n_audit"]),
        "content_free": bool(phases["content_free"]),
        "phase0_exact": bool(phases["phase0_exact"]),
    }


def _choose(rows):
    passing = [row for row in rows if row["ok"]]
    if not passing:
        return None
    passing.sort(key=lambda row: float(row["span"]))
    return passing[0]


def _fmt(value, digits=3):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _render_prereg(result, grid_rows, chosen):
    cost = result["cost"]
    phases = result["phases"]
    exact = result["exact"]
    lines = [
        "# Circadian clock preregistration",
        "",
        "Content-free circadian clock. The held-out bank has not been scored.",
        "",
        "There is no train bank. The fit bank selects the span. The held-out bank "
        "is the only later score.",
        "",
        "No feature flag. `gap_tau` stays 0.80. Polarity gain stays "
        "`1 + 2 tanh(w · normalize(k))`. Valence stays `u = 1 + 1 * tanh(valence)` "
        "with step 0.25. Phone and birthday spelling stays the exact stored answer, "
        "and still runs only when the winning relation is the one the prompt is "
        "completing. A continuation still encodes the query once. Readout stays "
        "hard top-1. Replay stays every 32 accepted writes, refreshing the top 1 "
        "slot, decay 0.50. The surprise gate and the retrieval-induced-forgetting "
        "gate are not in this change. The valence formula is not in this change.",
        "",
        "The clock is one global angle on the store. `phase = sin(angle)`. "
        "Angle 0 leaves the write strength at 1 and the decay at `0.5 ** age`. "
        "A nonzero phase scales that write's strength by `1 + span * phase` and "
        "the decay exponent by `1 - span * phase`. The angle is not read from "
        "fact text, and the log stores kind, angle, phase, and gain. Replay "
        "refresh still sets the top slot back to strength 1. The tick is `pi/16` "
        "and is not on the grid. `remember` does not tick.",
        "",
        "Prefer, avoid, and forbidden on the replay held-out bank stay in scope. "
        "They are not out of scope for this change. #47 held matched the #44 and "
        "#46 numbers.",
        "",
        "## Banks",
        "",
        f"- fit `{bank_sha256('fit')}`",
        f"- held `{bank_sha256('held')}`",
        "- frozen `e3cdd7fd595d8a76af03a276c2ad750b25b68e1bd8460c9e22e95aa12a820e66`",
        "- #39 fit `20b7cb3f9742709d320dc5913cf62675316673d7b457c55b6c78cf4de587e5ce`",
        "- #39 held `ea0a15e5edf0dcab53fede8fad1964b61c23b52c3df55c5f8e00865e59cb053a`",
        "- #40 held `cbca1f62c152eb928f498629e8d39a79cdcd9a144589cf2ad8484b65806dd464`",
        "- #41 train `626ec7cb9ac7824666be6a94ab8e6dc428b7e092ccfc7d8db689fde9c92be4d9`",
        "- #41 fit `ee75f1857bfb6b7e812dfb0ef4012af1dc6efb33ca17799f1211b570ad1c0c1d`",
        "- #41 held `e074b2ad7960bb81bbf64ae0ccd03dbeb0f358428a8a5c6c56e8f373a38d952b`",
        f"- #42 fit `{SURPRISE_FIT_SHA}`",
        f"- #42 held `{SURPRISE_HELD_SHA}`",
        f"- #43 fit `{RIF_FIT_SHA}`",
        f"- #43 held `{RIF_HELD_SHA}`",
        f"- #44 fit `{REPLAY_FIT_SHA}`",
        f"- #44 held `{REPLAY_HELD_SHA}`",
        f"- #45 fit `{PRIOR_VALENCE_FIT_SHA}`",
        f"- #45 held `{PRIOR_VALENCE_HELD_SHA}`",
        f"- #46 fit `{SPELL_FIT_SHA}`",
        f"- #46 held `{SPELL_HELD_SHA}`",
        f"- #47 fit `{VALENCE_FIT_SHA}`",
        f"- #47 held `{VALENCE_HELD_SHA}`",
        "- literal overlap 0 and token Jaccard under 0.50 against those banks and between fit and held",
        "",
        "## Rule",
        "",
        f"- clock span: {_fmt_num(chosen['span'])}",
        "- clock step: pi/16",
        f"- valence span: {_fmt_num(VALENCE_SPAN)}",
        f"- valence step: {_fmt_num(VALENCE_STEP)}",
        f"- replay every {int(REPLAY_EVERY)}, m {int(REPLAY_M)}, decay {REPLAY_DECAY:.2f}",
        f"- new parameters: {cost['parameters_new']}",
        f"- MLP parameters: {cost['mlp_parameters']}",
        f"- polarity parameters: {cost['polarity_parameters']}",
        f"- total parameters: {cost['parameters']}",
        f"- buffers: {cost['buffers']}",
        f"- forwards inside one remember: {cost['remember_forwards']}",
        f"- forwards inside one armed query: {cost['query_forwards']}",
        f"- forwards inside one clock tick: {cost['clock_tick_forwards']}",
        "",
        "## Fit selection",
        "",
        "The grid is span in {0.25, 0.5, 1}. "
        "A point passes when at least 6/8 high-phase margins move up, at least "
        "6/8 low-phase margins move down, decay order holds on at least 6/8 with "
        "phase 0 strength exactly 0.5, every write strength matches the clock "
        "gain, and the log is content-free. The smallest span is the one scored "
        "below. The full fit score then runs once. It is not repeated on a larger "
        "span if a non-interference floor fails. Mean floors are held-out gates. "
        "The held-out bank has not been scored.",
        "",
        "| span | pass | high | low | decay | content-free |",
        "|---|---|---|---|---|---|",
    ]
    for row in grid_rows:
        lines.append(
            f"| {_fmt_num(row['span'])} | {'yes' if row['ok'] else 'no'} | "
            f"{row['high_pos']}/{row['n']} {_fmt(row['high_mean'])} | "
            f"{row['low_neg']}/{row['n']} {_fmt(row['low_mean'])} | "
            f"{row['decay_order']}/{row['n']} gap {_fmt(row['decay_gap'])} | "
            f"{'yes' if row['content_free'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Fit bank `{result['bank_sha256']}`.",
            "",
            "| gate | pass | detail |",
            "|---|---|---|",
        ]
    )
    for name, ok, detail in gate_rows(result, CLOCK_GATES, held=False):
        lines.append(f"| {name} | {'yes' if ok else 'no'} | {detail} |")
    lines.extend(
        [
            "",
            "Fit does not require the held-out mean floors "
            f"(high mean {CLOCK_GATES['high_mean_min']}, "
            f"low mean {CLOCK_GATES['low_mean_max']}, "
            f"decay gap mean {CLOCK_GATES['decay_gap_mean_min']}). Those are held-out gates.",
            "",
            f"Fit high mean {_fmt(phases['mean_high'])} "
            f"({phases['n_high_pos']}/{phases['n']} > 0).",
            f"Fit low mean {_fmt(phases['mean_low'])} "
            f"({phases['n_low_neg']}/{phases['n']} < 0).",
            f"Fit decay gap {_fmt(phases['mean_decay_gap'])}.",
            "",
            "Fit on-relation phone "
            f"{exact['phone_question_text']!r} equals the stored {exact['phone_stored']!r}: "
            f"{exact['phone_exact']}.",
            "Fit on-relation birthday "
            f"{exact['birthday_question_text']!r} equals the stored {exact['birthday_stored']!r}: "
            f"{exact['birthday_exact']}.",
            f"Prefix detector on those questions: phone {exact['phone_prefix']}, "
            f"birthday {exact['birthday_prefix']}. It is not the gate.",
            "",
            "## Held-out gates",
            "",
            "The held-out bank has not been scored. It is merge-critical.",
            "",
            "- high phase moves the margin up on at least 6/8, "
            f"with mean at least {CLOCK_GATES['high_mean_min']}",
            "- low phase moves the margin down on at least 6/8, "
            f"with mean at most {CLOCK_GATES['low_mean_max']}",
            "- decay order high > phase 0 > low on at least 6/8, phase 0 strength exactly 0.5, "
            f"and mean decay gap at least {CLOCK_GATES['decay_gap_mean_min']}",
            "- write strength matches the content-free clock gain on every item",
            "- the clock log has no fact text, and two facts at one phase share one gain",
            "- absolute mean neutral ΔNLL ≤ 0.05, and at most 1 item with ΔNLL > 0.5",
            "- choice flips 0",
            "- leak 0",
            "- phone and birthday takeover 0 off-relation, including the cross question",
            "- on-relation generated text equals the stored number and the stored date",
            "- off-relation text is not the stored number and not the stored date",
            "- the prefix detector is recorded and is not the on-relation gate",
            "- incidental store keeps 32 slots",
            f"- prefer margin mean at least {REPLAY_PREFER_MEAN_MIN:.1f} on the #44 held bank "
            f"(#47, #44, and #46 held were {REPLAY_HELD_PREFER_MEAN:.3f}; "
            f"#41 held was {DIRECTION_HELD_PREFER_MEAN:.3f})",
            f"- avoid margin mean at least {REPLAY_AVOID_MEAN_MIN:.1f} on the #44 held bank "
            f"(#47, #44, and #46 held were {REPLAY_HELD_AVOID_MEAN:.3f}; "
            f"#41 held was {DIRECTION_HELD_AVOID_MEAN:.3f})",
            f"- forbidden logprob down on at least {int(REPLAY_FORBIDDEN_DOWN_MIN)}/8 on the #44 held bank "
            f"(#47, #44, and #46 held mean stored lift {REPLAY_HELD_FORBIDDEN_LIFT:.3f}; "
            f"#41 held mean stored lift {DIRECTION_HELD_FORBIDDEN_LIFT:.3f})",
            "",
            "Report the prefer, avoid, and forbidden deltas against the #47/#44 and #41 "
            "held numbers above. Those floors stay in scope. Do not retune the span on this bank.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(os.cpu_count() or 1)
    source_span = float(episodic.CLOCK_SPAN)
    print("loading SmolLM2-135M float32", flush=True)
    mem = LLMMemory(dtype=torch.float32)
    bank = fit_bank()
    rows = []
    try:
        for span in GRID:
            print(f"clock span={_fmt_num(span)}", flush=True)
            row = _probe(mem, bank, span)
            rows.append(row)
            print(
                f"  ok {row['ok']} high {row['high_pos']}/{row['n']} {row['high_mean']} "
                f"low {row['low_neg']}/{row['n']} {row['low_mean']} "
                f"decay {row['decay_order']}/{row['n']} content {row['content_free']}",
                flush=True,
            )
    finally:
        episodic.CLOCK_SPAN = source_span
    chosen = _choose(rows)
    if chosen is None:
        raise SystemExit("no grid point cleared the clock floors")
    if float(chosen["span"]) != source_span:
        print(f"set CLOCK_SPAN to {chosen['span']}", flush=True)
        mem.close()
        raise SystemExit(2)
    print("full fit score", flush=True)
    result = score_bank(mem, bank, progress=lambda msg: print(msg, flush=True))
    out = ROOT / "reports" / "subconscious_circadian_fit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "grid": rows,
        "chosen": {"span": chosen["span"]},
        "result": result,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", out, flush=True)
    cost = result["cost"]
    if (
        int(cost["parameters_new"]) != 0
        or int(cost["remember_forwards"]) != 2
        or int(cost["query_forwards"]) != 1
        or int(cost["clock_tick_forwards"]) != 0
    ):
        print("fit cost is not the fixed budget", cost, flush=True)
        mem.close()
        raise SystemExit(1)
    if not passes(result, CLOCK_GATES, held=False):
        print("fit gates failed", flush=True)
        for name, ok, detail in gate_rows(result, CLOCK_GATES, held=False):
            print(f"  {name}: {'yes' if ok else 'NO'} {detail}", flush=True)
        mem.close()
        raise SystemExit(1)
    prereg = ROOT / "reports" / "subconscious_circadian_prereg.md"
    prereg.write_text(_render_prereg(result, rows, chosen), encoding="utf-8")
    print("wrote", prereg, flush=True)
    mem.close()


if __name__ == "__main__":
    main()
