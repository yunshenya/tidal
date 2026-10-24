"""Fit the surprise quantile on the fit bank, then score that bank once.

The train bank is the post-warmup keep floor. Does not import the held-out
bank and does not score it. Prints the constant to set when the source does
not already match the chosen quantile, and exits before the full fit score
in that case.
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
    LLMMemory,
    REPLAY_DECAY,
    REPLAY_EVERY,
    REPLAY_M,
    VALENCE_SPAN,
    VALENCE_STEP,
)
from tidal.subconscious_circadian_data import CLOCK_GATES, fit_bank as clock_fit_bank
from tidal.subconscious_circadian_score import passes as clock_passes
from tidal.subconscious_circadian_score import score_bank as clock_score_bank
from tidal.subconscious_surprise_rel_data import (
    CIRCADIAN_FIT_SHA,
    CIRCADIAN_HELD_SHA,
    DIRECTION_FIT_SHA,
    DIRECTION_HELD_AVOID_MEAN,
    DIRECTION_HELD_FORBIDDEN_LIFT,
    DIRECTION_HELD_PREFER_MEAN,
    DIRECTION_HELD_SHA,
    DIRECTION_TRAIN_SHA,
    FIT39_SHA,
    FROZEN_SHA,
    GATE_SHA,
    HELD39_SHA,
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
    SURPRISE_GATES,
    SURPRISE_HELD_SHA,
    VALENCE_FIT_SHA,
    VALENCE_HELD_SHA,
    bank_sha256,
    fit_bank,
    train_bank,
)
from tidal.subconscious_surprise_rel_score import gate_rows, passes, score_bank, score_gate, score_keeps
from tidal.subconscious_valence_data import VALENCE_GATES, fit_bank as valence_fit_bank
from tidal.subconscious_valence_score import passes as valence_passes
from tidal.subconscious_valence_score import score_bank as valence_score_bank

# Smallest quantile that clears the fit count floors. The window length is not on the grid.
GRID = (0.0, 0.25, 0.5)


def _fmt_num(value):
    return f"{float(value):g}"


def _fmt(value, digits=3):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _probe(mem, bank, train, quantile):
    llm_memory.SURPRISE_QUANTILE = float(quantile)
    gate = score_gate(mem, bank, progress=lambda msg: print(msg, flush=True))
    keeps = score_keeps(mem, train, progress=lambda msg: print(msg, flush=True))
    ok = (
        int(gate["low_written"]) <= int(SURPRISE_GATES["low_written_max"])
        and int(gate["high_written"]) >= int(SURPRISE_GATES["high_written_min"])
        and int(gate["seed_written"]) == int(gate["seed_n"])
        and int(gate["seed_slots"]) == int(SURPRISE_GATES["incidental_slots"])
        and int(gate["window"]) == int(SURPRISE_GATES["window"])
        and gate["bar"] is not None
        and int(gate["window_has_text"]) == 0
        and int(keeps["seed_written"]) == int(keeps["seed_n"])
        and int(keeps["seed_slots"]) == int(SURPRISE_GATES["incidental_slots"])
        and int(keeps["keep_written"]) >= int(SURPRISE_GATES["keep_written_min"])
        and int(keeps["keep_written"]) == int(keeps["keep_n"])
    )
    return {
        "quantile": float(quantile),
        "ok": bool(ok),
        "low_written": int(gate["low_written"]),
        "low_n": int(gate["low_n"]),
        "high_written": int(gate["high_written"]),
        "high_n": int(gate["high_n"]),
        "low_mean_surprise": gate["low_mean_surprise"],
        "high_mean_surprise": gate["high_mean_surprise"],
        "bar": gate["bar"],
        "seed_slots": int(gate["seed_slots"]),
        "window": int(gate["window"]),
        "keep_written": int(keeps["keep_written"]),
        "keep_n": int(keeps["keep_n"]),
        "train_slots": int(keeps["seed_slots"]),
    }


def _choose(rows):
    passing = [row for row in rows if row["ok"]]
    if not passing:
        return None
    passing.sort(key=lambda row: float(row["quantile"]))
    return passing[0]


def _render_prereg(result, grid_rows, chosen, keeps):
    cost = result["cost"]
    gate = result["gate"]
    exact = result["exact"]
    lines = [
        "# Relative surprise preregistration",
        "",
        "Surprise-gated writes, relative to the store's own window. "
        "The held-out bank has not been scored.",
        "",
        "The train bank is the post-warmup keep floor: after 32 accepted "
        "incidental surprises, ordinary keeps must still be written. The fit "
        "bank selects the quantile. The held-out bank is the only later score.",
        "",
        "No feature flag. `gap_tau` stays 0.80. Polarity gain stays "
        "`1 + 2 tanh(w · normalize(k))`. Valence stays `u = 1 + 1 * tanh(valence)` "
        "with step 0.25. The clock stays `phase = sin(angle)` with span "
        f"{_fmt_num(episodic.CLOCK_SPAN)} and step pi/16. Phone and birthday "
        "spelling stays the exact stored answer, and still runs only when the "
        "winning relation is the one the prompt is completing. A continuation "
        "still encodes the query once. Readout stays hard top-1. Replay stays "
        "every 32 accepted writes, refreshing the top 1 slot, decay 0.50. "
        "The retrieval-induced-forgetting gate is not in this change. The "
        "valence formula is not in this change. The circadian formula is not "
        "in this change.",
        "",
        "A fact's surprise is the total NLL, in nats, of its answer tokens "
        "under the logits the value forward already returned. The window stores "
        "the last 32 accepted write surprises and no text. Fewer than 32 "
        "accepted writes leaves the salient fact accepted. After that, the fact "
        "is written when its surprise is at least the quantile of the last 32 "
        "accepted surprises. A rejected attempt does not enter the window. Chat "
        "and noise still return before any forward. The key forward runs only "
        "when the write is kept. There is no third forward and no new parameter. "
        "The window length is not on the grid.",
        "",
        "On the fit and held banks the incidental facts are written first, in "
        "listed order, and that fills the window. Lows are attempted next, then "
        "highs, with no clear between them. The train keeps are attempted only "
        "after the train incidental window is full.",
        "",
        "Prefer, avoid, and forbidden on the replay held-out bank stay in scope. "
        "They are not out of scope for this change. #48 held matched the #47, "
        "#44, and #46 numbers.",
        "",
        "## Banks",
        "",
        f"- train `{bank_sha256('train')}`",
        f"- fit `{bank_sha256('fit')}`",
        f"- held `{bank_sha256('held')}`",
        f"- frozen `{FROZEN_SHA}`",
        f"- #39 fit `{FIT39_SHA}`",
        f"- #39 held `{HELD39_SHA}`",
        f"- #40 held `{GATE_SHA}`",
        f"- #41 train `{DIRECTION_TRAIN_SHA}`",
        f"- #41 fit `{DIRECTION_FIT_SHA}`",
        f"- #41 held `{DIRECTION_HELD_SHA}`",
        f"- #42 fit `{SURPRISE_FIT_SHA}`",
        f"- #42 held `{SURPRISE_HELD_SHA}`",
        f"- #43 fit `{RIF_FIT_SHA}`",
        f"- #43 held `{RIF_HELD_SHA}`",
        f"- #44 fit `{REPLAY_FIT_SHA}`",
        f"- #44 held `{REPLAY_HELD_SHA}`",
        f"- closed valence fit `{PRIOR_VALENCE_FIT_SHA}`",
        f"- closed valence held `{PRIOR_VALENCE_HELD_SHA}`",
        f"- #46 fit `{SPELL_FIT_SHA}`",
        f"- #46 held `{SPELL_HELD_SHA}`",
        f"- #47 fit `{VALENCE_FIT_SHA}`",
        f"- #47 held `{VALENCE_HELD_SHA}`",
        f"- #48 fit `{CIRCADIAN_FIT_SHA}`",
        f"- #48 held `{CIRCADIAN_HELD_SHA}`",
        "",
        "## Quantile grid",
        "",
        "| quantile | low written | high written | train keeps | bar | pass |",
        "|---|---|---|---|---|---|",
    ]
    for row in grid_rows:
        lines.append(
            f"| {_fmt_num(row['quantile'])} | {row['low_written']}/{row['low_n']} | "
            f"{row['high_written']}/{row['high_n']} | {row['keep_written']}/{row['keep_n']} | "
            f"{_fmt(row['bar'])} | {'yes' if row['ok'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Chosen surprise quantile: {_fmt_num(chosen['quantile'])}.",
            f"surprise quantile: {_fmt_num(chosen['quantile'])}",
            "",
            f"Fit bank `{result['bank_sha256']}`. Train keeps "
            f"{keeps['keep_written']}/{keeps['keep_n']}.",
            "",
            "| gate | pass | detail |",
            "|---|---|---|",
        ]
    )
    for name, ok, detail in gate_rows(result, SURPRISE_GATES, held=False):
        lines.append(f"| {name} | {'yes' if ok else 'no'} | {detail} |")
    lines.extend(
        [
            "",
            "Fit uses the same count gates as held-out for low-surprise writes "
            f"(at most {SURPRISE_GATES['low_written_max']} of 8) and high-surprise "
            f"writes (at least {SURPRISE_GATES['high_written_min']} of 8). "
            "There is no absolute surprise threshold and no held-out mean floor "
            "on the raw NLL, because an absolute threshold did not transfer.",
            "",
            f"Fit low mean surprise {_fmt(gate['low_mean_surprise'])} "
            f"({gate['low_written']}/{gate['low_n']} written).",
            f"Fit high mean surprise {_fmt(gate['high_mean_surprise'])} "
            f"({gate['high_written']}/{gate['high_n']} written).",
            f"Fit window bar {_fmt(gate['bar'])}.",
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
            "- low-surprise facts are not written on at least 6 of 8 "
            f"(written at most {SURPRISE_GATES['low_written_max']})",
            "- high-surprise facts are written on at least "
            f"{SURPRISE_GATES['high_written_min']} of 8",
            "- the incidental seed fills 32 slots and a window of 32 numbers, "
            "and that window does not contain fact text",
            "- absolute mean neutral ΔNLL ≤ 0.05, and at most 1 item with ΔNLL > 0.5",
            "- choice flips 0",
            "- leak 0",
            "- phone and birthday takeover 0 off-relation, including the cross question",
            "- on-relation generated text equals the stored number and the stored date",
            "- off-relation text is not the stored number and not the stored date",
            "- the prefix detector is recorded and is not the on-relation gate",
            "- incidental store keeps 32 slots",
            "- chat does 0 forwards, a below-window fact does 1 forward and is not written, "
            "an above-window fact does 2 forwards and is written, new parameters 0",
            f"- prefer margin mean at least {REPLAY_PREFER_MEAN_MIN:.1f} on the #44 held bank "
            f"(#48, #47, #44, and #46 held were {REPLAY_HELD_PREFER_MEAN:.3f}; "
            f"#41 held was {DIRECTION_HELD_PREFER_MEAN:.3f})",
            f"- avoid margin mean at least {REPLAY_AVOID_MEAN_MIN:.1f} on the #44 held bank "
            f"(#48, #47, #44, and #46 held were {REPLAY_HELD_AVOID_MEAN:.3f}; "
            f"#41 held was {DIRECTION_HELD_AVOID_MEAN:.3f})",
            f"- forbidden logprob down on at least {int(REPLAY_FORBIDDEN_DOWN_MIN)}/8 on the #44 held bank "
            f"(#48, #47, #44, and #46 held mean stored lift {REPLAY_HELD_FORBIDDEN_LIFT:.3f}; "
            f"#41 held mean stored lift {DIRECTION_HELD_FORBIDDEN_LIFT:.3f})",
            "- valence fit gates already passed on the #47 fit bank; the held-out "
            "score must pass the #47 valence gates, including the held mean floors",
            "- circadian fit gates already passed on the #48 fit bank; the held-out "
            "score must pass the #48 circadian gates, including the held mean floors",
            "",
            "Report the prefer, avoid, and forbidden deltas against the #48/#47/#44 "
            "and #41 held numbers above. Those floors stay in scope. Do not retune "
            "the quantile on the held-out bank.",
            "",
            f"New parameters on the fit cost probe: {cost['parameters_new']}. "
            f"Chat forwards {cost['chat_forwards']}. "
            f"Below-window forwards {cost['below_forwards']}, written {cost['below_written']}. "
            f"Above-window forwards {cost['above_forwards']}, written {cost['above_written']}.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(os.cpu_count() or 1)
    source_q = float(llm_memory.SURPRISE_QUANTILE)
    print("loading SmolLM2-135M float32", flush=True)
    mem = LLMMemory(dtype=torch.float32)
    bank = fit_bank()
    train = train_bank()
    rows = []
    try:
        for quantile in GRID:
            print(f"surprise quantile={_fmt_num(quantile)}", flush=True)
            row = _probe(mem, bank, train, quantile)
            rows.append(row)
            print(
                f"  ok {row['ok']} low {row['low_written']}/{row['low_n']} "
                f"high {row['high_written']}/{row['high_n']} "
                f"keeps {row['keep_written']}/{row['keep_n']} bar {row['bar']}",
                flush=True,
            )
    finally:
        llm_memory.SURPRISE_QUANTILE = source_q
    chosen = _choose(rows)
    if chosen is None:
        raise SystemExit("no grid point cleared the surprise floors")
    if float(chosen["quantile"]) != source_q:
        print(f"set SURPRISE_QUANTILE to {chosen['quantile']}", flush=True)
        mem.close()
        raise SystemExit(2)
    print("full fit score", flush=True)
    result = score_bank(mem, bank, progress=lambda msg: print(msg, flush=True))
    print("train keeps", flush=True)
    keeps = score_keeps(mem, train, progress=lambda msg: print(msg, flush=True))
    print("valence fit floor", flush=True)
    valence = valence_score_bank(mem, valence_fit_bank(), progress=lambda msg: print(msg, flush=True))
    valence_ok = valence_passes(valence, VALENCE_GATES, held=False)
    print("valence fit", "yes" if valence_ok else "no", flush=True)
    print("circadian fit floor", flush=True)
    clock = clock_score_bank(mem, clock_fit_bank(), progress=lambda msg: print(msg, flush=True))
    clock_ok = clock_passes(clock, CLOCK_GATES, held=False)
    print("circadian fit", "yes" if clock_ok else "no", flush=True)
    out = ROOT / "reports" / "subconscious_surprise_rel_fit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "grid": rows,
        "chosen": {"quantile": chosen["quantile"]},
        "keeps": keeps,
        "result": result,
        "valence_fit_pass": bool(valence_ok),
        "circadian_fit_pass": bool(clock_ok),
        "constants": {
            "surprise_quantile": float(llm_memory.SURPRISE_QUANTILE),
            "surprise_window": int(llm_memory.SURPRISE_WINDOW),
            "valence_span": float(VALENCE_SPAN),
            "valence_step": float(VALENCE_STEP),
            "clock_span": float(episodic.CLOCK_SPAN),
            "replay_every": int(REPLAY_EVERY),
            "replay_m": int(REPLAY_M),
            "replay_decay": float(REPLAY_DECAY),
        },
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", out, flush=True)
    cost = result["cost"]
    keep_ok = (
        int(keeps["keep_written"]) == int(keeps["keep_n"])
        and int(keeps["keep_written"]) >= int(SURPRISE_GATES["keep_written_min"])
        and int(keeps["seed_slots"]) == int(SURPRISE_GATES["incidental_slots"])
    )
    if (
        int(cost["parameters_new"]) != 0
        or int(cost["chat_forwards"]) != 0
        or int(cost["below_forwards"]) != 1
        or int(cost["below_written"]) != 0
        or int(cost["above_forwards"]) != 2
        or int(cost["above_written"]) != 1
        or int(cost["warmup_forwards"]) != 2
        or int(cost["window_has_text"]) != 0
    ):
        print("fit cost is not the fixed budget", cost, flush=True)
        mem.close()
        raise SystemExit(1)
    if not passes(result, SURPRISE_GATES, held=False) or not keep_ok or not valence_ok or not clock_ok:
        print("fit gates failed", flush=True)
        for name, ok, detail in gate_rows(result, SURPRISE_GATES, held=False):
            print(f"  {name}: {'yes' if ok else 'NO'} {detail}", flush=True)
        print("keeps", keeps["keep_written"], "/", keeps["keep_n"], "slots", keeps["seed_slots"], flush=True)
        print("valence", valence_ok, "circadian", clock_ok, flush=True)
        mem.close()
        raise SystemExit(1)
    prereg = ROOT / "reports" / "subconscious_surprise_rel_prereg.md"
    prereg.write_text(_render_prereg(result, rows, chosen, keeps), encoding="utf-8")
    print("wrote", prereg, flush=True)
    mem.close()


if __name__ == "__main__":
    main()
