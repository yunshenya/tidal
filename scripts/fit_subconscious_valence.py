"""Fit valence span and step on the fit bank, then score that bank once.

Does not import the held-out bank and does not score it. Prints the constants
to set when the source does not already match the chosen rule, and exits
before the full fit score in that case.
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

import tidal.llm_memory as llm_memory
from tidal.llm_memory import LLMMemory, REPLAY_DECAY, VALENCE_SPAN, VALENCE_STEP
from tidal.subconscious_valence_data import (
    DIRECTION_HELD_AVOID_MEAN,
    DIRECTION_HELD_FORBIDDEN_LIFT,
    DIRECTION_HELD_PREFER_MEAN,
    PRIOR_VALENCE_FIT_SHA,
    PRIOR_VALENCE_HELD_SHA,
    REPLAY_AVOID_MEAN_MIN,
    REPLAY_FIT_SHA,
    REPLAY_FORBIDDEN_DOWN_MIN,
    REPLAY_HELD_SHA,
    REPLAY_HELD_AVOID_MEAN,
    REPLAY_HELD_FORBIDDEN_LIFT,
    REPLAY_HELD_PREFER_MEAN,
    REPLAY_PREFER_MEAN_MIN,
    RIF_FIT_SHA,
    RIF_HELD_SHA,
    SPELL_FIT_SHA,
    SPELL_HELD_SHA,
    SURPRISE_FIT_SHA,
    SURPRISE_HELD_SHA,
    VALENCE_GATES,
    bank_sha256,
    fit_bank,
)
from tidal.subconscious_valence_score import gate_rows, passes, score_bank, score_revise, score_signed

# Smallest span, then smallest step. Decay is not on the grid.
GRID = tuple((span, step) for span in (1.0, 2.0) for step in (0.25, 0.5, 1.0))


def _fmt_num(value):
    return f"{float(value):g}"


def _probe(mem, bank, span, step):
    llm_memory.VALENCE_SPAN = float(span)
    llm_memory.VALENCE_STEP = float(step)
    repeats = int(bank.REPEATS)
    toward = score_signed(mem, bank.TOWARD, 1.0, repeats)
    away = score_signed(mem, bank.AWAY, -1.0, repeats)
    revise = score_revise(mem, bank.TOWARD, repeats)
    ok = (
        int(toward["n_pos"]) >= int(VALENCE_GATES["toward_pos_min"])
        and int(toward["n_audit"]) == int(toward["n"])
        and int(away["n_neg"]) >= int(VALENCE_GATES["away_neg_min"])
        and int(away["n_audit"]) == int(away["n"])
        and int(revise["n_moved_back"]) >= int(VALENCE_GATES["revise_min"])
        and int(revise["n_valence_revised"]) >= int(VALENCE_GATES["revise_min"])
    )
    return {
        "span": float(span),
        "step": float(step),
        "ok": bool(ok),
        "toward_pos": int(toward["n_pos"]),
        "toward_n": int(toward["n"]),
        "toward_mean": toward["mean_delta"],
        "toward_audit": int(toward["n_audit"]),
        "away_neg": int(away["n_neg"]),
        "away_n": int(away["n"]),
        "away_mean": away["mean_delta"],
        "away_audit": int(away["n_audit"]),
        "revise_back": int(revise["n_moved_back"]),
        "revise_valence": int(revise["n_valence_revised"]),
        "revise_n": int(revise["n"]),
    }


def _choose(rows):
    passing = [row for row in rows if row["ok"]]
    if not passing:
        return None
    passing.sort(key=lambda row: (float(row["span"]), float(row["step"])))
    return passing[0]


def _fmt(value, digits=3):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _render_prereg(result, grid_rows, chosen):
    cost = result["cost"]
    toward = result["toward"]
    away = result["away"]
    exact = result["exact"]
    lines = [
        "# Experience valence preregistration",
        "",
        "Experience-grown valence. The held-out bank has not been scored.",
        "",
        "No feature flag. `gap_tau` stays 0.80. Polarity gain stays "
        "`1 + 2 tanh(w · normalize(k))`. Phone and birthday spelling stays the "
        "exact stored answer, and still runs only when the winning relation is "
        "the one the prompt is completing. A continuation still encodes the query "
        "once. Readout stays hard top-1. Replay stays every 32 accepted writes, "
        "refreshing the top 1 slot, decay 0.50. The surprise gate and the "
        "retrieval-induced-forgetting gate are not in this change.",
        "",
        "The add is `c * s * u * gain * (scale(v) + MLP(scale(v)))`. "
        "`u = 1 + span * tanh(valence)`. A write's experience field contributes "
        "its sign times the step. The fact text is not scored, and the language "
        "model does not supply a group prior. Valence 0 leaves `u = 1`. "
        "The same replay cycle multiplies a nonzero valence by 0.50 and appends "
        "an audit row. A correction replaces the scalar and appends its own row.",
        "",
        "Prefer, avoid, and forbidden on the replay held-out bank stay in scope. "
        "They are not out of scope for this change. #46 held matched the #44 numbers.",
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
        "- literal overlap 0 and token Jaccard under 0.50 against those banks and between fit and held",
        "",
        "## Rule",
        "",
        f"- valence span: {_fmt_num(chosen['span'])}",
        f"- valence step: {_fmt_num(chosen['step'])}",
        f"- decay: {REPLAY_DECAY:.2f}",
        f"- repeats: {int(result['repeats'])}",
        f"- new parameters: {cost['parameters_new']}",
        f"- MLP parameters: {cost['mlp_parameters']}",
        f"- polarity parameters: {cost['polarity_parameters']}",
        f"- total parameters: {cost['parameters']}",
        f"- buffers: {cost['buffers']}",
        f"- forwards inside one remember: {cost['remember_forwards']}",
        f"- forwards inside one armed query: {cost['query_forwards']}",
        "",
        "## Fit selection",
        "",
        "The grid is span in {1, 2} and step in {0.25, 0.5, 1}. "
        "A point passes when at least 6/8 positive experiences move the fit margin "
        "up, at least 6/8 negative experiences move it down, each of those rows "
        "has a complete audit, and at least 6/8 contrary experiences pull the "
        "margin and the scalar back. The smallest span, then the smallest step, "
        "is the one scored below. The full fit score then runs once. It is not "
        "repeated on a larger rule if a non-interference floor fails. "
        "The held-out bank has not been scored.",
        "",
        "| span | step | pass | toward | away | revise |",
        "|---|---|---|---|---|---|",
    ]
    for row in grid_rows:
        lines.append(
            f"| {_fmt_num(row['span'])} | {_fmt_num(row['step'])} | "
            f"{'yes' if row['ok'] else 'no'} | "
            f"{row['toward_pos']}/{row['toward_n']} {_fmt(row['toward_mean'])} | "
            f"{row['away_neg']}/{row['away_n']} {_fmt(row['away_mean'])} | "
            f"{row['revise_back']}/{row['revise_n']} |"
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
    for name, ok, detail in gate_rows(result, VALENCE_GATES, held=False):
        lines.append(f"| {name} | {'yes' if ok else 'no'} | {detail} |")
    lines.extend(
        [
            "",
            "Fit does not require the held-out mean floors "
            f"(toward mean {VALENCE_GATES['toward_mean_min']}, "
            f"away mean {VALENCE_GATES['away_mean_max']}). Those are held-out gates.",
            "",
            f"Fit toward mean {_fmt(toward['mean_delta'])} "
            f"({toward['n_pos']}/{toward['n']} > 0), audit {toward['n_audit']}/{toward['n']}.",
            f"Fit away mean {_fmt(away['mean_delta'])} "
            f"({away['n_neg']}/{away['n']} < 0), audit {away['n_audit']}/{away['n']}.",
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
            "- repeated positive experiences move the margin up on at least 6/8, "
            f"with mean at least {VALENCE_GATES['toward_mean_min']}, and a full audit",
            "- repeated negative experiences move the margin down on at least 6/8, "
            f"with mean at most {VALENCE_GATES['away_mean_max']}, and a full audit",
            "- a contrary experience pulls the margin and the scalar back on at least 6/8",
            "- absolute mean neutral ΔNLL ≤ 0.05, and at most 1 item with ΔNLL > 0.5",
            "- choice flips 0",
            "- leak 0",
            "- phone and birthday takeover 0 off-relation, including the cross question",
            "- on-relation generated text equals the stored number and the stored date",
            "- off-relation text is not the stored number and not the stored date",
            "- the prefix detector is recorded and is not the on-relation gate",
            "- incidental store keeps 32 slots",
            f"- prefer margin mean at least {REPLAY_PREFER_MEAN_MIN:.1f} on the #44 held bank "
            f"(#44 and #46 held were {REPLAY_HELD_PREFER_MEAN:.3f}; #41 held was {DIRECTION_HELD_PREFER_MEAN:.3f})",
            f"- avoid margin mean at least {REPLAY_AVOID_MEAN_MIN:.1f} on the #44 held bank "
            f"(#44 and #46 held were {REPLAY_HELD_AVOID_MEAN:.3f}; #41 held was {DIRECTION_HELD_AVOID_MEAN:.3f})",
            f"- forbidden logprob down on at least {int(REPLAY_FORBIDDEN_DOWN_MIN)}/8 on the #44 held bank "
            f"(#44 and #46 held mean stored lift {REPLAY_HELD_FORBIDDEN_LIFT:.3f}; "
            f"#41 held mean stored lift {DIRECTION_HELD_FORBIDDEN_LIFT:.3f})",
            "",
            "Report the prefer, avoid, and forbidden deltas against the #44 and #41 held numbers above. "
            "Those floors stay in scope. Do not retune span or step on this bank.",
            "",
        ]
    )
    return "\n".join(lines)


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(os.cpu_count() or 1)
    source_span = float(VALENCE_SPAN)
    source_step = float(VALENCE_STEP)
    print("loading SmolLM2-135M float32", flush=True)
    mem = LLMMemory(model_name=DEFAULT_MODEL, dtype=torch.float32)
    bank = fit_bank()
    rows = []
    try:
        for span, step in GRID:
            print(f"valence span={_fmt_num(span)} step={_fmt_num(step)}", flush=True)
            row = _probe(mem, bank, span, step)
            rows.append(row)
            print(
                f"  ok {row['ok']} toward {row['toward_pos']}/{row['toward_n']} "
                f"{row['toward_mean']} away {row['away_neg']}/{row['away_n']} "
                f"{row['away_mean']} revise {row['revise_back']}/{row['revise_n']}",
                flush=True,
            )
    finally:
        llm_memory.VALENCE_SPAN = source_span
        llm_memory.VALENCE_STEP = source_step
    chosen = _choose(rows)
    if chosen is None:
        raise SystemExit("no grid point cleared the valence floors")
    if float(chosen["span"]) != source_span or float(chosen["step"]) != source_step:
        print(
            f"set VALENCE_SPAN / VALENCE_STEP to {chosen['span']} and {chosen['step']}",
            flush=True,
        )
        mem.close()
        raise SystemExit(2)
    print("full fit score", flush=True)
    result = score_bank(mem, bank, progress=lambda msg: print(msg, flush=True))
    out = ROOT / "reports" / "subconscious_valence_fit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "grid": rows,
        "chosen": {"span": chosen["span"], "step": chosen["step"]},
        "result": result,
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", out, flush=True)
    cost = result["cost"]
    if int(cost["parameters_new"]) != 0 or int(cost["remember_forwards"]) != 2 or int(cost["query_forwards"]) != 1:
        print("fit cost is not the fixed budget", cost, flush=True)
        mem.close()
        raise SystemExit(1)
    if not passes(result, VALENCE_GATES, held=False):
        print("fit gates failed", flush=True)
        for name, ok, detail in gate_rows(result, VALENCE_GATES, held=False):
            print(f"  {name}: {'yes' if ok else 'NO'} {detail}", flush=True)
        mem.close()
        raise SystemExit(1)
    prereg = ROOT / "reports" / "subconscious_valence_prereg.md"
    prereg.write_text(_render_prereg(result, rows, chosen), encoding="utf-8")
    print("wrote", prereg, flush=True)
    mem.close()


if __name__ == "__main__":
    main()
