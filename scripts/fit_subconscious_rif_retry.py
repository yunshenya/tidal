"""Fit the retrieval span on the fit bank, then score that bank once.

Does not import the held-out bank and does not score it. Prints the constant
to set when the source does not already match the chosen span, and exits
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
from tidal.subconscious_circadian_score import gate_rows as clock_gate_rows
from tidal.subconscious_circadian_score import passes as clock_passes
from tidal.subconscious_circadian_score import score_bank as clock_score_bank
from tidal.subconscious_rif_retry_data import (
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
    HUMAN_EFFECT,
    RIF_GATES,
    RIF_HELD_SHA,
    SPELL_FIT_SHA,
    SPELL_HELD_SHA,
    SURPRISE_FIT_SHA,
    SURPRISE_HELD_SHA,
    SURPRISE_REL_FIT_SHA,
    SURPRISE_REL_HELD_SHA,
    SURPRISE_REL_TRAIN_SHA,
    VALENCE_FIT_SHA,
    VALENCE_HELD_SHA,
    bank_sha256,
    fit_bank,
)
from tidal.subconscious_rif_retry_score import gate_rows, passes, score_bank, score_quartets
from tidal.subconscious_surprise_rel_data import SURPRISE_GATES, fit_bank as surprise_fit_bank
from tidal.subconscious_surprise_rel_score import gate_rows as surprise_gate_rows
from tidal.subconscious_surprise_rel_score import passes as surprise_passes
from tidal.subconscious_surprise_rel_score import score_bank as surprise_score_bank
from tidal.subconscious_valence_data import VALENCE_GATES, fit_bank as valence_fit_bank
from tidal.subconscious_valence_score import gate_rows as valence_gate_rows
from tidal.subconscious_valence_score import passes as valence_passes
from tidal.subconscious_valence_score import score_bank as valence_score_bank

# Smallest span that clears the fit quartet floors. The set size is not on the grid.
GRID = (0.05, 0.10, 0.20)


def _fmt_num(value):
    return f"{float(value):g}"


def _fmt(value, digits=3):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _rows(gate_fn, result, gates, held):
    return [
        {"name": name, "ok": bool(ok), "detail": detail}
        for name, ok, detail in gate_fn(result, gates, held=held)
    ]


def _weaken_ok(quartets) -> bool:
    if not quartets["geometry_ok"]:
        return False
    return (
        float(quartets["competitor_drop_min"]) >= float(RIF_GATES["competitor_drop_min"])
        and float(quartets["competitor_drop_mean"]) >= float(RIF_GATES["competitor_mean_drop_min"])
        and float(quartets["winner_abs_drop_max"]) <= float(RIF_GATES["winner_abs_drop_max"])
        and float(quartets["noncompetitor_abs_drop_max"]) <= float(RIF_GATES["noncompetitor_abs_drop_max"])
    )


def _render_prereg(result, grid_rows, chosen, floors):
    cost = result["cost"]
    quartets = result["quartets"]
    lines = [
        "# Retrieval-induced forgetting retry preregistration",
        "",
        "Retrieval-induced forgetting. Retrieving a target weakens the other "
        "slots that share the winner's relation. The aim is the human pattern "
        "of a partial, fuzzy loss of access, not a more useful memory. "
        "The held-out bank has not been scored.",
        "",
        "No feature flag. `gap_tau` stays 0.80. Polarity gain stays "
        "`1 + 2 tanh(w · normalize(k))`. Valence stays `u = 1 + 1 * tanh(valence)` "
        "with step 0.25. The clock stays `phase = sin(angle)` with span "
        f"{_fmt_num(episodic.CLOCK_SPAN)} and step pi/16. Phone and birthday "
        "spelling stays the exact stored answer, and still runs only when the "
        "winning relation is the one the prompt is completing. A continuation "
        "still encodes the query once. Readout stays hard top-1. Replay stays "
        "every 32 accepted writes, refreshing the top 1 slot, decay 0.50. "
        "The surprise quantile stays 0 over the last 32 accepted write surprises. "
        "The valence formula is not in this change. The circadian formula is not "
        "in this change. The relative surprise gate is not in this change. "
        "The inject MLP is not in this change.",
        "",
        "After a strict retrieval win, and after that winner's strength has been "
        "copied for the current inject, other slots that share the winner's relation "
        "are competitors when there are 1 to 3 of them. A larger shared relation "
        "is background and is not a cue. A tie adds nothing and forgets nothing. "
        "A slot whose salience reaches 0.5 is left alone. The keep is "
        "`1 - span * weight`. Weight is 1 for the competitor whose key is closest "
        "to the winner and at least 1/2 for every other competitor in that set. "
        "The winner is not scaled. There is no extra forward and no new parameter. "
        "The set size and the salience exempt are not on the grid.",
        "",
        "Prefer, avoid, and forbidden on the replay held-out bank stay in scope. "
        "They are not out of scope for this change. Suppression does not run on a "
        "one-slot read, and a shared relation larger than three slots is not a cue, "
        "so the note stores and the polarity items are not competitors.",
        "",
        "## Banks",
        "",
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
        f"- #49 train `{SURPRISE_REL_TRAIN_SHA}`",
        f"- #49 fit `{SURPRISE_REL_FIT_SHA}`",
        f"- #49 held `{SURPRISE_REL_HELD_SHA}`",
        "- literal overlap 0 and token Jaccard under 0.50 against those banks and between fit and held",
        "",
        "## Span grid",
        "",
        "| span | min drop | mean drop | winner abs | outsider abs | geometry | pass |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in grid_rows:
        lines.append(
            f"| {_fmt_num(row['span'])} | {_fmt(row['competitor_drop_min'])} | "
            f"{_fmt(row['competitor_drop_mean'])} | {_fmt(row['winner_abs_drop_max'])} | "
            f"{_fmt(row['noncompetitor_abs_drop_max'])} | "
            f"{'yes' if row['geometry_ok'] else 'no'} | {'yes' if row['ok'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            f"Chosen retrieval span: {_fmt_num(chosen['span'])}.",
            f"rif span: {_fmt_num(chosen['span'])}",
            "",
            "The grid is 0.05, 0.10, 0.20. The smallest span whose fit quartets have "
            "minimum competitor drop at least 0.10, mean competitor drop at least 0.20, "
            "absolute winner drop at most 0.05, and absolute noncompetitor drop at most "
            "0.05 is the one scored below. The full fit score then runs once. It is not "
            "repeated at a larger span if a non-interference floor fails. The held-out "
            "bank has not been scored.",
            "",
            f"Fit bank `{result['bank_sha256']}`.",
            "",
            "| gate | pass | detail |",
            "|---|---|---|",
        ]
    )
    for name, ok, detail in gate_rows(result, RIF_GATES, held=False):
        lines.append(f"| {name} | {'yes' if ok else 'no'} | {detail} |")
    for label, rows in floors.items():
        for row in rows:
            lines.append(
                f"| {label} {row['name']} | {'yes' if row['ok'] else 'no'} | {row['detail']} |"
            )
    lines.extend(
        [
            "",
            "Fit does not require the held-out direction mean floors "
            f"(prefer mean {REPLAY_PREFER_MEAN_MIN:.1f}, avoid mean {REPLAY_AVOID_MEAN_MIN:.1f}) "
            "or forbidden-down. Those are held-out gates. Competitor mean drop is a fit gate.",
            "",
            f"Fit competitor min drop {_fmt(quartets['competitor_drop_min'])}, "
            f"mean {_fmt(quartets['competitor_drop_mean'])}, "
            f"winner abs {_fmt(quartets['winner_abs_drop_max'])}, "
            f"outsider abs {_fmt(quartets['noncompetitor_abs_drop_max'])}.",
            "",
            "## Human effect",
            "",
            HUMAN_EFFECT.strip(),
            "",
            "## Held-out gates",
            "",
            "The held-out bank has not been scored. It is merge-critical. "
            "These gates are the human effect in the section above, plus the "
            "regression floors already required. The numeric floors are unchanged.",
            "",
            "- Rp- competitors that share the winner's relation are recalled less than "
            "their strength-1 Nrp control after one target hit: minimum drop at least "
            "0.10, mean drop at least 0.20",
            "- the Rp+ target is not weakened: absolute winner drop at most 0.05 "
            "(human Rp+ recall was facilitated; this store does not add that facilitation)",
            "- Nrp noncompetitors are unaffected: absolute outsider drop at most 0.05",
            "- geometry: five slots, a strict gap, the target wins, every item hits, "
            "and exactly the three competitors are scaled",
            f"- prefer margin delta > 0 on 8/8 and mean at least {REPLAY_PREFER_MEAN_MIN:.1f} "
            f"on the replay held-out bank "
            f"(#49/#48/#47/#44/#46 held were {REPLAY_HELD_PREFER_MEAN:.3f}; "
            f"#41 held was {DIRECTION_HELD_PREFER_MEAN:.3f})",
            f"- avoid margin delta > 0 on 8/8 and mean at least {REPLAY_AVOID_MEAN_MIN:.1f} "
            f"on the replay held-out bank "
            f"(#49/#48/#47/#44/#46 held were {REPLAY_HELD_AVOID_MEAN:.3f}; "
            f"#41 held was {DIRECTION_HELD_AVOID_MEAN:.3f})",
            f"- forbidden logprob down on at least {int(REPLAY_FORBIDDEN_DOWN_MIN)}/8 "
            f"on the replay held-out bank "
            f"(#49/#48/#47/#44/#46 held mean stored lift {REPLAY_HELD_FORBIDDEN_LIFT:.3f}; "
            f"#41 held mean stored lift {DIRECTION_HELD_FORBIDDEN_LIFT:.3f})",
            "- those direction floors are bit-identical or not regressed; avoid 8/8 is required",
            "- surprise held gates, including neutral NLL, choice flips 0, leak 0, "
            "and phone and birthday text equal to the stored answer",
            "- valence held gates, including phone and birthday exact",
            "- circadian held gates, including phone and birthday exact",
            "- new parameters 0 and one forward inside the armed query",
            "",
            "Prefer, avoid, and forbidden are not out of scope. Report their deltas "
            "against the #49 and #41 held numbers above. Do not retune the span on "
            "the held-out bank.",
            "",
            f"New parameters on the fit cost probe: {cost['parameters_new']}. "
            f"Query forwards {cost['rank_forwards']}. "
            f"Forgetting record contains text: {cost['rif_has_text']}.",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(os.cpu_count() or 1)
    source = float(llm_memory.RIF_SPAN)
    if int(llm_memory.RIF_MAX_SET) != 3:
        raise SystemExit("rif set size is fixed at 3")
    if float(llm_memory.RIF_SALIENCE_EXEMPT) != 0.5:
        raise SystemExit("rif salience exempt is fixed at 0.5")
    bank = fit_bank()
    print("loading SmolLM2-135M float32", flush=True)
    mem = LLMMemory(dtype=torch.float32)
    rows = []
    try:
        for span in GRID:
            llm_memory.RIF_SPAN = float(span)
            print(f"quartet span {_fmt_num(span)}", flush=True)
            quartets = score_quartets(mem, bank)
            ok = _weaken_ok(quartets)
            row = {
                "span": float(span),
                "ok": bool(ok),
                "geometry_ok": bool(quartets["geometry_ok"]),
                "competitor_drop_min": quartets["competitor_drop_min"],
                "competitor_drop_mean": quartets["competitor_drop_mean"],
                "winner_abs_drop_max": quartets["winner_abs_drop_max"],
                "noncompetitor_abs_drop_max": quartets["noncompetitor_abs_drop_max"],
            }
            rows.append(row)
            print(
                "  "
                f"ok {ok} geometry {quartets['geometry_ok']} "
                f"min {_fmt(quartets['competitor_drop_min'])} "
                f"mean {_fmt(quartets['competitor_drop_mean'])} "
                f"winner {_fmt(quartets['winner_abs_drop_max'])} "
                f"outsider {_fmt(quartets['noncompetitor_abs_drop_max'])}",
                flush=True,
            )
            if ok:
                break
    finally:
        llm_memory.RIF_SPAN = source
    chosen = next((row for row in rows if row["ok"]), None)
    if chosen is None:
        print("no span on the grid passed the weaken gate", flush=True)
        mem.close()
        raise SystemExit(2)
    if abs(float(chosen["span"]) - source) > 1e-12:
        print(f"set RIF_SPAN = {_fmt_num(chosen['span'])}", flush=True)
        mem.close()
        raise SystemExit(2)
    print("full fit score", flush=True)
    result = score_bank(mem, bank, progress=lambda msg: print(msg, flush=True))
    print("valence fit floor", flush=True)
    valence = valence_score_bank(mem, valence_fit_bank(), progress=lambda msg: print(msg, flush=True))
    valence_ok = valence_passes(valence, VALENCE_GATES, held=False)
    print("valence fit", "yes" if valence_ok else "no", flush=True)
    print("circadian fit floor", flush=True)
    clock = clock_score_bank(mem, clock_fit_bank(), progress=lambda msg: print(msg, flush=True))
    clock_ok = clock_passes(clock, CLOCK_GATES, held=False)
    print("circadian fit", "yes" if clock_ok else "no", flush=True)
    print("surprise fit floor", flush=True)
    surprise = surprise_score_bank(mem, surprise_fit_bank(), progress=lambda msg: print(msg, flush=True))
    surprise_ok = surprise_passes(surprise, SURPRISE_GATES, held=False)
    print("surprise fit", "yes" if surprise_ok else "no", flush=True)
    floors = {
        "valence fit": _rows(valence_gate_rows, valence, VALENCE_GATES, False),
        "circadian fit": _rows(clock_gate_rows, clock, CLOCK_GATES, False),
        "surprise fit": _rows(surprise_gate_rows, surprise, SURPRISE_GATES, False),
    }
    out = ROOT / "reports" / "subconscious_rif_retry_fit.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "grid": rows,
        "chosen": {"span": float(chosen["span"])},
        "result": result,
        "floors": floors,
        "valence_fit_pass": bool(valence_ok),
        "circadian_fit_pass": bool(clock_ok),
        "surprise_fit_pass": bool(surprise_ok),
        "constants": {
            "rif_span": float(llm_memory.RIF_SPAN),
            "rif_max_set": int(llm_memory.RIF_MAX_SET),
            "rif_salience_exempt": float(llm_memory.RIF_SALIENCE_EXEMPT),
            "valence_span": float(VALENCE_SPAN),
            "valence_step": float(VALENCE_STEP),
            "clock_span": float(episodic.CLOCK_SPAN),
            "replay_every": int(REPLAY_EVERY),
            "replay_m": int(REPLAY_M),
            "replay_decay": float(REPLAY_DECAY),
            "surprise_quantile": float(llm_memory.SURPRISE_QUANTILE),
            "surprise_window": int(llm_memory.SURPRISE_WINDOW),
        },
    }
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("wrote", out, flush=True)
    if not passes(result, RIF_GATES, held=False) or not valence_ok or not clock_ok or not surprise_ok:
        print("fit gates failed; preregistration was not written", flush=True)
        for name, ok, detail in gate_rows(result, RIF_GATES, held=False):
            print(f"  {name}: {'yes' if ok else 'NO'} {detail}", flush=True)
        print("valence", valence_ok, "circadian", clock_ok, "surprise", surprise_ok, flush=True)
        mem.close()
        raise SystemExit(1)
    prereg = ROOT / "reports" / "subconscious_rif_retry_prereg.md"
    prereg.write_text(_render_prereg(result, rows, chosen, floors), encoding="utf-8")
    print("wrote", prereg, flush=True)
    mem.close()


if __name__ == "__main__":
    main()
