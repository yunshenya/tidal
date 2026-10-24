"""Score the fit spelling bank once and write the preregistration.

Does not score the held-out bank. Exits before the preregistration file when
a fit gate fails.
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

from tidal.llm_memory import DEFAULT_GAP_TAU, LLMMemory, POLARITY_SPAN, REPLAY_DECAY, REPLAY_EVERY, REPLAY_M
from tidal.subconscious_spell_data import (
    DIRECTION_HELD_AVOID_MEAN,
    DIRECTION_HELD_FORBIDDEN_LIFT,
    DIRECTION_HELD_PREFER_MEAN,
    FROZEN_SHA,
    REPLAY_FIT_SHA,
    REPLAY_HELD_AVOID_MEAN,
    REPLAY_HELD_FORBIDDEN_LIFT,
    REPLAY_HELD_PREFER_MEAN,
    REPLAY_HELD_SHA,
    RIF_FIT_SHA,
    RIF_HELD_SHA,
    SPELL_GATES,
    SURPRISE_FIT_SHA,
    SURPRISE_HELD_SHA,
    VALENCE_FIT_SHA,
    VALENCE_HELD_SHA,
    bank_sha256,
    fit_bank,
)
from tidal.subconscious_spell_score import gate_rows, passes, score_spell


def _fmt(value, digits=3):
    if value is None:
        return "—"
    return f"{float(value):.{digits}f}"


def _render_prereg(result):
    exact = result["exact"]
    neutral = result["neutral"]
    cost = result["cost"]
    nll = neutral["nll"]
    lines = [
        "# Exact phone and birthday spelling preregistration",
        "",
        "The held-out bank has not been scored.",
        "",
        "No feature flag. `gap_tau` stays 0.80. Polarity gain stays "
        "`1 + 2 tanh(w · normalize(k))`. Replay stays every 32 accepted writes, "
        "m = 1, decay 0.5. Readout stays hard top-1. A continuation still encodes "
        "the query once.",
        "",
        "Phone and birthday spelling copies the winning slot's stored answer, "
        "the full number or the full date. It does not spell a trained prefix "
        "such as `+1-415-555-` and it does not spell a digit readout. It runs "
        "only when the prompt is completing that relation.",
        "",
        "## Banks",
        "",
        f"- fit `{bank_sha256('fit')}`",
        f"- held `{bank_sha256('held')}`",
        f"- frozen `{FROZEN_SHA}`",
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
        f"- valence fit `{VALENCE_FIT_SHA}`",
        f"- valence held `{VALENCE_HELD_SHA}`",
        "- literal overlap 0 and token Jaccard under 0.50 against those banks and between fit and held",
        "",
        "## Rule",
        "",
        "- on-relation phone: generated text equals the stored number",
        "- on-relation birthday: generated text equals the stored date",
        "- prefix takeover is not the on-relation gate",
        "- off-relation phone takeover: 0",
        "- off-relation birthday takeover: 0",
        "- off-relation text is not the stored string",
        f"- new parameters: {cost['parameters_new']}",
        f"- forwards inside one remember: {cost['remember_forwards']}",
        f"- forwards inside one armed phone question: {cost['query_forwards']}",
        f"- gap_tau: {DEFAULT_GAP_TAU:.2f}",
        f"- polarity span: {POLARITY_SPAN:.0f}",
        f"- replay every: {int(REPLAY_EVERY)}",
        f"- replay m: {int(REPLAY_M)}",
        f"- replay decay: {REPLAY_DECAY:.1f}",
        "",
        "## Fit",
        "",
        f"- phone exact {exact['phone_exact']}: {exact['phone_question_text']}",
        f"- stored phone {exact['phone_stored']}",
        f"- phone prefix detector {exact['phone_prefix']}",
        f"- birthday exact {exact['birthday_exact']}: {exact['birthday_question_text']}",
        f"- stored birthday {exact['birthday_stored']}",
        f"- birthday prefix detector {exact['birthday_prefix']}",
        f"- phone off-relation takeover {_fmt(exact['phone_off_takeover'])}",
        f"- birthday off-relation takeover {_fmt(exact['birthday_off_takeover'])}",
        f"- phone off-relation equal to stored {exact['phone_off_equal']}",
        f"- birthday off-relation equal to stored {exact['birthday_off_equal']}",
        f"- neutral NLL mean {_fmt(nll['mean_delta'])}, items above 0.5: {nll['n_bad']}",
        f"- choice flips {neutral['choice']['flips']}/{neutral['choice']['n']}",
        f"- leak {neutral['leak']['n_leak']}/{neutral['leak']['n']}",
        f"- incidental slots {neutral['incidental_slots']}",
        "",
        "## Held-out gates",
        "",
        "The same spelling gates as the fit bank. In addition, rescore the #44 "
        "held-out prefer/avoid items and report them against that run:",
        "",
        f"- prefer mean at least {SPELL_GATES['prefer_mean_min']}, #44 held {REPLAY_HELD_PREFER_MEAN:.3f}, #41 held {DIRECTION_HELD_PREFER_MEAN:.3f}",
        f"- avoid mean at least {SPELL_GATES['avoid_mean_min']}, #44 held {REPLAY_HELD_AVOID_MEAN:.3f}, #41 held {DIRECTION_HELD_AVOID_MEAN:.3f}",
        f"- forbidden down at least {SPELL_GATES['forbidden_down_min']} of 8, #44 held lift {REPLAY_HELD_FORBIDDEN_LIFT:.3f}, #41 held lift {DIRECTION_HELD_FORBIDDEN_LIFT:.3f}",
        "",
        "One held-out pass. No retune.",
        "",
    ]
    return "\n".join(lines)


def main():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.set_num_threads(os.cpu_count() or 1)
    print("loading SmolLM2-135M float32", flush=True)
    mem = LLMMemory(model_name=DEFAULT_MODEL, dtype=torch.float32)
    result = score_spell(mem, fit_bank(), progress=lambda msg: print(msg, flush=True))
    rows = gate_rows(result, SPELL_GATES)
    for name, ok, detail in rows:
        print(f"  {name}: {'yes' if ok else 'NO'} {detail}", flush=True)
    if not passes(result, SPELL_GATES):
        mem.close()
        raise SystemExit("fit gates failed; preregistration was not written")
    out = ROOT / "reports" / "subconscious_spell_fit.json"
    out.write_text(json.dumps({"result": result}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    prereg = ROOT / "reports" / "subconscious_spell_prereg.md"
    prereg.write_text(_render_prereg(result), encoding="utf-8")
    print("wrote", out, flush=True)
    print("wrote", prereg, flush=True)
    mem.close()


if __name__ == "__main__":
    main()
