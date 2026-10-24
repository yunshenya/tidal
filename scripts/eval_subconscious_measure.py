"""Measurement-only pass. Does not train, retune, or change a parameter.

Writes ``reports/subconscious_measure.json`` and ``.md``. Refuses a second
pass when that JSON already exists. Partial sections resume from
``--checkpoint``.

    HF_HUB_OFFLINE=1 python scripts/eval_subconscious_measure.py
"""
from __future__ import annotations

import argparse
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

import numpy as np
import torch


CITATIONS = """\
## Citations

Choice is the associative-priming analogue. Meyer and Schvaneveldt (1971), Experiment II, found that “same” responses were 117 ± 18 ms faster for associated word pairs than for unassociated pairs, F(1, 11) = 42.6, p < .001. The margin here is a mean log-probability, not a reaction time, and the associate is a residual rather than a string on the screen.

Tulving and Schacter (1990) describe priming as a nonconscious form of human memory, separate from other memory systems. Perceptual priming, in their account, is pre-semantic. Conceptual priming seems based on semantic memory. The implicit prompts do not ask for the stored fact.

Preference is the mere-exposure analogue. Zajonc (1968) hypothesized that mere repeated exposure of a person to a stimulus enhances attitude toward it, where “mere” means the stimulus is accessible to perception. This probe stores the word once and does not show it in the prompt. That is one encoding, not Zajonc’s exposure-frequency schedule. No F statistic from that monograph is used here.

Tone is the evaluative-priming analogue. Fazio, Sanbonmatsu, Powell, and Kardes (1986) reported that an attitude with a strong object–evaluation association is activated automatically on mere presentation of the object, and that a following adjective is facilitated when its valence matches, provided the association is strong. The event word in this probe is not presented. Only the residual is. This is an analogue, not a replication.

Meyer, D. E., & Schvaneveldt, R. W. (1971). Facilitation in recognizing pairs of words: Evidence of a dependence between retrieval operations. Journal of Experimental Psychology, 90(2), 227–234. https://doi.org/10.1037/h0031564

Tulving, E., & Schacter, D. L. (1990). Priming and human memory systems. Science, 247(4940), 301–306. https://doi.org/10.1126/science.2296719

Zajonc, R. B. (1968). Attitudinal effects of mere exposure. Journal of Personality and Social Psychology, 9(2, Pt. 2), 1–27. https://doi.org/10.1037/h0025848

Fazio, R. H., Sanbonmatsu, D. M., Powell, M. C., & Kardes, F. R. (1986). On the automatic activation of attitudes. Journal of Personality and Social Psychology, 50(2), 229–238. https://doi.org/10.1037/0022-3514.50.2.229

The retrieval-induced-forgetting rates use the same sources already cited for that mechanism in `tidal/subconscious_rif_retry_data.py`: Anderson, Bjork, and Bjork (1994), https://doi.org/10.1037/0278-7393.20.5.1063, and Anderson (2003), https://doi.org/10.1016/j.jml.2003.08.006. Those papers measure category-cued recall after retrieval practice. The rate in this report is a log-probability margin.
"""


class BGE:
    """CLS top-1. Same pooling as ``scripts/eval_subconscious.py``, without a prefix."""

    def __init__(self):
        from transformers import AutoModel, AutoTokenizer

        self.tok = AutoTokenizer.from_pretrained("BAAI/bge-small-en-v1.5")
        self.model = AutoModel.from_pretrained("BAAI/bge-small-en-v1.5").eval()

    @torch.no_grad()
    def encode(self, texts):
        rows = [str(text) for text in texts]
        batch = self.tok(rows, padding=True, truncation=True, return_tensors="pt")
        hidden = self.model(**batch).last_hidden_state[:, 0]
        return torch.nn.functional.normalize(hidden, dim=-1)

    def top1(self, corpus, texts, query: str) -> str:
        q = self.encode([query])[0]
        scores = corpus @ q
        return texts[int(scores.argmax())]


def _lines(path: Path) -> list:
    return path.read_text(encoding="utf-8").splitlines()


def _function(path: Path, name: str, indent: int = 0) -> str:
    lines = _lines(path)
    prefix = (" " * indent) + "def " + name + "("
    start = next(i for i, line in enumerate(lines) if line.startswith(prefix))
    end = len(lines)
    for i in range(start + 1, len(lines)):
        stripped = lines[i]
        if stripped.startswith(prefix[:indent] + "def ") or (indent == 0 and stripped.startswith("def ")):
            end = i
            break
        if indent and stripped.startswith("def "):
            end = i
            break
        if indent and stripped.startswith("class "):
            end = i
            break
    while end > start and lines[end - 1].strip() == "":
        end -= 1
    body = lines[start:end]
    return "\n".join(body)


def _method(path: Path, class_name: str, method: str) -> str:
    lines = _lines(path)
    class_at = next(i for i, line in enumerate(lines) if line.startswith("class " + class_name))
    start = None
    for i in range(class_at + 1, len(lines)):
        if lines[i].startswith("class "):
            break
        if lines[i].startswith("    def " + method + "("):
            start = i
            break
    if start is None:
        raise RuntimeError(f"{class_name}.{method} not found")
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].startswith("    def ") or lines[i].startswith("class "):
            end = i
            break
    while end > start and lines[end - 1].strip() == "":
        end -= 1
    return "\n".join(lines[start:end])


def injection_quotes() -> dict:
    """Source of the path that arms a prompt. Read at report time, not copied by hand."""
    mem = ROOT / "tidal" / "llm_memory.py"
    epi = ROOT / "tidal" / "episodic.py"
    quotes = {
        "inject_forward": _method(mem, "InjectMLP", "forward"),
        "retrieval_confidence": _function(mem, "retrieval_confidence"),
        "on_layer": _method(mem, "LLMMemory", "_on_layer"),
        "rank": _method(mem, "LLMMemory", "_rank"),
        "set_delta": _method(mem, "LLMMemory", "_set_delta"),
        "broadcast": _function(mem, "_broadcast"),
        "mix_visible": _method(epi, "EpisodicStore", "_mix_visible"),
        "read": _method(epi, "EpisodicStore", "read"),
    }
    required = {
        "rank": 'mode="top1"',
        "mix_visible": "return s0.v.copy(), info",
        "broadcast": "return add.view(*shape)",
        "set_delta": "self._compose(value, key=rank.get(\"key\"), grad=False)",
        "read": 'mode: str = "softtopk"',
    }
    for key, needle in required.items():
        if needle not in quotes[key]:
            raise RuntimeError(f"injection quote {key} lost {needle}")
    return quotes


def _fmt(value, digits=4):
    if value is None:
        return "n/a"
    return f"{float(value):.{digits}f}"


def _rate(row) -> str:
    if not row:
        return "n/a"
    lo, hi = row["wilson95"]
    return f"{row['k']}/{row['n']} = {row['rate']:.4f} (95% Wilson {lo:.4f}–{hi:.4f})"


def _dist_table(dist: dict) -> str:
    if not dist or not dist.get("n_pairs"):
        return "No pairs.\n"
    keys = ["min", "p10", "p25", "p50", "p75", "p90", "p95", "p99", "max", "mean"]
    cells = " | ".join(_fmt(dist.get(key), 4) for key in keys)
    header = " | ".join(keys)
    bins = dist.get("hist_bins") or []
    counts = dist.get("hist_counts") or []
    hist = ", ".join(
        f"[{bins[i]:.2f}, {bins[i+1]:.2f}] {counts[i]}" for i in range(len(counts))
    )
    return (
        f"| {header} |\n| {' | '.join(['---'] * len(keys))} |\n| {cells} |\n\n"
        f"Pairs {dist['n_pairs']}. Fraction > 0.99: {dist.get('frac_gt_0.99')} "
        f"({dist.get('n_gt_0.99')}). Fraction > 0.9: {dist.get('frac_gt_0.9')} "
        f"({dist.get('n_gt_0.9')}). All pairs > 0.99: {dist.get('all_gt_0.99')}.\n\n"
        f"Histogram: {hist}. Overflow outside [-1, 1]: {dist.get('overflow')}.\n"
    )


def _quote_block(title: str, body: str) -> str:
    return f"### {title}\n\n```python\n{body}\n```\n"


def render_report(result: dict) -> str:
    inj = result.get("injection") or {}
    quotes = inj.get("quotes") or {}
    cos = result.get("key_cosine") or {}
    banks = result.get("banks") or {}
    implicit = result.get("implicit") or {}
    explicit = result.get("explicit") or {}
    cost = result.get("compute") or {}
    lines = []
    lines.append("# Subconscious measurement\n")
    lines.append(
        "This pass does not change retrieval, injection, surprise, valence, "
        "retrieval-induced forgetting, polarity, or spelling. The locked banks "
        "are untouched. The new banks are an additional measurement, 50 items "
        f"in each family. Bank sha256 `{result.get('bank_sha256')}`. "
        f"Protocol `{result.get('protocol')}`. Model `{result.get('model')}`. "
        f"Commit `{result.get('git_sha')}`.\n"
    )
    lines.append("## How retrieval feeds injection\n")
    lines.append(
        "Readout on this path is hard top-1. `LLMMemory._rank` calls "
        "`EpisodicStore.read` with `mode=\"top1\"`. For that mode `_mix_visible` "
        "returns `s0.v.copy()`, the entire stored value of the winning slot, "
        "not a mixture. `EpisodicStore.read` still defaults to `softtopk`; "
        "that default is the continuum path, and `generate` does not read "
        "`soft_k` or `soft_temp`.\n\n"
        "The vector that is added is not the raw value. `_set_delta` passes "
        "the value and the winning key through `InjectMLP.forward`: "
        "`scaled = value * scale`, `residual = scaled + net(scaled)`, then "
        "the polarity gain when the polarity vector is nonzero. One "
        "hidden-sized residual comes out. It is multiplied by "
        "`confidence * strength * valence_gain`. `retrieval_confidence` "
        "returns 1 when the store has fewer than two slots, and 0 on a tie.\n\n"
        "`_on_layer` adds that one residual through `_broadcast`. The "
        "broadcast reshape is `[1] * (hidden.dim() - 1) + [hidden]`, so the "
        "same full vector is added at every position of the layer output. "
        "It is not a per-position retrieval and it is not a slice of the vector.\n"
    )
    for key, title in (
        ("rank", "LLMMemory._rank"),
        ("mix_visible", "EpisodicStore._mix_visible"),
        ("read", "EpisodicStore.read"),
        ("inject_forward", "InjectMLP.forward"),
        ("retrieval_confidence", "retrieval_confidence"),
        ("set_delta", "LLMMemory._set_delta"),
        ("on_layer", "LLMMemory._on_layer"),
        ("broadcast", "_broadcast"),
    ):
        if quotes.get(key):
            lines.append(_quote_block(title, quotes[key]))
    lines.append("## Stored key cosines\n")
    lines.append(
        "The question is whether stored keys are still all above 0.99. "
        "The encoder distribution is pairwise cosine of `encode_key` on the "
        "new prefer and avoid items plus the locked replay held prefer/avoid "
        "items and the locked valence held toward/away items. `remember` is "
        "then run on the new prefer and avoid items only. `overwrite_only` "
        "writes those same encoder keys through `store.write`, so cosine "
        f"overwrite at {cos.get('overwrite_cos')} is visible without the surprise gate.\n"
    )
    encoder = cos.get("encoder") or {}
    lines.append(f"Encoder vectors: {encoder.get('n_vectors')}.\n")
    lines.append(_dist_table(encoder))
    remembered = cos.get("remember") or {}
    lines.append(
        f"After `remember`, attempted {remembered.get('attempted')}, "
        f"`remember` returned true {remembered.get('written')} times, "
        f"slots left {remembered.get('n_slots')}.\n"
    )
    lines.append(_dist_table(remembered.get("distribution") or {}))
    direct = cos.get("overwrite_only") or {}
    lines.append(
        f"Overwrite-only write, attempted {direct.get('attempted')}, "
        f"accepted {direct.get('accepted')}, slots left {direct.get('n_slots')}.\n"
    )
    lines.append(_dist_table(direct.get("distribution") or {}))
    by_source = cos.get("encoder_by_source") or {}
    if by_source:
        lines.append("Encoder pairs within each source:\n")
        lines.append("| source | vectors | pairs | min | median | max | frac > 0.99 |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- |")
        for name in sorted(by_source):
            dist = by_source[name]
            lines.append(
                f"| {name} | {dist.get('n_vectors')} | {dist.get('n_pairs')} | "
                f"{_fmt(dist.get('min'))} | {_fmt(dist.get('p50'))} | {_fmt(dist.get('max'))} | "
                f"{_fmt(dist.get('frac_gt_0.99'))} |"
            )
        lines.append("")
    lines.append("## Expanded banks\n")
    lines.append(
        "Each family has 50 items, disjoint from the existing data-module "
        "surfaces at token Jaccard under 0.50. The locked n=8 banks were not "
        "edited and were not the rows scored here. Rates use the existing "
        "scorers. A 95% Wilson interval is on the rate, not on the mean margin.\n"
    )
    prefer = (banks.get("direction") or {}).get("prefer") or {}
    avoid = (banks.get("direction") or {}).get("avoid") or {}
    lines.append("### Prefer and avoid\n")
    lines.append(
        "Prefer stores “favors ANSWER” and scores good minus bad against an "
        "empty store. Avoid stores “dodges ANSWER”; the good continuation is "
        "the other word and the bad continuation is the stored word. "
        "`stored_down` counts items whose stored-side log-probability fell.\n"
    )
    lines.append(f"Prefer margin up: {_rate(prefer.get('delta_pos'))}. Mean delta {_fmt(prefer.get('mean_delta'), 4)}.\n")
    lines.append(f"Avoid margin up: {_rate(avoid.get('delta_pos'))}. Mean delta {_fmt(avoid.get('mean_delta'), 4)}.\n")
    lines.append(
        f"Avoid stored-side down: {_rate(avoid.get('stored_down'))}. "
        f"Mean stored lift {_fmt(avoid.get('mean_stored_lift'), 4)}.\n"
    )
    valence = banks.get("valence") or {}
    toward = valence.get("toward") or {}
    away = valence.get("away") or {}
    lines.append("### Valence\n")
    lines.append(
        f"`score_signed` repeats the shipped step {valence.get('repeats')} times. "
        "Toward wants a positive margin change. Away wants a negative one.\n"
    )
    lines.append(f"Toward positive: {_rate(toward.get('signed_rate'))}. Mean delta {_fmt(toward.get('mean_delta'), 4)}.\n")
    lines.append(f"Away negative: {_rate(away.get('signed_rate'))}. Mean delta {_fmt(away.get('mean_delta'), 4)}.\n")
    surprise = banks.get("surprise") or {}
    lines.append("### Surprise\n")
    lines.append(
        "Thirty-two seed notes fill the window. Quantile 0 sets the bar at "
        "the minimum seed NLL. A low is rejected only when its answer-token "
        "NLL is strictly below every seed. Highs are the same frames with "
        "the answer replaced.\n"
    )
    lines.append(
        f"Seeds written {surprise.get('seed_written')}/{surprise.get('seed_n')}. "
        f"Bar {_fmt(surprise.get('bar'), 4)}. "
        f"Low mean surprise {_fmt(surprise.get('low_mean_surprise'), 4)}. "
        f"High mean surprise {_fmt(surprise.get('high_mean_surprise'), 4)}.\n"
    )
    lines.append(f"Lows not written: {_rate(surprise.get('low_rejected'))}.\n")
    lines.append(f"Highs written: {_rate(surprise.get('high_written'))}.\n")
    rif = banks.get("rif") or {}
    lines.append("### Retrieval-induced forgetting\n")
    lines.append(
        "Fifty quartets, each a target, three same-relation competitors, and "
        "one outsider. `score_quartets` compares one top-1 with a strength-1 "
        "control. `geometry_ok` requires five slots, the target first, three "
        "competitors, one outsider, a strict win, a hit on every row, winner "
        "and outsider strength 1, every competitor below 1, and `forgotten` "
        "equal to 3.\n\n"
        "`score_quartets` zips the written facts onto the surviving slots. "
        "When cosine overwrite leaves fewer than five slots, later roles drop "
        "out of that zip. The rows that remain are not a competitor or "
        "outsider sample. Drop rates are reported only when `geometry_ok` is "
        "nonzero.\n"
    )
    lines.append(f"Geometry ok: {_rate(rif.get('geometry_ok'))}.\n")
    counts = rif.get("slot_counts") or {}
    if counts:
        lines.append("| slots left | quartets |")
        lines.append("| --- | --- |")
        for nslot in range(1, 6):
            lines.append(f"| {nslot} | {int(counts.get(str(nslot), 0))} |")
        lines.append("")
    survivor = rif.get("survivor_max_key_cosine") or {}
    if survivor.get("n"):
        lines.append(
            f"Quartets with at least two slots: max key cosine "
            f"n={survivor.get('n')}, min {_fmt(survivor.get('min'))}, "
            f"median {_fmt(survivor.get('median'))}, max {_fmt(survivor.get('max'))}. "
            "A one-slot quartet records max key cosine 0. That 0 is the "
            "scorer’s n<2 sentinel.\n"
        )
    emitted = rif.get("emitted_role_counts") or {}
    if emitted:
        lines.append(
            f"The zip emitted target {emitted.get('target', 0)}, "
            f"competitor {emitted.get('competitor', 0)}, outsider "
            f"{emitted.get('outsider', 0)}. Those counts are not a rate "
            "denominator.\n"
        )
    forgotten = rif.get("forgotten_counts") or {}
    if forgotten:
        parts = ", ".join(
            f"{key}: {forgotten[key]}" for key in sorted(forgotten, key=lambda item: int(item))
        )
        lines.append(f"Forgotten counter by quartet: {parts}.\n")
    if rif.get("rates_estimated"):
        lines.append(
            f"Competitor drop > 0, geometry-ok quartets only: {_rate(rif.get('competitor_drop_pos'))}. "
            f"Mean {_fmt(rif.get('competitor_drop_mean'), 4)}. "
            f"Min {_fmt(rif.get('competitor_drop_min'), 4)}.\n"
        )
        lines.append(
            f"Winner not weakened: {_rate(rif.get('winner_not_weakened'))}. "
            f"Mean winner drop {_fmt(rif.get('winner_drop_mean'), 4)}. "
            f"Max absolute winner drop {_fmt(rif.get('winner_abs_max'), 4)}.\n"
        )
        lines.append(
            f"Outsider unaffected: {_rate(rif.get('outsider_unaffected'))}. "
            f"Max absolute outsider drop {_fmt(rif.get('outsider_abs_max'), 4)}.\n"
        )
    elif rif:
        lines.append(
            "Competitor and outsider rates are not estimated. No quartet kept "
            "the five-slot geometry the scorer requires.\n"
        )
    lines.append("## Implicit influence\n")
    lines.append(
        "The subconscious score is the unweighted mean of three paired "
        "Cohen’s d_z values: tone, choice, and preference, inject minus "
        "no-memory. d_z is the mean paired difference divided by its sample "
        "standard deviation. Explicit phone and birthday exact-match rates "
        "are not in that mean.\n\n"
        "The prompt never contains the name, the relation, the stored word, "
        "or the foil. Inject writes one slot, so confidence is 1 even if the "
        "query cosine is low; the mean query cosine is reported. No-memory "
        "is an empty store. In-context prepends the fact text and a newline. "
        "Simple RAG is BGE CLS top-1 over that family’s 50 fact lines. The "
        "query is the uninformative prompt, so a hit is not guaranteed.\n"
    )
    lines.append(
        f"Subconscious score (mean d_z, inject versus no-memory): "
        f"{_fmt(implicit.get('subconscious_score'), 4)}.\n"
    )
    families = implicit.get("families") or {}
    lines.append(
        "| family | d_z inject−none | d_z context−none | d_z RAG−none | "
        "d_z inject−context | d_z inject−RAG | inject Δ>0 | RAG hit | mean maxcos |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for family in ("tone", "choice", "preference"):
        block = families.get(family) or {}
        contrasts = block.get("contrasts") or {}
        lines.append(
            "| {family} | {a} | {b} | {c} | {d} | {e} | {f} | {g} | {h} |".format(
                family=family,
                a=_fmt((contrasts.get("inject_vs_nomem") or {}).get("dz")),
                b=_fmt((contrasts.get("context_vs_nomem") or {}).get("dz")),
                c=_fmt((contrasts.get("rag_vs_nomem") or {}).get("dz")),
                d=_fmt((contrasts.get("inject_vs_context") or {}).get("dz")),
                e=_fmt((contrasts.get("inject_vs_rag") or {}).get("dz")),
                f=_rate((contrasts.get("inject_vs_nomem") or {}).get("delta_pos")),
                g=_rate(block.get("rag_hit")),
                h=_fmt(block.get("inject_maxcos_mean")),
            )
        )
    lines.append("")
    for family in ("tone", "choice", "preference"):
        block = families.get(family) or {}
        contrasts = block.get("contrasts") or {}
        inj_c = contrasts.get("inject_vs_nomem") or {}
        lines.append(
            f"{family}: n={block.get('n')}, inject−none mean {_fmt(inj_c.get('mean'), 4)}, "
            f"sd {_fmt(inj_c.get('sd'), 4)}, d_z {_fmt(inj_c.get('dz'), 4)}, "
            f"rate {_rate(inj_c.get('delta_pos'))}. "
            f"Mean margins nomem {_fmt((block.get('mean_margin') or {}).get('nomem'), 4)}, "
            f"inject {_fmt((block.get('mean_margin') or {}).get('inject'), 4)}, "
            f"context {_fmt((block.get('mean_margin') or {}).get('context'), 4)}, "
            f"RAG {_fmt((block.get('mean_margin') or {}).get('rag'), 4)}. "
            f"Inject confidence mean {_fmt(block.get('inject_confidence_mean'), 4)}. "
            f"RAG unique lines {block.get('rag_unique')}."
        )
    lines.append("")
    lines.append("## Added compute per token\n")
    lines.append(
        "Wall clock on this CPU, median over the repeats below. An armed "
        "forward minus an empty forward at lengths 16, 32, and 64 changes "
        "sign and stays under a millisecond. That difference is timer noise, "
        "not a measured speedup and not a measured residual cost. The "
        f"structural cost inside the noise is one {cost.get('hidden')}-d add "
        f"at every position ({cost.get('residual_adds_per_token')} scalar "
        "adds) and one InjectMLP "
        f"({cost.get('inject_parameters')} parameters) once per prompt. "
        "Retrieval is not inside that forward.\n\n"
        "The added time that clears the noise is one `_rank` (the query "
        "forward and the top-1 read) plus one `InjectMLP` evaluation. Those "
        "seconds are also divided by the continuation-token count of one "
        "implicit pair (good plus bad). The inject total uses that amortized "
        "retrieval. It does not add the noisy residual mean.\n\n"
        "In-context is the longer forward minus the short forward. The table "
        "below keeps the undivided extra seconds, the added prompt tokens, "
        "and the continuation-token count. The summary row averages the "
        "per-continuation-token extras of one tone, one choice, and one "
        "preference item. RAG adds one uncached BGE top-1 over the 50 tone "
        "lines on top of that longer-prompt cost.\n"
    )
    retrieval = cost.get("retrieval") or {}
    bge = cost.get("bge") or {}
    lines.append(
        f"Repeats {cost.get('repeats')}, warmup {cost.get('warmup')}. "
        f"Signed mean residual extra per token across 16/32/64: "
        f"{_fmt(cost.get('residual_extra_per_token_s'), 6)} s. "
        "The sign flips with length.\n"
    )
    lines.append("| piece | seconds | per continuation token |")
    lines.append("| --- | --- | --- |")
    lines.append(
        "| residual add | within timer noise | within timer noise |"
    )
    lines.append(
        f"| `_rank` | {_fmt(retrieval.get('rank_median_s'), 6)} | "
        f"{_fmt(retrieval.get('rank_per_continuation_token_s'), 6)} |"
    )
    lines.append(
        f"| InjectMLP | {_fmt(retrieval.get('compose_median_s'), 6)} | "
        f"{_fmt(retrieval.get('compose_per_continuation_token_s'), 6)} |"
    )
    rank_tok = retrieval.get("rank_per_continuation_token_s")
    compose_tok = retrieval.get("compose_per_continuation_token_s")
    inject_tok = None if rank_tok is None or compose_tok is None else float(rank_tok) + float(compose_tok)
    lines.append(
        f"| inject total (`_rank` + InjectMLP) |  | {_fmt(inject_tok, 6)} |"
    )
    lines.append(
        f"| in-context extra forward |  | {_fmt(cost.get('context_extra_per_continuation_token_s'), 6)} |"
    )
    lines.append(
        f"| BGE corpus + query | {_fmt(bge.get('full_median_s'), 6)} | "
        f"{_fmt(bge.get('full_per_continuation_token_s'), 6)} |"
    )
    lines.append(
        f"| RAG total (uncached BGE + in-context extra) |  | "
        f"{_fmt(cost.get('rag_added_per_continuation_token_s'), 6)} |"
    )
    lines.append("")
    if retrieval.get("inject_added_per_continuation_token_s") is not None:
        lines.append(
            "The JSON field `inject_added_per_continuation_token_s` "
            f"({_fmt(retrieval.get('inject_added_per_continuation_token_s'), 6)}) "
            "adds the signed residual mean onto the inject total. The table "
            "leaves the residual out.\n"
        )
    forwards = cost.get("forward_by_length") or {}
    if forwards:
        lines.append("| length | empty s | armed s | extra s | extra per token s |")
        lines.append("| --- | --- | --- | --- | --- |")
        for length in ("16", "32", "64"):
            row = forwards.get(length) or {}
            lines.append(
                f"| {length} | {_fmt(row.get('empty_median_s'), 6)} | "
                f"{_fmt(row.get('armed_median_s'), 6)} | {_fmt(row.get('extra_s'), 6)} | "
                f"{_fmt(row.get('extra_per_token_s'), 6)} |"
            )
        lines.append("")
    prompts = cost.get("prompt_length") or []
    if prompts:
        lines.append(
            "| family | short tokens | long tokens | added prompt tokens | "
            "continuation tokens | extra s | extra per added prompt token s | "
            "extra per continuation token s |"
        )
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
        for row in prompts:
            lines.append(
                f"| {row.get('family')} | {row.get('short_tokens')} | {row.get('long_tokens')} | "
                f"{row.get('added_prompt_tokens')} | {row.get('continuation_tokens')} | "
                f"{_fmt(row.get('extra_s'), 6)} | {_fmt(row.get('extra_per_added_prompt_token_s'), 6)} | "
                f"{_fmt(row.get('extra_per_continuation_token_s'), 6)} |"
            )
        lines.append("")
    lines.append("## Explicit recall\n")
    lines.append(
        "Phone and birthday exact string match. These rates are not in the "
        "subconscious score. `generate` copies the stored answer when the "
        "prompt is completing that relation and the winning slot stores it, "
        "so a rate of 1.0 on a one-slot store is the spelling shortcut, not "
        "evidence that the residual spelled the string.\n"
    )
    for family in ("phone", "birthday"):
        block = explicit.get(family) or {}
        lines.append(f"{family}: {_rate(block.get('exact'))}.")
    lines.append("")
    lines.append(CITATIONS)
    lines.append(f"\nWall clock { _fmt(result.get('elapsed_s'), 1) } s. Load { _fmt(result.get('load_s'), 1) } s.\n")
    return "\n".join(lines) + "\n"


def _default(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(f"not JSON serializable: {type(obj).__name__}")


def _dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_default) + "\n", encoding="utf-8")
    tmp.replace(path)


def _load(path: Path):
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _preflight(mem) -> None:
    from tidal.llm_memory import _answer_from_offsets, _offset_pairs
    from tidal.subconscious_eval import token_ids
    from tidal.subconscious_measure_data import (
        avoid_items,
        away_items,
        birthday_items,
        choice_items,
        phone_items,
        prefer_items,
        preference_items,
        rif_quartets,
        surprise_highs,
        surprise_lows,
        surprise_seeds,
        tone_items,
        toward_items,
    )

    rows = (
        tone_items()
        + choice_items()
        + preference_items()
        + prefer_items()
        + avoid_items()
        + toward_items()
        + away_items()
    )
    for item in rows:
        prompts = [item["prompt"], item["text"] + "\n" + item["prompt"]]
        for prompt in prompts:
            for cont in (item["good"], item["bad"]):
                prefix = token_ids(mem.tokenizer, prompt)
                full = token_ids(mem.tokenizer, prompt + cont)
                if full[: len(prefix)] != prefix:
                    raise RuntimeError(f"{item['id']} retokenizes {cont!r}")
    facts = list(rows)
    facts.extend(phone_items())
    facts.extend(birthday_items())
    facts.extend(surprise_seeds())
    facts.extend(surprise_lows())
    facts.extend(surprise_highs())
    for group in rif_quartets():
        facts.extend(group)
    for item in facts:
        enc = mem._tokenize(item["text"])
        offsets = _offset_pairs(enc)
        if offsets is None:
            raise RuntimeError("tokenizer did not return offsets")
        mask = _answer_from_offsets(offsets, item["text"], item["answer"])
        if not any(mask):
            raise RuntimeError(f"{item['id']} answer mask is empty")


def main() -> None:
    parser = argparse.ArgumentParser(description="measurement-only subconscious pass")
    parser.add_argument("--json", default="reports/subconscious_measure.json")
    parser.add_argument("--md", default="reports/subconscious_measure.md")
    parser.add_argument("--checkpoint", default="/tmp/subconscious_measure")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmup", type=int, default=2)
    args = parser.parse_args()
    out_json = Path(args.json)
    if out_json.exists():
        raise SystemExit(f"{out_json} already exists; this pass is not rerun")
    torch.set_num_threads(os.cpu_count() or 1)
    ckpt = Path(args.checkpoint)
    ckpt.mkdir(parents=True, exist_ok=True)
    from tidal.llm_memory import LLMMemory
    from tidal.subconscious_measure_data import PROTOCOL, bank_sha256
    from tidal.subconscious_measure_score import (
        measure_compute,
        score_explicit,
        score_implicit,
        score_key_cosines,
        score_prefer_avoid,
        score_rif,
        score_surprise,
        score_valence,
    )

    print("loading SmolLM2-135M float32", flush=True)
    t0 = time.perf_counter()
    mem = LLMMemory(dtype=torch.float32)
    load_s = time.perf_counter() - t0
    print(f"loaded in {load_s:.1f}s layer={mem.layer} hidden={mem.hidden}", flush=True)
    _preflight(mem)
    print("preflight ok", flush=True)
    bge = BGE()
    print("bge loaded", flush=True)

    def progress(msg: str) -> None:
        print(msg, flush=True)

    def section(name, fn):
        final = ckpt / f"{name}.json"
        partial = ckpt / f"{name}.partial.json"
        hit = _load(final)
        if hit is not None and hit.get("complete"):
            print(f"resume {name}", flush=True)
            return hit["result"]
        seed = _load(partial)
        started = time.perf_counter()

        def save(obj):
            _dump(partial, obj)

        result = fn(save, seed)
        _dump(final, {"complete": True, "result": result})
        print(f"done {name} in {time.perf_counter() - started:.1f}s", flush=True)
        return result

    cosine = section(
        "cosine",
        lambda save, seed: score_key_cosines(mem, progress=progress),
    )
    direction = section(
        "direction",
        lambda save, seed: score_prefer_avoid(mem, progress=progress, save=save, seed=seed),
    )
    valence = section(
        "valence",
        lambda save, seed: score_valence(mem, progress=progress, save=save, seed=seed),
    )
    surprise = section(
        "surprise",
        lambda save, seed: score_surprise(mem, progress=progress),
    )
    rif = section(
        "rif",
        lambda save, seed: score_rif(mem, progress=progress, save=save, seed=seed),
    )
    implicit = section(
        "implicit",
        lambda save, seed: score_implicit(mem, bge, progress=progress, save=save, seed=seed),
    )
    explicit = section(
        "explicit",
        lambda save, seed: score_explicit(mem, progress=progress),
    )
    compute = section(
        "compute",
        lambda save, seed: measure_compute(mem, bge, repeats=args.repeats, warmup=args.warmup, progress=progress),
    )
    result = {
        "protocol": PROTOCOL,
        "bank_sha256": bank_sha256(),
        "model": "HuggingFaceTB/SmolLM2-135M",
        "dtype": "float32",
        "git_sha": _git_sha(),
        "load_s": float(load_s),
        "cpu_count": int(os.cpu_count() or 0),
        "torch_threads": int(torch.get_num_threads()),
        "injection": {
            "top1": True,
            "full_vector_every_position": True,
            "quotes": injection_quotes(),
        },
        "key_cosine": cosine,
        "banks": {
            "direction": direction,
            "valence": valence,
            "surprise": surprise,
            "rif": rif,
        },
        "implicit": implicit,
        "explicit": explicit,
        "compute": compute,
        "elapsed_s": float(time.perf_counter() - t0),
    }
    text = render_report(result)
    _dump(out_json, result)
    md_path = Path(args.md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(text, encoding="utf-8")
    print(f"wrote {out_json} and {md_path}", flush=True)


if __name__ == "__main__":
    main()
