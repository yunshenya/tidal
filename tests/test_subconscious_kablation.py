"""K-ablation bank, top-p rule, and the pre-score preregistration."""
import ast
import inspect
from collections import defaultdict
from pathlib import Path

import numpy as np

from tidal.episodic import EpisodicStore, token_jaccard
from tidal.llm_memory import MIX_K, MIX_TEMP
from tidal.subconscious_kablation_data import (
    COMPANION_NS,
    FAMILIES,
    FIT_N,
    HELD_N,
    STORE_SIZES,
    TOP_P,
    all_probes,
    bank_sha256,
    companion_facts,
    fit_probes,
    held_probes,
    store_facts,
)
from tidal.subconscious_kablation_score import (
    CELLS,
    Selector,
    cell_key,
    cumulative_weights,
    held_allowed,
    pearson,
    production_weights_unchanged,
    summarize_split,
)

ROOT = Path(__file__).resolve().parent.parent
PREREG = ROOT / "reports" / "subconscious_kablation_prereg.md"
TOKEN_RE = __import__("re").compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+(?:'[A-Za-z]+)?")


def _tokens(text: str) -> frozenset:
    out = set()
    for match in TOKEN_RE.finditer(str(text)):
        seg = match.group(0)
        if "\u4e00" <= seg[0] <= "\u9fff":
            out.add(seg)
        else:
            out.add(seg.lower())
    return frozenset(out)


def test_bank_splits_and_store_shapes():
    assert len(fit_probes()) == FIT_N * len(FAMILIES)
    assert len(held_probes()) == HELD_N * len(FAMILIES)
    assert {probe["split"] for probe in fit_probes()} == {"fit"}
    assert {probe["split"] for probe in held_probes()} == {"held"}
    fit_ids = {probe["id"] for probe in fit_probes()}
    held_ids = {probe["id"] for probe in held_probes()}
    assert fit_ids.isdisjoint(held_ids)
    assert len(bank_sha256()) == 64
    assert TOP_P == 0.9
    assert [cell_key(cell) for cell in CELLS] == [
        "fixed:4:0.05",
        "fixed:4:0.10",
        "fixed:4:0.20",
        "top_p:16:0.05",
        "top_p:16:0.10",
        "top_p:16:0.20",
        "top_p:32:0.05",
        "top_p:32:0.10",
        "top_p:32:0.20",
    ]
    ones = np.ones(4, dtype=np.float64)
    for probe in all_probes():
        assert probe["target"]["answer"] == probe["answer"]
        for fact in probe["similar"] + probe["fillers"]:
            assert fact["answer"] != probe["answer"]
            assert fact["rel"] == probe["rel"]
            assert token_jaccard(probe["target"]["text"], fact["text"]) < 0.85
        for size in STORE_SIZES:
            facts = store_facts(probe, size)
            assert len(facts) == size
            assert facts[0]["role"] == "target"
            store = EpisodicStore(capacity=64, dk=4, dv=4)
            for fact in facts:
                store.write(
                    ones,
                    ones,
                    name=fact["name"],
                    rel=fact["rel"],
                    group=probe["group"],
                    answer=fact["answer"],
                    text=fact["text"],
                )
            assert len(store) == size
        copy = EpisodicStore(capacity=8, dk=4, dv=4)
        target = probe["target"]
        copy.write(ones, ones, name=target["name"], rel=target["rel"], group=probe["group"], answer=target["answer"], text=target["text"])
        copy.write(ones, ones, name="Zkother", rel=target["rel"], group=probe["group"], answer=target["answer"], text=target["text"])
        assert len(copy) == 1
        high = companion_facts(probe, "high", 7)
        low = companion_facts(probe, "low", 7)
        assert len(high) == 8 and len(low) == 8
        assert all(fact["role"] == "similar" for fact in high[1:])
        assert all(fact["role"] == "filler" for fact in low[1:])
    assert COMPANION_NS == (1, 3, 7)


def test_surfaces_stay_disjoint_from_the_other_banks():
    import importlib

    fresh = []
    for probe in all_probes():
        fresh.append(probe["prompt"])
        fresh.append(probe["rel"])
        fresh.append(probe["group"])
        for fact in store_facts(probe, 32):
            fresh.append(fact["text"])
            fresh.append(fact["name"])
    old_sets = []
    for path in sorted((ROOT / "tidal").glob("*data.py")):
        if path.stem == "subconscious_kablation_data":
            continue
        mod = importlib.import_module(f"tidal.{path.stem}")
        found = []

        def walk(obj, _found=found):
            if isinstance(obj, str):
                _found.append(obj)
            elif isinstance(obj, dict):
                for value in obj.values():
                    walk(value)
            elif isinstance(obj, (list, tuple)):
                for value in obj:
                    walk(value)

        for key, value in vars(mod).items():
            if key.startswith("__"):
                continue
            walk(value)
        for text in found:
            tokens = _tokens(text)
            if tokens:
                old_sets.append(tokens)
    index = defaultdict(list)
    for tokens in old_sets:
        for token in tokens:
            index[token].append(tokens)
    worst = 0.0
    pair = None
    seen = set()
    for text in fresh:
        left = _tokens(text)
        if not left or id(text) in seen:
            continue
        seen.add(id(text))
        candidates = []
        marked = set()
        for token in left:
            for right in index.get(token, ()):
                mark = id(right)
                if mark in marked:
                    continue
                marked.add(mark)
                candidates.append(right)
        for right in candidates:
            score = len(left & right) / len(left | right)
            if score > worst:
                worst = score
                pair = (text[:80], " ".join(sorted(right))[:80])
    assert worst < 0.50, (worst, pair)


def test_top_p_keeps_the_shortest_prefix_and_renormalizes():
    uniform = np.zeros(10, dtype=np.float64)
    weights = cumulative_weights(uniform, np.ones(10), temperature=0.2, k_max=32, p=0.9)
    assert int(np.count_nonzero(weights)) == 9
    assert abs(float(weights.sum()) - 1.0) < 1e-12
    assert abs(float(weights[0]) - 1.0 / 9.0) < 1e-12
    assert weights[9] == 0.0
    capped = cumulative_weights(uniform, np.ones(10), temperature=0.2, k_max=4, p=0.9)
    assert int(np.count_nonzero(capped)) == 4
    assert abs(float(capped[:4].sum()) - 1.0) < 1e-12
    peaked = cumulative_weights(np.array([1.0, 0.0, 0.0, 0.0]), np.ones(4), temperature=0.05, k_max=32, p=0.9)
    assert peaked[0] == 1.0
    assert int(np.count_nonzero(peaked)) == 1
    masked = cumulative_weights(np.array([0.0, 5.0]), np.array([0.0, 1.0]), temperature=0.1, k_max=16, p=0.9)
    assert masked[0] == 0.0 and masked[1] == 1.0
    tied = cumulative_weights(np.array([1.0, 1.0]), np.ones(2), temperature=0.05, k_max=1, p=0.9)
    assert tied.tolist() == [1.0, 0.0]


def test_selector_patches_top_p_only_and_restores_the_default():
    import tidal.llm_memory as lm

    cos = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    strength = np.ones(5)
    before = lm.activation_weights
    assert MIX_K == 4 and MIX_TEMP == 0.05
    assert lm.MIX_K == 4 and lm.MIX_TEMP == 0.05
    fixed = {"mode": "fixed", "k": 4, "k_max": 4, "temperature": 0.2}
    with Selector(fixed):
        assert lm.activation_weights is before
        assert lm.MIX_K == 4
        assert abs(lm.MIX_TEMP - 0.2) < 1e-12
    assert lm.activation_weights is before
    assert lm.MIX_TEMP == 0.05
    top = {"mode": "top_p", "k": None, "k_max": 32, "temperature": 0.05}
    with Selector(top):
        assert lm.activation_weights is not before
        assert lm.MIX_K == 4
        weights = lm.activation_weights(cos, strength, temperature=lm.MIX_TEMP, k=lm.MIX_K)
    assert weights[0] == 1.0
    assert int(np.count_nonzero(weights)) == 1
    assert lm.activation_weights is before
    assert lm.MIX_TEMP == 0.05
    assert production_weights_unchanged(cos, strength)


def test_hypothesis_comparison_rejects_a_higher_score_at_lower_alignment():
    records = []
    families = ("tone", "choice", "preference")
    for family_index, family in enumerate(families):
        for replicate in (0, 1):
            implicit = []
            interference = []
            for size in STORE_SIZES:
                for cell in CELLS:
                    key = cell_key(cell)
                    if cell["mode"] == "fixed":
                        margin = 0.1 + 0.05 * replicate
                        alignment = 0.9
                    else:
                        margin = 0.4 + 0.1 * replicate
                        alignment = 0.2
                    implicit.append(
                        {
                            "size": size,
                            "cell": key,
                            "margin": margin,
                            "participation": 2.0,
                            "active": 4,
                            "mlp_runs": 4,
                            "mlp_runs_per_neighbor": 1.0,
                            "mlp_runs_per_broadcast_token": 0.1,
                            "alignment": alignment,
                            "hit": 1,
                        }
                    )
            for cell in CELLS:
                key = cell_key(cell)
                for band in ("high", "low"):
                    for n_companions in COMPANION_NS:
                        high_margin = {1: 1.0, 3: 0.5, 7: 0.0}[n_companions]
                        low_margin = {1: 2.0, 3: 1.5, 7: 1.0}[n_companions]
                        margin = high_margin if band == "high" else low_margin
                        interference.append(
                            {
                                "cell": key,
                                "band": band,
                                "n_companions": n_companions,
                                "margin": margin,
                                "mean_key_cosine": 0.95 if band == "high" else 0.4,
                                "participation": 1.5,
                                "active": 2,
                                "mlp_runs": 2,
                            }
                        )
            records.append(
                {
                    "id": f"fake-{family}-{replicate}",
                    "family": family,
                    "nomem_margin": 0.0,
                    "target_only_margin": 0.0,
                    "target_only_mlp_runs": 1,
                    "nomem_mlp_runs": 0,
                    "implicit": implicit,
                    "interference": interference,
                }
            )
        del family_index
    summary = summarize_split(records)
    assert len(summary["implicit_rows"]) == 27
    assert all(row["implication_holds"] is False for row in summary["comparisons"])
    assert summary["target_only_mlp_runs"] == 1
    assert summary["nomem_mlp_runs"] == 0
    similarity = [row for row in summary["contrasts"] if row["name"] == "similarity_n1"]
    assert similarity and all(row["k"] == row["n"] == 6 for row in similarity)
    high_count = [row for row in summary["contrasts"] if row["name"] == "count_high_7_below_1"]
    assert high_count and all(row["k"] == 6 for row in high_count)
    corr, n = pearson([0.2, 0.9, 0.3], [-1.0, 1.0, -0.5])
    assert n == 3 and corr is not None and corr > 0.0
    assert pearson([1.0, 1.0, 1.0], [0.0, 1.0, 2.0])[0] is None


def test_fit_scorer_does_not_read_held_probes_or_score_geometry_margins():
    from tidal import subconscious_kablation_score as score

    fit_source = inspect.getsource(score.score_fit)
    assert "held_probes" not in fit_source
    geometry_source = inspect.getsource(score.collect_geometry)
    assert "continuation_logprobs" not in geometry_source
    assert "score_margin" not in geometry_source
    tree = ast.parse((ROOT / "scripts" / "eval_subconscious_kablation.py").read_text(encoding="utf-8"))
    fit_fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_fit")
    calls = [node.func.id for node in ast.walk(fit_fn) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
    assert "_held" not in calls


def test_prereg_locks_the_grid_and_waits_for_scores():
    text = PREREG.read_text(encoding="utf-8")
    assert "Geometry has not been recorded." not in text
    assert "Fit has not been scored." not in text
    assert "Held has not been scored." in text
    assert "c00bfc8621bd6ada2a3a35f1b5d74645086c99fc" in text
    assert "0.15461428702409788" in text
    assert "0.07515952280613003" in text
    assert "72/72" in text
    assert "no direct paper" in text
    assert bank_sha256() in text
    assert "0.9" in text
    for cell in CELLS:
        assert cell_key(cell) in text
