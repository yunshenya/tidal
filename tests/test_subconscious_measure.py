"""Measurement banks and the implicit-score definition. No SmolLM load."""
import importlib.util
import statistics
from pathlib import Path

from tidal.question_heldout_data import _tokens
from tidal.subconscious_measure_data import (
    N_ITEMS,
    SEED_N,
    avoid_items,
    away_items,
    bank_sha256,
    birthday_items,
    choice_items,
    cohens_dz,
    phone_items,
    prefer_items,
    preference_items,
    rate_row,
    rif_quartets,
    subconscious_score,
    surprise_highs,
    surprise_lows,
    surprise_seeds,
    tokens,
    tone_items,
    toward_items,
    wilson_interval,
)
from tidal.subconscious_measure_score import cosine_distribution, summarize_implicit

ROOT = Path(__file__).resolve().parent.parent
SURFACE = {"text", "prompt", "good", "bad", "answer", "other", "name"}


def _eval_module():
    spec = importlib.util.spec_from_file_location(
        "eval_subconscious_measure",
        ROOT / "scripts" / "eval_subconscious_measure.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _builder():
    spec = importlib.util.spec_from_file_location(
        "build_subconscious_measure_data",
        ROOT / "scripts" / "build_subconscious_measure_data.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _surfaces():
    import tidal.subconscious_measure_data as data

    out = []

    def walk(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in SURFACE and isinstance(value, str):
                    out.append(value)
                else:
                    walk(value)
        elif isinstance(obj, list):
            for value in obj:
                walk(value)

    walk(data._PAYLOAD)
    return out


def _names(rows):
    return [row["name"] for row in rows]


def test_family_sizes():
    assert N_ITEMS >= 50
    assert len(prefer_items()) >= 50
    assert len(avoid_items()) >= 50
    assert len(toward_items()) >= 50
    assert len(away_items()) >= 50
    assert len(surprise_lows()) >= 50
    assert len(surprise_highs()) >= 50
    assert len(surprise_seeds()) == SEED_N == 32
    assert len(rif_quartets()) >= 50
    assert len(tone_items()) >= 50
    assert len(choice_items()) >= 50
    assert len(preference_items()) >= 50
    assert len(phone_items()) >= 50
    assert len(birthday_items()) >= 50
    assert bank_sha256() == bank_sha256()


def test_names_and_rif_shape_are_unique():
    groups = (
        prefer_items(),
        avoid_items(),
        toward_items(),
        away_items(),
        surprise_lows(),
        surprise_highs(),
        surprise_seeds(),
        tone_items(),
        choice_items(),
        preference_items(),
        phone_items(),
        birthday_items(),
    )
    names = []
    for rows in groups:
        names.extend(_names(rows))
    for group in rif_quartets():
        assert [row["role"] for row in group] == [
            "target",
            "competitor",
            "competitor",
            "competitor",
            "outsider",
        ]
        rels = {row["rel"] for row in group if row["role"] != "outsider"}
        assert len(rels) == 1
        assert group[-1]["rel"] not in rels
        names.extend(_names(group))
    assert len(names) == len(set(names))


def test_implicit_prompts_do_not_mention_the_fact():
    reserved = {"weigh", "unlabeled", "cadence", "tone", "choice", "lean"}
    for row in tone_items() + choice_items() + preference_items():
        prompt = tokens(row["prompt"])
        fact = tokens(row["text"]) | tokens(row["name"]) | tokens(row["answer"]) | tokens(row.get("other", ""))
        assert prompt.isdisjoint(fact)
        assert reserved.isdisjoint(fact)
        assert "phone" not in prompt
        assert "birthday" not in prompt


def test_explicit_prompts_name_the_relation_and_avoid_415():
    for row in phone_items():
        assert "phone" in tokens(row["prompt"])
        assert "415" not in row["answer"]
        assert row["answer"].startswith("+1-")
    for row in birthday_items():
        assert "birthday" in tokens(row["prompt"])
        year = int(row["answer"].rsplit(", ", 1)[1])
        assert year < 1970 or year > 2019


def test_token_regex_matches_the_existing_helper():
    samples = ["Don't shout.", "中文 phone", "April 4, 1996", "mqtag10"]
    for text in samples:
        assert tokens(text) == _tokens(text)


def test_wilson_bounds_and_cohens_dz():
    low, high = wilson_interval(0, 50)
    assert low == 0.0
    low, high = wilson_interval(50, 50)
    assert high == 1.0
    low, high = wilson_interval(25, 50)
    assert low < 0.5 < high
    row = rate_row(25, 50)
    assert row["wilson95"][0] < row["rate"] < row["wilson95"][1]
    diffs = [1.0, 2.0, 3.0]
    assert cohens_dz(diffs) == statistics.mean(diffs) / statistics.stdev(diffs)
    assert cohens_dz([1.0, 1.0]) is None
    assert cohens_dz([1.0]) is None


def test_subconscious_score_excludes_explicit_recall():
    assert subconscious_score({"tone": 1.0, "choice": 3.0, "preference": 5.0, "phone": 100.0, "birthday": -100.0}) == 3.0
    assert subconscious_score({"tone": 1.0, "choice": None, "preference": 5.0}) is None
    rows = []
    for family, base in (("tone", 1.0), ("choice", 2.0), ("preference", 3.0)):
        for offset in (0.0, 1.0):
            rows.append(
                {
                    "family": family,
                    "nomem_margin": 0.0,
                    "inject_margin": base + offset,
                    "context_margin": 0.0,
                    "rag_margin": 0.0,
                    "inject_maxcos": 0.2,
                    "inject_confidence": 1.0,
                    "rag_hit": 0,
                    "rag_retrieved": "other",
                }
            )
    summary = summarize_implicit(rows)
    assert "phone" not in summary["score_dz"]
    assert summary["subconscious_score"] == subconscious_score(summary["score_dz"])
    assert set(summary["score_dz"]) == {"tone", "choice", "preference"}


def test_key_rows_include_the_locked_banks():
    from tidal.subconscious_measure_score import _key_rows

    rows = _key_rows()
    sources = {source for source, _item in rows}
    assert sources == {
        "measure-prefer",
        "measure-avoid",
        "replay-held-prefer",
        "replay-held-avoid",
        "valence-held-toward",
        "valence-held-away",
    }
    assert sum(source.startswith("measure-") for source, _item in rows) == 100


def test_cosine_distribution_reports_the_threshold():
    above = cosine_distribution([0.991, 0.995, 1.0])
    assert above["all_gt_0.99"] is True
    assert above["n_pairs"] == 3
    mixed = cosine_distribution([0.2, 0.995])
    assert mixed["all_gt_0.99"] is False
    assert mixed["frac_gt_0.99"] == 0.5
    assert sum(mixed["hist_counts"]) == mixed["n_pairs"]


def test_quotes_and_citations_match_the_shipped_path():
    module = _eval_module()
    quotes = module.injection_quotes()
    assert 'mode="top1"' in quotes["rank"]
    assert "return s0.v.copy(), info" in quotes["mix_visible"]
    assert "return add.view(*shape)" in quotes["broadcast"]
    assert 'mode: str = "softtopk"' in quotes["read"]
    text = module.render_report({})
    for doi in (
        "10.1037/h0031564",
        "10.1126/science.2296719",
        "10.1037/h0025848",
        "10.1037/0022-3514.50.2.229",
    ):
        assert doi in text
    assert "excluded" in text.lower() or "not in the subconscious score" in text
    source = (ROOT / "tidal" / "subconscious_measure_score.py").read_text(encoding="utf-8")
    for forbidden in ("RIF_SPAN =", "SURPRISE_QUANTILE =", "VALENCE_SPAN =", "POLARITY_SPAN ="):
        assert forbidden not in source


def test_surfaces_are_disjoint_from_existing_banks():
    builder = _builder()
    old_texts = [text for text in builder.existing_strings() if str(text).strip()]
    old = [tokens(text) for text in old_texts]
    worst = 0.0
    pair = ("", "")
    for text in _surfaces():
        left = tokens(text)
        if not left:
            continue
        for right, old_text in zip(old, old_texts):
            if not right:
                continue
            score = len(left & right) / len(left | right)
            if score > worst:
                worst = score
                pair = (text, old_text)
    assert worst < 0.5, pair
