"""A dismissed clause drops a unique cue. The cue list does not grow.

The class is ignore / skip / forget / exclude, aside, a decoy noun, and the
Chinese marks. It ends at the next clause break when another clause follows.
The probe, the surface corpus, and every non-misleading bank keep their
cues. The frozen adversarial scores are not a gate.
"""
import ast
from pathlib import Path

import numpy as np

from tidal.question_abstain_dev_data import BANK as DEV_BANK
from tidal.question_adversarial_data import BANK as TEST_BANK
from tidal.question_cjk_dev_data import NAMES as CJK_NAMES
from tidal.question_cjk_dev_data import questions as cjk_questions
from tidal.question_cue_dev_data import NAMES, build_rows, kind_counts, questions_sha256
from tidal.question_heldout_data import ABSENT, PEOPLE, _tokens, build_cases, probe_strings
from tidal.question_indirect_dev_data import ABSENT as INDIRECT_ABSENT
from tidal.question_indirect_dev_data import OTHER, SELF, THIRD, bank_strings, questions as indirect_questions
from tidal.question_nickname_dev_data import build_rows as nick_rows
from tidal.question_nickname_dev_data import questions as nick_questions
from tidal.question_stress import build_cases as build_stress
from tidal.question_stress import case_strings
from tidal.question_surface import (
    ABSTAIN_MARGIN,
    DEV_NAMES,
    FIT_NAMES,
    ABSENT_NAMES,
    LABELS,
    LEX_RIDGE_GAP,
    STORED_DIM,
    corpus_strings,
    cue_set_aside,
    lex_labels,
    surface_abstain,
)

# sha256 of the question list only. A wording change moves this.
CUE_DEV_SHA = "ed3bd6aadfa92ed4d9843260ef1788875c83019fcb0046491d3928394c08628f"


def _typo_questions():
    src = Path("tests/test_question_indirect_dev.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if getattr(target, "id", "") == "_TYPO_DEV_QUESTIONS":
                return tuple(ast.literal_eval(node.value))
    raise RuntimeError("typo dev questions missing")


def test_cue_dev_sha_and_labels():
    rows = build_rows()
    assert questions_sha256(rows) == CUE_DEV_SHA
    assert len(rows) == 44
    assert kind_counts(rows) == {
        "set_aside": 20,
        "scoped_after": 12,
        "genuine": 6,
        "polysemy": 6,
    }
    for row in rows:
        assert row.lang in ("en", "zh")
        assert row.cue in LABELS
        assert lex_labels(row.question) == (row.cue,)
        assert cue_set_aside(row.question, row.cue) is (row.kind == "set_aside")


def _jaccard(a: str, b: str) -> float:
    left, right = _tokens(a), _tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def test_cue_dev_is_disjoint():
    rows = build_rows()
    qs = [row.question for row in rows]
    bags = {
        "probe": probe_strings(),
        "surface": corpus_strings(),
        "abstain_dev": case_strings(build_stress(DEV_BANK)),
        "test": case_strings(build_stress(TEST_BANK)),
        "cjk": cjk_questions(),
        "nick": nick_questions(),
        "indirect": bank_strings(),
        "indirect_q": indirect_questions(),
        "typo": _typo_questions(),
    }
    for name, texts in bags.items():
        bag = set(texts)
        assert not any(question in bag for question in qs), name
        best = max(_jaccard(a, b) for a in qs for b in texts)
        assert best < 0.50, (name, best)
    banned = set(CJK_NAMES)
    for lang in ("en", "zh"):
        for bank in (PEOPLE, ABSENT, FIT_NAMES, DEV_NAMES, ABSENT_NAMES):
            banned.update(bank[lang])
        banned.update(DEV_BANK["stored"][lang])
        banned.update(DEV_BANK["absent"][lang])
        banned.update(TEST_BANK["stored"][lang])
        banned.update(TEST_BANK["absent"][lang])
    for row in nick_rows():
        banned.update(row.roster)
    for group in (SELF, OTHER, THIRD, INDIRECT_ABSENT):
        banned.update(group.values())
    for name in NAMES["en"] + NAMES["zh"]:
        assert name not in banned


def _unique_set_aside(texts):
    found = []
    for text in texts:
        lex = lex_labels(text)
        if len(lex) == 1 and cue_set_aside(text, lex[0]):
            found.append(text)
    return found


def test_live_cues_outside_the_misleading_slice_stay():
    """A cue after the break, or with no dismissal, is still the ask."""
    assert _unique_set_aside(case.question for case in build_cases()) == []
    assert _unique_set_aside(corpus_strings()) == []
    dev = build_stress(DEV_BANK)
    test = build_stress(TEST_BANK)
    assert _unique_set_aside(c.question for c in dev if c.slice != "misleading_cue") == []
    assert _unique_set_aside(c.question for c in test if c.slice != "misleading_cue") == []
    assert _unique_set_aside(cjk_questions()) == []
    assert _unique_set_aside(nick_questions()) == []
    assert _unique_set_aside(indirect_questions()) == []
    assert _unique_set_aside(bank_strings()) == []
    assert _unique_set_aside(_typo_questions()) == []

    kept = (
        ("Ignore the side topics and read Owen's birthday.", "birthday"),
        ("Ignore the feast and keep the birthday Ingrid claims.", "birthday"),
        ("Skip the party and say which day Ingrid was born.", "birthday"),
        ("Leave William aside. The birthday to read belongs with Mary Jane.", "birthday"),
        ("Setting the other details aside, what is Owen's phone number?", "phone"),
        ("Skipping the filler, what's Owen's telephone?", "phone"),
        ("The extra names are a distraction: what is Owen's phone?", "phone"),
        ("Those other records aren't the one I want. What places did Ravi store?", "places"),
        ("Read the places Arthur set down, earliest stop first.", "places"),
        ("别管旁边的话题，把陈可的电话号码读出来。", "phone"),
        ("跳过那些填充，陈可的电话是什么？", "phone"),
        ("先放下李娜。要回读的电话是王伟的。", "phone"),
        ("先别看胡斌。值班要念的电话是孙悦的。", "phone"),
        ("别看成孙悦，问的是钟灵，电话记过吗？", "phone"),
        ("Set Harold aside. The telephone the porter should read is Imogen's.", "phone"),
        ("Skip Harold. For Quentin, which telephone was written?", "phone"),
    )
    for text, rel in kept:
        assert lex_labels(text) == (rel,), text
        assert cue_set_aside(text, rel) is False, text
    aside = "Leave William aside. The birthday to read belongs with Mary Jane."
    assert lex_labels(aside) == ("birthday",)


def _weights(column, scale):
    wide = np.zeros((STORED_DIM + 1, 3), np.float64)
    wide[0, column] = scale
    return {
        "Wa": wide,
        "abstain_margin": 0.1,
        "W": wide,
        "alpha": 0.0,
        "labels": LABELS,
    }


def _ask(text, rels, names, weights):
    query = np.zeros(STORED_DIM, np.float64)
    query[0] = 1.0
    keys = np.zeros((len(names), STORED_DIM), np.float64)
    return surface_abstain(query, keys, rels, names, weights, query_text=text)


def test_a_dismissed_cue_uses_the_next_ridge_label_and_the_margin():
    text = "Forget Eudora's candle tally and say how a voice lands after dusk."
    assert lex_labels(text) == ("birthday",)
    assert cue_set_aside(text, "birthday") is True
    # Birthday is the top. Phone leads places by 0.5, so the margin abstains
    # when phone is not stored and answers when it is.
    strong = np.zeros((STORED_DIM + 1, 3), np.float64)
    strong[0, 1] = 1.0
    strong[0, 0] = 0.5
    weights = _weights(0, 0.0)
    weights["Wa"] = strong
    assert _ask(text, ["birthday"], ["Eudora"], weights) == "missing_rel"
    assert _ask(text, ["phone"], ["Eudora"], weights) is None
    # The replacement lead is under the margin. Do not abstain a stored places
    # fact just because the dismissed cue was confident.
    weak = np.zeros((STORED_DIM + 1, 3), np.float64)
    weak[0, 1] = 1.0
    weak[0, 0] = 0.05
    weights["Wa"] = weak
    assert _ask(text, ["places"], ["Eudora"], weights) is None

    zh = "别管闵舟的蜡烛页，夜里怎么找人落脚。"
    assert lex_labels(zh) == ("birthday",)
    assert cue_set_aside(zh, "birthday") is True
    # The ridge already prefers places. A gap under the margin does not abstain.
    places = _weights(2, 0.05)
    assert _ask(zh, ["places"], ["闵舟"], places) is None
    places = _weights(2, 0.2)
    assert _ask(zh, ["birthday", "phone"], ["闵舟", "闵舟"], places) == "missing_rel"


def test_ravi_places_cue_still_beats_a_narrow_phone_ridge():
    assert LEX_RIDGE_GAP == 0.15
    assert ABSTAIN_MARGIN == 0.1
    text = "Those other records aren't the one I want. What places did Ravi store?"
    assert lex_labels(text) == ("places",)
    assert cue_set_aside(text, "places") is False
    # Phone leads by 0.12, under the yield gap. The places cue still wins.
    weights = _weights(0, 0.12)
    assert _ask(text, ["places"], ["Ravi"], weights) is None
    assert _ask(text, ["phone"], ["Ravi"], weights) == "missing_rel"


def test_a_second_live_cue_is_not_cleared():
    text = "Forget Eudora's birthday and read the telephone."
    assert lex_labels(text) == ("phone", "birthday")
    assert cue_set_aside(text, "birthday") is True
    assert cue_set_aside(text, "phone") is False
    # Two cues. The ridge decides. Birthday on top does not leave the phone cue
    # in charge, and it does not abstain when birthday is stored.
    weights = _weights(1, 1.0)
    assert _ask(text, ["phone"], ["Eudora"], weights) == "missing_rel"
    assert _ask(text, ["birthday"], ["Eudora"], weights) is None


def test_a_dismissed_cue_without_a_ridge_does_not_abstain():
    text = "Forget Eudora's candle tally and say how a voice lands after dusk."
    weights = {"abstain_margin": 0.1, "labels": LABELS}
    assert _ask(text, ["phone"], ["Eudora"], weights) is None


def test_surface_does_not_import_the_cue_dev_bank():
    blob = Path("tidal/question_surface.py").read_text(encoding="utf-8")
    assert "question_cue_dev" not in blob
    assert "enable_abstain" not in blob
    assert "use_abstain" not in blob
