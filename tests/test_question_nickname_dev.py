"""Suppletive diminutives. The split dev bank chooses the alias list.

The frozen adversarial test is scored after the list is fixed. It is not a
row in the bank, and it is not used to drop or add a form.
"""
import numpy as np

from tidal.question_abstain_dev_data import BANK as DEV_BANK
from tidal.question_adversarial_data import BANK as TEST_BANK
from tidal.question_cjk_dev_data import NAMES as CJK_NAMES
from tidal.question_cjk_dev_data import build_rows as cjk_rows
from tidal.question_heldout_data import ABSENT, PEOPLE, _tokens, probe_strings
from tidal.question_name_align import (
    ALIAS_GROUPS,
    _score_questions,
    best_name,
    load_aligner,
    propose_spans,
    rank_span,
)
from tidal.question_nickname_dev_data import (
    build_rows,
    kind_counts,
    questions_sha256,
    rejections,
    selected_groups,
)
from tidal.question_stress import build_cases, case_strings
from tidal.question_surface import (
    ABSENT_NAMES,
    DEV_NAMES,
    FIT_NAMES,
    STORED_DIM,
    corpus_strings,
    load_corpus_table,
    load_weights,
    name_in_text,
    score_abstain_margin,
    surface_abstain,
)

# sha256 of the question list only. A wording change moves this.
SPLIT_SHA = "a766f748cfc2bcc28f8c9b468c9822003798b3de377df15daa8a0633eb1aec98"


def test_the_shipped_list_is_the_dev_filter():
    assert questions_sha256() == SPLIT_SHA
    assert len(build_rows()) == 43
    assert kind_counts() == {
        "positive": 20,
        "reverse": 1,
        "honorific": 2,
        "possessive": 2,
        "prefix_kept": 3,
        "negative": 6,
        "ambiguous": 1,
        "compound": 3,
        "zh_keep": 3,
        "zh_drop": 2,
    }
    assert selected_groups() == ALIAS_GROUPS
    assert rejections() == (
        ("greta", "margaret", "known person"),
        ("ed", "edward", "shorter than 3"),
        ("anne", "ann", "edit-1 of ann"),
        ("al", "albert", "shorter than 3"),
        ("nellie", "helen", "known person"),
    )
    forms = [form for _cans, dims in ALIAS_GROUPS for form in dims]
    assert forms and all(len(form) >= 3 for form in forms)
    assert len(ALIAS_GROUPS) <= 24
    # The head is unchanged. The list is not a new feature.
    from tidal.question_name_align import FEATURES

    assert "alias" not in FEATURES
    from pathlib import Path

    blob = Path("tidal/question_name_align.py").read_text(encoding="utf-8")
    assert "question_nickname_dev" not in blob
    assert "enable_alias" not in blob


def test_every_dev_row_matches_its_label():
    for row in build_rows():
        from tidal.question_name_align import matched_people

        got = tuple(matched_people(row.question, row.roster))
        assert got == row.expect, (row.kind, row.question, got, row.expect)


def test_split_is_disjoint_from_the_other_banks():
    questions = [row.question for row in build_rows()]
    bags = {
        "probe": probe_strings(),
        "surface": corpus_strings(),
        "dev": case_strings(build_cases(DEV_BANK)),
        "test": case_strings(build_cases(TEST_BANK)),
        "cjk": [row[-1] for row in cjk_rows()],
    }
    assert not any(question in set(texts) for texts in bags.values() for question in questions)

    def jaccard(a, b):
        left, right = _tokens(a), _tokens(b)
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    for name, texts in bags.items():
        best = max(jaccard(a, b) for a in questions for b in texts)
        assert best <= 0.50, (name, best)
    banned = []
    for lang in ("en", "zh"):
        banned.extend(PEOPLE[lang])
        banned.extend(ABSENT[lang])
        banned.extend(DEV_BANK["stored"][lang])
        banned.extend(DEV_BANK["absent"][lang])
        banned.extend(TEST_BANK["stored"][lang])
        banned.extend(TEST_BANK["absent"][lang])
        banned.extend(FIT_NAMES[lang])
        banned.extend(DEV_NAMES[lang])
        banned.extend(ABSENT_NAMES[lang])
    banned.extend(CJK_NAMES)
    roster = {name for row in build_rows() for name in row.roster}
    assert not roster & set(banned)
    for question in questions:
        for name in banned:
            assert not name_in_text(name, question), (name, question)


def test_rejected_forms_do_not_match_and_a_compound_does_not_inherit():
    assert best_name("Greta", ["Margaret"]) is None
    assert best_name("Anne", ["Ann"]) is None
    assert best_name("Ed", ["Edward"]) is None
    assert best_name("Al", ["Albert"]) is None
    assert best_name("Nellie", ["Helen"]) is None
    assert best_name("ted", ["Edward"]) is None
    assert best_name("Dolly", ["Dorothy Quinn"]) is None
    assert best_name("Dot", ["Dorothy Quinn"]) is None
    assert best_name("Nell", ["Helen Grace"]) is None
    # Jane stays the head's substring hit. It is not an alias of Mary.
    assert best_name("Jane", ["Mary Jane"]) == "Mary Jane"
    assert best_name("Ms. Jane", ["Mary Jane", "Margaret"]) == "Mary Jane"
    assert best_name("Mary", ["Mary Jane", "Margaret"]) is None
    # Maggie shares a two-letter prefix with Mary Jane. The list picks Margaret.
    assert best_name("Maggie", ["Margaret", "Mary Jane"]) == "Margaret"
    # A diminutive of two full names matches only when one of them is stored.
    assert best_name("Ted", ["Edward"]) == "Edward"
    assert best_name("Ted", ["Theodore"]) == "Theodore"
    assert best_name("Ted", ["Edward", "Theodore"]) is None
    assert best_name("Teddy", ["Ted"]) is None
    assert best_name("Edward", ["Ted"]) == "Ted"


def test_shipped_pairs_resolve_both_ways():
    for cans, dims in ALIAS_GROUPS:
        for can in cans:
            stored = can.title() if can != "katherine" else "Katherine"
            if can == "catherine":
                stored = "Catherine"
            for dim in dims:
                surface = dim.title()
                assert best_name(surface, [stored]) == stored, (surface, stored)
                assert best_name(stored, [surface]) == surface, (stored, surface)
                assert best_name(f"Mr. {surface}" if surface != "Ginny" else f"Ms. {surface}", [stored]) == stored
    # Kate is a prefix of Katherine, so the head wins when both spellings are stored.
    assert best_name("Kate", ["Catherine"]) == "Catherine"
    assert best_name("Kate", ["Katherine"]) == "Katherine"
    assert best_name("Kate", ["Catherine", "Katherine"]) == "Katherine"
    assert best_name("Bob's", ["Robert"]) == "Robert"
    assert best_name("Ack", ["John"]) is None


def _alias_additions(texts, roster):
    weights = load_aligner()
    found = []
    for text in texts:
        for _start, _end, span in propose_spans(text):
            got = best_name(span, roster, weights)
            if got is None:
                continue
            ranked = rank_span(span, roster, weights)
            accepts = False
            if ranked and ranked[0][1] >= weights["threshold"]:
                if not (len(ranked) > 1 and ranked[0][2] - ranked[1][2] < weights["margin"]):
                    accepts = True
            if not accepts:
                found.append((span, got, text))
    return found


def test_the_list_does_not_fire_outside_the_suppletive_gap():
    probe = list(PEOPLE["en"]) + list(PEOPLE["zh"]) + list(ABSENT["en"]) + list(ABSENT["zh"])
    surface = [name for lang in ("en", "zh") for name in (*FIT_NAMES[lang], *DEV_NAMES[lang], *ABSENT_NAMES[lang])]
    dev = [name for lang in ("en", "zh") for name in (*DEV_BANK["stored"][lang], *DEV_BANK["absent"][lang])]
    test = [name for lang in ("en", "zh") for name in (*TEST_BANK["stored"][lang], *TEST_BANK["absent"][lang])]
    assert _alias_additions(probe_strings(), probe) == []
    assert _alias_additions(corpus_strings(), surface) == []
    assert _alias_additions(case_strings(build_cases(DEV_BANK)), dev) == []
    assert _alias_additions([row[-1] for row in cjk_rows()], list(CJK_NAMES)) == []
    added = _alias_additions(case_strings(build_cases(TEST_BANK)), test)
    assert sorted((span, got) for span, got, _text in added) == [
        ("Bill", "William"),
        ("Bob", "Robert"),
        ("Maggie", "Margaret"),
        ("Mr. Bill", "William"),
        ("Mr. Bob", "Robert"),
    ]


def _reason(question, writes):
    weights = dict(load_weights())
    weights["Wa"] = np.zeros_like(weights["Wa"])
    names = [row.name for row in writes]
    rels = [row.rel for row in writes]
    keys = np.zeros((len(names), STORED_DIM), dtype=np.float64)
    query = np.zeros(STORED_DIM, dtype=np.float64)
    query[0] = 1.0
    return surface_abstain(query, keys, rels, names, weights, query_text=question)


def test_frozen_english_nicknames_match_and_the_kept_slices_stay():
    """Generalization. These strings are not in the split bank."""
    test = build_cases(TEST_BANK)
    en_nick = [
        case for case in test
        if case.lang == "en" and case.slice == "name_variant" and case.subtype == "nickname"
    ]
    assert len(en_nick) == 6
    assert all(_reason(case.question, case.writes) is None for case in en_nick)
    short = [
        case for case in test
        if case.lang == "en" and case.slice == "name_variant" and case.subtype == "honorific_short"
    ]
    assert len(short) == 3
    assert all(_reason(case.question, case.writes) is None for case in short)
    typos = [case for case in test if case.subtype == "typo"]
    assert len(typos) == 16
    assert all(_reason(case.question, case.writes) == "absent_person" for case in typos)
    en_near = [case for case in test if case.lang == "en" and case.slice == "near_dup_absent"]
    assert len(en_near) == 16
    assert all(_reason(case.question, case.writes) == "absent_person" for case in en_near)
    two = [case for case in test if case.slice == "two_names" and case.label == "absent_person"]
    assert len(two) == 12
    assert all(_reason(case.question, case.writes) == "absent_person" for case in two)
    zh_keep = [
        case for case in test
        if case.lang == "zh" and case.slice == "name_variant"
        and case.subtype in ("nickname", "no_surname", "honorific_short")
    ]
    assert len(zh_keep) == 18
    assert all(_reason(case.question, case.writes) is None for case in zh_keep)
    longer = [
        case for case in test
        if case.subtype == "cjk_superstring" and case.focus in ("王伟强", "李娜儿", "张强生", "刘洋子")
    ]
    prefixes = [
        case for case in test
        if case.subtype == "cjk_superstring" and case.focus in ("老王伟", "小李娜", "大张强", "老赵敏")
    ]
    assert len(longer) == 4 and len(prefixes) == 4
    assert all(_reason(case.question, case.writes) == "absent_person" for case in longer + prefixes)


def test_old_dev_name_score_and_surface_curve_stay():
    dev = build_cases(DEV_BANK)
    scored = _score_questions(dev, load_aligner())
    assert scored["points"] == 126
    assert scored["n"] == 143
    assert scored["guard"] == 0
    table = load_corpus_table()
    curve = score_abstain_margin(table, load_weights(), 0.1)
    assert curve["false_abstain"] == {"en": 0, "zh": 0}
    assert curve["false_n"] == {"en": 192, "zh": 192}
    assert curve["missing"] == {"en": 188, "zh": 189}
    assert curve["absent"] == {"en": 36, "zh": 36}
    assert curve["deictic_false"] == {"en": 0, "zh": 0}
