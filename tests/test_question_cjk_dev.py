"""CJK superstring versus verb attachment. The split dev bank is the input.

The frozen adversarial test is scored after the rule is fixed. It is not a
row in the bank, and it is not used to choose a character.
"""
from collections import defaultdict

import numpy as np

from tidal.question_abstain_dev_data import BANK as DEV_BANK
from tidal.question_adversarial_data import BANK as TEST_BANK
from tidal.question_cjk_dev_data import (
    DROP,
    KEEP,
    NAMES,
    build_rows,
    class_counts,
    label_of,
    questions_sha256,
)
from tidal.question_heldout_data import ABSENT, PEOPLE, probe_strings
from tidal.question_name_align import _score_questions, load_aligner, matched_people
from tidal.question_stress import build_cases, case_strings
from tidal.question_surface import (
    ABSENT_NAMES,
    DEV_NAMES,
    FIT_NAMES,
    STORED_DIM,
    _CJK_AMBIVALENT,
    _CJK_ASPECT,
    _CJK_PARTICLE,
    _CJK_PREDICATE,
    _CJK_TIME_WORD,
    corpus_strings,
    load_corpus_table,
    load_weights,
    name_in_text,
    score_abstain_margin,
    surface_abstain,
)

# sha256 of the question list only. A wording change moves this.
SPLIT_SHA = "5e8b86f694d8a26a3505230fd3fd8f1e80e5261751026c57203dcf2e5ad508ee"
_NOMINAL = frozenset("的呢吧吗啊")


def _derived():
    """Predicate, ambivalent syllable, and time word, from labels only."""
    groups = defaultdict(dict)
    controls = []
    for cls, name, _tail, _kind, question in build_rows():
        end = question.index(name) + len(name)
        extra = question[end]
        nxt = question[end + 1]
        label = label_of(cls)
        if nxt not in _CJK_PARTICLE:
            controls.append((extra, nxt, label))
            continue
        seen = groups[extra]
        if nxt in seen and seen[nxt] != label:
            raise AssertionError(f"both labels for {extra}+{nxt}")
        seen[nxt] = label
    predicate = set()
    ambivalent = set()
    time_word = set()
    for extra, by in groups.items():
        keeps = {nxt for nxt, label in by.items() if label == "keep"}
        drops = {nxt for nxt, label in by.items() if label == "drop"}
        if "时" in keeps:
            time_word.add(extra + "时")
            keeps.remove("时")
        if keeps and drops:
            if not keeps <= _CJK_ASPECT or not drops <= (_NOMINAL | {"时"}):
                raise AssertionError(f"inseparable {extra} keep={keeps} drop={drops}")
            ambivalent.add(extra)
        elif keeps:
            predicate.add(extra)
    return frozenset(predicate), frozenset(ambivalent), frozenset(time_word), controls


def test_bank_is_the_split_and_the_rule_is_derived_from_it():
    rows = build_rows()
    assert questions_sha256(rows) == SPLIT_SHA
    assert class_counts(rows) == {"longer_name": 81, "verb": 68, "near_miss": 52}
    assert len(rows) == 201
    predicate, ambivalent, time_word, controls = _derived()
    assert predicate == _CJK_PREDICATE
    assert ambivalent == _CJK_AMBIVALENT
    assert time_word == _CJK_TIME_WORD
    assert controls and all(label == "keep" and nxt not in _CJK_PARTICLE for _extra, nxt, label in controls)
    # 的话 is one closer. The frozen template is not a row.
    assert any(tail.endswith("的话") and cls == "longer_name" for cls, _n, tail, _k, _q in rows)
    assert any(cls == "verb" and tail[1] in "的过了呢吧吗啊" for cls, _n, tail, _k, _q in rows)
    assert not any("名下是" in question for *_head, question in rows)
    # Same stored name, both labels: the next character is the split.
    both = {
        tail: label_of(cls)
        for cls, name, tail, _k, _q in rows
        if name == "田栖" and tail in ("平的", "平时")
    }
    assert both == {"平的": "drop", "平时": "keep"}


def test_every_dev_row_separates_and_the_head_does_not_undo_it():
    for cls, name, tail, _kind, question in build_rows():
        keep = label_of(cls) == "keep"
        assert name_in_text(name, question) is keep, (cls, name, tail, question)
        got = matched_people(question, NAMES)
        expect = [name] if keep else []
        assert got == expect, (cls, name, tail, got, question)
    assert KEEP == frozenset({"verb", "near_miss"})
    assert DROP == frozenset({"longer_name"})


def test_split_is_disjoint_from_probe_surface_dev_and_the_frozen_test():
    rows = build_rows()
    questions = [row[-1] for row in rows]
    bags = {
        "probe": probe_strings(),
        "surface": corpus_strings(),
        "dev": case_strings(build_cases(DEV_BANK)),
        "test": case_strings(build_cases(TEST_BANK)),
    }
    assert not any(question in set(texts) for texts in bags.values() for question in questions)
    from tidal.question_heldout_data import _tokens

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
    for question in questions:
        for name in banned:
            assert not name_in_text(name, question), (name, question)


def _reason(question, writes):
    weights = dict(load_weights())
    weights["Wa"] = np.zeros_like(weights["Wa"])
    names = [row.name for row in writes]
    rels = [row.rel for row in writes]
    keys = np.zeros((len(names), STORED_DIM), dtype=np.float64)
    query = np.zeros(STORED_DIM, dtype=np.float64)
    query[0] = 1.0
    return surface_abstain(query, keys, rels, names, weights, query_text=question)


def test_frozen_chinese_superstrings_abstain_and_the_kept_slices_stay():
    """Generalization. These strings are not in the split bank."""
    test = build_cases(TEST_BANK)
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
    ping = [case for case in test if "平时" in case.question]
    assert len(ping) == 2
    assert all(name_in_text(case.canonical, case.question) for case in ping)
    answerable = [case for case in ping if case.label == "answerable"]
    assert len(answerable) == 1
    assert _reason(answerable[0].question, answerable[0].writes) is None

    zh_keep = [
        case for case in test
        if case.lang == "zh" and case.slice == "name_variant"
        and case.subtype in ("nickname", "no_surname", "honorific_short")
    ]
    assert len(zh_keep) == 18
    assert all(_reason(case.question, case.writes) is None for case in zh_keep)
    en_near = [case for case in test if case.lang == "en" and case.slice == "near_dup_absent"]
    assert len(en_near) == 16
    assert all(_reason(case.question, case.writes) == "absent_person" for case in en_near)
    two = [case for case in test if case.slice == "two_names" and case.label == "absent_person"]
    assert len(two) == 12
    assert all(_reason(case.question, case.writes) == "absent_person" for case in two)


def test_probe_and_surface_verb_glue_still_matches():
    names = list(PEOPLE["zh"]) + list(FIT_NAMES["zh"]) + list(DEV_NAMES["zh"])
    glue = []
    for text in list(probe_strings()) + list(corpus_strings()):
        for name in names:
            if name not in text:
                continue
            start = 0
            while True:
                idx = text.find(name, start)
                if idx < 0:
                    break
                end = idx + len(name)
                extra = text[end] if end < len(text) else ""
                nxt = text[end + 1] if end + 1 < len(text) else ""
                if extra and nxt and nxt in _CJK_PARTICLE and extra not in _CJK_PARTICLE:
                    glue.append((name, extra, nxt, text))
                    assert name_in_text(name, text), (name, extra, nxt, text)
                start = idx + 1
    extras = {extra for _name, extra, _nxt, _text in glue}
    assert {"选", "报", "要", "给", "留", "说"} <= extras


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
