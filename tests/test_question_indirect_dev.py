"""Indirect hearer descriptions. The dev bank is the labeled store policy.

Continuum has no speaker and no addressee. A hearer description resolves
only when exactly one stored person has a first-person surface. The frozen
adversarial questions are scored after that rule. They are not rows here,
and the twelve strings are not a word list.
"""
import ast
import hashlib
from pathlib import Path

import numpy as np
import torch

from tidal.question_abstain_dev_data import BANK as DEV_BANK
from tidal.question_adversarial_data import BANK as TEST_BANK
from tidal.question_cjk_dev_data import build_rows as cjk_rows
from tidal.question_heldout_data import ABSENT, PEOPLE, _tokens, probe_strings
from tidal.question_indirect_dev_data import (
    bank_sha256,
    bank_strings,
    build_rows,
    label_counts,
    questions,
    questions_sha256,
    setup_counts,
)
from tidal.question_name_align import ALIAS_GROUPS
from tidal.question_nickname_dev_data import build_rows as nick_rows
from tidal.question_stress import build_cases, case_strings
from tidal.question_surface import (
    ABSENT_NAMES,
    DEV_NAMES,
    EXPECTED_BUNDLE_SHA256,
    FIT_NAMES,
    STORED_DIM,
    bundle_sha256,
    corpus_strings,
    dialogue_selves,
    is_deictic,
    is_indirect_hearer,
    ask_people,
    lex_labels,
    load_weights,
    matched_fact_rows,
    name_in_text,
    surface_abstain,
)

# Question list only. A wording change moves this. The bank sha also covers
# the store slots, so a surface edit moves the second digest and not the first.
QUESTIONS_SHA = "daf4619f2d24db78396f2516e7a4d315bb71eefb04dba825347c6fc755874b93"
BANK_SHA = "18b5e2d128fb6b626f28644b851d405581169c4befc842c96536cb84b5f6efd6"
# Closed PR #24. Not in this tree. Disjointness still has to name it.
TYPO_SHA = "68260c01d5f130c46285348f1ef9886f14e9aa2b47f5606c9357ecdc521f5f99"
LLM_INJECT_SHA = "cba711e03c09faf649a82e8922d5a604e01ad6970558002e9bdc11fa37974539"
LLM_SOURCE_SHA = "ac4fc7951d05b09c047960afa24175ce367844acf0596615ab1bf1a0fa8c5a60"

# The twelve frozen questions use addressee / interlocutor / 跟前这位 / 对面那位.
# This bank must not.
_FROZEN_WORDS = ("addressee", "interlocutor", "跟前这位", "对面那位", "spoken to", "addressed")

_TYPO_DEV_QUESTIONS = (
    "brumaire xylograph slip lists Hollid beside q01.",
    "brumaire xylograph ledger under Hollid keeps q02.",
    "The brumaire chit for Hollid is xylograph q03.",
    "brumaire xylograph slip lists Hollid beside q04.",
    "brumaire xylograph ledger under Hollid keeps q05.",
    "The brumaire chit for Hollid is xylograph q06.",
    "brumaire xylograph slip lists Branwell beside q07.",
    "brumaire xylograph ledger under Branwell keeps q08.",
    "The brumaire chit for Branwell is xylograph q09.",
    "brumaire xylograph slip lists Branwell beside q10.",
    "brumaire xylograph ledger under Branwell keeps q11.",
    "The brumaire chit for Branwell is xylograph q12.",
    "brumaire xylograph slip lists Bramwall beside q13.",
    "brumaire xylograph ledger under Bramwall keeps q14.",
    "The brumaire chit for Bramwall is xylograph q15.",
    "brumaire xylograph slip lists Bramwall beside q16.",
    "brumaire xylograph ledger under Bramwall keeps q17.",
    "The brumaire chit for Bramwall is xylograph q18.",
    "brumaire xylograph slip lists Pembrok beside q19.",
    "brumaire xylograph ledger under Pembrok keeps q20.",
    "The brumaire chit for Pembrok is xylograph q21.",
    "brumaire xylograph slip lists Pembrok beside q22.",
    "brumaire xylograph ledger under Pembrok keeps q23.",
    "The brumaire chit for Pembrok is xylograph q24.",
    "鼷册写着阮峥时，鼢页空着，参见鼩鼹q25。",
    "若鼷册是阮峥的话，鼢页没抄，参见鼩鼹q26。",
    "阮峥，鼷鼢页仍空，参见鼩鼹q27。",
    "阮峥呢，鼷页先翻鼢，参见鼩鼹q28。",
    "鼷册写着阮峥时，鼢页空着，参见鼩鼹q29。",
    "若鼷册是阮峥的话，鼢页没抄，参见鼩鼹q30。",
    "阮峥，鼷鼢页仍空，参见鼩鼹q31。",
    "阮峥呢，鼷页先翻鼢，参见鼩鼹q32。",
    "鼷册写着庞琨时，鼢页空着，参见鼩鼹q33。",
    "若鼷册是庞琨的话，鼢页没抄，参见鼩鼹q34。",
    "庞琨，鼷鼢页仍空，参见鼩鼹q35。",
    "庞琨呢，鼷页先翻鼢，参见鼩鼹q36。",
    "鼷册写着庞琨时，鼢页空着，参见鼩鼹q37。",
    "若鼷册是庞琨的话，鼢页没抄，参见鼩鼹q38。",
    "庞琨，鼷鼢页仍空，参见鼩鼹q39。",
    "庞琨呢，鼷页先翻鼢，参见鼩鼹q40。",
    "brumaire xylograph slip lists Hollid beside q41.",
    "brumaire xylograph ledger under Hollid keeps q42.",
    "The brumaire chit for Hollid is xylograph q43.",
    "鼷册写着阮峥时，鼢页空着，参见鼩鼹q44。",
    "若鼷册是阮峥的话，鼢页没抄，参见鼩鼹q45。",
    "阮峥，鼷鼢页仍空，参见鼩鼹q46。",
    "阮峥呢，鼷页先翻鼢，参见鼩鼹q47。",
    "brumaire xylograph slip lists Hollis-Jane beside q48.",
    "brumaire xylograph ledger under Hollis-Jane keeps q49.",
    "The brumaire chit for Hollis-Jane is xylograph q50.",
    "brumaire xylograph slip lists Hollis Jane beside q51.",
    "brumaire xylograph ledger under Hollis Jane keeps q52.",
    "The brumaire chit for Hollis Jane is xylograph q53.",
    "brumaire xylograph slip lists Bramwello beside q54.",
    "brumaire xylograph ledger under Bramwello keeps q55.",
    "The brumaire chit for Bramwello is xylograph q56.",
    "鼷册写着老阮铮时，鼢页空着，参见鼩鼹q57。",
    "若鼷册是老阮铮的话，鼢页没抄，参见鼩鼹q58。",
    "老阮铮，鼷鼢页仍空，参见鼩鼹q59。",
    "老阮铮呢，鼷页先翻鼢，参见鼩鼹q60。",
    "鼷册写着阮小铮时，鼢页空着，参见鼩鼹q61。",
    "若鼷册是阮小铮的话，鼢页没抄，参见鼩鼹q62。",
    "阮小铮，鼷鼢页仍空，参见鼩鼹q63。",
    "阮小铮呢，鼷页先翻鼢，参见鼩鼹q64。",
    "鼷册写着阮铮然时，鼢页空着，参见鼩鼹q65。",
    "若鼷册是阮铮然的话，鼢页没抄，参见鼩鼹q66。",
    "阮铮然，鼷鼢页仍空，参见鼩鼹q67。",
    "阮铮然呢，鼷页先翻鼢，参见鼩鼹q68。",
)


def _jaccard(a, b) -> float:
    left, right = _tokens(a), _tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _zero_weights():
    weights = dict(load_weights())
    weights["Wa"] = np.zeros_like(weights["Wa"])
    return weights


def _reason(question, names, rels, surfaces):
    keys = np.zeros((len(names), STORED_DIM), dtype=np.float64)
    query = np.zeros(STORED_DIM, dtype=np.float64)
    query[0] = 1.0
    return surface_abstain(
        query, keys, rels, names, _zero_weights(),
        query_text=question, surfaces=surfaces,
    )


def _expect_hearer(row) -> bool:
    """Non-hearer rows are the unique-self store asked about a neighbor or a bare demonstrative."""
    return not (row.setup == "unique_self" and row.label == "absent_person")


def test_bank_is_the_labeled_store_policy():
    rows = build_rows()
    assert questions_sha256(rows) == QUESTIONS_SHA
    assert bank_sha256(rows) == BANK_SHA
    assert len(rows) == 50
    assert label_counts(rows) == {"answerable": 24, "absent_person": 24, "missing_rel": 2}
    assert setup_counts(rows) == {
        "unique_self": 34,
        "no_self": 2,
        "two_selves": 2,
        "single_third": 2,
        "single_self": 2,
        "missing_rel": 2,
        "multi_fact": 2,
        "name_wins": 2,
        "absent_name": 2,
    }
    assert sum(row.lang == "en" for row in rows) == 27
    assert sum(row.lang == "zh" for row in rows) == 23
    asked = questions(rows)
    for needle in _FROZEN_WORDS:
        assert not any(needle in question for question in asked)
    # Role nouns and frozen locatives are the same class, and they are not dev strings.
    assert is_indirect_hearer("the addressee")
    assert is_indirect_hearer("the interlocutors")
    assert is_indirect_hearer("the person who is spoken to")
    assert is_indirect_hearer("that person across the desk")
    assert is_indirect_hearer("someone standing opposite")
    assert is_indirect_hearer("跟前这位")
    assert is_indirect_hearer("对面那位")
    assert not is_indirect_hearer("the one in front of the porter")
    assert not is_indirect_hearer(
        "You are the one in front of the porter. Read the telephone he copied."
    )
    assert not is_indirect_hearer("隔壁那位")
    assert not is_indirect_hearer("另一位")
    assert not is_indirect_hearer("那位")
    assert not is_indirect_hearer("When is your birthday?")
    assert not is_indirect_hearer("你的生日是哪一天？")
    assert is_deictic("When is your birthday?")
    assert is_deictic("你的生日是哪一天？")


def test_every_row_resolves_only_the_unique_self():
    rows = build_rows()
    for row in rows:
        assert is_indirect_hearer(row.question) is _expect_hearer(row), row.question
        assert is_deictic(row.question) is False
        assert lex_labels(row.question) == (row.rel,)
        got = ask_people(row.question, row.names, row.surfaces)
        expect = [row.referent] if row.referent else []
        assert got == expect, (row.setup, row.kind, got, expect, row.question)
        reason = _reason(row.question, row.names, row.rels, row.surfaces)
        expect_reason = None if row.label == "answerable" else row.label
        assert reason == expect_reason, (row.setup, row.kind, reason, row.question)
        bare = ask_people(row.question, row.names, None)
        if row.setup == "name_wins":
            assert bare == [row.referent]
        else:
            assert bare == [], row.question
        if row.setup == "unique_self" and row.label == "answerable":
            # The other person is written last and holds the same relation.
            assert row.names[-1] != row.referent
            assert row.rels[-1] == row.rel
        if row.setup == "multi_fact":
            assert matched_fact_rows(row.question, row.names, row.surfaces) == [0, 1]
            assert row.names[2] != row.referent
        if row.setup == "two_selves":
            assert len(dialogue_selves(row.names, row.surfaces)) == 2
        if row.setup == "no_self" or row.setup == "single_third":
            assert dialogue_selves(row.names, row.surfaces) == tuple()
        if row.setup == "missing_rel":
            assert row.rels[0] != row.rel
            assert row.names[0] == row.referent
    sample = rows[0]
    holed = list(sample.surfaces)
    holed[1] = None
    assert dialogue_selves(sample.names, holed) is None
    assert ask_people(sample.question, sample.names, holed) == []


def test_bank_is_disjoint_from_the_other_splits():
    rows = build_rows()
    left = bank_strings(rows)
    bags = {
        "probe": probe_strings(),
        "surface": corpus_strings(),
        "adv_dev": case_strings(build_cases(DEV_BANK)),
        "adv_test": case_strings(build_cases(TEST_BANK)),
        "cjk": [row[-1] for row in cjk_rows()],
        "nick": [row.question for row in nick_rows()],
        "typo": _TYPO_DEV_QUESTIONS,
    }
    assert hashlib.sha256("\n".join(_TYPO_DEV_QUESTIONS).encode("utf-8")).hexdigest() == TYPO_SHA
    assert len(_TYPO_DEV_QUESTIONS) == 68
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
    for name, texts in bags.items():
        assert not any(text in set(texts) for text in left), name
        best = max(_jaccard(a, b) for a in left for b in texts)
        assert best < 0.50, (name, best)
    for text in left:
        for name in banned:
            assert not name or not name_in_text(name, text), (name, text)
    frozen = [case.question for case in build_cases(TEST_BANK) if case.subtype == "second_indirect"]
    assert len(frozen) == 12
    for question in questions(rows):
        assert not any(old in question or question in old for old in frozen)


def test_pattern_leaves_the_fixed_slices_alone():
    test = build_cases(TEST_BANK)
    dev = build_cases(DEV_BANK)
    indirect = [case for case in test if case.subtype == "second_indirect"]
    assert len(indirect) == 12
    assert all(is_indirect_hearer(case.question) and not is_deictic(case.question) for case in indirect)
    others = [case for case in test if case.subtype != "second_indirect"]
    assert not any(is_indirect_hearer(case.question) for case in others)
    assert not any(is_indirect_hearer(write.text) for case in test for write in case.writes)

    dev_hit = [case for case in dev if is_indirect_hearer(case.question)]
    assert len(dev_hit) == 11
    assert all(case.subtype == "second_indirect" for case in dev_hit)
    missed = [
        case.question for case in dev
        if case.subtype == "second_indirect" and not is_indirect_hearer(case.question)
    ]
    assert missed == [
        "Across the desk sits someone whose birthday is in the green diary. Which date?",
    ]

    for text in list(probe_strings()) + list(corpus_strings()):
        assert not is_indirect_hearer(text)
    for row in cjk_rows():
        assert not is_indirect_hearer(row[-1])
    for row in nick_rows():
        assert not is_indirect_hearer(row.question)
    for question in _TYPO_DEV_QUESTIONS:
        assert not is_indirect_hearer(question)

    def reason_of(case):
        return _reason(
            case.question,
            [write.name for write in case.writes],
            [write.rel for write in case.writes],
            [write.text for write in case.writes],
        )

    en_near = [case for case in test if case.lang == "en" and case.slice == "near_dup_absent"]
    zh_near = [case for case in test if case.lang == "zh" and case.slice == "near_dup_absent"]
    assert len(en_near) == 16 and len(zh_near) == 14
    assert all(reason_of(case) == "absent_person" for case in en_near + zh_near)
    two = [case for case in test if case.slice == "two_names" and case.label == "absent_person"]
    assert len(two) == 12
    assert all(reason_of(case) == "absent_person" for case in two)
    longer = [case for case in test if case.subtype == "cjk_superstring"]
    assert len(longer) == 8
    assert all(reason_of(case) == "absent_person" for case in longer)
    typos = [case for case in test if case.subtype == "typo"]
    assert len(typos) == 16
    assert all(reason_of(case) == "absent_person" for case in typos)
    pinyin = [case for case in test if case.subtype == "pinyin"]
    assert len(pinyin) == 8
    for case in pinyin:
        names = [write.name for write in case.writes]
        surfaces = [write.text for write in case.writes]
        assert ask_people(case.question, names, surfaces) == [case.canonical]
        assert reason_of(case) is None
    maggie = [case for case in test if "Maggie" in case.question]
    assert len(maggie) == 1
    assert maggie[0].canonical == "Margaret"
    assert ask_people(
        maggie[0].question,
        [write.name for write in maggie[0].writes],
        [write.text for write in maggie[0].writes],
    ) == ["Margaret"]


def test_frozen_indirect_needs_one_first_person_surface():
    """The same twelve strings abstain when the store is not a unique self."""
    from tidal.question_adversarial import load_table

    table = load_table()
    weights = load_weights()
    cases = [case for case in build_cases(TEST_BANK) if case.subtype == "second_indirect"]
    assert len(cases) == 12
    for case in cases:
        names = [write.name for write in case.writes]
        rels = [write.rel for write in case.writes]
        surfaces = [write.text for write in case.writes]
        assert ask_people(case.question, names, surfaces) == [case.canonical]
        assert matched_fact_rows(case.question, names, surfaces) == [0]
        assert case.writes[0].name == case.canonical
        assert case.writes[0].rel == case.rel == case.writes[1].rel
        assert _reason(case.question, names, rels, surfaces) is None
        assert _reason(case.question, names, rels, None) == "absent_person"
        third = [f"The card kept under {name} stays." for name in names]
        assert ask_people(case.question, names, third) == []
        assert _reason(case.question, names, rels, third) == "absent_person"
        copied = list(surfaces)
        copied[1] = surfaces[1] + " I keep my own copy as well."
        assert ask_people(case.question, names, copied) == []
        keys = np.stack([table[write.text] for write in case.writes], 0)
        assert surface_abstain(
            table[case.question], keys, rels, names, weights,
            query_text=case.question, surfaces=surfaces,
        ) is None
    matcher = Path("tidal/question_surface.py").read_text(encoding="utf-8")
    for case in cases:
        assert case.question not in matcher


def test_ask_scopes_to_the_self_when_the_other_sentence_is_closer():
    """Ranking scope, not only the abstain bit. The other person's sentence is nearer."""
    from tidal.continuum import Continuum, _cfg
    from tidal.vap import VAPModel

    torch.manual_seed(4)
    model = VAPModel(8, d=16, hidden=16, layers=1, dropout=0).eval()
    model.tproj = torch.nn.Sequential(torch.nn.Dropout(0.0), torch.nn.Linear(512, 16))
    model.tproj.eval()
    cont = Continuum(model, _cfg(dict(levels=(), episodic_cfg={"overwrite_cos": 1.0})))
    rng = np.random.default_rng(4)

    def flat():
        return rng.choice([0.2, 0.4, 0.6, 0.8], size=8).astype(np.float32)

    def axis(i):
        vec = np.zeros(512, np.float32)
        vec[i] = 1.0
        return vec

    def near(i):
        """Closer to ``axis(i)`` than to any other axis, and not an exact copy.

        An exact copy of a stored sentence cancels abstention. These asks
        have to abstain, so the query stays under that gate.
        """
        vec = axis(i)
        vec[i + 3] = 0.2
        vec /= np.linalg.norm(vec)
        return vec

    def reset():
        cont.episodic.clear()
        cont._surface_by_state.clear()
        cont._sentence_by_state.clear()

    def store(name, rel, value, emb, surface):
        event = cont.embed_event(flat(), emb, surface=surface) if surface is not None else cont.embed_event(flat(), emb)
        assert cont.remember(event, value, name=name, rel=rel, group=None)

    def ask(question, emb):
        query = cont.embed_event(flat(), emb, surface=question)
        _value, info = cont.query(query, group=None, mode="top1")
        scores = cont.rank_question_scores(query, group=None)
        return cont.ask(query, group=None), info["abstain"], scores

    self_en = "Nephoscope ledger: I circled my own natal dawn on the mark, yaffingale."
    other_en = "Nephoscope ledger: the natal dawn kept under Torvald stays on the yaffingale card."
    self_zh = "龖页写着我自己的生辰，爩边没有第二份。"
    other_zh = "龖页写着卞菽的生辰，爩边没有第二份。"

    reset()
    store("Saffi", "birthday", "self-day", axis(2), self_en)
    store("Torvald", "birthday", "other-day", axis(1), other_en)
    got, reason, scores = ask("Which birthday is filed for the hearer?", axis(1))
    assert got == "self-day" and reason is None
    assert int(np.argmax(scores)) == 0

    reset()
    store("倪棠", "birthday", "self-day-zh", axis(2), self_zh)
    store("卞菽", "birthday", "other-day-zh", axis(1), other_zh)
    got, reason, scores = ask("眼前这位的生日落在哪一天？", axis(1))
    assert got == "self-day-zh" and reason is None
    assert int(np.argmax(scores)) == 0

    reset()
    store("Saffi", "birthday", "self-day", axis(2), self_en)
    store("Torvald", "phone", "other-phone", axis(4), other_en.replace("natal dawn", "callback"))
    got, reason, scores = ask("Which telephone is filed for the hearer?", axis(4))
    assert got is None and reason == "missing_rel"
    assert int(np.argmax(scores)) == 0

    reset()
    store("Saffi", "birthday", "self-day", axis(2), self_en)
    store("Torvald", "birthday", "other-day", axis(1), other_en + " I kept my own copy.")
    got, reason, _scores = ask("Which birthday is filed for the hearer?", near(1))
    assert got is None and reason == "absent_person"

    reset()
    store("Saffi", "birthday", "self-day", axis(2), other_en.replace("Torvald", "Saffi"))
    store("Torvald", "birthday", "other-day", axis(1), other_en)
    got, reason, _scores = ask("Which birthday is filed for the hearer?", near(1))
    assert got is None and reason == "absent_person"

    reset()
    store("Saffi", "birthday", "named-day", axis(2), other_en.replace("Torvald", "Saffi"))
    store("Torvald", "birthday", "self-day", axis(1), self_en)
    got, reason, scores = ask("Which birthday is filed for Saffi, not the hearer?", axis(1))
    assert got == "named-day" and reason is None
    assert int(np.argmax(scores)) == 0

    reset()
    store("Saffi", "birthday", "self-day", axis(1), self_en)
    store("Torvald", "birthday", "other-day", axis(2), other_en)
    got, reason, _scores = ask("Which birthday is filed for Ysolde, not the hearer?", near(1))
    assert got is None and reason == "absent_person"

    reset()
    store("Saffi", "birthday", "self-day", axis(2), self_en)
    store("Torvald", "phone", "other-phone", axis(4), "The card kept under Torvald stays.")
    got, reason, _scores = ask(
        "You are the one in front of the porter. Read the telephone he copied.",
        axis(4),
    )
    assert got == "other-phone" and reason is None

    reset()
    store("Saffi", "birthday", "self-day", axis(2), self_en)
    store("Torvald", "birthday", "other-day", axis(1), None)
    got, reason, _scores = ask("Which birthday is filed for the hearer?", near(1))
    assert got is None and reason == "absent_person"


def test_sources_do_not_import_the_bank_or_grow_the_alias_list():
    surface = Path("tidal/question_surface.py").read_text(encoding="utf-8")
    continuum = Path("tidal/continuum.py").read_text(encoding="utf-8")
    data = Path("tidal/question_indirect_dev_data.py").read_text(encoding="utf-8")
    for blob in (surface, continuum, data):
        assert "enable_abstain" not in blob and "use_abstain" not in blob
        assert "enable_hearer" not in blob and "use_hearer" not in blob
        assert "llm_memory" not in blob and "LLMMemory" not in blob
    for needle in ("question_indirect_dev", "question_adversarial", "question_stress", "question_abstain"):
        assert needle not in surface
    tree = ast.parse(data)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert set(imported) <= {"__future__", "hashlib", "dataclasses", "typing"}
    assert "question_surface" not in data
    for needle in ("Saffi", "Torvald", "Ysolde", "nephoscope", "yaffingale", "倪棠", "郗茉"):
        assert needle not in surface
    assert ALIAS_GROUPS[0] == (("robert",), ("bob", "bobby"))
    assert len(ALIAS_GROUPS) == 22
    assert bundle_sha256() == EXPECTED_BUNDLE_SHA256
    assert float(load_weights()["alpha"]) == 0.3
    root = Path("tidal")
    assert hashlib.sha256((root / "llm_inject_smollm2_135m.pt").read_bytes()).hexdigest() == LLM_INJECT_SHA
    assert hashlib.sha256((root / "llm_memory.py").read_bytes()).hexdigest() == LLM_SOURCE_SHA
