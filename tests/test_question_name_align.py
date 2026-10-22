"""Name alignment head. Dev names and the surface roster only.

The adversarial test is not an input. Edit distance 1 stays a non-match on
both sides, because the dev bank labels that edit both ways.
"""
from pathlib import Path

import numpy as np

from tidal.question_name_align import (
    FEATURES,
    best_name,
    load_aligner,
    matched_people,
)
from tidal.question_surface import DEV_NAMES, name_in_text

DEV_SHA = "26889048ad9340726568ec7c187da4d2614c40a800128c5580f519ae7ff86b2a"
EN = ("Harold", "Imogen", "Judith", "Kenneth", "Lorraine", "Marcus", "Nellie", "Anna Louise")
ZH = ("孙悦", "胡斌", "郭燕", "高远", "吴迪", "郑凯", "徐峰", "马丽")


def test_bundle_is_the_dev_fit_and_is_tiny():
    weights = load_aligner()
    assert weights is not None
    assert weights["features"] == FEATURES
    assert weights["coef"].shape == (len(FEATURES),)
    assert len(FEATURES) <= 12
    blob = np.load(Path("tidal/question_name_align.npz"), allow_pickle=False)
    assert str(blob["dev_sha256"].item()) == DEV_SHA
    assert int(blob["dev_n"]) == 143
    assert int(blob["baseline_points"]) == 78
    # Score of these weights on the shipped proposer. The dev grid picked the
    # threshold; this number is that score, not a test-set tune.
    assert int(blob["dev_points"]) == 126
    # Shipped decision. A two-letter prefix is not a nickname: Immy and a stub
    # of Imogen are the same feature on the dev bank.
    assert float(blob["threshold"]) == 0.8
    assert float(blob["margin"]) == 4.0


def test_dev_nicknames_pinyin_and_titles_align():
    assert best_name("Harry", EN) == "Harold"
    assert best_name("Judy", EN) == "Judith"
    assert best_name("Ken", EN) == "Kenneth"
    assert best_name("Lorrie", EN) == "Lorraine"
    assert best_name("Marc", EN) == "Marcus"
    assert best_name("Nell", EN) == "Nellie"
    assert best_name("Lou", EN) == "Anna Louise"
    assert best_name("Mr. Ken", EN) == "Kenneth"
    assert best_name("Ms. Louise", EN) == "Anna Louise"
    assert best_name("Immy", EN) is None
    assert best_name("Ha", EN) is None

    assert best_name("小悦", ZH) == "孙悦"
    assert best_name("悦", ZH) == "孙悦"
    assert best_name("孙女士", ZH) == "孙悦"
    assert best_name("Sun Yue", ZH) == "孙悦"
    assert best_name("Hu Bin", ZH) == "胡斌"
    assert best_name("zheng kai", ZH) == "郑凯"
    assert best_name("gaoyuan", ZH) == "高远"
    assert best_name("阿斌", ZH) == "胡斌"
    assert matched_people("值班簿写着Hu Bin时，生日该念哪一条？", ZH) == ["胡斌"]
    assert matched_people("值班簿写着zheng kai时，地方该念哪一条？", ZH) == ["郑凯"]


def test_edit1_and_extensions_do_not_align():
    assert best_name("Harols", EN) is None
    assert best_name("Juduth", EN) is None
    assert best_name("Marcas", EN) is None
    assert best_name("Anna Louize", EN) is None
    assert best_name("Kennth", EN) is None
    assert best_name("Harrold", EN) is None
    assert best_name("Nellie Jane", EN) is None
    assert best_name("吴笛", ZH) is None
    assert best_name("郑恺", ZH) is None
    assert best_name("徐锋", ZH) is None
    assert best_name("孙悦然", ZH) is None
    assert best_name("老孙悦", ZH) is None
    assert best_name("小胡斌", ZH) is None
    assert best_name("孙小悦", ZH) is None


def test_a_leading_adverb_and_a_possessive_stay_on_the_stored_name():
    assert matched_people(
        "Roughly which birthday does the file keep for Harold?",
        ["Harold", "Imogen", "Judith"],
    ) == ["Harold"]
    assert best_name("Harold's", EN) is None
    assert matched_people("Harold's telephone is which?", EN) == ["Harold"]


def test_surface_absent_names_and_the_word_one_do_not_match():
    text = "Which handset would raise Quentin if one was stored?"
    assert matched_people(text, DEV_NAMES["en"]) == []
    assert matched_people("Has the yard ever filed a telephone for Soren?", DEV_NAMES["en"]) == []
    assert matched_people("场院有没有给龚涟存过电话？", DEV_NAMES["zh"]) == []
    assert matched_people("如果我们听过，方甯的地点会是哪些？", DEV_NAMES["zh"]) == []


def test_lexical_verb_and_superstring_stay_the_stored_name():
    """The head does not undo a lexical hit the probe relies on.

    王伟强的 is the same shape as 唐宁选的, so 王伟 still matches. That
    superstring remains an answered near-duplicate.
    """
    assert name_in_text("唐宁", "唐宁选的碰面点是哪里？")
    assert matched_people("唐宁选的碰面点是哪里？", ["唐宁", "姜北"]) == ["唐宁"]
    assert name_in_text("王伟", "名下是王伟强的话，电话存了没有？")
    assert matched_people("名下是王伟强的话，电话存了没有？", ["王伟", "李娜"]) == ["王伟"]
    assert matched_people("老王伟，电话是哪一部？", ["王伟"]) == []


def test_dev_questions_abstain_only_when_the_person_is_absent():
    from tidal.question_abstain_dev_data import BANK as DEV
    from tidal.question_stress import build_cases
    from tidal.question_surface import STORED_DIM, load_weights, surface_abstain

    weights = dict(load_weights())
    # A one-hot query is not a sentence vector. Zero the abstain ridge so the
    # lexical cue decides and this test is only about who was matched.
    weights["Wa"] = np.zeros_like(weights["Wa"])
    cases = build_cases(DEV)

    def reason(case):
        names = [row.name for row in case.writes]
        rels = [row.rel for row in case.writes]
        keys = np.zeros((len(names), STORED_DIM), dtype=np.float64)
        query = np.zeros(STORED_DIM, dtype=np.float64)
        query[0] = 1.0
        return surface_abstain(
            query, keys, rels, names, weights, query_text=case.question,
        )

    nick = [case for case in cases if case.subtype == "nickname" and case.canonical == "Harold"]
    assert len(nick) == 1
    assert reason(nick[0]) is None
    immy = [case for case in cases if case.subtype == "nickname" and case.canonical == "Imogen"]
    assert len(immy) == 1
    assert reason(immy[0]) == "absent_person"
    typos = [case for case in cases if case.subtype == "typo"]
    assert typos and all(reason(case) == "absent_person" for case in typos)
    absent = [
        case for case in cases
        if case.slice == "two_names" and case.label == "absent_person"
    ]
    assert len(absent) == 12
    assert all(reason(case) == "absent_person" for case in absent)
    both = [
        case for case in cases
        if case.slice == "two_names" and case.label == "answerable"
    ]
    assert both and all(reason(case) is None for case in both)
