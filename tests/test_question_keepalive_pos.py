"""Positive keep-alive slice.

Listener labels were committed before the score. The score test locks the
measured misses. It does not require them to be zero, and it does not
relabel a row.
"""
from collections import Counter
from pathlib import Path

from tidal.dialogue_state import TURN_WINDOW
from tidal.question_dialogue_eval import jaccard_report
from tidal.question_keepalive_expand_data import bank_sha256 as expand_bank_sha
from tidal.question_keepalive_expand_data import questions_sha256 as expand_questions_sha
from tidal.question_keepalive_heldout_data import bank_sha256 as frozen_bank_sha
from tidal.question_keepalive_heldout_data import questions_sha256 as frozen_questions_sha
from tidal.question_keepalive_pos_eval import prior_bags, replay_rows, score_rows
from tidal.question_keepalive_pos_data import Ask, Turn, bank_sha256, bank_strings, build_rows, kind_counts, questions_sha256
from tidal.question_surface import ABSTAIN_MARGIN, LEX_RIDGE_GAP

QUESTIONS_SHA = "322afe2bba895dae80beda74db92272356732a771e561e6bb2d2b5b983c47038"
BANK_SHA = "b32f13929119c17b1cc4dfcc81b534f21e7c953a6c5904eba87a7062a5f064af"
EXPAND_QUESTIONS_SHA = "9af36973b625ce4ae30f516fafdc3a4aec5c37fbfcaf2a8fc31ec423025ee19c"
EXPAND_BANK_SHA = "30af4bb72c1fbe346dd6b5572ebb60d811b860b71ad060385654f3a7d35f1604"
FROZEN_QUESTIONS_SHA = "092252c79252651ad6cdee2c63783a445ca1a2e4e15804732c955a8407de1a03"
FROZEN_BANK_SHA = "694fed5e12b4e4dbbcd831c8394e7101cbcab908d6406dd5ac5611a4e2c85b25"
LISTENERS = {
    "pos_keep": {"Ranulf": 10, "Isolde": 8, "Petronilla": 6, "祁松": 10, "康梧": 8, "钮芮": 6},
    "pos_voc": {"Ranulf": 10, "Isolde": 8, "Petronilla": 6, "祁松": 10, "康梧": 10, "钮芮": 4},
    "neg_near": {"ABSENT": 20},
}
_FORBIDDEN = (
    "apply_utterance",
    "resolve_deixis",
    "vocative_hearer",
    "refresh_resolved",
    "_predict",
    "DialogueState",
    "ask_people",
    "name_spans",
    "selected_fact_rows",
)


def _ask(row):
    return next(event for event in row.events if isinstance(event, Ask))


def test_pos_bank_matches_the_preregistered_labels():
    rows = build_rows()
    assert len(rows) == 116
    assert kind_counts(rows) == {"pos_keep": 48, "pos_voc": 48, "neg_near": 20}
    assert questions_sha256(rows) == QUESTIONS_SHA
    assert bank_sha256(rows) == BANK_SHA
    assert len(bank_strings(rows)) == 2546
    assert frozen_questions_sha() == FROZEN_QUESTIONS_SHA
    assert frozen_bank_sha() == FROZEN_BANK_SHA
    assert expand_questions_sha() == EXPAND_QUESTIONS_SHA
    assert expand_bank_sha() == EXPAND_BANK_SHA
    assert TURN_WINDOW == 4
    assert ABSTAIN_MARGIN == 0.1
    assert LEX_RIDGE_GAP == 0.15
    langs = Counter()
    listeners = {kind: Counter() for kind in LISTENERS}
    gaps = Counter()
    voc_turns = Counter()
    for row in rows:
        ask = _ask(row)
        langs[(ask.kind, ask.lang)] += 1
        listeners[ask.kind][ask.listener] += 1
        nturns = sum(isinstance(event, Turn) for event in row.events)
        if ask.kind == "pos_keep":
            gaps[nturns - 1] += 1
        elif ask.kind == "pos_voc":
            voc_turns[nturns] += 1
            assert nturns in (1, 2)
        else:
            assert nturns == 1
            assert ask.listener == "ABSENT"
    assert dict(langs) == {
        ("pos_keep", "en"): 24,
        ("pos_keep", "zh"): 24,
        ("pos_voc", "en"): 24,
        ("pos_voc", "zh"): 24,
        ("neg_near", "en"): 10,
        ("neg_near", "zh"): 10,
    }
    assert {kind: dict(counts) for kind, counts in listeners.items()} == LISTENERS
    assert dict(gaps) == {2: 8, 3: 8, 4: 8, 5: 8, 6: 16}
    assert dict(voc_turns) == {1: 30, 2: 18}
    by_id = {row.eid: row for row in rows}
    titled = {
        "pos_voc_zh_lao_qi_phone": ("老祁", "祁松"),
        "pos_voc_zh_teacher_qi_birthday": ("祁老师", "祁松"),
        "pos_voc_zh_lao_kang_later_phone": ("老康", "康梧"),
        "pos_voc_zh_teacher_kang_later_birthday": ("康老师", "康梧"),
        "pos_voc_en_dr_ranulf_phone": ("Dr. Ranulf", "Ranulf"),
        "pos_voc_en_dr_later_petronilla_birthday": ("Dr. Petronilla", "Petronilla"),
        "pos_voc_zh_a_qi_phone": ("祁松啊", "祁松"),
        "pos_voc_zh_ya_kang_birthday": ("康梧呀", "康梧"),
        "pos_voc_zh_bare_ni_kang_phone": ("康梧你把", "康梧"),
        "pos_voc_en_bare_you_later_isolde_phone": ("Isolde you left", "Isolde"),
    }
    for eid, (fragment, listener) in titled.items():
        row = by_id[eid]
        anchor = next(event.text for event in row.events if isinstance(event, Turn))
        assert fragment in anchor
        assert _ask(row).listener == listener
    assert "Swithin," in next(event.text for event in by_id["neg_near_en_unknown_phone"].events if isinstance(event, Turn))
    assert "鄂棠，" in next(event.text for event in by_id["neg_near_zh_unknown_phone"].events if isinstance(event, Turn))
    quote = next(event.text for event in by_id["neg_near_en_quote_phone"].events if isinstance(event, Turn))
    assert quote.startswith("The margin slip reads")
    assert not quote.startswith("Ranulf")


def test_pos_bank_is_disjoint_from_earlier_banks():
    report = jaccard_report(bank_strings(build_rows()), prior_bags())
    assert report["keepalive_42_questions"]["sha"] == FROZEN_QUESTIONS_SHA
    assert report["keepalive_42_bank"]["sha"] == FROZEN_BANK_SHA
    assert report["keepalive_expand_questions"]["sha"] == EXPAND_QUESTIONS_SHA
    assert report["keepalive_expand_bank"]["sha"] == EXPAND_BANK_SHA
    assert report["dialogue_real_questions"]["sha"].startswith("117ef589")
    assert report["dialogue_real_bank"]["sha"].startswith("0be42845")
    assert report["frozen"]["sha"].startswith("cce20b6b")
    for name, row in report.items():
        assert row["literal"] == 0, name
        assert row["max_jaccard"] < 0.50, (name, row["max_jaccard"], row["pair"])


def test_pos_bank_does_not_replay_the_rule_and_production_files_do_not_import_it():
    blob = Path("tidal/question_keepalive_pos_data.py").read_text(encoding="utf-8")
    for name in _FORBIDDEN:
        assert name not in blob
    root = Path("tidal")
    for name in ("question_surface.py", "dialogue_state.py", "continuum.py", "llm_memory.py"):
        text = (root / name).read_text(encoding="utf-8")
        assert "question_keepalive_pos" not in text
        assert "enable_keepalive" not in text


# Measured once against the frozen rule. A later edit that removes a miss
# has to change this list on purpose. This run does not.
LISTENER_MISS_EIDS = (
    "pos_keep_en_he6a_phone",
    "pos_keep_en_he6a_birthday",
    "pos_voc_en_dr_ranulf_phone",
    "pos_voc_en_dr_ranulf_birthday",
    "pos_voc_en_dr_isolde_phone",
    "pos_voc_en_dr_isolde_birthday",
    "pos_voc_en_dr_later_ranulf_phone",
    "pos_voc_en_dr_later_ranulf_birthday",
    "pos_voc_en_bare_you_later_isolde_phone",
    "pos_voc_en_bare_you_later_isolde_birthday",
    "pos_voc_en_dr_later_petronilla_phone",
    "pos_voc_en_dr_later_petronilla_birthday",
    "pos_keep_zh_ta6a_phone",
    "pos_keep_zh_ta6a_birthday",
    "pos_voc_zh_a_qi_phone",
    "pos_voc_zh_a_qi_birthday",
    "pos_voc_zh_ya_kang_phone",
    "pos_voc_zh_ya_kang_birthday",
    "pos_voc_zh_lao_qi_phone",
    "pos_voc_zh_lao_qi_birthday",
    "pos_voc_zh_teacher_qi_phone",
    "pos_voc_zh_teacher_qi_birthday",
    "pos_voc_zh_bare_ni_kang_phone",
    "pos_voc_zh_bare_ni_kang_birthday",
    "pos_voc_zh_lao_kang_later_phone",
    "pos_voc_zh_lao_kang_later_birthday",
    "pos_voc_zh_a_niu_later_phone",
    "pos_voc_zh_a_niu_later_birthday",
    "pos_voc_zh_teacher_kang_later_phone",
    "pos_voc_zh_teacher_kang_later_birthday",
)


def test_pos_score_locks_the_measured_misses():
    scored = score_rows(replay_rows())
    assert scored["encoder_misses"] == []
    assert [(item["eid"], item["rule_label"], item["listener_label"], item["state_reason"])
            for item in scored["listener_misses"]] == [
        (eid, "absent_person", "answerable", "absent_person") for eid in LISTENER_MISS_EIDS
    ]
    assert scored["kind"]["pos_keep"] == {
        "n": 48, "state": 48, "empty": 48, "human": 44, "state_fa": 0, "empty_fa": 0,
    }
    assert scored["kind"]["pos_voc"] == {
        "n": 48, "state": 48, "empty": 48, "human": 22, "state_fa": 0, "empty_fa": 0,
    }
    assert scored["kind"]["neg_near"] == {
        "n": 20, "state": 20, "empty": 20, "human": 20, "state_fa": 0, "empty_fa": 0,
    }
    assert scored["lang"]["en"]["n"] == 58 and scored["lang"]["en"]["human"] == 46
    assert scored["lang"]["zh"]["n"] == 58 and scored["lang"]["zh"]["human"] == 40
    assert scored["lang"]["en"]["state"] == 58 and scored["lang"]["zh"]["state"] == 58
    assert scored["lang"]["en"]["state_fa"] == 0 and scored["lang"]["zh"]["empty_fa"] == 0
