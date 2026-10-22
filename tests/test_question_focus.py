"""A focus restrictor keeps the named person when another name is present.

``only`` / ``just`` / ``solely`` (optional ``about``) and ``只问`` / ``只看`` /
``单独问`` govern the following name. ``not only``, a coordinated pair, and an
open ``between`` / ``among`` / ``和`` do not narrow. Dismiss and a unique
possessor still decide first. The margin stays 0.1 and the ridge gap stays
0.15. Frozen scores are not a gate.
"""
import ast
import hashlib
from pathlib import Path

import numpy as np

from tidal.question_abstain_dev_data import BANK as DEV_BANK
from tidal.question_adversarial_data import BANK as TEST_BANK
from tidal.question_bare_dev_data import NAMES as BARE_NAMES
from tidal.question_bare_dev_data import questions as bare_questions
from tidal.question_cjk_dev_data import NAMES as CJK_NAMES
from tidal.question_cjk_dev_data import questions as cjk_questions
from tidal.question_cue_dev_data import NAMES as CUE_NAMES
from tidal.question_cue_dev_data import questions as cue_questions
from tidal.question_focus_dev_data import (
    NAMES,
    bank_strings,
    build_rows,
    kind_counts,
    questions_sha256,
)
from tidal.question_heldout_data import ABSENT, PEOPLE, _tokens, build_cases, probe_strings
from tidal.question_indirect_dev_data import ABSENT as INDIRECT_ABSENT
from tidal.question_indirect_dev_data import OTHER, SELF, THIRD
from tidal.question_indirect_dev_data import bank_strings as indirect_bank
from tidal.question_indirect_dev_data import questions as indirect_questions
from tidal.question_name_align import diagnose
from tidal.question_nickname_dev_data import build_rows as nick_rows
from tidal.question_nickname_dev_data import questions as nick_questions
from tidal.question_stress import build_cases as build_stress
from tidal.question_stress import case_strings
from tidal.question_surface import (
    ABSTAIN_MARGIN,
    ABSENT_NAMES,
    DEV_NAMES,
    FIT_NAMES,
    LABELS,
    LEX_RIDGE_GAP,
    STORED_DIM,
    _focus_names,
    ask_people,
    ask_targets,
    corpus_strings,
    lex_labels,
    load_weights,
    name_in_text,
    surface_abstain,
)
from tidal.question_two_name_dev_data import ABSENT as TWO_ABSENT
from tidal.question_two_name_dev_data import NAMES as TWO_NAMES
from tidal.question_two_name_dev_data import bank_strings as two_strings

# sha256 of the question list only. A wording change moves this.
FOCUS_DEV_SHA = "03dea71d6a47343fc6410c21df19868e824818e83b48e61ae2aaffc45e188c7c"
RESIDUAL_SHA = "b35a4c42560a98a237ae30f2d1d4f68dc5a9987dadadc18e58da1875624cf4a5"
RESIDUAL_NAMES = ("Godfrey", "Linnea", "Hadrian", "Odette", "宓珩", "邹岚", "庞屹", "岑书")

_FROZEN_BITS = (
    "ask only about",
    "which birthday is filed",
    "里面，只问",
    "porter asks only",
    "brinekeel",
    "cedar slip",
)


def _literal_questions(path: str, name: str):
    src = Path(path).read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if getattr(target, "id", "") == name:
                return tuple(ast.literal_eval(node.value))
    raise RuntimeError(f"{name} missing in {path}")


def _typo_questions():
    return _literal_questions("tests/test_question_indirect_dev.py", "_TYPO_DEV_QUESTIONS")


def _residual_questions():
    return _literal_questions("tests/test_question_two_name.py", "_RESIDUAL_QUESTIONS")


def _jaccard(a: str, b: str) -> float:
    left, right = _tokens(a), _tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _targets_without_restrictor(question, names):
    import tidal.question_surface as qs

    saved = qs._restrict_targets
    qs._restrict_targets = lambda text, matched: []
    try:
        return qs.ask_targets(question, names)
    finally:
        qs._restrict_targets = saved


def _abstain_without_restrictor(row):
    import tidal.question_surface as qs

    saved = qs._restrict_targets
    qs._restrict_targets = lambda text, matched: []
    try:
        return _zero_abstain(row)
    finally:
        qs._restrict_targets = saved


def test_focus_dev_sha_and_shape():
    rows = build_rows()
    assert questions_sha256(rows) == FOCUS_DEV_SHA
    assert len(rows) == 96
    assert sum(row.lang == "en" for row in rows) == 51
    assert sum(row.lang == "zh" for row in rows) == 45
    assert kind_counts(rows) == {
        "focus_hit": 21,
        "focus_miss": 21,
        "open_either": 12,
        "open_miss": 6,
        "dismiss_hit": 6,
        "dismiss_miss": 6,
        "coord_either": 6,
        "not_only_either": 6,
        "one_hit": 6,
        "one_miss": 6,
    }
    patterns = {(row.lang, row.kind, row.pattern) for row in rows}
    for pattern in ("only_about", "just_about", "solely", "just"):
        assert ("en", "focus_hit", pattern) in patterns
        assert ("en", "focus_miss", pattern) in patterns
    for pattern in ("zhiwen", "zhikan", "dandu"):
        assert ("zh", "focus_hit", pattern) in patterns
        assert ("zh", "focus_miss", pattern) in patterns
    assert ("en", "open_either", "between") in patterns
    assert ("en", "open_either", "among") in patterns
    assert ("zh", "open_either", "he") in patterns
    assert ("zh", "open_either", "yu") in patterns
    assert ABSTAIN_MARGIN == 0.1
    assert LEX_RIDGE_GAP == 0.15
    for row in rows:
        assert row.rel in LABELS
        assert lex_labels(row.question) == (row.rel,)
        for bit in _FROZEN_BITS:
            assert bit not in row.question
        for slot in row.slots:
            assert slot.value not in row.question
            assert slot.value not in slot.text
        names = [slot.name for slot in row.slots]
        info = diagnose(row.question, names)
        assert info["added"] == [] and info["unknown"] == [] and info["force_absent"] is False
        if row.expect in ("answer", "either"):
            assert row.golds
        else:
            assert row.golds == ()


def test_focus_dev_is_disjoint():
    rows = build_rows()
    own = bank_strings(rows)
    residual = _residual_questions()
    bags = {
        "probe": probe_strings(),
        "surface": corpus_strings(),
        "abstain_dev": case_strings(build_stress(DEV_BANK)),
        "frozen": case_strings(build_stress(TEST_BANK)),
        "cjk": cjk_questions(),
        "nick": nick_questions(),
        "indirect": indirect_bank(),
        "indirect_q": indirect_questions(),
        "typo": _typo_questions(),
        "cue": cue_questions(),
        "bare": bare_questions(),
        "residual": residual,
        "two_name": two_strings(),
    }
    assert hashlib.sha256("\n".join(residual).encode("utf-8")).hexdigest() == RESIDUAL_SHA
    assert len(residual) == 100
    for name, texts in bags.items():
        bag = set(texts)
        assert not any(text in bag for text in own), name
        score = 0.0
        for left in own:
            for right in texts:
                score = max(score, _jaccard(left, right))
        assert score < 0.50, (name, score)
    banned = set(CJK_NAMES)
    banned.update(RESIDUAL_NAMES)
    for lang in ("en", "zh"):
        banned.update(CUE_NAMES[lang])
        banned.update(BARE_NAMES[lang])
        banned.update(TWO_NAMES[lang])
        for bank in (PEOPLE, ABSENT, FIT_NAMES, DEV_NAMES, ABSENT_NAMES):
            banned.update(bank[lang])
        banned.update(DEV_BANK["stored"][lang])
        banned.update(DEV_BANK["absent"][lang])
        banned.update(TEST_BANK["stored"][lang])
        banned.update(TEST_BANK["absent"][lang])
    banned.add(TWO_ABSENT["en"])
    banned.add(TWO_ABSENT["zh"])
    for row in nick_rows():
        banned.update(row.roster)
    for group in (SELF, OTHER, THIRD, INDIRECT_ABSENT):
        banned.update(group.values())
    for name in NAMES["en"] + NAMES["zh"]:
        assert name not in banned, name
    for text in own:
        for name in banned:
            assert not name_in_text(name, text), (name, text)


def _zero_abstain(row):
    names = [slot.name for slot in row.slots]
    rels = [slot.rel for slot in row.slots]
    keys = np.zeros((len(names), STORED_DIM), np.float64)
    return surface_abstain(
        np.zeros(STORED_DIM), keys, rels, names, load_weights(), query_text=row.question,
    )


def test_restrictor_is_what_separates_the_focus():
    """Dismiss and possessive already ran. Only the restrictor moves focus rows.

    On a focus miss the other name is nearer the cue and stores the relation,
    so the previous rule answers. After the restrictor the focus lacks it.
    Open, coordinated, and not-only rows stay both names.
    """
    expect = {"missing_rel": "missing_rel", "answer": None, "either": None}
    for row in build_rows():
        names = [slot.name for slot in row.slots]
        after = ask_targets(row.question, names)
        before = _targets_without_restrictor(row.question, names)
        reason = _zero_abstain(row)
        assert reason == expect[row.expect], (row.kind, row.pattern, reason, row.question)
        if row.kind.startswith("focus_"):
            assert after == [row.target]
            assert set(before) == {row.target, row.other}
            nearest = _focus_names(row.question, before, lex_labels(row.question))
            assert set(nearest) == {row.other}
            if row.kind == "focus_miss":
                assert _abstain_without_restrictor(row) is None
        elif row.kind in ("open_either", "open_miss", "coord_either", "not_only_either"):
            assert set(after) == {row.target, row.other}
            assert set(before) == set(after)
        else:
            assert after == [row.target]
            assert before == [row.target]
        if row.kind == "one_hit":
            assert ask_people(row.question, names) == [row.target]


def test_a_closer_other_fact_does_not_answer_the_focus():
    """Axis 121 is a phone call the shipped ridge agrees with.

    The other person's sentence is the query vector. Without the restrictor
    that fact wins. With it, the focus's fact wins, and a focus who lacks
    the relation abstains.
    """
    from tidal.continuum import Continuum, _cfg
    from tidal.question_surface import _abstain_pred
    from tidal.vap import VAPModel
    import torch

    hit = next(
        row for row in build_rows()
        if row.kind == "focus_hit" and row.lang == "en" and row.pattern == "only_about" and row.rel == "phone"
    )
    miss = next(
        row for row in build_rows()
        if row.kind == "focus_miss" and row.lang == "en" and row.pattern == "only_about" and row.rel == "phone"
    )
    weights = load_weights()
    phone = np.zeros(STORED_DIM, np.float64)
    phone[121] = 1.0
    pred, gap = _abstain_pred(phone, weights["Wa"])
    assert pred == "phone" and gap >= LEX_RIDGE_GAP

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

    def store(name, rel, value, text, emb):
        assert cont.remember(
            cont.embed_event(flat(), emb, surface=text),
            value, name=name, rel=rel, group=None,
        )

    import tidal.question_surface as qs

    saved = qs._restrict_targets

    def run(row, restrict):
        qs._restrict_targets = saved if restrict else (lambda text, matched: [])
        cont.episodic.clear()
        return cont

    try:
        run(hit, False)
        store(hit.target, "phone", "target-phone", hit.slots[0].text, axis(2))
        store(hit.other, "phone", "other-phone", hit.slots[1].text, axis(121))
        q = cont.embed_event(flat(), axis(121), surface=hit.question)
        assert cont.ask(q, group=None) == "other-phone"
        run(hit, True)
        store(hit.target, "phone", "target-phone", hit.slots[0].text, axis(2))
        store(hit.other, "phone", "other-phone", hit.slots[1].text, axis(121))
        q = cont.embed_event(flat(), axis(121), surface=hit.question)
        assert cont.ask(q, group=None) == "target-phone"

        run(miss, False)
        for slot in miss.slots:
            emb = axis(121) if slot.name == miss.other and slot.rel == "phone" else axis(2)
            store(slot.name, slot.rel, f"{slot.name}-{slot.rel}", slot.text, emb)
        q = cont.embed_event(flat(), axis(121), surface=miss.question)
        assert cont.ask(q, group=None) == f"{miss.other}-phone"
        run(miss, True)
        for slot in miss.slots:
            emb = axis(121) if slot.name == miss.other and slot.rel == "phone" else axis(2)
            store(slot.name, slot.rel, f"{slot.name}-{slot.rel}", slot.text, emb)
        q = cont.embed_event(flat(), axis(121), surface=miss.question)
        assert cont.ask(q, group=None) is None
        _value, info = cont.query(q, group=None, mode="top1")
        assert info["abstain"] == "missing_rel"
    finally:
        qs._restrict_targets = saved


def test_surface_does_not_import_the_focus_bank():
    surface = Path("tidal/question_surface.py").read_text(encoding="utf-8")
    continuum = Path("tidal/continuum.py").read_text(encoding="utf-8")
    data = Path("tidal/question_focus_dev_data.py").read_text(encoding="utf-8")
    assert "question_focus_dev" not in surface
    assert "ask only about" not in surface
    assert "Griselda" not in surface
    assert "单独问|只问|只看" in surface
    for blob in (surface, continuum):
        assert "enable_abstain" not in blob and "use_abstain" not in blob
        assert "llm_memory" not in blob and "LLMMemory" not in blob
    tree = ast.parse(data)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert set(imported) <= {"__future__", "hashlib", "dataclasses", "typing"}
