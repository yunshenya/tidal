"""Last-token reads of a multiword given name stay a head hit.

``Ms. Crowe`` / ``Crowe`` against stored ``Philomena Crowe`` is the shape
``Ms. Jane`` / ``Jane`` uses for ``Mary Jane``. The abstain-dev fit labels
the same shape as a match (``Ms. Louise`` / ``Anna Louise``), and the frozen
adversarial row does too. A hard reject of that token was measured on this
bank and left out of ``_featurize``. The margin stays 0.1 and the ridge gap
stays 0.15. Frozen scores are not a gate.
"""
import ast
import hashlib
import re
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
from tidal.question_focus_dev_data import NAMES as FOCUS_NAMES
from tidal.question_focus_dev_data import bank_strings as focus_strings
from tidal.question_heldout_data import ABSENT, PEOPLE, _tokens, probe_strings
from tidal.question_indirect_dev_data import ABSENT as INDIRECT_ABSENT
from tidal.question_indirect_dev_data import OTHER, SELF, THIRD
from tidal.question_indirect_dev_data import bank_strings as indirect_bank
from tidal.question_indirect_dev_data import questions as indirect_questions
from tidal.question_lastname_dev_data import (
    NAMES,
    bank_strings,
    build_rows,
    kind_counts,
    questions_sha256,
)
from tidal.question_name_align import _core, best_name, diagnose, matched_people
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
    _abstain_pred,
    corpus_strings,
    lex_labels,
    load_weights,
    name_in_text,
)
from tidal.question_two_name_dev_data import ABSENT as TWO_ABSENT
from tidal.question_two_name_dev_data import NAMES as TWO_NAMES
from tidal.question_two_name_dev_data import bank_strings as two_strings
import tidal.question_name_align as qa

# sha256 of the question list only. A wording change moves this.
LASTNAME_DEV_SHA = "064d7052e2caad6db71272315fb8af2bbfbb246a1c474970548bf424217e8920"
RESIDUAL_SHA = "b35a4c42560a98a237ae30f2d1d4f68dc5a9987dadadc18e58da1875624cf4a5"
RESIDUAL_NAMES = ("Godfrey", "Linnea", "Hadrian", "Odette", "宓珩", "邹岚", "庞屹", "岑书")

_ABSENT_KINDS = ("honor_absent", "bare_absent")
_STATUS_DESIRED = {
    "honor_absent": 0,
    "bare_absent": 0,
    "honor_sep": 12,
    "bare_sep": 6,
    "full_only": 6,
    "full_both": 0,
    "single_hit": 12,
    "given_keep": 3,
    "nick_hit": 5,
    "nick_miss": 2,
    "cjk_super": 8,
    "cjk_hit": 8,
}
_GATE_DESIRED = dict(_STATUS_DESIRED)
_GATE_DESIRED["honor_absent"] = 24
_GATE_DESIRED["bare_absent"] = 9


def _literal_questions(path, name):
    src = Path(path).read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if getattr(target, "id", "") == name:
                return tuple(ast.literal_eval(node.value))
    raise RuntimeError(f"{name} missing in {path}")


def _jaccard(a, b):
    left, right = _tokens(a), _tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _last_token_only(span, name):
    name_toks = [tok.lower() for tok in re.findall(r"[A-Za-z]+", name)]
    if len(name_toks) < 2:
        return False
    core_toks = [tok.lower() for tok in re.findall(r"[A-Za-z]+", _core(span))]
    return core_toks == [name_toks[-1]]


def _install_gate():
    orig = qa._featurize

    def gated(span, name):
        if _last_token_only(span, name):
            return None
        return orig(span, name)

    qa._featurize = gated
    return orig


def test_lastname_dev_sha_and_shape():
    rows = build_rows()
    assert questions_sha256(rows) == LASTNAME_DEV_SHA
    assert len(rows) == 101
    assert sum(row.lang == "en" for row in rows) == 85
    assert sum(row.lang == "zh" for row in rows) == 16
    assert kind_counts(rows) == {
        "honor_absent": 24,
        "bare_absent": 9,
        "honor_sep": 12,
        "bare_sep": 6,
        "full_only": 6,
        "full_both": 6,
        "single_hit": 12,
        "given_keep": 3,
        "nick_hit": 5,
        "nick_miss": 2,
        "cjk_super": 8,
        "cjk_hit": 8,
    }
    assert ABSTAIN_MARGIN == 0.1
    assert LEX_RIDGE_GAP == 0.15
    titles = {"Ms.", "Mr.", "Mrs.", "Miss"}
    assert {row.pattern for row in rows if row.kind == "honor_absent"} == titles
    for row in rows:
        assert row.rel in LABELS
        assert lex_labels(row.question) == (row.rel,)
        for slot in row.slots:
            assert slot.value not in row.question
            assert slot.value not in slot.text
        if row.expect == "answer":
            assert len(row.golds) == 1
        else:
            assert row.expect == "absent"
            assert row.golds == ()


def test_lastname_dev_is_disjoint():
    rows = build_rows()
    own = bank_strings(rows)
    assert len(own) == 309
    residual = _literal_questions("tests/test_question_two_name.py", "_RESIDUAL_QUESTIONS")
    bags = {
        "probe": probe_strings(),
        "surface": corpus_strings(),
        "abstain_dev": case_strings(build_stress(DEV_BANK)),
        "frozen": case_strings(build_stress(TEST_BANK)),
        "cjk": cjk_questions(),
        "nick": nick_questions(),
        "indirect": indirect_bank(),
        "indirect_q": indirect_questions(),
        "typo": _literal_questions("tests/test_question_indirect_dev.py", "_TYPO_DEV_QUESTIONS"),
        "cue": cue_questions(),
        "bare": bare_questions(),
        "residual": residual,
        "two_name": two_strings(),
        "focus": focus_strings(),
    }
    assert hashlib.sha256("\n".join(residual).encode("utf-8")).hexdigest() == RESIDUAL_SHA
    for name, texts in bags.items():
        bag = set(texts)
        assert not any(text in bag for text in own), name
        score = max(_jaccard(left, right) for left in own for right in texts)
        assert score < 0.50, (name, score)
    banned = set(CJK_NAMES)
    banned.update(RESIDUAL_NAMES)
    for lang in ("en", "zh"):
        banned.update(CUE_NAMES[lang])
        banned.update(BARE_NAMES[lang])
        banned.update(TWO_NAMES[lang])
        banned.update(FOCUS_NAMES[lang])
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


def test_production_head_still_accepts_the_last_token():
    """The shipped head. Jane is Mary Jane. Louise is Anna Louise."""
    assert best_name("Ms. Jane", ["Mary Jane"]) == "Mary Jane"
    assert best_name("Jane", ["Mary Jane"]) == "Mary Jane"
    assert best_name("Ms. Louise", ["Anna Louise"]) == "Anna Louise"
    assert best_name("Louise", ["Anna Louise"]) == "Anna Louise"
    assert best_name("Lou", ["Anna Louise"]) == "Anna Louise"
    assert best_name("Ms. Mary", ["Mary Jane"]) == "Mary Jane"
    assert best_name("Bob", ["Robert"]) == "Robert"
    assert best_name("Mr. Bob", ["Robert"]) == "Robert"
    assert best_name("Gwendolyn", ["Gwendolyn Hesketh"]) == "Gwendolyn Hesketh"
    assert best_name("Crowe", ["Crowe"]) == "Crowe"
    assert best_name("王先生", ["王伟"]) == "王伟"
    assert best_name("小岐", ["傅岐"]) == "傅岐"
    assert best_name("王伟强", ["王伟"]) is None
    # Both people stored: the exact token leads, and the margin keeps the head quiet.
    # The string matcher still has the short name.
    assert best_name("Ms. Jane", ["Mary Jane", "Jane"]) is None
    assert diagnose("Ms. Jane keeps a token.", ["Mary Jane", "Jane"])["lexical"] == ["Jane"]


def _desired(row, answer, reason):
    if row.expect == "absent":
        return answer is None and reason == "absent_person"
    return answer == row.golds[0] and reason is None


def _ask_bank(gate):
    import torch

    from tidal.continuum import Continuum, _cfg
    from tidal.vap import VAPModel

    orig = qa._featurize
    if gate:
        _install_gate()
    torch.manual_seed(4)
    model = VAPModel(8, d=64, hidden=64, layers=1, dropout=0).eval()
    model.tproj = torch.nn.Sequential(torch.nn.Dropout(0.0), torch.nn.Linear(512, 64))
    model.tproj.eval()
    cont = Continuum(model, _cfg(dict(levels=(), episodic_cfg={"overwrite_cos": 1.0})))
    axis_of = {"phone": 121, "birthday": 261, "places": 13}

    def flat():
        return np.full(8, 0.5, np.float32)

    def axis(i):
        vec = np.zeros(512, np.float32)
        vec[i] = 1.0
        return vec

    def near_vec(i):
        vec = axis(i)
        vec[2] = np.float32(0.05)
        vec /= np.linalg.norm(vec)
        return vec

    out = []
    try:
        for row in build_rows():
            cont.episodic.clear()
            for slot in row.slots:
                use = near_vec(axis_of[row.rel]) if slot.name == row.near and slot.rel == row.rel else axis(2)
                assert cont.remember(
                    cont.embed_event(flat(), use, surface=slot.text),
                    slot.value, name=slot.name, rel=slot.rel, group=None,
                )
            query = cont.embed_event(flat(), axis(axis_of[row.rel]), surface=row.question)
            answer = cont.ask(query, group=None)
            _value, info = cont.query(query, group=None, mode="top1")
            out.append((row, answer, info.get("abstain")))
    finally:
        qa._featurize = orig
    return out


def _tally(results):
    counts = {}
    for row, answer, reason in results:
        slot = counts.setdefault(row.kind, [0, 0])
        slot[1] += 1
        slot[0] += int(_desired(row, answer, reason))
    return {kind: hit for kind, (hit, _n) in counts.items()}


def test_unshipped_last_token_reject_is_separable_on_dev_only():
    """Status quo returns the multiword fact. The reject abstains those rows.

    The reject is installed only for this assertion, then taken back out.
    ``full_both`` still returns the separately stored last name: that hit is
    the string matcher, and this reject does not edit it. Nickname, given-name,
    single-token, and CJK controls stay.
    """
    weights = load_weights()
    for rel, index in (("phone", 121), ("birthday", 261), ("places", 13)):
        vec = np.zeros(STORED_DIM, np.float64)
        vec[index] = 1.0
        pred, gap = _abstain_pred(vec, weights["Wa"])
        assert pred == rel and gap >= LEX_RIDGE_GAP
    query = np.zeros(512, np.float64)
    query[121] = 1.0
    near = query.copy()
    near[2] = 0.05
    near /= np.linalg.norm(near)
    assert 0.99 < float(query @ near) < 0.999

    rows = build_rows()
    status_match = []
    for row in rows:
        names = [slot.name for slot in row.slots]
        status_match.append(matched_people(row.question, names))
    orig = _install_gate()
    try:
        gate_match = []
        for row in rows:
            names = [slot.name for slot in row.slots]
            gate_match.append(matched_people(row.question, names))
        assert best_name("Ms. Jane", ["Mary Jane"]) is None
        assert best_name("Jane", ["Mary Jane"]) is None
        assert best_name("Ms. Louise", ["Anna Louise"]) is None
        assert best_name("Louise", ["Anna Louise"]) is None
        assert best_name("Lou", ["Anna Louise"]) == "Anna Louise"
        assert best_name("Bob", ["Robert"]) == "Robert"
        assert best_name("Ms. Mary", ["Mary Jane"]) == "Mary Jane"
        assert best_name("Gwendolyn", ["Gwendolyn Hesketh"]) == "Gwendolyn Hesketh"
        assert best_name("Crowe", ["Crowe"]) == "Crowe"
        assert best_name("王先生", ["王伟"]) == "王伟"
        assert best_name("小岐", ["傅岐"]) == "傅岐"
    finally:
        qa._featurize = orig
    assert best_name("Ms. Jane", ["Mary Jane"]) == "Mary Jane"
    assert best_name("Ms. Louise", ["Anna Louise"]) == "Anna Louise"

    moved = []
    for row, before, after in zip(rows, status_match, gate_match):
        if before != after:
            moved.append(row.kind)
            assert row.kind in _ABSENT_KINDS
            assert before == [row.near]
            assert after == []
        elif row.kind in _ABSENT_KINDS:
            raise AssertionError(row.question)
        elif row.kind == "full_both":
            assert set(before) == {row.target, row.near}
            assert before == after
        elif row.kind in ("nick_miss", "cjk_super"):
            assert before == []
            assert after == []
        else:
            assert before == [row.target]
            assert after == [row.target]
    assert moved.count("honor_absent") == 24
    assert moved.count("bare_absent") == 9

    assert _tally(_ask_bank(False)) == _STATUS_DESIRED
    assert _tally(_ask_bank(True)) == _GATE_DESIRED
    assert best_name("Ms. Jane", ["Mary Jane"]) == "Mary Jane"
    src = Path("tidal/question_name_align.py").read_text(encoding="utf-8")
    assert "_extension(_core(span), name) or _ambiguous_edit(span, name)" in src
    assert "last_token_only" not in src


def test_surface_does_not_import_the_lastname_bank():
    surface = Path("tidal/question_surface.py").read_text(encoding="utf-8")
    align = Path("tidal/question_name_align.py").read_text(encoding="utf-8")
    continuum = Path("tidal/continuum.py").read_text(encoding="utf-8")
    data = Path("tidal/question_lastname_dev_data.py").read_text(encoding="utf-8")
    assert "question_lastname_dev" not in surface
    assert "question_lastname_dev" not in align
    assert "Philomena" not in align
    assert "Crowe" not in surface
    for blob in (surface, align, continuum):
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
