"""Dialogue state licenses one edit-1 name and leaves an empty log unchanged.

Fact writes are logged and do not license. An utterance or an earlier ask
does, inside a window of four turns. A longer name stays rejected. The
margin stays 0.1 and the ridge gap stays 0.15. Frozen questions are not an
input to the wording.
"""
import ast
import hashlib
from pathlib import Path

import numpy as np

from tidal.dialogue_state import TURN_WINDOW, DialogueState
from tidal.question_dialogue_dev_data import (
    bank_sha256,
    bank_strings,
    build_rows,
    kind_counts,
    questions_sha256,
)
from tidal.question_dialogue_eval import jaccard_report, prior_corpora
from tidal.question_name_align import dialogue_licensed_edit
from tidal.question_surface import (
    ABSTAIN_MARGIN,
    LEX_RIDGE_GAP,
    LABELS,
    ask_people,
    ask_targets,
    load_weights,
    score_abstain_margin,
    load_corpus_table,
)

QUESTIONS_SHA = "11bc041256941969c1055bb5bfc3cd491407718131689aa2f253a2d38e669f2a"
BANK_SHA = "69bd3ed6327625afc07cd06a46cbe771d998e806f4a430b3da2aeba014f3ec98"

# Question lists of the banks this split has to stay off. A wording change
# there moves the sha; this file does not edit those strings.
PRIOR_SHA = {
    "abstain_dev": "26889048ad9340726568ec7c187da4d2614c40a800128c5580f519ae7ff86b2a",
    "frozen": "cce20b6bf29f594992adac193ebcf05323fdd31cbe8c9f695a94d88a564723a5",
    "cjk": "5e8b86f694d8a26a3505230fd3fd8f1e80e5261751026c57203dcf2e5ad508ee",
    "nickname": "a766f748cfc2bcc28f8c9b468c9822003798b3de377df15daa8a0633eb1aec98",
    "typo": "68260c01d5f130c46285348f1ef9886f14e9aa2b47f5606c9357ecdc521f5f99",
    "indirect_questions": "daf4619f2d24db78396f2516e7a4d315bb71eefb04dba825347c6fc755874b93",
    "indirect_bank": "18b5e2d128fb6b626f28644b851d405581169c4befc842c96536cb84b5f6efd6",
    "cue": "ed3bd6aadfa92ed4d9843260ef1788875c83019fcb0046491d3928394c08628f",
    "bare": "a0d7309555ecb504c1f44bc382786c0ccff35f9a339d95b685b7a148cb465d23",
    "residual": "b35a4c42560a98a237ae30f2d1d4f68dc5a9987dadadc18e58da1875624cf4a5",
    "two_name": "9c72a8b3f65eff7e6fdc436e906f769de6f9e85451284464f23eb338296d02d8",
    "focus": "03dea71d6a47343fc6410c21df19868e824818e83b48e61ae2aaffc45e188c7c",
    "typo_edit": "3447aaaa0b649dc602f139cedb660bb92bace99f9de3f22d1aa3ad0f261b70fd",
    "lastname_questions": "064d7052e2caad6db71272315fb8af2bbfbb246a1c474970548bf424217e8920",
    "lastname_bank": "626514af8f1f4399c23f47e1b59ae7676e711d450e1e66ea0d0e96b5df4aa22e",
}

KINDS = {
    "typo_hit_utterance": 16,
    "typo_abstain_silent": 16,
    "typo_abstain_other": 16,
    "typo_hit_prior_ask": 8,
    "typo_abstain_stale": 4,
    "typo_abstain_tie": 2,
    "spelling_hit": 8,
    "nickname_hit": 6,
    "latin_extension": 2,
    "cjk_superstring": 4,
    "cjk_prefix": 2,
    "focus_hit": 4,
    "dismiss_hit": 4,
    "open_either": 4,
    "indirect_hit": 2,
    "indirect_noself": 2,
    "indirect_missing": 2,
}

# Kinds whose question is answerable with no mention list. The rest of the
# answerable rows need the person to have been said.
_CONTROLS = {
    "spelling_hit",
    "nickname_hit",
    "focus_hit",
    "dismiss_hit",
    "open_either",
    "indirect_hit",
    "indirect_noself",
    "indirect_missing",
}
_TYPO_HITS = {"typo_hit_utterance", "typo_hit_prior_ask"}


def _active(row):
    """The mention list ``note_turn`` and earlier asks would have built."""
    state = DialogueState()
    names = [slot.name for slot in row.slots]
    surfaces = [slot.text for slot in row.slots]
    for turn in row.turns:
        mentioned = ask_people(turn, names, surfaces, state.conversational())
        state.note_utterance(mentioned, surface=turn)
    for prior in row.prior_asks:
        live = state.conversational(ignore_surface=prior)
        mentioned = ask_people(prior, names, surfaces, live)
        state.note_ask(mentioned, prior)
    return state.conversational(ignore_surface=row.question)


def _targets(row, active):
    names = [slot.name for slot in row.slots]
    surfaces = [slot.text for slot in row.slots]
    if row.kind.startswith("indirect"):
        return ask_people(row.question, names, surfaces, active)
    return ask_targets(row.question, names, surfaces, active)


def test_bank_sha_and_shape():
    rows = build_rows()
    assert questions_sha256(rows) == QUESTIONS_SHA
    assert bank_sha256(rows) == BANK_SHA
    assert len(rows) == 102
    assert kind_counts(rows) == KINDS
    assert sum(row.lang == "en" for row in rows) == 50
    assert sum(row.lang == "zh" for row in rows) == 52
    assert TURN_WINDOW == 4
    assert ABSTAIN_MARGIN == 0.1
    assert LEX_RIDGE_GAP == 0.15
    hit = {row.who for row in rows if row.kind == "typo_hit_utterance"}
    silent = {row.who for row in rows if row.kind == "typo_abstain_silent"}
    other = {row.who for row in rows if row.kind == "typo_abstain_other"}
    assert hit and hit == silent == other
    assert len(bank_strings(rows)) == len(set(bank_strings(rows)))


def test_bank_is_disjoint_from_earlier_splits():
    rows = build_rows()
    bags = prior_corpora()
    assert set(PRIOR_SHA) <= set(bags)
    for name, sha in PRIOR_SHA.items():
        assert bags[name]["sha"] == sha, name
    report = jaccard_report(bank_strings(rows), bags)
    for name, row in report.items():
        assert row["literal"] == 0, name
        assert row["max_jaccard"] <= 0.50, (name, row["max_jaccard"], row["pair"])


def test_edit_is_licensed_only_for_one_active_person():
    assert dialogue_licensed_edit("Swithan", ["Swithun"], ()) is None
    assert dialogue_licensed_edit("Swithan", ["Swithun"], ["Swithun"]) == "Swithun"
    assert dialogue_licensed_edit("Swithan", ["Swithun", "Wulfrun"], ["Wulfrun"]) is None
    assert dialogue_licensed_edit("Swithuno", ["Swithun"], ["Swithun"]) is None
    assert dialogue_licensed_edit("Swithu", ["Swithun"], ["Swithun"]) == "Swithun"
    assert dialogue_licensed_edit(
        "Hrothgar", ["Hrathgar", "Hrethgar"], ["Hrathgar", "Hrethgar"],
    ) is None
    assert dialogue_licensed_edit("Hrothgar", ["Hrathgar", "Hrethgar"], ["Hrathgar"]) == "Hrathgar"
    assert dialogue_licensed_edit("储宴", ["储晏"], ["储晏"]) == "储晏"
    assert dialogue_licensed_edit("储晏然", ["储晏"], ["储晏"]) is None
    assert dialogue_licensed_edit("老储晏", ["储晏"], ["储晏"]) is None

    rows = build_rows()
    for row in rows:
        names = [slot.name for slot in row.slots]
        surfaces = [slot.text for slot in row.slots]
        live = _active(row)
        got = _targets(row, live)
        bare = _targets(row, ())
        if row.kind in _TYPO_HITS:
            assert row.target in live, (row.kind, row.lang, live)
            assert got == [row.target], (row.kind, row.question, got)
            assert bare == []
        elif row.kind in _CONTROLS:
            assert got == bare, (row.kind, got, bare)
            if row.kind == "open_either":
                assert set(got) == {slot.name for slot in row.slots}
            elif row.kind == "indirect_noself":
                assert got == []
            elif row.kind != "indirect_noself":
                assert got == [row.target], (row.kind, got, row.question)
        else:
            assert got == [] and bare == [], (row.kind, got, bare, live)
        # The same call with the argument omitted is the empty list.
        if row.kind.startswith("indirect"):
            assert ask_people(row.question, names, surfaces) == ask_people(
                row.question, names, surfaces, (),
            )
        else:
            assert ask_targets(row.question, names) == ask_targets(
                row.question, names, surfaces, (),
            )


def test_writes_do_not_count_and_the_window_closes():
    state = DialogueState()
    state.note_write("Swithun")
    state.note_write("")
    assert state.conversational() == ()
    assert state.recent() == ("Swithun",)
    assert state.turn == 0
    state.note_utterance(["Swithun"], surface="spoke of Swithun")
    assert state.conversational() == ("Swithun",)
    for i in range(TURN_WINDOW):
        state.note_utterance(["Wulfrun"], surface=f"spoke of Wulfrun {i}")
    assert state.conversational() == ("Wulfrun",)
    assert state.turn == TURN_WINDOW + 1

    repeat = DialogueState()
    repeat.note_ask(["Swithun"], "telephone for Swithun")
    repeat.note_ask(["Swithun"], "telephone for Swithun")
    assert repeat.turn == 1
    assert repeat.conversational(ignore_surface="telephone for Swithun") == ()
    assert repeat.conversational() == ("Swithun",)
    repeat.note_ask(["Wulfrun"], "telephone for Wulfrun")
    assert repeat.turn == 2
    assert repeat.conversational(ignore_surface="telephone for Wulfrun") == ("Swithun",)


def test_matcher_does_not_import_the_dev_bank():
    root = Path("tidal")
    banned = ("question_dialogue_dev", "llm_memory", "LLMMemory", "transformers")
    for name in ("dialogue_state.py", "continuum.py", "question_name_align.py"):
        blob = (root / name).read_text(encoding="utf-8")
        for needle in banned:
            assert needle not in blob, (name, needle)
    surface = (root / "question_surface.py").read_text(encoding="utf-8")
    assert "question_dialogue_dev" not in surface
    assert "llm_memory" not in surface and "LLMMemory" not in surface
    data = (root / "question_dialogue_dev_data.py").read_text(encoding="utf-8")
    tree = ast.parse(data)
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            imported.append(node.module)
        elif isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
    assert set(imported) <= {"__future__", "hashlib", "dataclasses", "typing"}
    for blob in (surface, (root / "continuum.py").read_text(encoding="utf-8")):
        assert "enable_abstain" not in blob and "use_abstain" not in blob
        assert "enable_dialogue" not in blob


def test_surface_margin_is_unchanged():
    row = score_abstain_margin(load_corpus_table(), load_weights(), 0.1)
    assert row["false_abstain"] == {"en": 0, "zh": 0}
    assert row["false_n"] == {"en": 192, "zh": 192}
    assert ABSTAIN_MARGIN == 0.1
    assert LEX_RIDGE_GAP == 0.15
    assert LABELS == ("phone", "birthday", "places")


def test_ask_uses_the_log_and_ignores_a_bare_write():
    """Tiny encoder. Sentence vectors are relation directions, not e5.

    A fact write leaves the typo abstaining. A turn, or an earlier ask,
    makes that same typo return the stored value. Asking the typo again
    does not open a second turn. ``new_session`` drops the mention and the
    typo abstains; the exact name still answers. ``clear`` drops both.
    """
    import torch

    from tidal.continuum import Continuum, _cfg
    from tidal.episodic import decode_span
    from tidal.question_heldout import surface_feats
    from tidal.question_surface import LABELS as RELS
    from tidal.vap import VAPModel

    rows = build_rows()
    weights = load_weights()
    Wa = np.asarray(weights["Wa"], dtype=np.float64)
    directions = {}
    for i, lab in enumerate(RELS):
        direction = Wa[:-1, i].astype(np.float64)
        directions[lab] = direction / np.linalg.norm(direction)

    def slot_vec(rel: str, i: int) -> np.ndarray:
        base = directions[rel]
        noise = np.zeros_like(base)
        noise[i % 500] = 1.0
        noise -= base * float(np.dot(noise, base))
        noise /= np.linalg.norm(noise)
        vec = 0.95 * base + 0.05 * noise
        vec /= np.linalg.norm(vec)
        return vec.astype(np.float32)

    torch.manual_seed(4)
    prev = torch.backends.mkldnn.enabled
    torch.backends.mkldnn.enabled = False
    model = VAPModel(8, d=16, hidden=16, layers=1, dropout=0).eval()
    model.tproj = torch.nn.Sequential(torch.nn.Dropout(0.0), torch.nn.Linear(512, 16))
    model.tproj.eval()
    cont = Continuum(model, _cfg(dict(levels=(), episodic_cfg={"overwrite_cos": 1.0})))

    def run(row, with_state: bool):
        cont.episodic.clear()
        cont._mem_group = None
        for i, slot in enumerate(row.slots):
            key = cont.embed_event(
                surface_feats(slot.text, "statement"),
                slot_vec(slot.rel, i + 1),
                surface=slot.text,
            )
            assert cont.remember(
                key, slot.value, name=slot.name, rel=slot.rel, kind="fact", group=None,
            ) is True
        assert cont._dialogue_for(None).conversational() == ()
        if with_state:
            for turn in row.turns:
                cont.note_turn(turn, group=None)
            for prior in row.prior_asks:
                pq = cont.embed_event(
                    surface_feats(prior, "question"),
                    directions[row.rel].astype(np.float32),
                    surface=prior,
                )
                cont.query(pq, group=None, mode="top1")
        qq = cont.embed_event(
            surface_feats(row.question, "question"),
            directions[row.rel].astype(np.float32),
            surface=row.question,
        )
        value, info = cont.query(qq, group=None, mode="top1")
        answer = None if value is None else decode_span(value)
        return answer, info.get("abstain"), qq

    failures = []
    try:
        sample = next(
            row for row in rows if row.kind == "typo_hit_utterance" and row.lang == "en"
        )
        for row in rows:
            for mode in ("empty", "state"):
                answer, reason, _qq = run(row, mode == "state")
                expect = row.label if mode == "state" else row.empty_label
                golds = row.golds if mode == "state" else row.empty_golds
                good = (
                    (expect == "answerable" and reason is None and answer in golds)
                    or (expect == "absent_person" and reason == "absent_person" and answer is None)
                    or (expect == "missing_rel" and reason == "missing_rel" and answer is None)
                )
                if not good:
                    failures.append(
                        f"{row.kind} {row.lang} {mode}: expect {expect} "
                        f"got {reason!r} {answer!r}"
                    )
        assert not failures, "\n".join(failures[:12])

        answer, reason, sample_q = run(sample, True)
        assert reason is None and answer in sample.golds
        state = cont._dialogue_for(None)
        turn = state.turn
        again, info = cont.query(sample_q, group=None, mode="top1")
        assert state.turn == turn
        assert decode_span(again) in sample.golds and info.get("abstain") is None

        cont.new_session(None)
        assert None not in cont._dialogues
        value, info = cont.query(sample_q, group=None, mode="top1")
        assert value is None and info["abstain"] == "absent_person"
        exact = f"barmkin oxter telephone for {sample.target} thrid99 㐀㐁"
        eq = cont.embed_event(
            surface_feats(exact, "question"),
            directions["phone"].astype(np.float32),
            surface=exact,
        )
        value, info = cont.query(eq, group=None, mode="top1")
        assert info.get("abstain") is None and decode_span(value) in sample.golds

        cont.episodic.clear()
        assert cont._dialogues == {}
        assert cont.ask(eq, group=None) is None
        key = cont.embed_event(
            surface_feats("chat line", "chat"),
            directions["phone"].astype(np.float32),
            surface="chat line",
        )
        assert cont.remember(
            key, "qxv0000", name="Swithun", rel="phone", kind="chat", group=None,
        ) is False
        assert cont._dialogue_for(None).recent() == ()

        key = cont.embed_event(
            surface_feats(sample.slots[0].text, "statement"),
            slot_vec("phone", 7),
            surface=sample.slots[0].text,
        )
        assert cont.remember(
            key, sample.slots[0].value, name=sample.target, rel="phone", kind="fact", group="room",
        )
        cont.note_turn(sample.turns[0], group="room")
        assert sample.target in cont._active_names("room", sample.question)
        assert cont._active_names(None, sample.question) == ()
    finally:
        torch.backends.mkldnn.enabled = prev
