"""Two-name asks name the target, then check that person's facts.

A dismissed name is not the ask target when another name remains. A unique
possessor of the live cue is the ask target (``Name's cue``, ``Name的cue``,
``is Name's``, ``是Name的``). ``X and Y's`` stays both. Cue-nearest still
applies only among the names that remain, and only for abstain. The margin
stays 0.1 and the ridge gap stays 0.15. The frozen adversarial scores are
not a gate.
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
from tidal.question_heldout_data import ABSENT, PEOPLE, _tokens, build_cases, probe_strings
from tidal.question_indirect_dev_data import ABSENT as INDIRECT_ABSENT
from tidal.question_indirect_dev_data import OTHER, SELF, THIRD
from tidal.question_indirect_dev_data import bank_strings as indirect_bank
from tidal.question_indirect_dev_data import build_rows as indirect_rows
from tidal.question_indirect_dev_data import questions as indirect_questions
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
    ask_people,
    ask_targets,
    corpus_strings,
    cue_set_aside,
    iter_corpus,
    lex_labels,
    load_weights,
    matched_fact_rows,
    surface_abstain,
)
from tidal.question_two_name_dev_data import (
    ABSENT as TWO_ABSENT,
    NAMES,
    bank_strings,
    build_rows,
    kind_counts,
    questions_sha256,
)

# sha256 of the question list only. A wording change moves this.
TWO_DEV_SHA = "9c72a8b3f65eff7e6fdc436e906f769de6f9e85451284464f23eb338296d02d8"
RESIDUAL_SHA = "b35a4c42560a98a237ae30f2d1d4f68dc5a9987dadadc18e58da1875624cf4a5"
RESIDUAL_NAMES = ("Godfrey", "Linnea", "Hadrian", "Odette", "宓珩", "邹岚", "庞屹", "岑书")

_RESIDUAL_QUESTIONS = (
    "Godfrey's voice route sits inside a locations folder at the west loft, futtock carling.",
    'A locations ledger in the chart room is where Linnea is reached by voice, kneetimber hawsehole.',
    'Hadrian is reached by voice from the locations slip on the second shelf, garboard deadwood.',
    'The locations chit that lets a voice find Odette is the one the clerk keeps, breasthook knighthead.',
    'Inside the locations dossier, the voice path for Godfrey is the one to read, limberhole maststep.',
    "Linnea's voice path was copied onto a locations binder, kelson stemson.",
    'The morning Hadrian grows a year older is written in a locations folio, treenail rudderpost.',
    'Odette adds a year on the date a locations logbook still holds, gudgeon pintle.',
    '宓珩的声音路径夹在地点卷宗里，齮齯。',
    '地点档案里记着怎么用声音找到邹岚，齰齱。',
    '庞屹入夜后的声音落在哪一本地方簿册，齲齳。',
    '地方案卷里留着岑书的声音去向，齴齵。',
    '宓珩满岁的那天写在地点档册上，齶齷。',
    '哪一本地方卷夹记着邹岚长一岁的日子，齸齹。',
    '庞屹添岁的页在地点册页里，齺齻。',
    '岑书又大一岁的日子收在地方卷宗，齼齽。',
    'Which locations folder in the west loft holds the voice route for Godfrey, fashionpiece cantframe.',
    "Linnea's voice route, the one a locations slip still carries, is which, hawsepipe countertimber.",
    'A locations binder carries how Odette is reached by voice, orlop becket.',
    'The locations logbook that names the day Hadrian grows a year older is which, thole davit.',
    "Godfrey's added year is the date kept in a locations dossier, binnacle kedge.",
    'From a locations chit, a voice finds Linnea after dusk, marline ratline.',
    '宓珩的声音去向收在地点卷宗的哪一页，齮齰。',
    '邹岚的声音去向抄在地方档案里了吗，齱齲。',
    '用声音找庞屹时，要翻的是哪本地方案卷，齳齴。',
    '岑书满岁写在哪一本地点簿册，齵齶。',
    '宓珩长一岁的日子压在地方档册下，齷齸。',
    '地点卷夹里是邹岚入夜后的声音路径，齹齺。',
    "Read the spoken halts filed in Godfrey's locations folder, first halt onward, sheave whisker.",
    "Linnea's locations ledger keeps the halts in the order they were spoken, yardarm bobstay.",
    "Which locations slip lists Hadrian's halts without reordering, bumpkin cathead.",
    'The locations dossier for Odette is a sequence of halts, chainplate deadeye.',
    '宓珩的地点卷宗按口述先后有哪些，齻齼。',
    '邹岚的地点档案里，落脚顺序怎么念，齽龕。',
    '庞屹的地方簿册从第一处到最后一处是什么，龢龣。',
    '岑书的地方案卷依口述留下了哪些落脚，龤龥。',
    "Godfrey's locations folio, the spoken halts in order, are which, footrope gammoning.",
    "Which locations logbook holds Linnea's halts from first to last, jackstay kevel.",
    'The halts Hadrian dictated sit in a locations binder, manrope parrel.',
    "Odette's locations chit is the spoken sequence of halts, snatchblock traveller.",
    '哪一本地点档册写着宓珩的落脚先后，齮齱。',
    '邹岚歇脚的站序抄进地方卷夹，口述时怎么排，齰齲。',
    '庞屹的地点册页要按口述念完，有哪些，齳齵。',
    '地方卷宗上岑书的落脚顺序是什么，齴齶。',
    "Read Godfrey's locations folder aloud by voice, first halt onward, lanyard topsail.",
    "By voice, recite the halts Linnea's locations ledger holds, spritsail lateen.",
    '用声音把宓珩的地点卷宗按先后念出来，齮齴。',
    '邹岚的地点档案，用声音依口述念落脚，齯齵。',
    'The locations slip of halts for Hadrian is to be read by voice, mizzen bowsprit.',
    "Recite by voice the halts kept in Odette's locations dossier, jibboom martingale.",
    '庞屹的地方簿册用声音按落脚先后念，齰齳。',
    '声音只用来念岑书地点案卷里的落脚顺序，齱齴。',
    'The way to get through to Godfrey is the entry a locations folder still holds, gammon cranse.',
    'How Linnea is reached after dusk is written on a locations slip, catharpin futtockplate.',
    '地点卷宗里写着怎么联络上宓珩，齵齷。',
    '哪一本地方簿册记着庞屹入夜后怎么找得着，齸齹。',
    'A locations ledger holds how Hadrian is reached once the loft is quiet, seizing whipping.',
    'The entry for reaching Odette after dusk sits in a locations chit, fidspike marlinespike.',
    '怎么联络上邹岚，写在地点档案的哪一页，齺齻。',
    '岑书入夜后怎么找得着，地方案卷里有没有写，齼齽。',
    'Ignore the birthday supper and leave it; how is Godfrey reached after dusk, servingboard twine.',
    'Skip the candle line for Linnea; keep the spoken halts in order, beeswax nettles.',
    'Forget the rendezvous card, then the morning Hadrian grows a year older, foxes sennit.',
    'Exclude the handset chit; the halts for Odette stay in spoken order, grommet cringle.',
    '别管宓珩的蜡烛页，入夜后怎么找得着，齮齹。',
    '先合上号码夹子。邹岚满岁记在第几格，齯齺。',
    '地点清册一边去，庞屹入夜后该把声音送去哪一条，齰齻。',
    '先放下岑书的电话条，落脚按什么先后，齱齼。',
    'The birthday board goes aside. A voice is how Godfrey gets found after dusk, reefpoint earring.',
    "Ignoring the candle tally, keep Linnea's spoken halts as given, clewgarnet buntline.",
    'The rendezvous card for Hadrian is a distraction: which morning does the year get added, leechline bowline.',
    'That handset note is the wrong drawer. Read the morning Odette grows a year older, vangpeak peakhalyard.',
    '别提蜡烛页。夜里把声音送到宓珩的哪一条，齵齸。',
    '生日册先放一边，邹岚的落脚顺序怎么念，齹齽。',
    '庞屹的号码条先收到柜里，满岁写在哪一页，齺龕。',
    '跳过岑书的地点页，入夜后怎么找得着，龢齮。',
    'Which telephone does the loft keep for Godfrey, halyard futtock.',
    "On the loft card, Linnea's birthday is the date to read, carling kneetimber.",
    "The spoken queue still has Hadrian's places, hawsehole garboard.",
    'Which handset reaches Odette after dusk, deadwood breasthook.',
    '藤匣侧格有没有夹着宓珩的手机，龣龤。',
    '邹岚的出生日写在哪一页，龥齯。',
    '藤匣里还留着庞屹的地点没有，齰齱。',
    '岑书的号码留在哪一条，齲齳。',
    'The loft telephone for Godfrey is which string, knighthead limberhole.',
    'Which birthday does Linnea claim on the loft card, maststep kelson.',
    "Hadrian's places on the loft card stay in spoken order, stemson treenail.",
    'The handset that reaches Odette from the loft is which, rudderpost gudgeon.',
    '宓珩的手机留在柜侧哪一条，齴齵。',
    '柜侧夹着邹岚落地那天的出生页，齶齷。',
    '庞屹的地点留在藤匣里，先后怎么念，齸齹。',
    '岑书认的出生日在哪一页，齺齻。',
    "Ignore the side chatter. Godfrey's telephone is the line the loft should read, pintle fashionpiece.",
    "Filler notes are for skipping. Linnea's birthday is the claim to read, cantframe hawsepipe.",
    "Park every scrap aside. Hadrian's places want a reading, countertimber orlop.",
    'Forget that feast. Odette is reached on which handset, becket thole.',
    '闲篇跳过就算。邹岚的手机夹在柜侧哪一格，齼齽。',
    '闲谈别管。出生日要看庞屹的那一页，龕龢。',
    '别人的条子先放下。宓珩的地点依口述有哪些，龣龤。',
    '闲话留到以后再说。岑书的号码夹在哪一格，龥齮。',
)


def _typo_questions():
    src = Path("tests/test_question_indirect_dev.py").read_text(encoding="utf-8")
    for node in ast.parse(src).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if getattr(target, "id", "") == "_TYPO_DEV_QUESTIONS":
                return tuple(ast.literal_eval(node.value))
    raise RuntimeError("typo dev questions missing")


def _jaccard(a: str, b: str) -> float:
    left, right = _tokens(a), _tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def test_two_name_dev_sha_and_shape():
    rows = build_rows()
    assert questions_sha256(rows) == TWO_DEV_SHA
    assert len(rows) == 96
    assert sum(row.lang == "en" for row in rows) == 57
    assert sum(row.lang == "zh" for row in rows) == 39
    assert kind_counts(rows) == {
        "dismiss_miss": 18,
        "dismiss_hit": 18,
        "poss_miss": 12,
        "poss_hit": 12,
        "one_hit": 9,
        "one_miss": 9,
        "one_dismiss_hit": 6,
        "absent": 6,
        "open_either": 6,
    }
    assert ABSTAIN_MARGIN == 0.1
    assert LEX_RIDGE_GAP == 0.15
    weights = load_weights()
    pred, gap = _abstain_pred(np.zeros(STORED_DIM), weights["Wa"])
    assert pred in LABELS and gap < LEX_RIDGE_GAP
    for row in rows:
        assert row.lang in ("en", "zh")
        assert row.rel in LABELS
        assert lex_labels(row.question) == (row.rel,)
        assert cue_set_aside(row.question, row.rel) is False
        for slot in row.slots:
            assert slot.value not in row.question
            assert slot.value not in slot.text
        if row.expect in ("answer", "either"):
            assert row.golds
        else:
            assert row.golds == ()


def test_two_name_dev_is_disjoint():
    rows = build_rows()
    own = bank_strings(rows)
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
        "residual": _RESIDUAL_QUESTIONS,
    }
    assert hashlib.sha256("\n".join(_RESIDUAL_QUESTIONS).encode("utf-8")).hexdigest() == RESIDUAL_SHA
    assert len(_RESIDUAL_QUESTIONS) == 100
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
    banned.update(CUE_NAMES["en"])
    banned.update(CUE_NAMES["zh"])
    banned.update(BARE_NAMES["en"])
    banned.update(BARE_NAMES["zh"])
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
    from tidal.question_surface import name_in_text
    for name in NAMES["en"] + NAMES["zh"] + (TWO_ABSENT["en"], TWO_ABSENT["zh"]):
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


def test_target_scope_decides_abstain_on_a_zero_query():
    """The shipped bias gap is under 0.15, so the unique cue stands.

    The other person's fact is in the store. It does not answer a question
    whose target lacks the relation, and it is not the ranked person when
    the target has the relation.
    """
    expect = {
        "missing_rel": "missing_rel",
        "absent_person": "absent_person",
        "answer": None,
        "either": None,
    }
    for row in build_rows():
        names = [slot.name for slot in row.slots]
        reason = _zero_abstain(row)
        assert reason == expect[row.expect], (row.kind, row.lang, row.rel, reason, row.question)
        targets = ask_targets(row.question, names)
        if row.kind in ("dismiss_miss", "dismiss_hit", "poss_miss", "poss_hit"):
            idx = matched_fact_rows(row.question, names)
            assert idx is not None
            assert {names[i] for i in idx} == {row.target}
            assert targets == [row.target]
        elif row.kind == "open_either":
            assert set(targets) == {row.target, row.other}
            assert len(targets) == 2
        elif row.kind == "absent":
            assert targets == []
            assert matched_fact_rows(row.question, names) is None
        elif row.kind == "one_dismiss_hit":
            assert targets == [row.target]
        elif row.kind in ("one_hit", "one_miss"):
            assert targets == [row.target]


def test_coordination_keeps_both_possessors():
    en = "Ann and Catherine's birthday is which date?"
    assert set(ask_targets(en, ["Ann", "Catherine"])) == {"Ann", "Catherine"}
    zh = "嵇沅和逯荞的生日记在哪一条。"
    assert set(ask_targets(zh, ["嵇沅", "逯荞"])) == {"嵇沅", "逯荞"}
    # A single name inside a dismiss clause is still the ask.
    alone = "Ignore Cressida's supper. Which telephone does the cedar slip hold?"
    assert ask_targets(alone, ["Cressida"]) == ["Cressida"]


def test_other_banks_do_not_change_ask_targets():
    def _same(question, names, surfaces=None):
        assert ask_targets(question, names, surfaces) == ask_people(question, names, surfaces)

    for case in build_cases():
        names = [write.name for write in case.writes]
        surfaces = [write.text for write in case.writes]
        _same(case.question, names, surfaces)
    for _split, lang, _name, _rel, _role, text in iter_corpus():
        roster = list(FIT_NAMES[lang]) + list(DEV_NAMES[lang]) + list(ABSENT_NAMES[lang])
        _same(text, roster)
    for row in indirect_rows():
        _same(row.question, row.names, row.surfaces)
    for bank in (TEST_BANK, DEV_BANK):
        moved = 0
        for case in build_stress(bank):
            names = [write.name for write in case.writes]
            surfaces = [write.text for write in case.writes]
            targets = ask_targets(case.question, names, surfaces)
            people = ask_people(case.question, names, surfaces)
            if case.slice != "two_names":
                assert targets == people, case.question
            elif targets != people:
                assert targets == [case.focus]
                assert set(targets) < set(people)
                moved += 1
            else:
                assert case.focus in people or targets == people
        # 18 from the dismissed-clause / unique-possessor rule, plus 6 English
        # restrictor rows (only about / asks only) that narrow to the focus.
        assert moved == 24


def test_a_closer_other_fact_does_not_answer():
    """Axis 121 is a phone call the shipped ridge agrees with.

    The other person's sentence is an exact copy of the query. Ranking and
    abstain both stay on the possessor.
    """
    from tidal.continuum import Continuum, _cfg
    from tidal.vap import VAPModel
    import torch

    hit = next(row for row in build_rows() if row.kind == "poss_hit" and row.lang == "en" and row.rel == "phone")
    miss = next(row for row in build_rows() if row.kind == "poss_miss" and row.lang == "en" and row.rel == "phone")
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
        # The tiny value dim holds a short code. The surface string is the fact.
        assert cont.remember(
            cont.embed_event(flat(), emb, surface=text),
            value, name=name, rel=rel, group=None,
        )

    cont.episodic.clear()
    # Target's sentence is far. The other person's sentence copies the query.
    store(hit.target, "phone", "target-phone", hit.slots[0].text, axis(2))
    store(hit.other, "phone", "other-phone", hit.slots[1].text, axis(121))
    q = cont.embed_event(flat(), axis(121), surface=hit.question)
    assert cont.ask(q, group=None) == "target-phone"
    _value, info = cont.query(q, group=None, mode="top1")
    assert info["abstain"] is None

    cont.episodic.clear()
    for slot in miss.slots:
        emb = axis(121) if slot.name == miss.other else axis(2)
        store(slot.name, slot.rel, "short", slot.text, emb)
    q = cont.embed_event(flat(), axis(121), surface=miss.question)
    assert cont.ask(q, group=None) is None
    _value, info = cont.query(q, group=None, mode="top1")
    assert info["abstain"] == "missing_rel"


def test_surface_does_not_import_the_two_name_bank():
    surface = Path("tidal/question_surface.py").read_text(encoding="utf-8")
    continuum = Path("tidal/continuum.py").read_text(encoding="utf-8")
    data = Path("tidal/question_two_name_dev_data.py").read_text(encoding="utf-8")
    assert "question_two_name" not in surface
    assert "question_focus_dev" not in surface
    # The frozen sentences are not copied. The restrictor is an operator class.
    assert "ask only about" not in surface
    assert "里面，只问" not in surface
    assert "which birthday is filed" not in surface
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
