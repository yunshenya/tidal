"""One-character typos versus absent near-duplicates.

The split records both labels on one edit profile. No name rule is fit on
it. The frozen adversarial test is scored as it ships, and it is not an
input to the bank.
"""
from collections import defaultdict
from pathlib import Path

import numpy as np
from pypinyin import lazy_pinyin

from tidal.question_abstain_dev_data import BANK as DEV_BANK
from tidal.question_adversarial import load_table
from tidal.question_adversarial_data import BANK as TEST_BANK
from tidal.question_cjk_dev_data import NAMES as CJK_NAMES
from tidal.question_cjk_dev_data import build_rows as cjk_rows
from tidal.question_heldout_data import ABSENT, PEOPLE, _tokens, build_cases as probe_cases, probe_strings
from tidal.question_name_align import (
    _chars,
    _extension,
    _lev,
    best_name,
    matched_people,
    propose_spans,
)
from tidal.question_nickname_dev_data import build_rows as nick_rows
from tidal.question_stress import build_cases, case_strings
from tidal.question_surface import (
    ABSENT_NAMES,
    DEV_NAMES,
    FIT_NAMES,
    STORED_DIM,
    corpus_strings,
    load_weights,
    name_in_text,
    surface_abstain,
)
from tidal.question_typo_dev_data import build_rows, kind_counts, label_counts, questions_sha256

# sha256 of the question list only. A wording change moves this.
SPLIT_SHA = "68260c01d5f130c46285348f1ef9886f14e9aa2b47f5606c9357ecdc521f5f99"
_ROWS = ("qwertyuiop", "asdfghjkl", "zxcvbnm")
_POS = {ch: (r, c) for r, row in enumerate(_ROWS) for c, ch in enumerate(row)}


def _adjacent(a: str, b: str) -> bool:
    """QWERTY neighbors, including diagonals. y and i are two columns apart."""
    if a not in _POS or b not in _POS or a == b:
        return False
    ra, ca = _POS[a]
    rb, cb = _POS[b]
    return abs(ra - rb) <= 1 and abs(ca - cb) <= 1


def _sub_pair(surface: str, name: str):
    """Unordered pair of the one substituted character, or None."""
    left, right = _chars(surface), _chars(name)
    if len(left) != len(right) or _lev(left, right) != 1:
        return None
    diffs = [(a, b) for a, b in zip(left, right) if a != b]
    if len(diffs) != 1:
        return None
    a, b = diffs[0]
    return tuple(sorted((a, b)))


def _pinyin(text: str) -> str:
    if not any("\u4e00" <= ch <= "\u9fff" for ch in text):
        return ""
    return "".join(lazy_pinyin(text))


def _edit1(surface: str, roster):
    """Stored names at distance 1 that are not a longer copy of the name."""
    found = []
    for name in roster:
        if _lev(_chars(surface), _chars(name)) == 1 and not _extension(surface, name):
            found.append(name)
    return tuple(found)


def _second(surface: str, roster) -> int:
    dists = sorted(_lev(_chars(surface), _chars(name)) for name in roster)
    return dists[1] if len(dists) > 1 else 99


def _reason(question, writes, query=None):
    weights = load_weights()
    names = [row.name for row in writes]
    rels = [row.rel for row in writes]
    keys = np.zeros((len(names), STORED_DIM), dtype=np.float64)
    if query is None:
        query = np.zeros(STORED_DIM, dtype=np.float64)
        query[0] = 1.0
    got = surface_abstain(
        np.asarray(query, dtype=np.float64), keys, rels, names, weights, query_text=question,
    )
    return got or ""


def _unique(surface, roster, *, subs=False, adj=False, homo=False, deny=frozenset(), min_second=None):
    """The store-context gates that were measured. Not a matcher."""
    cands = []
    for name in _edit1(surface, roster):
        if subs or adj or homo:
            pair = _sub_pair(surface, name)
            if pair is None:
                continue
            if adj and not _adjacent(*pair):
                continue
            if homo and _pinyin(surface) != _pinyin(name):
                continue
            if pair in deny:
                continue
        cands.append(name)
    if len(cands) != 1:
        return None
    if min_second is not None and _second(surface, roster) < min_second:
        return None
    return cands[0]


def test_bank_is_the_split_and_both_labels_share_one_profile():
    rows = build_rows()
    assert questions_sha256(rows) == SPLIT_SHA
    assert len(rows) == 68
    assert kind_counts(rows) == {
        "collision_adj": 6,
        "collision_mn": 6,
        "collision_vowel": 6,
        "collision_del": 6,
        "collision_homo": 8,
        "collision_diffpy": 8,
        "tie": 7,
        "hyphen": 3,
        "space": 3,
        "extension": 3,
        "prefix": 4,
        "middle": 4,
        "superstring": 4,
    }
    assert label_counts(rows) == {"match": 20, "abstain": 48}
    groups = defaultdict(list)
    for row in rows:
        if not row.kind.startswith("collision"):
            continue
        groups[(row.kind, row.surface, row.roster)].append(row)
    assert len(groups) == 6
    for (kind, surface, roster), items in groups.items():
        assert {row.label for row in items} == {"match", "abstain"}
        assert len({_edit1(row.surface, row.roster) for row in items}) == 1
        assert len(_edit1(surface, roster)) == 1
        closers = {row.closer for row in items}
        if items[0].lang == "en":
            assert closers == {"slip", "ledger", "chit"}
        else:
            assert closers == {"shi", "dehua", "comma", "ne"}
        if kind == "collision_del":
            assert _sub_pair(surface, _edit1(surface, roster)[0]) is None
        elif kind == "collision_homo":
            assert _pinyin(surface) == _pinyin(_edit1(surface, roster)[0])
        elif kind == "collision_diffpy":
            assert _pinyin(surface) != _pinyin(_edit1(surface, roster)[0])
        elif kind == "collision_adj":
            pair = _sub_pair(surface, _edit1(surface, roster)[0])
            assert pair == ("d", "s") and _adjacent(*pair)
        elif kind == "collision_mn":
            pair = _sub_pair(surface, _edit1(surface, roster)[0])
            assert pair == ("m", "n") and _adjacent(*pair)
        elif kind == "collision_vowel":
            pair = _sub_pair(surface, _edit1(surface, roster)[0])
            assert pair == ("a", "e") and not _adjacent(*pair)
    # A tie is two stored names at distance 1. An extension is a longer span.
    ties = [row for row in rows if row.kind == "tie"]
    assert ties and all(len(_edit1(row.surface, row.roster)) == 2 for row in ties)
    longer = [row for row in rows if row.kind in ("extension", "hyphen", "space", "prefix", "middle", "superstring")]
    assert longer and all(
        any(_extension(row.surface, name) for name in row.roster) for row in longer
    )
    assert all(_edit1(row.surface, row.roster) == () for row in longer)


def test_shipped_matcher_abstains_on_both_labels():
    for row in build_rows():
        assert matched_people(row.question, row.roster) == []
        assert _reason(row.question, _writes(row)) == "absent_person"
    assert best_name("Hollid", ["Hollis", "Quillon", "Vesper"]) is None
    assert best_name("Branwell", ["Bramwell", "Isolde", "Lorcan"]) is None
    assert best_name("Bramwall", ["Bramwell", "Barnaby", "Romilly"]) is None
    assert best_name("Pembrok", ["Pembroke", "Marlow", "Quillon"]) is None
    assert best_name("阮峥", ["阮铮", "庞岚", "闵廷"]) is None
    assert best_name("庞琨", ["庞岚", "闵廷", "鲍麒"]) is None
    assert best_name("Hollid", ["Hollis", "Hollin", "Quillon"]) is None
    assert best_name("Bramwello", ["Bramwell"]) is None
    assert best_name("Hollis-Jane", ["Hollis"]) is None
    assert best_name("Hollis Jane", ["Hollis"]) is None
    assert best_name("老阮铮", ["阮铮"]) is None
    assert best_name("阮小铮", ["阮铮"]) is None
    assert best_name("阮铮然", ["阮铮"]) is None


def _writes(row):
    from tidal.question_stress import StressWrite

    return tuple(StressWrite(f"kept {name}", f"v{i}", name, "phone") for i, name in enumerate(row.roster))


def test_no_store_context_gate_separates_the_collision():
    """Same surface, same roster, both labels. The decision is one bit."""
    rows = [row for row in build_rows() if row.kind.startswith("collision")]
    deny = _dev_both_pairs()
    gates = {
        "edit1": lambda s, r: _unique(s, r),
        "sub": lambda s, r: _unique(s, r, subs=True),
        "second3": lambda s, r: _unique(s, r, subs=True, min_second=3),
        "adj": lambda s, r: _unique(s, r, adj=True),
        "adj_deny": lambda s, r: _unique(s, r, adj=True, deny=deny),
        "homo": lambda s, r: _unique(s, r, homo=True),
    }
    # Every gate misses at least one collision row. The abstain controls stay abstained.
    for name, gate in gates.items():
        wrong = [row for row in rows if not _label_ok(gate, row)]
        assert wrong, name
    # The fresh m/n pair is not on the abstain-dev denylist, and the gate still
    # matches the absent row.
    mn = [row for row in rows if row.kind == "collision_mn"]
    assert all(_unique(row.surface, row.roster, adj=True, deny=deny) == "Bramwell" for row in mn)
    adj = [row for row in rows if row.kind == "collision_adj"]
    assert all(_unique(row.surface, row.roster, adj=True, deny=deny) is None for row in adj)
    controls = [row for row in build_rows() if not row.kind.startswith("collision")]
    assert controls and all(row.label == "abstain" for row in controls)
    assert all(_unique(row.surface, row.roster) is None for row in controls)


def _label_ok(gate, row) -> bool:
    got = gate(row.surface, row.roster)
    if row.label == "match":
        return got is not None and got in _edit1(row.surface, row.roster)
    return got is None


def _dev_both_pairs():
    """Substitution pairs the abstain dev bank labels both ways."""
    both = defaultdict(set)
    dev = build_cases(DEV_BANK)
    for case in dev:
        if case.subtype not in ("typo", "edit1_sub", "cjk_edit1"):
            continue
        pair = _sub_pair(case.focus, case.canonical)
        if pair is not None:
            both[pair].add(case.label)
    pairs = {pair for pair, labels in both.items() if labels == {"answerable", "absent_person"}}
    assert pairs == {
        ("d", "s"), ("i", "u"), ("a", "u"), ("s", "z"),
        ("笛", "迪"), ("凯", "恺"), ("峰", "锋"),
    }
    # e↔a is typo-only on this bank (Imogan) and a frozen near-dup (Catherina).
    typo_only = {pair for pair, labels in both.items() if labels == {"answerable"}}
    assert ("a", "e") in typo_only
    return frozenset(pairs)


def test_split_is_disjoint_from_the_other_banks():
    questions = [row.question for row in build_rows()]
    bags = {
        "probe": probe_strings(),
        "surface": corpus_strings(),
        "dev": case_strings(build_cases(DEV_BANK)),
        "test": case_strings(build_cases(TEST_BANK)),
        "cjk": [row[-1] for row in cjk_rows()],
        "nick": [row.question for row in nick_rows()],
    }
    assert not any(question in set(texts) for texts in bags.values() for question in questions)

    def jaccard(a, b):
        left, right = _tokens(a), _tokens(b)
        if not left or not right:
            return 0.0
        return len(left & right) / len(left | right)

    best = {}
    for name, texts in bags.items():
        score = 0.0
        pair = ("", "")
        for question in questions:
            for other in texts:
                got = jaccard(question, other)
                if got > score:
                    score = got
                    pair = (question, other)
        assert score <= 0.50, (name, score, pair)
        best[name] = score
    assert best["probe"] <= 0.25
    assert best["nick"] <= 0.28
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
    for row in nick_rows():
        banned.extend(row.roster)
    roster = {name for row in build_rows() for name in row.roster}
    assert not roster & set(banned)
    for question in questions:
        for name in banned:
            assert not name_in_text(name, question), (name, question)


def test_matcher_does_not_import_the_split():
    root = Path("tidal")
    for name in (
        "question_name_align.py",
        "question_surface.py",
        "question_head.py",
        "llm_memory.py",
    ):
        assert "question_typo_dev" not in (root / name).read_text(encoding="utf-8")


def _focus_cases(bank):
    return [
        case for case in build_cases(bank)
        if case.subtype == "typo" or case.slice == "near_dup_absent"
    ]


def test_frozen_edit1_profiles_overlap_and_the_shipped_abstain_stays():
    """Baseline. These strings are not in the split bank."""
    table = load_table()
    test = build_cases(TEST_BANK)
    typos = [case for case in test if case.subtype == "typo"]
    assert len(typos) == 16
    en_near = [case for case in test if case.lang == "en" and case.slice == "near_dup_absent"]
    zh_near = [case for case in test if case.lang == "zh" and case.slice == "near_dup_absent"]
    assert len(en_near) == 16 and len(zh_near) == 14
    for case in typos + en_near + zh_near:
        assert _reason(case.question, case.writes, table[case.question]) == "absent_person"
        roster = [row.name for row in case.writes]
        if case.subtype == "typo":
            assert _edit1(case.focus, roster) == (case.canonical,)
            assert sum(1 for row in case.writes if row.name == case.canonical) == 1
            assert any(row.name == case.canonical and row.rel == case.rel and row.value in case.golds for row in case.writes)

    en_typo_second = {case.focus: _second(case.focus, [row.name for row in case.writes]) for case in typos if case.lang == "en"}
    en_sub_second = {
        case.focus: _second(case.focus, [row.name for row in case.writes])
        for case in en_near if case.subtype == "edit1_sub"
    }
    assert en_typo_second == {
        "Robart": 5, "Elizaboth": 7, "Willian": 6, "Catharine": 6,
        "Margoret": 5, "Patricis": 7, "Anm": 6, "Mary Jame": 6,
    }
    assert en_sub_second == {"Robern": 7, "Williem": 7, "Catherina": 7, "Mary Jone": 6}
    # Every near substitution is at least as close, in the second neighbor, as
    # the farthest typo is not. No cutoff accepts every typo and rejects every near-dup.
    assert min(en_typo_second.values()) <= min(en_sub_second.values())

    zh_shared = ("黄蕾", "周滔", "陈靖")
    for surface in zh_shared:
        pair = [case for case in typos + zh_near if case.focus == surface]
        assert {case.subtype for case in pair} == {"typo", "cjk_edit1"}
        for case in pair:
            roster = [row.name for row in case.writes]
            assert _edit1(case.focus, roster) == (case.canonical,)
            assert _second(case.focus, roster) == 2
            assert _pinyin(case.focus) == _pinyin(case.canonical)

    # 张疆 is the one Chinese typo whose pinyin differs. It is not a near-dup surface.
    zhang = next(case for case in typos if case.focus == "张疆")
    assert _pinyin(zhang.focus) != _pinyin(zhang.canonical)
    assert not any(case.focus == "张疆" for case in zh_near)

    def accepted(gate, cases):
        return tuple(
            case.focus for case in cases
            if gate(case.focus, [row.name for row in case.writes]) == case.canonical
        )

    edit1 = lambda s, r: _unique(s, r)
    sub = lambda s, r: _unique(s, r, subs=True)
    adj = lambda s, r: _unique(s, r, adj=True)
    deny = _dev_both_pairs()
    adj_deny = lambda s, r: _unique(s, r, adj=True, deny=deny)
    homo = lambda s, r: _unique(s, r, homo=True)
    assert accepted(edit1, typos) == tuple(case.focus for case in typos)
    assert set(accepted(edit1, en_near)) == {"Robern", "Williem", "Catherina", "Mary Jone", "Patrici"}
    assert set(accepted(edit1, zh_near)) == set(zh_shared)
    assert set(accepted(sub, en_near)) == {"Robern", "Williem", "Catherina", "Mary Jone"}
    assert set(accepted(adj, [case for case in typos if case.lang == "en"])) == {
        "Willian", "Patricis", "Anm", "Mary Jame",
    }
    assert accepted(adj, en_near) == ()
    assert accepted(adj, zh_near) == ()
    # Dropping the abstain-dev pairs leaves the frozen keyboard typos, and
    # Catherina is the dev's typo-only vowel pair used as an absent name.
    assert set(accepted(adj_deny, [case for case in typos if case.lang == "en"])) == {
        "Willian", "Patricis", "Anm", "Mary Jame",
    }
    assert _sub_pair("Catherina", "Catherine") == ("a", "e")
    assert set(accepted(homo, [case for case in typos if case.lang == "zh"])) == {
        "王玮", "李那", "刘样", "陈靖", "赵悯", "黄蕾", "周滔",
    }
    assert set(accepted(homo, zh_near)) == set(zh_shared)

    # Headline abstain, real embeddings, shipped matcher.
    fa = miss = absent = 0
    n_ans = n_miss = n_abs = 0
    for case in test:
        why = _reason(case.question, case.writes, table[case.question])
        if case.label == "answerable":
            n_ans += 1
            fa += int(bool(why))
        elif case.label == "missing_rel":
            n_miss += 1
            miss += int(why == "missing_rel")
        else:
            n_abs += 1
            absent += int(why == "absent_person")
    assert (fa, n_ans) == (39, 216)
    assert (miss, n_miss) == (53, 69)
    assert (absent, n_abs) == (42, 42)
    two = [case for case in test if case.slice == "two_names" and case.label == "absent_person"]
    assert len(two) == 12
    assert all(_reason(case.question, case.writes, table[case.question]) == "absent_person" for case in two)
    assert all(_unique(case.focus, [row.name for row in case.writes]) is None for case in two)


def test_a_deletion_gate_attaches_mei_on_one_probe_question():
    hits = []
    for case in probe_cases():
        roster = tuple(dict.fromkeys(row.name for row in case.writes if row.name))
        for _start, _end, span in propose_spans(case.question):
            if span in roster:
                continue
            found = _edit1(span, roster)
            if found:
                hits.append((span, found, case.question))
    assert hits == [("me", ("Mei",), "Where did Ada tell me to show up?")]
    assert _sub_pair("me", "Mei") is None
