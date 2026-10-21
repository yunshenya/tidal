"""Held-out facts for frozen-LM memory measurements.

This is the in-repo stand-in for the deleted neuromem ``data_v12`` generator
the SmolLM key-collapse numbers were measured with. Names and relations are
disjoint from the calibration names in ``tidal.llm_memory`` (``Sam``,
``Robin``, …). A session of distractor count ``N`` is one target fact plus
``N`` other turns. Each other turn is a fact or a chat line with equal
probability, which is why an ``N=4`` session contains about three facts.
Chat turns are not facts. Phone, birthday, and several other answers are
multi-token. ``match`` is alphanumeric containment, the same kind of check
as the old free-generation scorer.

No torch. Safe to import from tests.
"""
from __future__ import annotations

import random
import re
from typing import Optional

_TRAIN_NAMES = (
    "Ada", "Bea", "Cal", "Dee", "Ezra", "Finn", "Gia", "Hugo",
    "Ivy", "Jules", "Kai", "Lena", "Milo", "Nora", "Omar", "Pia",
    "Tara", "Uma", "Vera", "Wes", "Xena", "Yuri", "Zara", "Amos",
)
_TEST_NAMES = (
    "Nell", "Otto", "Priya", "Remy", "Soren", "Thea", "Ulric", "Veda",
    "Willa", "Yves", "Leif", "Maren", "Nico", "Orla", "Paolo", "Ruth",
    "Sami", "Tove", "Brigid", "Dorian", "Esme", "Farid", "Greta", "Hassan",
)
_RELS = ("phone", "birthday", "city", "color", "job", "team")
_QUESTION = {
    "phone": "What is {name}'s phone number?",
    "birthday": "When is {name}'s birthday?",
    "city": "Which city does {name} live in?",
    "color": "What is {name}'s favorite color?",
    "job": "What is {name}'s job?",
    "team": "Which team is {name} on?",
}
_TEXT = {
    "phone": "{name}'s phone number is {a}",
    "birthday": "{name}'s birthday is {a}",
    "city": "{name} lives in {a}",
    "color": "{name}'s favorite color is {a}",
    "job": "{name}'s job is {a}",
    "team": "{name}'s team is {a}",
}
_PARA = {
    "phone": "The phone number for {name} is {a}",
    "birthday": "The birthday of {name} is {a}",
    "city": "The city where {name} lives is {a}",
    "color": "The favorite color of {name} is {a}",
    "job": "The job held by {name} is {a}",
    "team": "The team that includes {name} is {a}",
}
_MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)
_CITIES = (
    "Lisbon", "Milan", "Oslo", "Bern", "Quito", "Tunis", "Riga", "Malmo",
    "Pune", "Lima", "Accra", "Basel", "Cadiz", "Dakar", "Essen", "Fargo",
    "Ghent", "Hilo", "Izmir", "Jena", "Kyoto", "Leeds", "Namur", "Omaha",
)
_COLORS = (
    "green", "amber", "cobalt", "ivory", "crimson", "navy",
    "olive", "silver", "violet", "teal", "copper", "indigo",
)
_JOBS = (
    "harbor pilot", "night clerk", "map binder", "choir teacher",
    "field medic", "glass blower", "line cook", "radio host",
    "seed keeper", "tour guide", "bridge welder", "copy editor",
)
_TEAMS = (
    "cedar lantern", "north harbor", "quiet orbit", "silver maple",
    "river glass", "amber yard", "little forge", "open ledger",
    "bright keel", "slow comet", "dust choir", "iron sparrow",
)
_CHAT = (
    "The kettle just boiled.",
    "I left the window open.",
    "That meeting ran long.",
    "The bus was late again.",
    "Could you pass the salt?",
    "I think it might rain.",
    "The lights in the hall flickered.",
    "Someone is at the door.",
    "I finished the chapter.",
    "The cat is on the chair.",
    "We should leave before dark.",
    "The soup needs another minute.",
)


def names(split: str) -> tuple:
    if split == "train":
        return _TRAIN_NAMES
    if split == "test":
        return _TEST_NAMES
    raise ValueError(f"split must be 'train' or 'test', got {split!r}")


def norm(text: str) -> str:
    """Lowercase letters and digits only, for containment checks."""
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def match(gen: str, answer: str) -> bool:
    """True when the normalized answer is contained in the normalized generation."""
    gold = norm(answer)
    return gold != "" and gold in norm(gen)


def _answer(rel: str, i: int) -> str:
    if rel == "phone":
        return f"+1-415-555-{1000 + (i % 8000):04d}"
    if rel == "birthday":
        return f"{_MONTHS[i % 12]} {(i % 27) + 1}, {1970 + (i % 50)}"
    if rel == "city":
        return _CITIES[i % len(_CITIES)]
    if rel == "color":
        return _COLORS[i % len(_COLORS)]
    if rel == "job":
        return _JOBS[i % len(_JOBS)]
    if rel == "team":
        return _TEAMS[i % len(_TEAMS)]
    raise KeyError(rel)


def fact(
    rng: random.Random,
    split: str,
    n: Optional[str] = None,
    rel: Optional[str] = None,
    c: Optional[str] = None,
) -> dict:
    """One fact. ``n`` / ``rel`` pin the identity. ``c`` is accepted and ignored.

    The old generator took ``c`` and could still repeat an answer, so callers
    loop on ``norm``. The same loop works here.
    """
    del c
    pool = names(split)
    name = n if n is not None else rng.choice(pool)
    relation = rel if rel is not None else rng.choice(_RELS)
    i = rng.randrange(10_000)
    answer = _answer(relation, i)
    return {
        "kind": "fact",
        "n": name,
        "rel": relation,
        "a": answer,
        "c": relation,
        "text": _TEXT[relation].format(name=name, a=answer),
        "para": _PARA[relation].format(name=name, a=answer),
        "q": _QUESTION[relation].format(name=name),
    }


def chat(rng: random.Random, split: str = "test") -> dict:
    del split
    text = rng.choice(_CHAT)
    return {"kind": "chat", "text": text, "a": "", "n": "", "rel": "", "q": ""}


def _fresh_fact(rng: random.Random, split: str, used: set, answers: set, tries: int = 80) -> dict:
    fallback = None
    last = None
    for _ in range(tries):
        last = fact(rng, split)
        if (last["n"], last["rel"]) in used:
            continue
        if fallback is None:
            fallback = last
        if norm(last["a"]) in answers:
            continue
        used.add((last["n"], last["rel"]))
        answers.add(norm(last["a"]))
        return last
    if fallback is None:
        raise RuntimeError("no unused (name, rel) left in this split")
    used.add((fallback["n"], fallback["rel"]))
    answers.add(norm(fallback["a"]))
    return fallback


def session(rng: random.Random, split: str, n_distract: int) -> list:
    """Target fact plus ``n_distract`` turns. Half of those turns are facts.

    Identities do not repeat, so a later collapse is cosine overwrite rather
    than a same-name update. Turn order is shuffled.
    """
    if int(n_distract) < 0:
        raise ValueError("n_distract must be >= 0")
    target = fact(rng, split)
    used = {(target["n"], target["rel"])}
    answers = {norm(target["a"])}
    turns = [target]
    for _ in range(int(n_distract)):
        if rng.random() < 0.5:
            turns.append(chat(rng, split))
        else:
            turns.append(_fresh_fact(rng, split, used, answers))
    rng.shuffle(turns)
    return turns


def fact_stream(rng: random.Random, split: str, k: int) -> list:
    """``k`` facts with distinct identities and distinct normalized answers."""
    if int(k) < 1:
        raise ValueError("k must be >= 1")
    cap = len(names(split)) * len(_RELS)
    if int(k) > cap:
        raise ValueError(f"only {cap} distinct identities in split {split}")
    used: set = set()
    answers: set = set()
    return [_fresh_fact(rng, split, used, answers) for _ in range(int(k))]
