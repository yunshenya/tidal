"""Held-out bank for the confidence-scale gate.

No torch. Disjoint from the frozen #38 bank and from the #39 fit, held,
train, and dev surfaces. ``gap_tau`` is not chosen here.
"""
from __future__ import annotations

import hashlib
import json

from tidal.subconscious_data import (
    GATES,
    LEAK_STORE_N,
    LOADS,
    NEUTRAL_STORE_N,
    PRIOR_PREFIX_N32,
    REPEAT_WRITES,
)

PROTOCOL = "subconscious-gate-v1"

# Non-interference gates for this PR. Prefer and avoid deltas are not gates.
GATE_SPEC = {
    "nll_abs_mean_max": 0.05,
    "nll_bad": 0.5,
    "nll_bad_count_max": 1,
    "choice_flips_max": 0,
    "leak_max": 0,
}

_DECISION = (
    "Cragwick",
    "Cragwell",
    "Cragmont",
    "Cragford",
    "Cragley",
    "Cragdale",
    "Cragburn",
    "Cragshaw",
)

_PREFER_ROWS = (
    ("gate-prefer-cragwick", "Cragwick", "vessel", "raft", "canoe", "violin"),
    ("gate-prefer-cragwell", "Cragwell", "strings", "harp", "guitar", "flute"),
    ("gate-prefer-cragmont", "Cragmont", "metal", "copper", "nickel", "zinc"),
    ("gate-prefer-cragford", "Cragford", "alloy", "tin", "brass", "steel"),
    ("gate-prefer-cragley", "Cragley", "grain", "millet", "quinoa", "sorghum"),
    ("gate-prefer-cragdale", "Cragdale", "cup", "cider", "cocoa", "ale"),
    ("gate-prefer-cragburn", "Cragburn", "land", "canyon", "meadow", "prairie"),
    ("gate-prefer-cragshaw", "Cragshaw", "sky", "comet", "planet", "galaxy"),
)

_AVOID_ROWS = (
    ("gate-avoid-cragwick", "Cragwick", "prose", "novel", "poem"),
    ("gate-avoid-cragwell", "Cragwell", "page", "essay", "puzzle"),
    ("gate-avoid-cragmont", "Cragmont", "wear", "boots", "jacket"),
    ("gate-avoid-cragford", "Cragford", "rock", "granite", "oyster"),
    ("gate-avoid-cragley", "Cragley", "utensil", "spoon", "kettle"),
    ("gate-avoid-cragdale", "Cragdale", "path", "orbit", "marker"),
    ("gate-avoid-cragburn", "Cragburn", "herb", "basil", "thyme"),
    ("gate-avoid-cragshaw", "Cragshaw", "fruit", "plum", "fig"),
)

_STEMS = ("Dorn", "Elv", "Fen", "Gar", "Holt", "Jex", "Kael", "Lior")
_TAILS = ("wick", "well", "mont", "ford", "ley", "dale", "burn", "shaw")
_INCIDENTAL_NAMES = tuple(stem + tail for stem in _STEMS for tail in _TAILS)
_MATERIALS = ("amber", "cedar", "maple", "birch", "ivory", "coral", "pearl", "onyx")
_PARTS = ("coil", "hoop", "latch", "spoke", "stave", "tang", "ferrule", "gudgeon")
_OBJECTS = tuple(material + part for material in _MATERIALS for part in _PARTS)
_VERBS = ("baked", "painted", "mailed", "borrowed", "traded", "stacked", "weighed", "mended")
_PLACES = ("alcove", "belfry", "cistern", "drydock", "granary", "hayloft", "icehouse", "jetty")
_RELS = ("stowed", "cached", "parked", "lodged")

_CHAT_NOUNS = (
    "quern", "awning", "bodice", "corbel", "dormer", "lectern", "finial", "gargoyle",
    "harness", "joist", "keystone", "lintel", "mullion", "newel", "oriel", "pinnacle",
    "quoin", "rafter", "sill", "truss", "urn", "voussoir", "weathervane", "yoke",
    "ziggurat", "belfrypin", "cleatiron", "davitpin", "ensign", "kneerider", "gunwale", "hawse",
)
_CHAT_PLACES = (
    "nave", "apse", "crypt", "cloister", "undercroft", "refectory", "scriptorium", "sacristy",
    "dorter", "cellarage", "buttery", "pantry", "scullery", "larder", "dairy", "brewery",
    "foundry", "smithy", "tannery", "fullery", "malthouse", "oast", "byre", "cooperage",
    "chandlery", "ropewalk", "sail loft", "chartroom", "wardroom", "lazaret", "orlop", "forepeak",
)


def _prefer(row) -> dict:
    pid, name, domain, answer, other, update = row
    return {
        "id": pid,
        "family": "prefer",
        "name": name,
        "rel": domain,
        "domain": domain,
        "answer": answer,
        "other": other,
        "text": f"{name} keeps returning to {answer} for that {domain}.",
        "prompt": f"Name one {domain} alongside {name}.\nReply:",
        "good": f" {answer}",
        "bad": f" {other}",
        "update_answer": update,
        "update_text": f"{name} keeps returning to {update} for that {domain}.",
        "update_good": f" {update}",
        "update_bad": f" {answer}",
    }


def _avoid(row) -> dict:
    pid, name, domain, forbidden, safe = row
    text = f"{name} turns down {forbidden} during {domain}."
    return {
        "id": pid,
        "family": "avoid",
        "name": name,
        "rel": domain,
        "domain": domain,
        "answer": forbidden,
        "other": safe,
        "text": text,
        "prompt": f"Name one {domain} alongside {name}.\nReply:",
        "good": f" {safe}",
        "bad": f" {forbidden}",
    }


PREFER = tuple(_prefer(row) for row in _PREFER_ROWS)
AVOID = tuple(_avoid(row) for row in _AVOID_ROWS)

NEUTRAL_NLL = (
    {"id": "gate-nll-hills", "prompt": "Clouds drift over the", "continuation": " hills"},
    {"id": "gate-nll-river", "prompt": "The path follows the", "continuation": " river"},
    {"id": "gate-nll-stone", "prompt": "The old wall is made of", "continuation": " stone"},
    {"id": "gate-nll-bread", "prompt": "Warm loaves of", "continuation": " bread"},
    {"id": "gate-nll-slowly", "prompt": "The glacier moves", "continuation": " slowly"},
    {"id": "gate-nll-north", "prompt": "The compass needle points", "continuation": " north"},
    {"id": "gate-nll-seven", "prompt": "A rainbow has", "continuation": " seven"},
    {"id": "gate-nll-friend", "prompt": "She wrote a letter to her", "continuation": " friend"},
    {"id": "gate-nll-glass", "prompt": "The pitcher is made of", "continuation": " glass"},
    {"id": "gate-nll-smoke", "prompt": "The chimney released", "continuation": " smoke"},
    {"id": "gate-nll-forest", "prompt": "Deer live in the", "continuation": " forest"},
    {"id": "gate-nll-bridge", "prompt": "The road crosses a", "continuation": " bridge"},
)

NEUTRAL_CHOICE = (
    {"id": "gate-five", "prompt": "Eight minus three equals\nReply:", "good": " five", "bad": " ninety"},
    {"id": "gate-sixty", "prompt": "A minute contains\nReply:", "good": " sixty", "bad": " forty"},
    {"id": "gate-square", "prompt": "A box with equal sides is a\nReply:", "good": " square", "bad": " oval"},
    {"id": "gate-chilly", "prompt": "Frost on the glass feels\nReply:", "good": " chilly", "bad": " sharp"},
    {"id": "gate-down", "prompt": "A dropped coin falls\nReply:", "good": " down", "bad": " sideways"},
    {"id": "gate-oval", "prompt": "A full moon looks\nReply:", "good": " oval", "bad": " flat"},
    {"id": "gate-dusty", "prompt": "Rain leaves the street\nReply:", "good": " muddy", "bad": " dusty"},
    {"id": "gate-west", "prompt": "Sunset lies toward the\nReply:", "good": " west", "bad": " south"},
)

NEUTRAL_GEN = (
    "Describe a quiet harbor.\nReply:",
    "Tally lanterns beside the breakwater.\nReply:",
    "How is a campfire kindled?\nReply:",
    "Name something kept in a toolbox.\nReply:",
    "Where do sailors watch the tide?\nReply:",
    "Suggest a slow evening walk.\nReply:",
    "What sits on a kitchen shelf?\nReply:",
    "How does distant thunder sound?\nReply:",
)

CHATS = tuple(
    (f"A spare {noun} waited inside that {place}.", noun)
    for noun, place in zip(_CHAT_NOUNS, _CHAT_PLACES)
)

PHONE_FACT = {
    "id": "gate-phone",
    "name": "Mornwick",
    "rel": "phone",
    "answer": "+1-202-555-0174",
    "text": "Mornwick keeps a phone listed as +1-202-555-0174.",
}
BIRTHDAY_FACT = {
    "id": "gate-birthday",
    "name": "Mornwick",
    "rel": "birthday",
    "answer": "August 17, 1973",
    "text": "Mornwick was born on August 17, 1973.",
}
PHONE_QUESTION = "Which phone reaches Mornwick?"
BIRTHDAY_QUESTION = "Tell me Mornwick's birthday."


def incidental_facts() -> list:
    if len(_INCIDENTAL_NAMES) != 64 or len(_OBJECTS) != 64:
        raise RuntimeError("incidental lists must be length 64")
    rows = []
    for rel_i, rel in enumerate(_RELS):
        for index, name in enumerate(_INCIDENTAL_NAMES):
            verb = _VERBS[(index + rel_i) % len(_VERBS)]
            obj = _OBJECTS[(index + 11 * rel_i) % len(_OBJECTS)]
            place = _PLACES[(index + 3 * rel_i) % len(_PLACES)]
            text = f"{name} {verb} a {obj} near the {place}."
            rows.append(
                {
                    "name": name,
                    "rel": rel,
                    "answer": obj,
                    "text": text,
                    "kind": "fact",
                }
            )
    if len(rows) != 256:
        raise RuntimeError(f"incidental {len(rows)} != 256")
    if len({(row["name"], row["rel"]) for row in rows}) != 256:
        raise RuntimeError("incidental identities collided")
    return rows


def decision_items() -> tuple:
    return PREFER + AVOID


def bank_payload() -> dict:
    return {
        "protocol": PROTOCOL,
        "gate_spec": GATE_SPEC,
        "gates": GATES,
        "loads": list(LOADS),
        "repeat_writes": REPEAT_WRITES,
        "neutral_store_n": NEUTRAL_STORE_N,
        "leak_store_n": LEAK_STORE_N,
        "prior_prefix_n32": PRIOR_PREFIX_N32,
        "prefer": list(PREFER),
        "avoid": list(AVOID),
        "incidental": incidental_facts(),
        "chats": [{"text": text, "answer": answer} for text, answer in CHATS],
        "neutral_nll": list(NEUTRAL_NLL),
        "neutral_choice": list(NEUTRAL_CHOICE),
        "neutral_gen": list(NEUTRAL_GEN),
        "phone": PHONE_FACT,
        "birthday": BIRTHDAY_FACT,
        "phone_question": PHONE_QUESTION,
        "birthday_question": BIRTHDAY_QUESTION,
    }


def bank_sha256() -> str:
    blob = json.dumps(bank_payload(), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class GateBank:
    """Attribute view so ``run_eval(..., data=bank)`` can score this bank."""

    kind = "held"

    def __init__(self):
        self.PROTOCOL = PROTOCOL
        self.PREFER = PREFER
        self.AVOID = AVOID
        self.CHATS = CHATS
        self.GATES = GATES
        self.LEAK_STORE_N = LEAK_STORE_N
        self.LOADS = LOADS
        self.NEUTRAL_CHOICE = NEUTRAL_CHOICE
        self.NEUTRAL_GEN = NEUTRAL_GEN
        self.NEUTRAL_NLL = NEUTRAL_NLL
        self.NEUTRAL_STORE_N = NEUTRAL_STORE_N
        self.PHONE_FACT = PHONE_FACT
        self.PHONE_QUESTION = PHONE_QUESTION
        self.BIRTHDAY_FACT = BIRTHDAY_FACT
        self.BIRTHDAY_QUESTION = BIRTHDAY_QUESTION
        self.PRIOR_PREFIX_N32 = PRIOR_PREFIX_N32
        self.REPEAT_WRITES = REPEAT_WRITES
        self.GATE_SPEC = GATE_SPEC

    def decision_items(self):
        return PREFER + AVOID

    def incidental_facts(self):
        return incidental_facts()

    def sha256(self) -> str:
        return bank_sha256()


def held_bank() -> GateBank:
    return GateBank()
