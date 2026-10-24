"""Fit and held stores for the K ablation. No continuation margin is stored here.

Each probe is one implicit judgment. The size-32 store holds the target, 15
same-relation prose companions, and 16 same-relation code-frame companions.
Sizes 8 and 16 keep a prefix of those two lists. Companions use a different
answer from the target, so none is a surface copy. Prose companions are the
high-similarity band. Code-frame companions are the low-similarity band.
"""
from __future__ import annotations

import hashlib
import json

PROTOCOL = "subconscious-kablation-v1"
FIT_N = 8
HELD_N = 16
N_PROBES = FIT_N + HELD_N
STORE_SIZES = (8, 16, 32)
SIMILAR_N = {8: 3, 16: 7, 32: 15}
FILLER_N = {8: 4, 16: 8, 32: 16}
COMPANION_NS = (1, 3, 7)
TOP_P = 0.9
TEMPERATURES = (0.05, 0.1, 0.2)
SELECTORS = (
    {"mode": "fixed", "k": 4},
    {"mode": "top_p", "k_max": 16},
    {"mode": "top_p", "k_max": 32},
)
FAMILIES = ("tone", "choice", "preference")

TONE_POS = (
    "festival",
    "lullaby",
    "harvest",
    "garden",
    "melody",
    "sunrise",
    "kitten",
    "picnic",
    "rainbow",
    "dolphin",
    "meadow",
    "ballet",
)
TONE_NEG = (
    "shipwreck",
    "drought",
    "landslide",
    "famine",
    "funeral",
    "blizzard",
    "venom",
    "riot",
    "plague",
    "tsunami",
    "dungeon",
    "volcano",
)
CHOICE_WORDS = (
    "sampan",
    "sandbar",
    "canoe",
    "kayak",
    "paddle",
    "rudder",
    "harbor",
    "beacon",
    "compass",
    "lantern",
    "satchel",
    "violin",
    "trumpet",
    "banjo",
    "flute",
    "marble",
    "pebble",
    "granite",
    "basalt",
    "copper",
    "bronze",
    "linen",
    "velvet",
    "canvas",
)
PREF_WORDS = (
    "stretcher",
    "hammock",
    "quilt",
    "pillow",
    "blanket",
    "teapot",
    "kettle",
    "ladle",
    "skillet",
    "apricot",
    "barley",
    "cinnamon",
    "nutmeg",
    "parsley",
    "turnip",
    "badger",
    "ferret",
    "otter",
    "heron",
    "maple",
    "cedar",
    "willow",
    "spruce",
    "juniper",
)

_FAMILY_LETTER = {"tone": "t", "choice": "c", "preference": "p"}
_PROMPT_PREFIX = {"tone": "Zkabltone", "choice": "Zkablchoice", "preference": "Zkablpref"}


def _other_word(pool: tuple, index: int, slot: int, banned: str) -> str:
    shift = 0
    while shift <= len(pool):
        word = pool[(index + slot + shift) % len(pool)]
        if word != banned:
            return word
        shift += 1
    raise RuntimeError("answer pool has no companion word")


def _tone_target(index: int) -> tuple:
    if index % 2 == 0:
        word = TONE_POS[index // 2]
        return word, " warm", " harsh", TONE_POS + TONE_NEG
    word = TONE_NEG[index // 2]
    return word, " harsh", " warm", TONE_POS + TONE_NEG


def _word_target(pool: tuple, index: int) -> tuple:
    word = pool[index]
    foil = pool[(index + 1) % len(pool)]
    return word, " " + word, " " + foil, pool


def _target_spec(family: str, index: int) -> tuple:
    if family == "tone":
        return _tone_target(index)
    if family == "choice":
        return _word_target(CHOICE_WORDS, index)
    if family == "preference":
        return _word_target(PREF_WORDS, index)
    raise KeyError(family)


def _probe(family: str, index: int) -> dict:
    letter = _FAMILY_LETTER[family]
    answer, good, bad, pool = _target_spec(family, index)
    anchor = f"zkanc{letter}{index:02d}"
    rel = f"zkrel{letter}{index:02d}"
    group = f"zkgrp{letter}{index:02d}"
    name = f"Zk{letter}{index:02d}00"
    target = {
        "role": "target",
        "name": name,
        "rel": rel,
        "answer": answer,
        "text": f"{name} zkablates {answer} beside {rel} {anchor}.",
    }
    similar = []
    for slot in range(1, 16):
        other = _other_word(pool, index, slot, answer)
        nonce = f"zkn{letter}{index:02d}{slot:02d}"
        slot_name = f"Zk{letter}{index:02d}{slot:02d}"
        similar.append(
            {
                "role": "similar",
                "name": slot_name,
                "rel": rel,
                "answer": other,
                "text": f"{slot_name} zkablates {other} beside {rel} {nonce}.",
            }
        )
    fillers = []
    for slot in range(16, 32):
        other = _other_word(pool, index, slot, answer)
        nonce = f"zkn{letter}{index:02d}{slot:02d}"
        slot_name = f"Zk{letter}{index:02d}{slot:02d}"
        fillers.append(
            {
                "role": "filler",
                "name": slot_name,
                "rel": rel,
                "answer": other,
                "text": f"zkfn {nonce}() {{ zkreturn {other}; }}",
            }
        )
    return {
        "id": f"kabl-{letter}{index:02d}",
        "family": family,
        "split": "fit" if index < FIT_N else "held",
        "index": index,
        "prompt": f"{_PROMPT_PREFIX[family]} {anchor}:",
        "good": good,
        "bad": bad,
        "answer": answer,
        "rel": rel,
        "group": group,
        "anchor": anchor,
        "target": target,
        "similar": similar,
        "fillers": fillers,
    }


def all_probes() -> tuple:
    return tuple(_probe(family, index) for family in FAMILIES for index in range(N_PROBES))


def fit_probes() -> tuple:
    return tuple(probe for probe in all_probes() if probe["split"] == "fit")


def held_probes() -> tuple:
    return tuple(probe for probe in all_probes() if probe["split"] == "held")


def store_facts(probe: dict, size: int) -> list:
    """Target, then the prose prefix, then the code-frame prefix."""
    if size not in SIMILAR_N:
        raise KeyError(size)
    return [probe["target"], *probe["similar"][: SIMILAR_N[size]], *probe["fillers"][: FILLER_N[size]]]


def companion_facts(probe: dict, band: str, n: int) -> list:
    """Target plus the first ``n`` companions of one similarity band."""
    if n not in COMPANION_NS:
        raise KeyError(n)
    if band == "high":
        companions = probe["similar"][:n]
    elif band == "low":
        companions = probe["fillers"][:n]
    else:
        raise KeyError(band)
    return [probe["target"], *companions]


def _payload() -> dict:
    return {
        "protocol": PROTOCOL,
        "fit_n": FIT_N,
        "held_n": HELD_N,
        "store_sizes": list(STORE_SIZES),
        "similar_n": {str(key): value for key, value in SIMILAR_N.items()},
        "filler_n": {str(key): value for key, value in FILLER_N.items()},
        "companion_ns": list(COMPANION_NS),
        "top_p": TOP_P,
        "temperatures": list(TEMPERATURES),
        "selectors": [dict(row) for row in SELECTORS],
        "probes": list(all_probes()),
    }


def bank_sha256() -> str:
    blob = json.dumps(_payload(), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
