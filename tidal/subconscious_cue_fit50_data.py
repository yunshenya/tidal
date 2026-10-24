"""Diagnosis-only direction fit bank. No torch.

Fifty prefer items in the n=8 direction-fit template. Content words are
disjoint from the held banks and from that n=8 fit bank. This bank does not
choose theta, half-life, lambda, or any other constant.
"""
from __future__ import annotations

import hashlib
import json
import re

PROTOCOL = "cue-direction-fit50-v1"
N = 50
PROMPT = "Hand {name} some {domain}.\nState:"
TEXT = "{name} circles toward {answer} about {domain}."

NAMES = tuple(f"Quelva{index:02d}" for index in range(N))
DOMAINS = (
    "cedar", "maple", "birch", "alder", "aspen", "willow", "cypress", "sequoia",
    "juniper", "fir", "larch", "yew", "holly", "myrtle", "olive", "fig",
    "quince", "plum", "mango", "guava", "melon", "oats", "millet", "sorghum",
    "flax", "hemp", "linen", "canvas", "denim", "velvet", "satin", "wool",
    "silk", "cotton", "twine", "spindle", "loom", "shuttle", "bobbin", "thimble",
    "needle", "awl", "mallet", "bellows", "forge", "kiln", "mortar", "pestle",
    "ladle", "griddle",
)
ANSWERS = (
    "kettle", "teapot", "saucer", "goblet", "flagon", "cruet", "decanter", "pitcher",
    "carafe", "compass", "sextant", "astrolabe", "hourglass", "metronome", "fiddle",
    "banjo", "zither", "harp", "clarinet", "piccolo", "flute", "cornet", "trombone",
    "tuba", "horn", "cymbal", "tambourine", "marimba", "xylophone", "vibraphone",
    "gong", "chime", "organ", "piano", "clavichord", "harpsichord", "accordion",
    "concertina", "badger", "mink", "weasel", "wolverine", "marmot", "mole",
    "shrew", "hedgehog", "porcupine", "chipmunk", "raccoon", "skunk",
)
FOILS = (
    "armadillo", "pangolin", "okapi", "gazelle", "impala", "kudu", "eland", "oryx",
    "chamois", "alpaca", "vicuna", "dromedary", "yak", "buffalo", "wildebeest",
    "quagga", "onager", "dingo", "hyena", "civet", "mongoose", "meerkat", "tarsier",
    "loris", "galago", "capuchin", "macaque", "baboon", "mandrill", "gibbon",
    "orangutan", "gorilla", "penguin", "puffin", "auk", "murre", "guillemot",
    "albatross", "shearwater", "fulmar", "gannet", "pelican", "egret", "stork",
    "ibis", "spoonbill", "flamingo", "crane", "coot", "loon",
)


def _items():
    rows = []
    for index, (name, domain, answer, foil) in enumerate(zip(NAMES, DOMAINS, ANSWERS, FOILS)):
        rows.append(
            {
                "id": f"cue-fit50-prefer-{index}",
                "family": "prefer",
                "name": name,
                "rel": domain,
                "domain": domain,
                "answer": answer,
                "other": foil,
                "text": TEXT.format(name=name, answer=answer, domain=domain),
                "prompt": PROMPT.format(name=name, domain=domain),
                "good": " " + answer,
                "bad": " " + foil,
            }
        )
    return tuple(rows)


PREFER = _items()
_WORD = re.compile(r"[a-z0-9]+")


def _harvest(obj, bag, depth=0) -> None:
    if depth > 8:
        return
    if isinstance(obj, str):
        bag.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            _harvest(value, bag, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            _harvest(value, bag, depth + 1)


def _held_tokens() -> set:
    from tidal.subconscious_circadian_data import held_bank as clock_held
    from tidal.subconscious_cue_fresh_data import fresh_bank
    from tidal.subconscious_direction_data import fit_bank, held_bank
    from tidal.subconscious_gate_data import held_bank as gate_held
    from tidal.subconscious_measure_data import (
        RifMeasure,
        avoid_items,
        away_items,
        birthday_items,
        choice_items,
        phone_items,
        prefer_items,
        preference_items,
        tone_items,
        toward_items,
    )
    from tidal.subconscious_replay_data import held_bank as replay_held
    from tidal.subconscious_rif_retry_data import held_bank as rif_held
    from tidal.subconscious_spell_data import held_bank as spell_held
    from tidal.subconscious_surprise_rel_data import held_bank as surprise_held
    from tidal.subconscious_valence_data import held_bank as valence_held

    bag = []
    for bank in (
        held_bank(),
        fit_bank(),
        fresh_bank(),
        valence_held(),
        replay_held(),
        spell_held(),
        clock_held(),
        surprise_held(),
        rif_held(),
        gate_held(),
    ):
        _harvest(bank, bag)
    for rows in (
        prefer_items(),
        avoid_items(),
        toward_items(),
        away_items(),
        choice_items(),
        tone_items(),
        preference_items(),
        phone_items(),
        birthday_items(),
        RifMeasure.QUARTETS,
    ):
        _harvest(rows, bag)
    return set(_WORD.findall("\n".join(bag).lower()))


def content_words() -> set:
    return set(NAMES) | set(DOMAINS) | set(ANSWERS) | set(FOILS) | {name.lower() for name in NAMES}


def assert_disjoint() -> None:
    """Content words do not occur in the held banks or the n=8 fit bank."""
    if len(PREFER) < 50:
        raise RuntimeError("fit50 bank is below 50")
    if len(set(DOMAINS)) != 50 or len(set(ANSWERS)) != 50 or len(set(FOILS)) != 50:
        raise RuntimeError("fit50 content words are not unique")
    shared = sorted(content_words() & _held_tokens())
    if shared:
        raise RuntimeError("fit50 bank overlaps held or the n=8 fit bank: " + ", ".join(shared[:12]))


def bank_sha256() -> str:
    payload = {"protocol": PROTOCOL, "prefer": list(PREFER)}
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class Fit50Bank:
    PROTOCOL = PROTOCOL
    PREFER = PREFER

    def sha256(self) -> str:
        return bank_sha256()


def fit50_bank() -> Fit50Bank:
    assert_disjoint()
    return Fit50Bank()
