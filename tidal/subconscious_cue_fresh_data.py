"""Fresh direction bank for the cue revision. No torch.

Fifty prefer items, fifty neutral-choice prompts, fifty neutral-NLL items,
and forty incidental facts. None of these strings are the direction fit bank
or the direction held bank. The gates were fixed before this bank is scored.
"""
from __future__ import annotations

import hashlib
import json

from tidal.subconscious_direction_data import fit_bank, held_bank

PROTOCOL = "cue-direction-fresh-v1"
N = 50
N_INCIDENTAL = 40


def _item(index: int) -> dict:
    name = f"Qorf{index:02d}"
    domain = f"vext{index:02d}"
    answer = f"lumen{index:02d}"
    other = f"shade{index:02d}"
    return {
        "id": f"cue-fresh-prefer-{index}",
        "family": "prefer",
        "name": name,
        "rel": domain,
        "domain": domain,
        "answer": answer,
        "other": other,
        "text": f"{name} fastens {answer} within {domain}.",
        "prompt": f"Assign {domain} for {name}.\nMark:",
        "good": " " + answer,
        "bad": " " + other,
    }


def _choice(index: int) -> dict:
    return {
        "id": f"cue-fresh-choice-{index}",
        "prompt": f"Sort glyph{index:02d}.\nMark:",
        "good": f" pave{index:02d}",
        "bad": f" mote{index:02d}",
    }


def _nll(index: int) -> dict:
    return {
        "id": f"cue-fresh-nll-{index}",
        "prompt": f"Carry wick{index:02d} toward",
        "continuation": f" brim{index:02d}",
    }


def _incidental(index: int) -> dict:
    name = f"Yorb{index:02d}"
    rel = f"wisp{index % 8:02d}"
    obj = f"cobb{index:02d}"
    place = f"pier{index:02d}"
    return {
        "name": name,
        "rel": rel,
        "answer": obj,
        "text": f"{name} parks one {obj} beside {place}.",
        "kind": "fact",
    }


PREFER = tuple(_item(index) for index in range(N))
NEUTRAL_CHOICE = tuple(_choice(index) for index in range(N))
NEUTRAL_NLL = tuple(_nll(index) for index in range(N))
INCIDENTAL = tuple(_incidental(index) for index in range(N_INCIDENTAL))


_IDENTITY = ("name", "rel", "answer", "other", "domain", "text", "prompt", "good", "bad", "continuation")


def _surface(bank) -> set:
    found = set()
    blobs = []
    blobs.extend(bank.PREFER)
    blobs.extend(bank.AVOID)
    blobs.extend(bank.NEUTRAL_CHOICE)
    blobs.extend(bank.NEUTRAL_NLL)
    blobs.extend(bank.incidental_facts())
    for row in blobs:
        for key, value in row.items():
            if key in _IDENTITY and isinstance(value, str):
                found.add(value)
                if key in ("name", "rel", "answer", "other", "domain"):
                    found.add(value.lower())
    return found


def _fresh_tokens() -> set:
    found = set()
    for row in list(PREFER) + list(NEUTRAL_CHOICE) + list(NEUTRAL_NLL) + list(INCIDENTAL):
        for key, value in row.items():
            if key in _IDENTITY and isinstance(value, str):
                found.add(value)
                if key in ("name", "rel", "answer", "other", "domain"):
                    found.add(value.lower())
    return found


def assert_disjoint() -> None:
    """The fresh strings do not reuse a fit or old-held direction string."""
    old = _surface(fit_bank()) | _surface(held_bank())
    shared = sorted(_fresh_tokens() & old)
    if shared:
        raise RuntimeError("fresh direction bank overlaps fit or held: " + ", ".join(shared[:12]))
    if len(PREFER) < 50 or len(NEUTRAL_CHOICE) < 50 or len(NEUTRAL_NLL) < 50:
        raise RuntimeError("fresh direction bank is below 50")
    if len(INCIDENTAL) < 31:
        raise RuntimeError("fresh incidental bank cannot fill a store of 32")


def bank_sha256() -> str:
    payload = {
        "protocol": PROTOCOL,
        "prefer": list(PREFER),
        "neutral_choice": list(NEUTRAL_CHOICE),
        "neutral_nll": list(NEUTRAL_NLL),
        "incidental": list(INCIDENTAL),
    }
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class FreshDirectionBank:
    PROTOCOL = PROTOCOL
    PREFER = PREFER
    NEUTRAL_CHOICE = NEUTRAL_CHOICE
    NEUTRAL_NLL = NEUTRAL_NLL

    def incidental_facts(self):
        return list(INCIDENTAL)

    def sha256(self) -> str:
        return bank_sha256()


def fresh_bank() -> FreshDirectionBank:
    assert_disjoint()
    return FreshDirectionBank()
