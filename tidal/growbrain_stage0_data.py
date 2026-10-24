"""Fit and held items for protocol ``growbrain-stage0-v1``.

Eight fit items, fifty held items. The id sets are disjoint. Implicit
families are tone, choice, and preference. Phone and birthday are explicit
strings and are not part of Cohen's ``d_z``. No torch.
"""
from __future__ import annotations

import hashlib
import json

PROTOCOL = "growbrain-stage0-v1"
FIT_N = 8
HELD_N = 50

_TONE = (
    ("bright", "bleak"),
    ("warm", "cold"),
    ("calm", "harsh"),
    ("kind", "cruel"),
    ("soft", "sharp"),
    ("clear", "dim"),
    ("gentle", "rough"),
    ("sweet", "bitter"),
    ("glad", "grim"),
    ("fair", "stark"),
    ("mild", "wild"),
    ("open", "shut"),
    ("safe", "dire"),
    ("fond", "curt"),
)
_CHOICE = (
    ("north", "south"),
    ("left", "right"),
    ("early", "late"),
    ("walk", "ride"),
    ("red", "blue"),
    ("aisle", "window"),
    ("paper", "glass"),
    ("river", "ridge"),
    ("dawn", "dusk"),
    ("inner", "outer"),
    ("upper", "lower"),
    ("first", "third"),
    ("local", "ferry"),
    ("plain", "fancy"),
)
_PREFER = (
    ("tea", "coffee"),
    ("pears", "grapes"),
    ("bread", "rice"),
    ("cats", "dogs"),
    ("lakes", "peaks"),
    ("violins", "drums"),
    ("wool", "silk"),
    ("mint", "clove"),
    ("boats", "sleds"),
    ("soups", "stews"),
    ("maps", "coins"),
    ("roses", "thorns"),
    ("honey", "brine"),
    ("quiet", "noise"),
)
_FIT_NAMES = ("Ansel", "Blythe", "Corin", "Dahlia", "Elowen", "Fletcher", "Gwyn", "Hollis")
_HELD_NAMES = (
    "Arden", "Briar", "Cassian", "Delphine", "Edric", "Faye", "Galen", "Hester",
    "Ivo", "Joss", "Keir", "Lark", "Maeve", "Nils", "Odette", "Perrin",
    "Quill", "Rowan", "Sable", "Tamsin", "Uriel", "Vesper", "Wynn", "Xyla",
    "Yael", "Zeke", "Anouk", "Bram", "Cress", "Dunlin", "Eland", "Fable",
    "Gita", "Hart", "Ines", "Jori", "Kit", "Lux", "Moss", "Nova",
    "Ori", "Pike", "Quest", "Rhett", "Sol", "Tess", "Umair", "Vale",
    "Wick", "Yarrow",
)
_FIT_PHONES = ("+1-415-555-2060",)
_HELD_PHONES = (
    "+1-415-555-4101",
    "+1-415-555-4102",
    "+1-415-555-4103",
    "+1-415-555-4104",
)
_FIT_DATES = ("April 9, 1988",)
_HELD_DATES = (
    "June 11, 1984",
    "August 2, 1977",
    "January 19, 2001",
    "October 27, 1996",
)


def _item(item_id, family, name, rel, answer, other):
    if family == "phone":
        text = f"{name}'s phone number is {answer}"
        prompt = f"{name}'s phone number is "
    elif family == "birthday":
        text = f"{name}'s birthday is {answer}"
        prompt = f"{name}'s birthday is "
    else:
        text = f"{name}'s {rel} is {answer}"
        prompt = f"{name}'s {rel} is "
    if not text.endswith(answer) or prompt + answer != text:
        raise RuntimeError(f"{item_id} prompt does not rebuild the fact")
    return {
        "id": item_id,
        "family": family,
        "name": name,
        "rel": rel,
        "text": text,
        "answer": answer,
        "prompt": prompt,
        "good": answer,
        "bad": other,
    }


def _bank(prefix, names, n_tone, n_choice, n_prefer, phones, dates):
    rows = []
    cursor = 0

    def take(count):
        nonlocal cursor
        picked = names[cursor : cursor + count]
        cursor += count
        return picked

    for name, pair in zip(take(n_tone), _TONE):
        rows.append(_item(f"{prefix}-{len(rows):03d}", "tone", name, "mood", pair[0], pair[1]))
    for name, pair in zip(take(n_choice), _CHOICE):
        rows.append(_item(f"{prefix}-{len(rows):03d}", "choice", name, "pick", pair[0], pair[1]))
    for name, pair in zip(take(n_prefer), _PREFER):
        rows.append(_item(f"{prefix}-{len(rows):03d}", "preference", name, "favorite", pair[0], pair[1]))
    for name, number in zip(take(len(phones)), phones):
        rows.append(_item(f"{prefix}-{len(rows):03d}", "phone", name, "phone", number, None))
    for name, date in zip(take(len(dates)), dates):
        rows.append(_item(f"{prefix}-{len(rows):03d}", "birthday", name, "birthday", date, None))
    if cursor != len(names):
        raise RuntimeError(f"{prefix} used {cursor} names of {len(names)}")
    return tuple(rows)


FIT = _bank("growbrain-fit", _FIT_NAMES, 2, 2, 2, _FIT_PHONES, _FIT_DATES)
HELD = _bank("growbrain-held", _HELD_NAMES, 14, 14, 14, _HELD_PHONES, _HELD_DATES)
FIT_IDS = tuple(row["id"] for row in FIT)
HELD_IDS = tuple(row["id"] for row in HELD)


def bank_sha256() -> str:
    payload = json.dumps(
        {"fit": list(FIT), "held": list(HELD), "protocol": PROTOCOL},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _check():
    if len(FIT) != FIT_N or len(HELD) != HELD_N:
        raise RuntimeError("bank size drifted")
    if set(FIT_IDS) & set(HELD_IDS):
        raise RuntimeError("fit and held ids overlap")
    if len(set(FIT_IDS)) != len(FIT_IDS) or len(set(HELD_IDS)) != len(HELD_IDS):
        raise RuntimeError("duplicate ids")
    names = [row["name"] for row in FIT + HELD]
    if len(set(names)) != len(names):
        raise RuntimeError("a name is in both banks or twice in one")
    for row in FIT + HELD:
        if row["family"] in ("phone", "birthday"):
            continue
        if row["good"] == row["bad"]:
            raise RuntimeError(f"{row['id']} has no contrast")


_check()
