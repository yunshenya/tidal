"""Decision data for the inject fix.

Train and dev teach the MLP. The fit bank chooses ``gap_tau`` only.
The held-out bank is scored once, after the gate file is committed.

No torch. Names, answer words, and full strings stay disjoint from
``tidal.subconscious_data`` (bank ``e3cdd7fd…``) and from each other.
"""
from __future__ import annotations

import hashlib
import itertools
import json

from tidal.subconscious_data import (
    DISTRACTOR_STRIDE,
    GATES,
    LEAK_STORE_N,
    LOADS,
    NEUTRAL_STORE_N,
    PRIOR_PREFIX_N32,
    REPEAT_WRITES,
)

# (domain, stored-or-safe word, foil-or-forbidden word)
_TRAIN_PAIRS = (
    ("snack", "crackers", "walnuts"),
    ("drink", "juice", "coffee"),
    ("pet", "rabbits", "snakes"),
    ("lunch", "pasta", "shrimp"),
    ("hue", "yellow", "orange"),
    ("sport", "soccer", "tennis"),
    ("route", "buses", "bikes"),
    ("crop", "peaches", "lemons"),
    ("veg", "carrots", "onions"),
    ("meat", "turkey", "salmon"),
    ("cup", "yogurt", "pickles"),
    ("spice", "garlic", "ginger"),
    ("grain", "barley", "oats"),
    ("meal", "pizza", "chili"),
    ("flock", "pigeons", "spiders"),
    ("play", "rugby", "golf"),
)
_TRAIN_NAMES = (
    "Corwin", "Delia", "Ellis", "Farrah", "Giles", "Hattie", "Inigo", "Jorie",
    "Klaus", "Leda", "Marcel", "Oswald", "Pippa", "Renata", "Silas", "Tilda",
)
_DEV_PAIRS = (
    ("snack", "puppies", "kittens"),
    ("drink", "goats", "sheep"),
    ("pet", "surfing", "skating"),
    ("lunch", "canoes", "wagons"),
    ("hue", "cookie", "syrup"),
    ("sport", "vanilla", "caramel"),
    ("route", "papaya", "lime"),
    ("crop", "apple", "banana"),
)
_DEV_NAMES = (
    "Bramley", "Cassian", "Elspeth", "Fenton", "Gresham", "Hadrian", "Ismay", "Jolene",
)
_TRAIN_PROMPTS = (
    "Pick one {domain} for {name}.\nAnswer:",
    "Select one {domain} for {name}.\nAnswer:",
    "Offer a {domain} to {name}.\nFinal:",
)
_AVOID_TEXTS = (
    "{name} is allergic to {answer}.",
    "{name} is afraid of {answer}.",
    "{name} cannot have {answer}.",
)

# Fit bank. Prompt and fact wording stay off the frozen bank's template.
_FIT_PAIRS = (
    ("beast", "horses", "cows", "cotton"),
    ("fabric", "wool", "silk", "silver"),
    ("textile", "velvet", "canvas", "green"),
    ("alley", "bowling", "boxing", "violet"),
    ("trail", "hiking", "rowing", "indigo"),
    ("arena", "wrestling", "fencing", "bronze"),
    ("kernel", "rice", "stew", "olive"),
    ("herb", "cloves", "beets", "brick"),
)
_FIT_NAMES = (
    "Kendrick", "Lavender", "Merrick", "Noland", "Orlaith", "Paget", "Quinlan", "Roscoe",
)
_FIT_PROMPT = "Select one {domain} for {name}.\nAnswer:"
_FIT_PREFER_TEXT = "{name} would rather pick {answer} as a {domain}."
_FIT_AVOID_TEXT = "{name} steers clear of {answer}."

_HELD_PAIRS = (
    ("sport", "geese", "ducks", "beetles"),
    ("cloth", "parrots", "turtles", "crickets"),
    ("vessel", "cycling", "skiing", "frogs"),
    ("toy", "mustard", "olives", "bats"),
    ("bloom", "lentils", "broccoli", "mice"),
    ("metal", "sandwich", "sushi", "rats"),
    ("dance", "tea", "berry", "toads"),
    ("craft", "golden", "scarlet", "lizards"),
)
_HELD_NAMES = (
    "Talbot", "Winslow", "Yarden", "Zarek", "Ameline", "Bexley", "Corliss", "Dunstan",
)
_HELD_PROMPT = "Offer a {domain} to {name}.\nFinal:"
_HELD_PREFER_TEXT = "{name} would rather receive {answer} as a {domain}."
_HELD_AVOID_TEXT = "{name} is allergic to {answer}."

_NAME_A = ("br", "cl", "dr", "fr", "gl", "kr", "pl", "sk", "tr", "vl")
_NAME_B = ("an", "el", "in", "or", "un")
_NAME_C = ("wick", "low", "dell", "ford", "mont", "well", "crest", "lan")

_FIT_OBJ_A = ("brass", "cork", "drift", "flint", "gorse", "hazel", "iron", "jute")
_FIT_OBJ_B = ("pin", "lug", "hook", "wedge", "bolt", "clamp", "rod", "cap")
_HELD_OBJ_A = ("larch", "mossy", "oaken", "piney", "reedy", "slate", "thorn", "viney")
_HELD_OBJ_B = ("knob", "stud", "nail", "plug", "post", "band", "coil", "disk")

# Rel labels stay stable. The surface verb rotates inside a frame so eight
# neighboring facts do not share a key and collapse under cosine overwrite.
_FIT_RELS = ("tucked", "wrapped", "hung", "sealed")
_HELD_RELS = ("buried", "tied", "slid", "propped")
_FIT_VERBS = ("hid", "sold", "found", "moved", "lost", "gave", "took", "sent")
_HELD_VERBS = ("dropped", "carried", "washed", "counted", "lifted", "turned", "rolled", "marked")

_FIT_CHATS = (
    ("lantern", "porch"), ("pebble", "step"), ("candle", "shelf"), ("basket", "nook"),
    ("pillow", "couch"), ("hammer", "bench"), ("button", "drawer"), ("pencil", "tray"),
    ("marble", "bowl"), ("ribbon", "box"), ("bucket", "yard"), ("sponge", "sink"),
    ("napkin", "table"), ("compass", "desk"), ("whistle", "hook"), ("wallet", "coat"),
    ("ticket", "pocket"), ("mirror", "hall"), ("blanket", "chair"), ("shovel", "shed"),
    ("ladder", "wall"), ("cushion", "seat"), ("bottle", "crate"), ("spool", "sill"),
    ("cleat", "path"), ("davit", "jar"), ("binnacle", "dock"), ("fathom", "pail"),
    ("gimbal", "lap"), ("keelson", "map"), ("thwart", "cord"), ("toggle", "locker"),
)
_HELD_CHATS = (
    ("anchor", "cellar"), ("barrel", "hearth"), ("cinder", "ramp"), ("dolly", "banner"),
    ("emblem", "cask"), ("funnel", "hatch"), ("goblet", "cloak"), ("hinge", "vase"),
    ("insect", "canoe"), ("jewel", "skiff"), ("lever", "rack"), ("mantle", "tide"),
    ("orchid", "hamper"), ("paddle", "forge"), ("quartz", "cliff"), ("rudder", "lane"),
    ("saddle", "stall"), ("urchin", "loft"), ("wicker", "pipe"), ("yarrow", "berth"),
    ("zipper", "quay"), ("anvil", "bilge"), ("beacon", "spar"), ("cobble", "clew"),
    ("draper", "scow"), ("easel", "winch"), ("gasket", "tiller"), ("marlin", "pulpit"),
    ("cairn", "transom"), ("kedge", "gunwale"), ("rowlock", "lazaret"), ("fid", "hawsepipe"),
)


def _names(start: int, count: int) -> tuple:
    pool = [f"{a}{b}{c}".capitalize() for a, b, c in itertools.product(_NAME_A, _NAME_B, _NAME_C)]
    return tuple(pool[start : start + count])


def _objects(prefixes: tuple, suffixes: tuple) -> tuple:
    rows = []
    for i, suffix in enumerate(suffixes):
        for prefix in prefixes:
            rows.append(prefix + suffix)
    # 8*8 = 64, stable order.
    del i
    if len(rows) != 64 or len(set(rows)) != 64:
        raise RuntimeError("object list must be 64 unique strings")
    return tuple(rows)


def _pair_rows(names, pairs, *, split: str, prompts, avoid_texts, swap: bool = False) -> list:
    """One prefer fact and one avoid fact per pair.

    ``swap`` stores the other word in the same frame and flips the
    continuation. A word is then the preferred answer under one key and the
    forbidden answer under the other, so the value alone cannot pick the target.
    """
    if len(names) != len(pairs):
        raise ValueError("one name per pair")
    rows = []
    for i, (domain, good, bad) in enumerate(pairs):
        prompt = prompts[i % len(prompts)].format(domain=domain, name=names[i])
        stored_prefer = bad if swap else good
        foil_prefer = good if swap else bad
        forbidden = good if swap else bad
        safe = bad if swap else good
        prefer = {
            "split": split,
            "family": "prefer",
            "name": names[i],
            "rel": domain + ("-swap" if swap else ""),
            "domain": domain,
            "text": f"{names[i]}'s favorite {domain} is {stored_prefer}.",
            "a": stored_prefer,
            "q": prompt,
            "good": f" {stored_prefer}",
            "bad": f" {foil_prefer}",
        }
        avoid = {
            "split": split,
            "family": "avoid",
            "name": names[i],
            "rel": domain + ("-no-swap" if swap else "-no"),
            "domain": domain,
            "text": avoid_texts[i % len(avoid_texts)].format(name=names[i], answer=forbidden),
            "a": forbidden,
            "q": prompt,
            "good": f" {safe}",
            "bad": f" {forbidden}",
        }
        if forbidden not in avoid["text"] or stored_prefer not in prefer["text"]:
            raise ValueError("answer missing from fact text")
        rows.append(prefer)
        rows.append(avoid)
    return rows


def train_rows() -> list:
    shared = dict(split="train", prompts=_TRAIN_PROMPTS, avoid_texts=_AVOID_TEXTS)
    return _pair_rows(_TRAIN_NAMES, _TRAIN_PAIRS, swap=False, **shared) + _pair_rows(
        _TRAIN_NAMES, _TRAIN_PAIRS, swap=True, **shared
    )


def dev_rows() -> list:
    return _pair_rows(
        _DEV_NAMES,
        _DEV_PAIRS,
        split="dev",
        prompts=("Pick one {domain} for {name}.\nAnswer:",),
        avoid_texts=_AVOID_TEXTS,
    )


def _decision(names, pairs, *, family_prefix, prompt_fmt, prefer_fmt, avoid_fmt) -> tuple:
    prefer, avoid = [], []
    for i, (domain, good, bad, update) in enumerate(pairs):
        name = names[i]
        prompt = prompt_fmt.format(domain=domain, name=name)
        pid = f"{family_prefix}-{name.lower()}"
        prefer.append(
            {
                "id": f"prefer-{pid}",
                "family": "prefer",
                "name": name,
                "rel": domain,
                "domain": domain,
                "answer": good,
                "other": bad,
                "text": prefer_fmt.format(name=name, domain=domain, answer=good),
                "prompt": prompt,
                "good": f" {good}",
                "bad": f" {bad}",
                "update_answer": update,
                "update_text": prefer_fmt.format(name=name, domain=domain, answer=update),
                "update_good": f" {update}",
                "update_bad": f" {good}",
            }
        )
        avoid.append(
            {
                "id": f"avoid-{pid}",
                "family": "avoid",
                "name": name,
                "rel": domain + "-no",
                "domain": domain,
                "answer": bad,
                "other": good,
                "text": avoid_fmt.format(name=name, answer=bad),
                "prompt": prompt,
                "good": f" {good}",
                "bad": f" {bad}",
            }
        )
    return tuple(prefer), tuple(avoid)


def _incidental(names, objects, rels, verbs) -> list:
    rows = []
    for rel in rels:
        for i, (name, obj) in enumerate(zip(names, objects)):
            verb = verbs[i % len(verbs)]
            text = f"{name} {verb} the {obj}."
            if obj not in text or verb not in text:
                raise RuntimeError("object missing from incidental text")
            rows.append({"name": name, "rel": rel, "answer": obj, "text": text, "kind": "fact"})
    return rows


def _chats(pairs, template: str) -> tuple:
    rows = []
    seen = set()
    for noun, place in pairs:
        text = template.format(noun=noun, place=place)
        if noun not in text or text in seen:
            raise RuntimeError("chat line is missing its noun or is repeated")
        seen.add(text)
        rows.append((text, noun))
    return tuple(rows)


def _payload(bank: dict) -> dict:
    body = {
        "prefer": list(bank["PREFER"]),
        "avoid": list(bank["AVOID"]),
        "incidental": list(bank["INCIDENTAL"]),
        "chats": [{"text": text, "answer": answer} for text, answer in bank["CHATS"]],
        "neutral_nll": list(bank["NEUTRAL_NLL"]),
        "neutral_choice": list(bank["NEUTRAL_CHOICE"]),
        "neutral_gen": list(bank["NEUTRAL_GEN"]),
        "phone": bank["PHONE_FACT"],
        "birthday": bank["BIRTHDAY_FACT"],
        "phone_question": bank["PHONE_QUESTION"],
        "birthday_question": bank["BIRTHDAY_QUESTION"],
        "loads": list(LOADS),
        "neutral_store_n": NEUTRAL_STORE_N,
        "leak_store_n": LEAK_STORE_N,
        "repeat_writes": REPEAT_WRITES,
        "distractor_stride": DISTRACTOR_STRIDE,
    }
    return body


def bank_sha256(bank: dict) -> str:
    blob = json.dumps(_payload(bank), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _make(kind: str) -> dict:
    if kind == "fit":
        prefer, avoid = _decision(
            _FIT_NAMES,
            _FIT_PAIRS,
            family_prefix="fit",
            prompt_fmt=_FIT_PROMPT,
            prefer_fmt=_FIT_PREFER_TEXT,
            avoid_fmt=_FIT_AVOID_TEXT,
        )
        names = _names(0, 64)
        objects = _objects(_FIT_OBJ_A, _FIT_OBJ_B)
        rels, verbs = _FIT_RELS, _FIT_VERBS
        chats = _chats(_FIT_CHATS, "The {noun} rested on the {place}.")
        nll = (
            {"id": "fit-sky", "prompt": "Birds fly across the", "continuation": " sky"},
            {"id": "fit-window", "prompt": "Unlatch the nearest", "continuation": " window"},
            {"id": "fit-cold", "prompt": "The opposite of hot is", "continuation": " cold"},
            {"id": "fit-sides", "prompt": "A triangle has three", "continuation": " sides"},
            {"id": "fit-white", "prompt": "Snow is usually", "continuation": " white"},
            {"id": "fit-story", "prompt": "We read a short", "continuation": " story"},
            {"id": "fit-sea", "prompt": "Fish swim in the", "continuation": " sea"},
            {"id": "fit-noon", "prompt": "Halfway through the day is", "continuation": " twelve"},
            {"id": "fit-oven", "prompt": "Bread is baked in an", "continuation": " oven"},
            {"id": "fit-kitten", "prompt": "A baby cat is a", "continuation": " kitten"},
            {"id": "fit-autumn", "prompt": "Leaves fall in the", "continuation": " autumn"},
            {"id": "fit-dusk", "prompt": "Stars come out at", "continuation": " dusk"},
        )
        choice = (
            {"id": "fit-six", "prompt": "Three plus three equals\nChoice:", "good": " six", "bad": " eight"},
            {"id": "fit-cold2", "prompt": "Ice feels\nChoice:", "good": " cold", "bad": " hot"},
            {"id": "fit-months", "prompt": "A year has twelve\nChoice:", "good": " months", "bad": " hours"},
            {"id": "fit-quiet", "prompt": "Libraries are usually\nChoice:", "good": " quiet", "bad": " spicy"},
            {"id": "fit-swim", "prompt": "Fish can\nChoice:", "good": " swim", "bad": " vote"},
            {"id": "fit-sunday", "prompt": "Saturday comes before\nChoice:", "good": " Sunday", "bad": " March"},
            {"id": "fit-hands", "prompt": "Gloves go on your\nChoice:", "good": " hands", "bad": " ears"},
            {"id": "fit-rain", "prompt": "Clouds can bring\nChoice:", "good": " rain", "bad": " sand"},
        )
        gen = (
            "Describe a calm morning.\nChoice:",
            "Name a tool for writing.\nChoice:",
            "What grows in a garden bed?\nChoice:",
            "How does a river move?\nChoice:",
            "Suggest a place to sit.\nChoice:",
            "What holds water in a kitchen?\nChoice:",
            "Name a color in a sunset.\nChoice:",
            "What sound does rain make?\nChoice:",
        )
        phone = {
            "id": "fit-phone",
            "name": "Paget",
            "rel": "phone",
            "answer": "8082727171",
            "text": "Paget phone 8082727171.",
        }
        birthday = {
            "id": "fit-birthday",
            "name": "Paget",
            "rel": "birthday",
            "answer": "June 9, 1988",
            "text": "Paget birthday is June 9, 1988.",
        }
        phone_q = "Phone for Paget?"
        birthday_q = "Birthday for Paget?"
    elif kind == "held":
        prefer, avoid = _decision(
            _HELD_NAMES,
            _HELD_PAIRS,
            family_prefix="held",
            prompt_fmt=_HELD_PROMPT,
            prefer_fmt=_HELD_PREFER_TEXT,
            avoid_fmt=_HELD_AVOID_TEXT,
        )
        names = _names(128, 64)
        objects = _objects(_HELD_OBJ_A, _HELD_OBJ_B)
        rels, verbs = _HELD_RELS, _HELD_VERBS
        chats = _chats(_HELD_CHATS, "Someone shifted the {noun} toward that {place}.")
        nll = (
            {"id": "held-road", "prompt": "Cars drive on the", "continuation": " road"},
            {"id": "held-blue", "prompt": "The clear sky looks", "continuation": " gray"},
            {"id": "held-warm", "prompt": "Soup should be served", "continuation": " warm"},
            {"id": "held-round", "prompt": "A ball is usually", "continuation": " round"},
            {"id": "held-paper", "prompt": "Letters are written on", "continuation": " paper"},
            {"id": "held-sleep", "prompt": "People sleep in a", "continuation": " bed"},
            {"id": "held-fast", "prompt": "Rabbits can run very", "continuation": " fast"},
            {"id": "held-green", "prompt": "Grass is often", "continuation": " brown"},
            {"id": "held-milk", "prompt": "Coffee is sometimes served with", "continuation": " cream"},
            {"id": "held-left", "prompt": "Your other hand is the", "continuation": " left"},
            {"id": "held-spring", "prompt": "Flowers bloom in the", "continuation": " spring"},
            {"id": "held-soft", "prompt": "A whisper is very", "continuation": " soft"},
        )
        choice = (
            {"id": "held-ten", "prompt": "Five plus five equals\nFinal:", "good": " ten", "bad": " three"},
            {"id": "held-wet", "prompt": "Rain makes the ground\nFinal:", "good": " damp", "bad": " arid"},
            {"id": "held-hours", "prompt": "A day has twenty four\nFinal:", "good": " minutes", "bad": " decades"},
            {"id": "held-slow", "prompt": "Turtles are usually\nFinal:", "good": " slow", "bad": " loud"},
            {"id": "held-read", "prompt": "Books are meant to be\nFinal:", "good": " read", "bad": " eaten"},
            {"id": "held-winter", "prompt": "Autumn comes before\nFinal:", "good": " winter", "bad": " June"},
            {"id": "held-head", "prompt": "A hat goes on your\nFinal:", "good": " head", "bad": " knee"},
            {"id": "held-wind", "prompt": "Kites need some\nFinal:", "good": " wind", "bad": " grit"},
        )
        gen = (
            "Count lamps along that pier.\nFinal:",
            "Where does a spare oar belong?\nFinal:",
            "Describe mud after a thaw.\nFinal:",
            "Which bird stays all winter?\nFinal:",
            "How are spare blankets folded?\nFinal:",
            "What fills a kettle before tea?\nFinal:",
            "Point to a northern constellation.\nFinal:",
            "When does the tide turn back?\nFinal:",
        )
        phone = {
            "id": "held-phone",
            "name": "Dunstan",
            "rel": "phone",
            "answer": "9091818181",
            "text": "Dunstan phone 9091818181.",
        }
        birthday = {
            "id": "held-birthday",
            "name": "Dunstan",
            "rel": "birthday",
            "answer": "April 12, 2001",
            "text": "Dunstan birthday is April 12, 2001.",
        }
        phone_q = "Read Dunstan phone aloud now?"
        birthday_q = "State Dunstan birthday next please?"
    else:
        raise ValueError(kind)
    incidental = _incidental(names, objects, rels, verbs)
    if len(incidental) != 256:
        raise RuntimeError(f"{kind} incidental {len(incidental)} != 256")
    if len({(row["name"], row["rel"]) for row in incidental}) != 256:
        raise RuntimeError(f"{kind} incidental identities collided")
    bank = {
        "PREFER": prefer,
        "AVOID": avoid,
        "INCIDENTAL": incidental,
        "CHATS": chats,
        "NEUTRAL_NLL": nll,
        "NEUTRAL_CHOICE": choice,
        "NEUTRAL_GEN": gen,
        "PHONE_FACT": phone,
        "BIRTHDAY_FACT": birthday,
        "PHONE_QUESTION": phone_q,
        "BIRTHDAY_QUESTION": birthday_q,
        "GATES": GATES,
        "LOADS": LOADS,
        "LEAK_STORE_N": LEAK_STORE_N,
        "NEUTRAL_STORE_N": NEUTRAL_STORE_N,
        "REPEAT_WRITES": REPEAT_WRITES,
        "PRIOR_PREFIX_N32": PRIOR_PREFIX_N32,
    }
    bank["sha256"] = bank_sha256(bank)
    return bank


class Bank:
    """Attribute view so ``run_eval(..., data=bank)`` can score it."""

    def __init__(self, kind: str):
        self.kind = kind
        self._bank = _make(kind)
        for key, value in self._bank.items():
            if key == "sha256":
                continue
            setattr(self, key, value)

    def decision_items(self):
        return self.PREFER + self.AVOID

    def incidental_facts(self):
        return list(self.INCIDENTAL)

    def sha256(self) -> str:
        return self._bank["sha256"]


def fit_bank() -> Bank:
    return Bank("fit")


def held_bank() -> Bank:
    return Bank("held")


def answer_words(rows) -> set:
    words = set()
    for row in rows:
        for key in ("a", "answer", "other", "update_answer"):
            if key in row and row[key]:
                words.add(str(row[key]))
        for key in ("good", "bad", "update_good", "update_bad", "continuation"):
            if key in row and isinstance(row[key], str):
                words.add(row[key].strip())
    return words
