"""Fit and held banks for the error-pattern measurement.

No torch. The ids, the words, and the recorded numbers are the
pre-registered probe. This module does not choose a constant.
"""
from __future__ import annotations

import hashlib

PROTOCOL = "error-pattern-v1"
FIT_N = 8
HELD_N = 50
PROMPT = "Hand {name} some {domain}.\nState:"
TEXT = "{name} circles toward {answer} about {domain}."

# Crossref on 2026-10-24. Empty abstracts stay title-only in the preregistration.
DOIS = (
    "10.1037/0278-7393.20.5.1063",
    "10.1037/0096-3445.128.2.186",
    "10.1177/1754073908100432",
    "10.1037/0278-7393.23.5.1083",
    "10.1038/nrn1764",
    "10.1146/annurev-psych-010213-115117",
    "10.1098/rstb.2019.0693",
)

# Shipped #65 record. The eval must match these and the live functions.
WALL_LEVELS = (1.0, 0.9885140203528962, 0.9548416039104165, 0.03125)
TOKEN_LEVELS = (1.0, 0.9771599684342459, 0.9117224885582168)
BOTH_LEVELS = (0.7071067811865476, 0.8312378961427878)
LIVE_INTERNAL_4 = 5.0
WAKE_PRESSURE_4 = 0.2990792952357356
RIF_COMPETITOR = 0.95
SPONTANEOUS_RATE = 0.0625

# Shipped #66 probe. Explicit report only. Not an input to d_z.
PHONE = "+1-617-555-4826"
BIRTHDAY = "April 21, 1984"
LURE_PHONE = "+1-617-555-9090"
LURE_BIRTHDAY = "June 9, 1972"
EXPECTED = {
    "phone_high": "+1-617-555-4826",
    "birthday_high": "April 21, 1984",
    "phone_low": "+1-617-555-5937",
    "birthday_low": "April 1, 1984",
    "phone_long": "+1-617-555-5937",
    "birthday_long": "April 1, 1984",
    "phone_emotion": "+1-617-555-4862",
    "birthday_emotion": "April 12, 1984",
    "phone_slip": "+1-617-555-4862",
    "birthday_slip": "April 12, 1984",
}

# One SmolLM token with a leading space. Alphabetical, disjoint, not a fit.
FIT_ANSWERS = ("actor", "alley", "anchor", "anger", "ant", "apple", "army", "artist")
FIT_FOILS = ("ash", "aunt", "autumn", "avenue", "basket", "beach", "bear", "bee")
FIT_DOMAINS = ("beetle", "bench", "berry", "bicycle", "bird", "black", "blood", "blue")
HELD_ANSWERS = (
    "boat", "bone", "book", "boot", "bowl", "brain", "brass", "bread", "bridge", "bronze",
    "brother", "brown", "bucket", "cabin", "camel", "canoe", "canyon", "castle", "chain", "chair",
    "chart", "cheese", "child", "cider", "cliff", "clock", "cloud", "coat", "coffee", "coin",
    "comedy", "comet", "copper", "cost", "cottage", "creek", "crown", "cup", "dance", "dancer",
    "deer", "desert", "desk", "dock", "doctor", "door", "drama", "drawer", "eagle", "east",
)
HELD_FOILS = (
    "elm", "evening", "famine", "farm", "fear", "fence", "fleet", "flood", "floor", "fog",
    "forest", "fork", "fox", "friend", "frost", "garden", "glacier", "globe", "glove", "goat",
    "gold", "granite", "graph", "gray", "green", "grief", "guilt", "hail", "hall", "harbor",
    "hat", "hawk", "health", "heart", "home", "honey", "honor", "hope", "house", "insect",
    "iris", "iron", "island", "jewel", "journal", "joy", "judge", "juice", "justice", "kitchen",
)
HELD_DOMAINS = (
    "knife", "ladder", "lake", "lamp", "lane", "lantern", "lawyer", "lead", "letter", "lightning",
    "lily", "lion", "lock", "map", "marble", "market", "meadow", "medal", "mercy", "mist",
    "money", "moon", "morning", "muscle", "music", "myth", "nail", "navy", "nerve", "north",
    "notebook", "novel", "oak", "ocean", "onion", "orbit", "owl", "page", "paper", "parent",
    "path", "peace", "peak", "pear", "pepper", "pig", "pilot", "pine", "plague", "planet",
)


def _items(prefix: str, name_prefix: str, answers, foils, domains) -> tuple:
    rows = []
    for index, (answer, foil, domain) in enumerate(zip(answers, foils, domains)):
        name = f"{name_prefix}{index:02d}"
        rows.append(
            {
                "id": f"{prefix}-{index:03d}",
                "name": name,
                "rel": domain,
                "domain": domain,
                "answer": answer,
                "foil": foil,
                "text": TEXT.format(name=name, answer=answer, domain=domain),
                "prompt": PROMPT.format(name=name, domain=domain),
                "good": " " + answer,
                "bad": " " + foil,
            }
        )
    return tuple(rows)


FIT_ITEMS = _items("error-fit", "Errfit", FIT_ANSWERS, FIT_FOILS, FIT_DOMAINS)
HELD_ITEMS = _items("error-held", "Errheld", HELD_ANSWERS, HELD_FOILS, HELD_DOMAINS)
FIT_IDS = tuple(item["id"] for item in FIT_ITEMS)
HELD_IDS = tuple(item["id"] for item in HELD_ITEMS)


def content_words() -> set:
    raw = set(FIT_ANSWERS) | set(FIT_FOILS) | set(FIT_DOMAINS) | set(HELD_ANSWERS) | set(HELD_FOILS) | set(HELD_DOMAINS)
    raw |= {item["name"].lower() for item in FIT_ITEMS + HELD_ITEMS}
    return raw


def bank_sha256() -> str:
    lines = [PROTOCOL, PROMPT, TEXT, *DOIS, *FIT_IDS, *HELD_IDS]
    for item in FIT_ITEMS + HELD_ITEMS:
        lines.append(
            "|".join(
                (
                    item["id"],
                    item["name"],
                    item["domain"],
                    item["answer"],
                    item["foil"],
                    item["good"],
                    item["bad"],
                    item["prompt"],
                    item["text"],
                )
            )
        )
    for key in sorted(EXPECTED):
        lines.append(key + "=" + EXPECTED[key])
    lines.append(",".join(str(item) for item in WALL_LEVELS))
    lines.append(",".join(str(item) for item in TOKEN_LEVELS))
    lines.append(",".join(str(item) for item in BOTH_LEVELS))
    lines.append(str(RIF_COMPETITOR))
    lines.append(str(SPONTANEOUS_RATE))
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
