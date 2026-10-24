"""Fit bank for the surface-repeat threshold. No log-prob margins.

Each case is a handful of writes. The threshold is chosen by whether a
true repeat collapses and a different fact stays. Held quartets are not
in this module.
"""
from __future__ import annotations

import hashlib
import json

PROTOCOL = "subconscious-interfere-v1"
GRID = (0.50, 0.67, 0.75, 0.85, 0.92, 1.0)

_EXACT = "qelvorn morvexil tundravo saxpenul lirvoten hendaxom."
_EXACT_CASE = "Qelvorn Morvexil Tundravo Saxpenul Lirvoten Hendaxom"
_EXTRA = "qelvorn morvexil tundravo saxpenul lirvoten hendaxom brontelu"
_SWAPPED = "qelvorn morvexil tundravo saxpenul lirvoten yivkarnox"
_OTHER = "cindrelum zolphexan wavqorin beltruny oxpindal qastuven."
_FAR = "jexwomal nurvapel skintora doldravem pixqenul habmivor."

CASES = (
    {
        "id": "exact-case",
        "rule": "merge",
        "writes": (
            {"name": "Anqelvor", "rel": "rfitexact", "text": _EXACT},
            {"name": "Benqelvor", "rel": "rfitexact", "text": _EXACT_CASE},
        ),
    },
    {
        "id": "extra-token",
        "rule": "merge",
        "writes": (
            {"name": "Cenqelvor", "rel": "rfitextra", "text": _EXACT},
            {"name": "Denqelvor", "rel": "rfitextra", "text": _EXTRA},
        ),
    },
    {
        "id": "one-word",
        "rule": "stay",
        "writes": (
            {"name": "Enqelvor", "rel": "rfitword", "text": _EXACT},
            {"name": "Fenqelvor", "rel": "rfitword", "text": _SWAPPED},
        ),
    },
    {
        "id": "disjoint-same-rel",
        "rule": "two",
        "writes": (
            {"name": "Genqelvor", "rel": "rfitfar", "text": _OTHER},
            {"name": "Henqelvor", "rel": "rfitfar", "text": _FAR},
        ),
    },
    {
        "id": "same-text-other-rel",
        "rule": "two",
        "writes": (
            {"name": "Inqelvor", "rel": "rfitleft", "text": _EXACT},
            {"name": "Jenqelvor", "rel": "rfitright", "text": _EXACT},
        ),
    },
    {
        "id": "close-keys-different-text",
        "rule": "two",
        "writes": (
            {
                "name": "Kenqelvor",
                "rel": "rfitkey",
                "text": _OTHER,
                "key": (1.0, 0.0, 0.0, 0.0),
                "value": (1.0, 0.0, 0.0, 0.0),
            },
            {
                "name": "Lenqelvor",
                "rel": "rfitkey",
                "text": _FAR,
                "key": (1.01, 0.0, 0.0, 0.0),
                "value": (0.0, 1.0, 0.0, 0.0),
            },
        ),
    },
    {
        "id": "empty-rel",
        "rule": "two",
        "writes": (
            {"name": "Menqelvor", "rel": None, "text": _EXACT},
            {"name": "Nenqelvor", "rel": None, "text": _EXACT},
        ),
    },
    {
        "id": "other-group",
        "rule": "two",
        "writes": (
            {"name": "Onqelvor", "rel": "rfitgrp", "text": _EXACT, "group": "g1"},
            {"name": "Penqelvor", "rel": "rfitgrp", "text": _EXACT, "group": "g2"},
        ),
    },
    {
        "id": "identity-update",
        "rule": "one",
        "writes": (
            {"name": "Qenqelvor", "rel": "rfitid", "text": _OTHER},
            {"name": "Qenqelvor", "rel": "rfitid", "text": _FAR},
        ),
    },
)


def bank_sha256() -> str:
    blob = json.dumps({"cases": CASES, "grid": GRID, "protocol": PROTOCOL}, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()
