"""Fit and held ids for pattern-separation v0. Numeric fixtures, not sentences.

No torch. The ids keep the banks disjoint. Every id runs the same
pre-registered fixtures, so the canonical scores do not depend on the id.
"""
from __future__ import annotations

import hashlib

import numpy as np

from tidal.episodic import cosine_scores
from tidal.pattern_separation import (
    SEPARATION_SPAN,
    SEPARATION_TAU,
    key_cosine,
    match_scores,
    separate_key,
    separate_keys,
)

PROTOCOL = "dg-pattern-separation-v0"
FIT_N = 8
HELD_N = 50
# mem_key_dim / fingerprint width. The live hook still runs at d_model.
SEPARATION_DIM = 32
GATES = (
    "identity",
    "similar",
    "confusion",
    "distinct",
    "equal",
    "write",
    "export",
    "item",
)


def _ids(prefix: str, count: int) -> tuple:
    return tuple(f"{prefix}-{index:03d}" for index in range(int(count)))


FIT_IDS = _ids("separation-fit", FIT_N)
HELD_IDS = _ids("separation-held", HELD_N)


def _axis(index: int) -> np.ndarray:
    row = np.zeros(SEPARATION_DIM, dtype=np.float64)
    row[int(index)] = 1.0
    return row


def similar_pair() -> tuple[np.ndarray, np.ndarray]:
    """Near-neighbor keys. Cosine is above TAU and below 1."""
    left = _axis(0)
    raw = np.zeros(SEPARATION_DIM, dtype=np.float64)
    raw[0] = 1.0
    raw[1] = 0.1
    right = raw / float(np.linalg.norm(raw))
    return left, right


def distinct_pair() -> tuple[np.ndarray, np.ndarray]:
    """Orthogonal keys. Cosine is 0, below TAU."""
    return _axis(0), _axis(2)


def equal_pair() -> tuple[np.ndarray, np.ndarray]:
    mark = _axis(0)
    return mark, mark.copy()


def query_toward_left(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Closer to ``left`` than to ``right``, still inside the collision band."""
    del right
    raw = left.copy()
    raw[1] = 0.02
    return raw / float(np.linalg.norm(raw))


def canonical_scores() -> dict:
    """Scores this fixture set must produce. Identical for every id."""
    similar_left, similar_right = similar_pair()
    distinct_left, distinct_right = distinct_pair()
    equal_left, equal_right = equal_pair()
    similar_bank = np.stack([similar_left, similar_right], 0)
    distinct_bank = np.stack([distinct_left, distinct_right], 0)
    equal_bank = np.stack([equal_left, equal_right], 0)
    query = query_toward_left(similar_left, similar_right)
    raw = cosine_scores(similar_bank, query)
    sep = match_scores(query, similar_bank)
    separated = separate_keys(similar_bank)
    return {
        "protocol": PROTOCOL,
        "dim": int(SEPARATION_DIM),
        "tau": float(SEPARATION_TAU),
        "span": float(SEPARATION_SPAN),
        "similar_raw": float(key_cosine(similar_left, similar_right)),
        "similar_sep": float(key_cosine(separated[0], separated[1])),
        "confusion_raw_gap": float(raw[0] - raw[1]),
        "confusion_sep_gap": float(sep[0] - sep[1]),
        "distinct_raw": float(key_cosine(distinct_left, distinct_right)),
        "distinct_sep": float(key_cosine(separate_keys(distinct_bank)[0], separate_keys(distinct_bank)[1])),
        "equal_sep": float(key_cosine(separate_keys(equal_bank)[0], separate_keys(equal_bank)[1])),
        "identity_norm": float(np.linalg.norm(separate_key(similar_left, ()))),
        "query_winner_raw": int(np.argmax(raw)),
        "query_winner_sep": int(np.argmax(sep)),
    }


def formula_gates() -> dict:
    """Pure-function gates. Live cue / export are scored in tests."""
    scores = canonical_scores()
    similar_left, similar_right = similar_pair()
    identity = (
        np.allclose(separate_key(similar_left, ()), similar_left)
        and np.allclose(separate_key(similar_left, [similar_right], tau=0.999), similar_left)
        and scores["identity_norm"] == float(np.linalg.norm(similar_left))
        and scores["distinct_sep"] == scores["distinct_raw"]
    )
    similar = scores["similar_raw"] >= float(SEPARATION_TAU) and scores["similar_sep"] < scores["similar_raw"]
    confusion = (
        scores["confusion_raw_gap"] > 0.0
        and scores["confusion_sep_gap"] > scores["confusion_raw_gap"]
        and scores["query_winner_raw"] == 0
        and scores["query_winner_sep"] == 0
    )
    distinct = scores["distinct_raw"] == 0.0 and scores["distinct_sep"] == 0.0
    equal = scores["equal_sep"] == 1.0
    return {
        "identity": bool(identity),
        "similar": bool(similar),
        "confusion": bool(confusion),
        "distinct": bool(distinct),
        "equal": bool(equal),
        "scores": scores,
    }


def bank_sha256() -> str:
    scores = canonical_scores()
    payload = "\n".join(
        (
            *FIT_IDS,
            *HELD_IDS,
            PROTOCOL,
            str(scores["dim"]),
            str(scores["tau"]),
            str(scores["span"]),
            str(scores["similar_raw"]),
            str(scores["similar_sep"]),
            str(scores["confusion_raw_gap"]),
            str(scores["confusion_sep_gap"]),
            *GATES,
        )
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
