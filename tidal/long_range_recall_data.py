"""Locked bank and gates for the long-range recall held-out eval.

Measurement only. No runtime flag. Criteria match
``docs/long_range_recall_prereg.md`` and are not retuned after a run.
"""
from __future__ import annotations

import hashlib
from typing import Any, Literal, Mapping, Never, Sequence

PROTOCOL = "long-range-recall-held-v0"
SEEDS = (0, 1, 2, 3, 4)
N_VALUES = (10, 30, 100)
ITEMS_N = 10
STUB_DIM = 64
STRONG_COSINE = 1.0
WEAK_COSINE = 0.25
SIMILAR_COSINE = 0.95
UNRELATED_COSINE = 0.0
BASELINE_TURN_DECAY = 0.5 ** (1.0 / 30.0)
TARGET_AROUSAL = 0.8
TARGET_EXPERIENCE = 1.0
WAKE_AFTER_PLANT = 1.0
WAKE_EVERY = 4
WAKE_STEP = 0.5
SLEEP_WAKE = 4.0
SLEEP_DURATION = 2.0
OFFLINE_N10 = 8
OFFLINE_N30 = 4
OFFLINE_PERIOD = 16
LONG_OFFLINE_TURN = 75
FROZEN_RECONSTRUCT_SHA = (
    "bad8e653287b996114e32b68920c53c35dfa142d16c6bfb85fd5b4a811f78589"
)
FROZEN_RECONSTRUCT_INTERFACE = 6
FROZEN_CUE_INTERFACE = 7
H4_MIN = 0.80
H5_MAX = 0.05

Condition = Literal["full", "off_77", "off_79", "off_81", "off_83", "baseline"]
Family = Literal["recall", "conflict", "unrelated"]
CONDITIONS: tuple[Condition, ...] = (
    "full",
    "off_77",
    "off_79",
    "off_81",
    "off_83",
    "baseline",
)
FAMILIES: tuple[Family, ...] = ("recall", "conflict", "unrelated")
PACKET_KEYS = (
    "interface_version",
    "cue_dim",
    "memory_id",
    "vec",
    "activation",
    "strength_confidence",
    "valence",
    "modality",
    "circadian_phase",
    "triggered",
    "gap",
    "winner_cosine",
    "sleep_pressure",
    "arousal",
    "source",
    "fingerprint",
    "fusion_confidence",
)

ITEMS = (
    {
        "id": 0,
        "name": "hari",
        "rel": "food",
        "text": "That day I ate matcha cake.",
        "answer": "matcha cake",
        "strong_cue": "That day I ate matcha cake.",
        "weak_cue": "I ate some again.",
        "conflict_text": "That day I ate strawberry cake.",
        "conflict_answer": "strawberry cake",
        "unrelated": "The printer is jammed again.",
        "gloss_plant": "那天吃了抹茶蛋糕",
        "gloss_weak": "我又吃到了",
    },
    {
        "id": 1,
        "name": "nora",
        "rel": "seat",
        "text": "I prefer sitting by the window.",
        "answer": "the window",
        "strong_cue": "I prefer sitting by the window.",
        "weak_cue": "I sat down again.",
        "conflict_text": "I prefer sitting by the aisle.",
        "conflict_answer": "the aisle",
        "unrelated": "Helsinki is cold in March.",
        "gloss_plant": "",
        "gloss_weak": "",
    },
    {
        "id": 2,
        "name": "ada",
        "rel": "pet",
        "text": "Ada's cat is named Mochi.",
        "answer": "Mochi",
        "strong_cue": "Ada's cat is named Mochi.",
        "weak_cue": "The cat came back.",
        "conflict_text": "Ada's cat is named Sable.",
        "conflict_answer": "Sable",
        "unrelated": "The violin needs new rosin.",
        "gloss_plant": "",
        "gloss_weak": "",
    },
    {
        "id": 3,
        "name": "tess",
        "rel": "meet",
        "text": "The meeting moved to Thursday.",
        "answer": "Thursday",
        "strong_cue": "The meeting moved to Thursday.",
        "weak_cue": "When did it move?",
        "conflict_text": "The meeting moved to Friday.",
        "conflict_answer": "Friday",
        "unrelated": "A red kite hangs in the hall.",
        "gloss_plant": "",
        "gloss_weak": "",
    },
    {
        "id": 4,
        "name": "cole",
        "rel": "drink",
        "text": "I cannot drink coffee after noon.",
        "answer": "coffee",
        "strong_cue": "I cannot drink coffee after noon.",
        "weak_cue": "I already had some.",
        "conflict_text": "I cannot drink tea after noon.",
        "conflict_answer": "tea",
        "unrelated": "The library closes at six.",
        "gloss_plant": "",
        "gloss_weak": "",
    },
    {
        "id": 5,
        "name": "bea",
        "rel": "mug",
        "text": "The blue mug lives on the second shelf.",
        "answer": "blue mug",
        "strong_cue": "The blue mug lives on the second shelf.",
        "weak_cue": "Where did the mug go?",
        "conflict_text": "The green mug lives on the second shelf.",
        "conflict_answer": "green mug",
        "unrelated": "Salt spilled on the counter.",
        "gloss_plant": "",
        "gloss_weak": "",
    },
    {
        "id": 6,
        "name": "remy",
        "rel": "train",
        "text": "We booked the late train home.",
        "answer": "late train",
        "strong_cue": "We booked the late train home.",
        "weak_cue": "The train again.",
        "conflict_text": "We booked the early train home.",
        "conflict_answer": "early train",
        "unrelated": "The radio plays a waltz.",
        "gloss_plant": "",
        "gloss_weak": "",
    },
    {
        "id": 7,
        "name": "mio",
        "rel": "soup",
        "text": "My favorite soup is miso.",
        "answer": "miso",
        "strong_cue": "My favorite soup is miso.",
        "weak_cue": "I had soup again.",
        "conflict_text": "My favorite soup is broth.",
        "conflict_answer": "broth",
        "unrelated": "Fog covers the south dock.",
        "gloss_plant": "",
        "gloss_weak": "",
    },
    {
        "id": 8,
        "name": "fern",
        "rel": "plant",
        "text": "The plant on the desk is a fern.",
        "answer": "fern",
        "strong_cue": "The plant on the desk is a fern.",
        "weak_cue": "The plant is still there.",
        "conflict_text": "The plant on the desk is a fig.",
        "conflict_answer": "fig",
        "unrelated": "Someone moved the chess clock.",
        "gloss_plant": "",
        "gloss_weak": "",
    },
    {
        "id": 9,
        "name": "kit",
        "rel": "key",
        "text": "I left the key under the mat.",
        "answer": "under the mat",
        "strong_cue": "I left the key under the mat.",
        "weak_cue": "Where is the key?",
        "conflict_text": "I left the key under the pot.",
        "conflict_answer": "under the pot",
        "unrelated": "The oven beeped twice already.",
        "gloss_plant": "",
        "gloss_weak": "",
    },
)


def _unknown_condition(condition: Never) -> Never:
    raise RuntimeError(f"unhandled condition: {condition}")


def _unknown_family(family: Never) -> Never:
    raise RuntimeError(f"unhandled family: {family}")


def item_by_id(item_id: int) -> dict[str, Any]:
    row = ITEMS[int(item_id)]
    if int(row["id"]) != int(item_id):
        raise RuntimeError("item id drifted")
    return dict(row)


def distractor_text(seed: int, item_id: int, turn: int) -> str:
    return f"Distractor {int(seed)}-{int(item_id)}-{int(turn)} notes the weather was mild."


def distractor_name(seed: int, item_id: int, turn: int) -> str:
    return f"d{int(seed):02d}{int(item_id):02d}{int(turn):03d}"


def similar_text(seed: int, item_id: int, turn: int) -> str:
    return f"Near {int(seed)}-{int(item_id)}-{int(turn)} shares the same day."


def similar_name(seed: int, item_id: int, turn: int) -> str:
    return f"n{int(seed):02d}{int(item_id):02d}{int(turn):03d}"


def similar_turns(n: int) -> tuple[int, int]:
    depth = int(n)
    return (1, max(2, depth // 3))


def bank_sha256() -> str:
    lines = [PROTOCOL, *map(str, SEEDS), *map(str, N_VALUES)]
    for row in ITEMS:
        lines.append(
            "|".join(
                str(row[key])
                for key in (
                    "id",
                    "name",
                    "rel",
                    "text",
                    "answer",
                    "strong_cue",
                    "weak_cue",
                    "conflict_text",
                    "conflict_answer",
                    "unrelated",
                )
            )
        )
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def mean_std(values: Sequence[float]) -> tuple[float, float]:
    nums = [float(value) for value in values]
    if not nums:
        raise ValueError("values must be nonempty")
    mean = sum(nums) / len(nums)
    if len(nums) == 1:
        return mean, 0.0
    var = sum((value - mean) ** 2 for value in nums) / (len(nums) - 1)
    return mean, var ** 0.5


def _cell(summary: Mapping[str, Any], condition: str, n: int, metric: str) -> Mapping[str, Any]:
    return summary["conditions"][condition][str(int(n))][metric]


def score_hypotheses(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Locked H1-H5. Ties fail the strict inequalities. Do not retune."""
    full_10 = float(_cell(summary, "full", 10, "recall_at_1_strong")["mean"])
    base_10 = float(_cell(summary, "baseline", 10, "recall_at_1_strong")["mean"])
    full_100 = float(_cell(summary, "full", 100, "recall_at_1_strong")["mean"])
    base_100 = float(_cell(summary, "baseline", 100, "recall_at_1_strong")["mean"])
    full_weak_100 = float(_cell(summary, "full", 100, "hit_rate_weak")["mean"])
    off83_weak_100 = float(_cell(summary, "off_83", 100, "hit_rate_weak")["mean"])
    conflict_means = [
        float(_cell(summary, "full", n, "conflict_newest_take")["mean"]) for n in N_VALUES
    ]
    false_means = [float(_cell(summary, "full", n, "false_recall")["mean"]) for n in N_VALUES]
    conflict = sum(conflict_means) / len(conflict_means)
    false_recall = sum(false_means) / len(false_means)
    # 1e-12 absorbs the 0.05+0.05+0.05 binary residue. The locked
    # floors stay 0.80 and 0.05.
    rows = {
        "H1": {
            "pass": bool(full_10 > base_10),
            "full": full_10,
            "baseline": base_10,
        },
        "H2": {
            "pass": bool(full_100 > base_100),
            "full": full_100,
            "baseline": base_100,
        },
        "H3": {
            "pass": bool(full_weak_100 > off83_weak_100),
            "full": full_weak_100,
            "off_83": off83_weak_100,
        },
        "H4": {
            "pass": bool(conflict + 1e-12 >= float(H4_MIN)),
            "full": conflict,
            "floor": float(H4_MIN),
        },
        "H5": {
            "pass": bool(false_recall <= float(H5_MAX) + 1e-12),
            "full": false_recall,
            "ceiling": float(H5_MAX),
        },
    }
    return rows


def hypotheses_pass(rows: Mapping[str, Any]) -> bool:
    return all(bool(rows[name]["pass"]) for name in ("H1", "H2", "H3", "H4", "H5"))
