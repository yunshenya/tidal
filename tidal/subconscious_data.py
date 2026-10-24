"""Implicit-influence bank for the frozen-LM inject path.

Disjoint from ``tidal.llm_memory_data`` and from the key-calibration names.
No torch. The sha256 of :func:`bank_payload` is the pre-registered bank id.

Decision prompts do not name either option. Continuations carry a leading
space so SmolLM2 scores each option as one token (checked at eval time: the
prompt ids must be a prefix of the prompt-plus-continuation ids).
"""
from __future__ import annotations

import hashlib
import json
import re

from tidal.subconscious_ids import LEAK_STORE_N

PROTOCOL = "subconscious-v1"

# Length-normalized natural log. n = 8 items in a family unless noted.
GATES = {
    "delta_min": 0.05,
    "delta_pos_min": 6,
    "acc_lift_min": 0.25,
    "salient_hit_at_64_min": 0.75,
    "repeat_abs_max": 1e-4,
    "chat_abs_max": 1e-4,
    "update_delta_min": 0.05,
    "update_delta_pos_min": 6,
    "nll_mean_max": 0.05,
    "nll_bad": 0.5,
    "nll_bad_count_max": 0,
    "choice_flips_max": 1,
    "leak_max": 0,
    "unrelated_abs_mean_max": 0.05,
    "unrelated_large": 0.1,
    "unrelated_large_count_max": 2,
}

# Quoted from docs/llm_memory.md, trained-inject prefix table. Not re-run here.
# 42/64 = 0.65625, reported there as 0.656. 30/64 = 0.46875, reported as 0.469.
PRIOR_PREFIX_N32 = {
    "source": "docs/llm_memory.md",
    "section": "Trained inject",
    "protocol": (
        "answer-stripped prefix free-gen, n=64, seeds 5000+N, "
        "soft top-4 at temperature 0.05, before the phone and birthday readouts"
    ),
    "inject_hits": 42,
    "n": 64,
    "context_hits": 30,
    "rag_hits": 30,
}

LOADS = (8, 32, 64)
RETRIEVAL_SIZES = (8, 32, 64, 128)
REPEAT_WRITES = 3
NEUTRAL_STORE_N = 32
DISTRACTOR_STRIDE = 17

_PREFER_ROWS = (
    ("prefer-mara", "Mara", "snack", "apples", "peanuts", "pears"),
    ("prefer-helio", "Helio", "drink", "water", "soda", "broth"),
    ("prefer-nia", "Nia", "pet", "cats", "dogs", "fish"),
    ("prefer-colm", "Colm", "lunch", "salad", "steak", "soup"),
    ("prefer-petra", "Petra", "wrap", "blue", "red", "linen"),
    ("prefer-idris", "Idris", "game", "chess", "poker", "dice"),
    ("prefer-sable", "Sable", "ride", "trains", "planes", "boats"),
    ("prefer-juniper", "Juniper", "fruit", "pears", "grapes", "mango"),
)

_AVOID_SPECS = (
    ("avoid-harlow", "Harlow", "snack", "peanuts", "apples", "Harlow is allergic to peanuts."),
    ("avoid-nestor", "Nestor", "drink", "milk", "water", "Nestor is allergic to milk."),
    ("avoid-cleo", "Cleo", "pet", "dogs", "cats", "Cleo is afraid of dogs."),
    ("avoid-bodhi", "Bodhi", "lunch", "steak", "salad", "Bodhi cannot eat steak."),
    ("avoid-ingrid", "Ingrid", "dessert", "honey", "berries", "Ingrid is allergic to honey."),
    ("avoid-tomas", "Tomas", "ride", "planes", "trains", "Tomas is afraid of planes."),
    ("avoid-yara", "Yara", "gift", "drums", "books", "Yara cannot use drums."),
    ("avoid-felix", "Felix", "fruit", "grapes", "pears", "Felix is allergic to grapes."),
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
        "text": f"{name}'s favorite {domain} is {answer}.",
        "prompt": f"Pick one {domain} for {name}.\nAnswer:",
        "good": f" {answer}",
        "bad": f" {other}",
        "update_answer": update,
        "update_text": f"{name}'s favorite {domain} is {update}.",
        "update_good": f" {update}",
        "update_bad": f" {answer}",
    }


def _avoid(row) -> dict:
    pid, name, domain, forbidden, safe, text = row
    if forbidden not in text:
        raise ValueError(f"{pid} answer is not in the fact text")
    return {
        "id": pid,
        "family": "avoid",
        "name": name,
        "rel": domain,
        "domain": domain,
        "answer": forbidden,
        "other": safe,
        "text": text,
        "prompt": f"Pick one {domain} for {name}.\nAnswer:",
        "good": f" {safe}",
        "bad": f" {forbidden}",
    }


PREFER = tuple(_prefer(row) for row in _PREFER_ROWS)
AVOID = tuple(_avoid(row) for row in _AVOID_SPECS)

_INCIDENTAL_NAMES = (
    "Alden", "Bram", "Cassia", "Doran", "Elowen", "Faye", "Gideon", "Hester",
    "Ione", "Joss", "Kestrel", "Linnea", "Moss", "Nadine", "Orson", "Pilar",
    "Quen", "Rowan", "Sigrid", "Talin", "Una", "Vesper", "Wren", "Yorick",
    "Ansel", "Bex", "Corin", "Dahlia", "Eamon", "Freya", "Galen", "Honora",
    "Isidore", "Juno", "Kit", "Liora", "Maeve", "Nils", "Oona", "Perrin",
    "Rhea", "Stig", "Tamsin", "Uriel", "Wynn", "Xenia", "Yates", "Zola",
    "Arlo", "Bibi", "Cressida", "Dax", "Elio", "Fiora", "Gust", "Hedy",
    "Ivo", "Josslyn", "Keir", "Lumen", "Marnie", "Newt", "Odile", "Pax",
)
_OBJECTS = (
    "thimble", "lodestone", "abacus", "sextant", "bodkin", "clepsydra", "inkwell", "astrolabe",
    "bellows", "clothespin", "doweling", "gimlet", "mandrel", "pendulum", "spindle", "trestle",
    "zither", "bobbin", "crucible", "distaff", "flagon", "grommet", "hawser", "lanyard",
    "nostrum", "obelisk", "pestle", "plinth", "quiver", "valise", "windlass", "yardarm",
    "bushel", "coffer", "dibble", "ferule", "mallet", "needle", "oarlock", "pulley",
    "quillon", "sheave", "andiron", "capstan", "futtock", "linchpin", "marlinspike", "outrigger",
    "pintle", "scupper", "tholepin", "backsaw", "drawknife", "eelgrass", "graving", "kevelpin",
    "navline", "winchpin", "xylocarp", "zaxhead", "bradawl", "caulker", "deadeye", "fidspike",
)
_FRAMES = (
    ("keeps", "{name} keeps one {obj} in a drawer."),
    ("folded", "{name} folded one {obj} today."),
    ("labels", "{name} labels one {obj} on a shelf."),
    ("stored", "{name} stored one {obj} upstairs."),
)

CHATS = (
    ("The kettle whistled softly.", "softly"),
    ("A window was left open.", "open"),
    ("The meeting ran long.", "long"),
    ("The bus arrived late.", "late"),
    ("Please pass the salt.", "salt"),
    ("It might rain tonight.", "tonight"),
    ("The hall lights flickered.", "flickered"),
    ("Someone knocked once.", "once"),
    ("I finished the chapter.", "chapter"),
    ("The cat is on the chair.", "chair"),
    ("We should leave before dark.", "dark"),
    ("The soup needs another minute.", "minute"),
    ("A clock chimed downstairs.", "downstairs"),
    ("The hallway smells like soap.", "soap"),
    ("I lost the spare key.", "key"),
    ("The radio played quietly.", "quietly"),
    ("Leaves blew across the step.", "step"),
    ("The lamp needs a new bulb.", "bulb"),
    ("Traffic was slow at noon.", "noon"),
    ("The package came this morning.", "morning"),
    ("Please water the fern.", "fern"),
    ("The stairs creaked again.", "again"),
    ("I set the table already.", "already"),
    ("The sink is dripping.", "dripping"),
    ("A neighbor waved from the yard.", "yard"),
    ("The notebook is in the bag.", "bag"),
    ("We missed the first song.", "song"),
    ("The heater clicked on.", "clicked"),
    ("I saved a seat by the aisle.", "aisle"),
    ("The paint on the sill is wet.", "wet"),
    ("Please lock the side gate.", "gate"),
    ("The story ended too soon.", "soon"),
)

NEUTRAL_NLL = (
    {"id": "nll-mat", "prompt": "The cat sat on the", "continuation": " mat"},
    {"id": "nll-there", "prompt": "Once upon a time", "continuation": " there"},
    {"id": "nll-much", "prompt": "Thank you very", "continuation": " much"},
    {"id": "nll-day", "prompt": "Have a nice", "continuation": " day"},
    {"id": "nll-begin", "prompt": "In the beginning", "continuation": " there"},
    {"id": "nll-rose", "prompt": "Roses are", "continuation": " red"},
    {"id": "nll-east", "prompt": "The sun rises in the", "continuation": " east"},
    {"id": "nll-four", "prompt": "One two three", "continuation": " four"},
    {"id": "nll-everyone", "prompt": "Good morning", "continuation": " everyone"},
    {"id": "nll-door", "prompt": "Please close the", "continuation": " door"},
    {"id": "nll-book", "prompt": "She opened the", "continuation": " book"},
    {"id": "nll-night", "prompt": "It was a dark and stormy", "continuation": " night"},
)

NEUTRAL_CHOICE = (
    {"id": "neu-four", "prompt": "Two plus two equals\nAnswer:", "good": " four", "bad": " nine"},
    {"id": "neu-water", "prompt": "Ice is frozen\nAnswer:", "good": " water", "bad": " fire"},
    {"id": "neu-days", "prompt": "A week has seven\nAnswer:", "good": " days", "bad": " dogs"},
    {"id": "neu-loudly", "prompt": "Dogs often bark\nAnswer:", "good": " loudly", "bad": " purple"},
    {"id": "neu-fly", "prompt": "Birds can\nAnswer:", "good": " fly", "bad": " type"},
    {"id": "neu-tuesday", "prompt": "Monday comes before\nAnswer:", "good": " Tuesday", "bad": " apples"},
    {"id": "neu-feet", "prompt": "Shoes go on your\nAnswer:", "good": " feet", "bad": " moon"},
    {"id": "neu-clouds", "prompt": "Rain falls from\nAnswer:", "good": " clouds", "bad": " shoes"},
)

NEUTRAL_GEN = (
    "What should we cook tonight?\nAnswer:",
    "Tell a very short joke.\nAnswer:",
    "How does a bedtime story start?\nAnswer:",
    "Name something seen in the sky.\nAnswer:",
    "What belongs on a breakfast table?\nAnswer:",
    "Suggest a quiet indoor hobby.\nAnswer:",
    "Where do people keep umbrellas?\nAnswer:",
    "What noise does a clock make?\nAnswer:",
)

PHONE_FACT = {
    "id": "script-phone",
    "name": "Quill",
    "rel": "phone",
    "answer": "+1-415-555-4242",
    "text": "Quill's phone number is +1-415-555-4242.",
}
BIRTHDAY_FACT = {
    "id": "script-birthday",
    "name": "Quill",
    "rel": "birthday",
    "answer": "March 4, 1991",
    "text": "Quill's birthday is March 4, 1991.",
}
PHONE_QUESTION = "What is Quill's phone number?"


def decision_items() -> tuple:
    return PREFER + AVOID


def incidental_facts() -> list:
    """256 facts. Four frames, 64 names, one object per name inside a frame."""
    if len(_INCIDENTAL_NAMES) != 64 or len(_OBJECTS) != 64:
        raise RuntimeError("incidental lists must be length 64")
    rows = []
    for rel, template in _FRAMES:
        for name, obj in zip(_INCIDENTAL_NAMES, _OBJECTS):
            text = template.format(name=name, obj=obj)
            if obj not in text:
                raise RuntimeError("object missing from incidental text")
            rows.append({
                "name": name,
                "rel": rel,
                "answer": obj,
                "text": text,
                "kind": "fact",
            })
    return rows


def distractors(item_index: int, n_distract: int, incidental: list | None = None) -> list:
    """Nested slice. ``n_distract`` rows starting at ``(index * 17) mod len``.

    The N=8 neighborhood is the first 7 rows of the N=64 neighborhood.
    """
    rows = incidental_facts() if incidental is None else incidental
    n = int(n_distract)
    if n < 0 or n > len(rows):
        raise ValueError(f"n_distract {n} is outside 0..{len(rows)}")
    start = (int(item_index) * DISTRACTOR_STRIDE) % len(rows)
    return [rows[(start + i) % len(rows)] for i in range(n)]


def completion_prompt(text: str, answer: str) -> str:
    """Fact with the answer word and the single space before it removed."""
    if not isinstance(answer, str) or answer == "":
        raise ValueError("answer is empty")
    j = text.rfind(answer)
    if j < 0:
        raise ValueError("answer is not in the fact text")
    prefix = text[:j]
    if prefix.endswith(" "):
        prefix = prefix[:-1]
    if prefix == "":
        raise ValueError("completion prompt is empty")
    return prefix


def word_hit(text: str, word: str) -> bool:
    """Whole-word match. ``red`` does not match ``registered``."""
    if not isinstance(text, str) or not isinstance(word, str) or word == "":
        return False
    return re.search(r"\b" + re.escape(word.lower()) + r"\b", text.lower()) is not None


def choice_words() -> set:
    words = set()
    for item in decision_items():
        words.add(item["answer"])
        words.add(item["other"])
        if "update_answer" in item:
            words.add(item["update_answer"])
    for row in NEUTRAL_CHOICE:
        words.add(row["good"].strip())
        words.add(row["bad"].strip())
    for row in NEUTRAL_NLL:
        words.add(row["continuation"].strip())
    return words


def bank_payload() -> dict:
    return {
        "protocol": PROTOCOL,
        "gates": GATES,
        "loads": list(LOADS),
        "retrieval_sizes": list(RETRIEVAL_SIZES),
        "repeat_writes": REPEAT_WRITES,
        "neutral_store_n": NEUTRAL_STORE_N,
        "leak_store_n": LEAK_STORE_N,
        "distractor_stride": DISTRACTOR_STRIDE,
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
    }


def bank_sha256() -> str:
    blob = json.dumps(bank_payload(), sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()
