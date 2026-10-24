"""Real-word direction held bank. No torch.

Fifty prefer items in the n=8 direction-fit template, plus the controls the
direction gates read. Content words are disjoint from the fit banks and from
the earlier held banks. This bank does not choose a constant.
"""
from __future__ import annotations

import hashlib
import json
import re
import types

PROTOCOL = "cue-direction-word-v1"
N = 50
PROMPT = "Hand {name} some {domain}.\nState:"
TEXT = "{name} circles toward {answer} about {domain}."
CHOICE_PROMPT = "Say the right one\nState:"
NLL_PROMPT = "Say nothing\n"
# Fixed frame around the content words. These tokens are not memory content.
SKELETON = frozenset({
    "hand", "some", "state", "circles", "toward", "about",
    "a", "an", "the", "is", "are", "of", "to", "and", "with", "one",
})
FRAME = frozenset({"say", "right", "today", "stows", "behind", "nothing"})
SENTINELS = ("beaver", "cedar", "bleryl", "quelva00", "lumen00")

NAMES = tuple(f"Sovrik{index:02d}" for index in range(N))
INC_NAMES = tuple(f"Tovren{index:02d}" for index in range(40))

DOMAINS = (
    "almond", "baobab", "beech", "blackberry", "blueberry", "boxwood", "buckeye", "cabbage",
    "chive", "cilantro", "clove", "cress", "daikon", "garlic", "ginger", "grapefruit",
    "huckleberry", "jujube", "kudzu", "maize", "milkweed", "mulberry", "parsley", "peach",
    "peanut", "pistachio", "potato", "pumpkin", "rosemary", "sorrel", "strawberry", "sugarcane",
    "sunflower", "tulip", "vanilla", "violet", "walnut", "wasabi", "yarrow", "amaranth",
    "anise", "artichoke", "arugula", "aubergine", "caraway", "cauliflower", "celeriac", "chamomile",
    "chard", "chayote",
)

ANSWERS = (
    "acorn", "bracelet", "bookmark", "buckle", "camera", "candle", "candlestick", "cellphone",
    "dagger", "dictionary", "eraser", "flashlight", "football", "hanger", "harmonica", "keyboard",
    "locket", "magazine", "microphone", "mirror", "monitor", "necktie", "notepad", "ocarina",
    "paintbrush", "paperclip", "pencil", "perfume", "pillow", "pinecone", "plate", "purse",
    "quilt", "sailboat", "sandal", "scarf", "screw", "screwdriver", "shampoo", "shawl",
    "shirt", "slipper", "spatula", "sponge", "thermometer", "thread", "toothbrush", "toothpaste",
    "toothpick", "towel",
)

FOILS = (
    "aardvark", "alligator", "antelope", "canary", "capybara", "chameleon", "cheetah", "chimpanzee",
    "chinchilla", "cougar", "coyote", "hippopotamus", "horse", "iguana", "jaguar", "koala",
    "marmoset", "moose", "mustang", "opossum", "panther", "parakeet", "parrot", "platypus",
    "porpoise", "reindeer", "rhinoceros", "sloth", "walrus", "warthog", "wildcat", "wombat",
    "woodchuck", "hamster", "panda", "rabbit", "baseball", "basketball", "blouse", "broccoli",
    "chicken", "chocolate", "crowbar", "incense", "keychain", "martini", "matchbook", "mobile",
    "pasta", "pearl",
)

CHOICE_GOOD = (
    "chickpea", "courgette", "endive", "horseradish", "paprika", "scallion", "spinach", "squash",
    "zucchini", "avocado", "bilberry", "breadfruit", "cantaloupe", "cherimoya", "clementine", "cloudberry",
    "cranberry", "damson", "dragonfruit", "durian", "feijoa", "gooseberry", "honeydew", "jackfruit",
    "passionfruit", "persimmon", "pomegranate", "pomelo", "rambutan", "satsuma", "tamarillo", "tangerine",
    "watermelon", "farro", "teff", "triticale", "buckwheat", "semolina", "tapioca", "arrowroot",
    "cassava", "plantain", "carob", "filbert", "cashew", "macadamia", "lingonberry", "boysenberry",
    "loganberry", "saffron",
)

CHOICE_BAD = (
    "turmeric", "cumin", "coriander", "cardamom", "mace", "allspice", "obsidian", "pumice",
    "basalt", "mica", "feldspar", "quartzite", "schist", "flint", "jasper", "agate",
    "onyx", "zircon", "spinel", "tourmaline", "amethyst", "citrine", "malachite", "turquoise",
    "jade", "amber", "coral", "graphite", "peat", "travertine", "alabaster", "serpentine",
    "soapstone", "talc", "anthracite", "bauxite", "kaolin", "bentonite", "diatomite", "hematite",
    "magnetite", "galena", "cinnabar", "laterite", "tweezers", "umbrella", "wallet", "washcloth",
    "whistle", "wishbone",
)

NLL = (
    "wrench", "wristwatch", "zipper", "hammer", "bottle", "button", "computer", "couch",
    "lemon", "orange", "shark", "beetroot", "salsify", "okra", "tahini", "molasses",
    "ketchup", "vinegar", "gelatin", "yeast", "whisk", "colander", "grater", "peeler",
    "corkscrew", "tureen", "cleaver", "gimlet", "caliper", "bodkin", "napkin", "clamp",
    "plane", "adze", "cowl", "jerkin", "gaiter", "sarong", "anorak", "beanie",
    "mitten", "loafer", "brogue", "galosh", "clog", "mule", "pump", "wedge",
    "bootie", "visor",
)

INC_OBJ = (
    "pocketknife", "pocketwatch", "scallop", "soccer", "spectacles", "spice", "spool", "statuette",
    "sticker", "stockings", "tissue", "lipstick", "robot", "binoculars", "earrings", "scissors",
    "sunglasses", "tongs", "snowglobe", "sandglass", "deodorant", "domino", "flyswatter", "cinnamon",
    "lilac", "nettle", "nightshade", "watercress", "azuki", "butternut", "cayenne", "chives",
    "delicata", "radicchio", "tabasco", "blackcurrant", "mangosteen", "redcurrant", "grape", "tarragon",
)

PLACES = (
    "attic", "cellar", "closet", "cupboard", "balcony", "terrace", "veranda", "courtyard",
)

RELS = (
    "folio", "quire", "vellum", "codex", "ledger", "gazette", "brochure", "dossier",
)


def _items():
    rows = []
    for index, (name, domain, answer, foil) in enumerate(zip(NAMES, DOMAINS, ANSWERS, FOILS)):
        rows.append(
            {
                "id": f"cue-word-prefer-{index}",
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


def _choice():
    rows = []
    for index, (good, bad) in enumerate(zip(CHOICE_GOOD, CHOICE_BAD)):
        rows.append(
            {
                "id": f"cue-word-choice-{index}",
                "prompt": CHOICE_PROMPT,
                "good": " " + good,
                "bad": " " + bad,
            }
        )
    return tuple(rows)


def _nll():
    rows = []
    for index, word in enumerate(NLL):
        rows.append(
            {
                "id": f"cue-word-nll-{index}",
                "prompt": NLL_PROMPT,
                "continuation": " " + word + " today",
            }
        )
    return tuple(rows)


def _incidental():
    rows = []
    for index, name in enumerate(INC_NAMES):
        obj = INC_OBJ[index]
        place = PLACES[index % len(PLACES)]
        rel = RELS[index % len(RELS)]
        rows.append(
            {
                "name": name,
                "rel": rel,
                "answer": obj,
                "text": f"{name} stows one {obj} behind {place}.",
                "kind": "fact",
            }
        )
    return tuple(rows)


PREFER = _items()
NEUTRAL_CHOICE = _choice()
NEUTRAL_NLL = _nll()
INCIDENTAL = _incidental()
_WORD = re.compile(r"[a-z0-9]+")


def content_words() -> set:
    raw = set(NAMES) | set(INC_NAMES) | set(DOMAINS) | set(ANSWERS) | set(FOILS) | set(CHOICE_GOOD) | set(CHOICE_BAD) | set(NLL) | set(INC_OBJ) | set(PLACES) | set(RELS)
    return {word.lower() for word in raw}


def _walk(obj, bag, seen, depth=0) -> None:
    if obj is None or depth > 8:
        return
    if isinstance(obj, (int, float, bool)):
        return
    if isinstance(obj, str):
        bag.append(obj)
        return
    mark = id(obj)
    if mark in seen:
        return
    seen.add(mark)
    if isinstance(obj, dict):
        for value in obj.values():
            _walk(value, bag, seen, depth + 1)
    elif isinstance(obj, (list, tuple, set, frozenset)):
        for value in obj:
            _walk(value, bag, seen, depth + 1)
    elif hasattr(obj, "__dict__"):
        for key, value in vars(obj).items():
            if str(key).startswith("_") or callable(value):
                continue
            _walk(value, bag, seen, depth + 1)


def prior_tokens() -> set:
    """Tokens from the fit banks and the earlier held banks."""
    bag = []
    seen = set()
    modules = (
        "tidal.subconscious_direction_data",
        "tidal.subconscious_cue_fit50_data",
        "tidal.subconscious_cue_fresh_data",
        "tidal.subconscious_valence_data",
        "tidal.subconscious_circadian_data",
        "tidal.subconscious_replay_data",
        "tidal.subconscious_rif_retry_data",
        "tidal.subconscious_spell_data",
        "tidal.subconscious_surprise_rel_data",
        "tidal.subconscious_gate_data",
        "tidal.subconscious_measure_data",
        "tidal.subconscious_data",
        "tidal.subconscious_interfere_data",
        "tidal.subconscious_softmix_data",
        "tidal.llm_memory_data",
    )
    for name in modules:
        mod = __import__(name, fromlist=["*"])
        for key, value in vars(mod).items():
            if key.startswith("_") or callable(value) or isinstance(value, types.ModuleType):
                continue
            _walk(value, bag, seen)
    from tidal.subconscious_circadian_data import fit_bank as clock_fit
    from tidal.subconscious_circadian_data import held_bank as clock_held
    from tidal.subconscious_cue_fit50_data import fit50_bank
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
    from tidal.subconscious_replay_data import fit_bank as replay_fit
    from tidal.subconscious_replay_data import held_bank as replay_held
    from tidal.subconscious_rif_retry_data import fit_bank as rif_fit
    from tidal.subconscious_rif_retry_data import held_bank as rif_held
    from tidal.subconscious_spell_data import fit_bank as spell_fit
    from tidal.subconscious_spell_data import held_bank as spell_held
    from tidal.subconscious_surprise_rel_data import fit_bank as surprise_fit
    from tidal.subconscious_surprise_rel_data import held_bank as surprise_held
    from tidal.subconscious_surprise_rel_data import train_bank
    from tidal.subconscious_valence_data import fit_bank as valence_fit
    from tidal.subconscious_valence_data import held_bank as valence_held

    for obj in (
        fit_bank(),
        held_bank(),
        fit50_bank(),
        fresh_bank(),
        valence_fit(),
        valence_held(),
        clock_fit(),
        clock_held(),
        replay_fit(),
        replay_held(),
        spell_fit(),
        spell_held(),
        train_bank(),
        surprise_fit(),
        surprise_held(),
        rif_fit(),
        rif_held(),
        gate_held(),
        RifMeasure.QUARTETS,
        prefer_items(),
        avoid_items(),
        toward_items(),
        away_items(),
        choice_items(),
        tone_items(),
        preference_items(),
        phone_items(),
        birthday_items(),
    ):
        _walk(obj, bag, seen)
    return set(_WORD.findall("\n".join(bag).lower()))


def _bank_tokens() -> set:
    found = set()
    for row in list(PREFER) + list(NEUTRAL_CHOICE) + list(NEUTRAL_NLL) + list(INCIDENTAL):
        for key, value in row.items():
            if key in ("id", "kind", "family"):
                continue
            if isinstance(value, str):
                found.update(_WORD.findall(value.lower()))
    return found


def assert_disjoint() -> None:
    """Content words do not occur in a fit bank or an earlier held bank."""
    if len(PREFER) < 50 or len(NEUTRAL_CHOICE) < 50 or len(NEUTRAL_NLL) < 50:
        raise RuntimeError("word direction bank is below 50")
    if len(INCIDENTAL) < 31:
        raise RuntimeError("word incidental bank cannot fill a store of 32")
    groups = {
        "domains": DOMAINS,
        "answers": ANSWERS,
        "foils": FOILS,
        "choice_good": CHOICE_GOOD,
        "choice_bad": CHOICE_BAD,
        "nll": NLL,
        "inc_obj": INC_OBJ,
        "places": PLACES,
        "rels": RELS,
        "names": NAMES,
        "inc_names": INC_NAMES,
    }
    for name, words in groups.items():
        if len(words) != len(set(words)):
            raise RuntimeError(f"{name} words are not unique")
    content = content_words()
    if len(content) != sum(len(words) for words in groups.values()):
        raise RuntimeError("word bank content words overlap across roles")
    if content & (SKELETON | FRAME):
        raise RuntimeError("a content word is also a frame word")
    prior = prior_tokens()
    missing = [word for word in SENTINELS if word not in prior]
    if missing:
        raise RuntimeError("prior token walker missed " + ", ".join(missing))
    shared = sorted(content & prior)
    if shared:
        raise RuntimeError("word bank overlaps a fit or held bank: " + ", ".join(shared[:12]))
    framed = sorted(FRAME & prior)
    if framed:
        raise RuntimeError("frame word is already in a prior bank: " + ", ".join(framed))
    stray = sorted(_bank_tokens() - content - SKELETON - FRAME)
    if stray:
        raise RuntimeError("word bank has an unlisted token: " + ", ".join(stray[:12]))
    identities = [(row["name"], row["rel"]) for row in list(PREFER) + list(INCIDENTAL)]
    if len(identities) != len(set(identities)):
        raise RuntimeError("word bank identities collided")


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


class WordDirectionBank:
    PROTOCOL = PROTOCOL
    PREFER = PREFER
    NEUTRAL_CHOICE = NEUTRAL_CHOICE
    NEUTRAL_NLL = NEUTRAL_NLL

    def incidental_facts(self):
        return list(INCIDENTAL)

    def sha256(self) -> str:
        return bank_sha256()


def word_bank() -> WordDirectionBank:
    assert_disjoint()
    return WordDirectionBank()
