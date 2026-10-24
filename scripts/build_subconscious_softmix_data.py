"""Build the fit probes and the multi-slot distractors. Does not score margins.

Fit items are disjoint from the measurement bank. Held probes stay in
``tidal/subconscious_measure_data.py``. Each probe gets three distractors
whose keys, together with the probe, stay under cosine 0.85 so cosine
overwrite does not collapse the four slots. That check is geometry.
"""
from __future__ import annotations

import importlib
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+(?:'[A-Za-z]+)?")
VOWELS = "aeiou"
CONSONANTS = "bcdfghjklmnpqrstvwxyz"
SURFACE = {"text", "prompt", "good", "bad", "answer", "other", "name", "rel"}
N_FIT = 16
N_DISTRACTORS = 3
COSINE_MAX = 0.85


def tokens(text: str) -> set:
    out = set()
    for match in TOKEN_RE.finditer(str(text)):
        seg = match.group(0)
        if "\u4e00" <= seg[0] <= "\u9fff":
            out.add(seg)
        else:
            out.add(seg.lower())
    return out


def walk(obj, out):
    if isinstance(obj, str):
        out.append(obj)
    elif isinstance(obj, dict):
        for value in obj.values():
            walk(value, out)
    elif isinstance(obj, (list, tuple)):
        for value in obj:
            walk(value, out)


def existing_strings():
    texts = []
    for path in sorted((ROOT / "tidal").glob("*data.py")):
        if path.stem == "subconscious_softmix_data":
            continue
        mod = importlib.import_module(f"tidal.{path.stem}")
        for key, value in vars(mod).items():
            if key.startswith("__"):
                continue
            walk(value, texts)
    return texts


def synth(index: int, forbidden: set, used: set) -> str:
    n = index
    while True:
        chars = []
        x = n
        for _ in range(3):
            chars.append(CONSONANTS[x % len(CONSONANTS)])
            x //= len(CONSONANTS)
            chars.append(VOWELS[x % len(VOWELS)])
            x //= len(VOWELS)
        word = "".join(chars)
        n += 1
        if word in forbidden or word in used:
            continue
        used.add(word)
        return word


def name_for(index: int, forbidden: set, used: set) -> str:
    word = synth(index + 1000, forbidden, used)
    name = word[:1].upper() + word[1:] + "qo"
    lowered = name.lower()
    if lowered in forbidden or lowered in used:
        return name_for(index + 17, forbidden, used)
    used.add(lowered)
    return name


def jaccard(left: set, right: set) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def worst_jaccard(texts, old_sets):
    worst = 0.0
    pair = None
    for text in texts:
        left = tokens(text)
        if not left:
            continue
        for old, old_text in old_sets:
            score = jaccard(left, old)
            if score > worst:
                worst = score
                pair = (text[:90], old_text[:90])
    return worst, pair


def fit_items(forbidden, used_words, used_names):
    families = ("tone", "choice", "preference")
    events = [synth(i, forbidden, used_words) for i in range(N_FIT)]
    answers = [synth(200 + i, forbidden, used_words) for i in range(N_FIT * 4)]
    cursor = 0
    grouped = {family: [] for family in families}
    for index in range(N_FIT):
        for family in families:
            name = name_for(index * 3 + ["tone", "choice", "preference"].index(family), forbidden, used_names)
            rel = f"mfit{family[0]}{index}"
            if family == "tone":
                sign = "pos" if index % 2 == 0 else "neg"
                event = events[index]
                text = f"{name} narrates {event} across {rel}."
                good, bad = (" kindly", " cruelly") if sign == "pos" else (" cruelly", " kindly")
                prompt = f"Balance quiet register.\nHue mfhue{index}:"
                answer = event
                other = ""
            elif family == "choice":
                answer = answers[cursor]
                other = answers[cursor + 1]
                cursor += 2
                text = f"{name} picks {answer} ahead of {other} inside {rel}."
                good, bad = f" {answer}", f" {other}"
                prompt = f"Balance quiet register.\nPick mfpick{index}:"
                sign = ""
            else:
                answer = answers[cursor]
                other = answers[cursor + 1]
                cursor += 2
                text = f"{name} echoes {answer} across {rel}."
                good, bad = f" {answer}", f" {other}"
                prompt = f"Balance quiet register.\nBend mfbend{index}:"
                sign = ""
            grouped[family].append(
                {
                    "id": f"softmix-fit-{family}-{index}",
                    "family": family,
                    "name": name,
                    "rel": rel,
                    "answer": answer,
                    "other": other,
                    "sign": sign,
                    "text": text,
                    "prompt": prompt,
                    "good": good,
                    "bad": bad,
                }
            )
    return grouped


# Three dissimilar frames per probe. Same-shape English sentences sat near
# cosine 0.85 of each other and of the probe, so a lexical swap could not
# keep four slots. These frames are code, a modular residue, and a printf
# line. Later frames are only retries when a group is still too close or
# the probe is not the query's cosine argmax.
FRAMES = (
    "fn {word}() {{ return {n}; }}",
    "Residue {n} modulo {m} equals {word}.",
    "printf {word} from sector {n}.",
    "0x{n} xor 0x{m} equals {word}.",
    "{word} lattice {n} {m} {k}.",
    "beaker {n} holds {word} near {m}.",
)


def frame_text(frame: int, word: str, n: int, m: int, k: int) -> str:
    return FRAMES[int(frame) % len(FRAMES)].format(word=word, n=n, m=m, k=k)


def _nums(nonce: int):
    n = 100 + (nonce % 800)
    m = 200 + ((nonce * 3) % 700)
    k = 300 + ((nonce * 7) % 600)
    return n, m, k


def distractor(probe_id: str, slot: int, frame: int, name: str, word: str, rel: str, nonce: int) -> dict:
    n, m, k = _nums(nonce)
    return {
        "id": f"{probe_id}-d{slot}",
        "name": name,
        "rel": rel,
        "answer": word,
        "text": frame_text(frame, word, n, m, k),
        "frame": int(frame) % len(FRAMES),
    }


def assign_distractors(probes, forbidden, used_words, used_names, nonce_base: int):
    assigned = {}
    nonce = nonce_base
    for probe in probes:
        group = []
        for slot in range(N_DISTRACTORS):
            name = name_for(nonce, forbidden, used_names)
            word = synth(nonce, forbidden, used_words)
            rel = f"msoftd{nonce}"
            group.append(distractor(probe["id"], slot, slot, name, word, rel, nonce))
            nonce += 1
        assigned[probe["id"]] = group
    return assigned


def surfaces(payload):
    found = []

    def walk_surface(obj):
        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in SURFACE and isinstance(value, str):
                    found.append(value)
                else:
                    walk_surface(value)
        elif isinstance(obj, list):
            for value in obj:
                walk_surface(value)

    walk_surface(payload)
    return found


def cosine(a, b) -> float:
    import numpy as np

    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def repair_geometry(payload, mem, forbidden, used_words, used_names):
    """Rewrite distractor frames until each group keeps four slots and the probe wins the query."""
    import numpy as np

    from tidal.llm_memory import activation_weights, participation_ratio

    cache = {}
    queries = {}

    def key_of(item):
        mark = (item["text"], item["answer"])
        if mark not in cache:
            cache[mark] = np.asarray(mem.encode_key(item["text"], item["answer"]), dtype=np.float64)
        return cache[mark]

    def query_of(prompt):
        if prompt not in queries:
            queries[prompt] = np.asarray(mem._query_vector(prompt), dtype=np.float64)
        return queries[prompt]

    nonce = 90000
    groups = []
    ordered = []
    for family, probes in payload["fit"].items():
        for probe in probes:
            ordered.append(probe)
    ordered.extend(payload["held_order"])
    for index, probe in enumerate(ordered):
        if index % 20 == 0:
            print(f"geometry {index}/{len(ordered)} {probe['id']}", flush=True)
        group = payload["distractors"][probe["id"]]
        row, nonce = _repair_group(
            probe, group, key_of, query_of, forbidden, used_words, used_names, nonce
        )
        groups.append(row)
        nonce += 30
    temps = (0.05, 0.10, 0.20, 0.40)
    ks = (2, 4, 8)
    preview = {}
    for temperature in temps:
        preview[f"{temperature:.2f}"] = {}
        for k in ks:
            parts = []
            for row in groups:
                weights = activation_weights(
                    row["query_cosine"],
                    [1.0, 1.0, 1.0, 1.0],
                    temperature=temperature,
                    k=k,
                )
                parts.append(participation_ratio(weights))
            preview[f"{temperature:.2f}"][str(k)] = {
                "mean": float(sum(parts) / len(parts)),
                "min": float(min(parts)),
            }
    worst = max(row["max_pair"] for row in groups)
    winners = sum(1 for row in groups if row["target_argmax"])
    return {
        "max_key_cosine": worst,
        "n_groups": len(groups),
        "target_argmax": winners,
        "participation": preview,
        "groups": [{key: row[key] for key in ("id", "max_pair", "query_cosine", "target_argmax")} for row in groups],
    }


def _repair_group(probe, group, key_of, query_of, forbidden, used_words, used_names, nonce):
    query = query_of(probe["prompt"])
    for _attempt in range(36):
        keys = [key_of(probe)] + [key_of(item) for item in group]
        scores = [cosine(query, key) for key in keys]
        pair_max = 0.0
        pair_slot = 0
        for left in range(len(keys)):
            for right in range(left + 1, len(keys)):
                close = cosine(keys[left], keys[right])
                if close >= pair_max:
                    pair_max = close
                    pair_slot = right - 1 if right > 0 else 0
                    if left > 0 and scores[left] >= scores[right]:
                        pair_slot = left - 1
        winner = max(range(len(scores)), key=lambda i: scores[i])
        target_wins = winner == 0 and scores[0] > max(scores[1:])
        if pair_max < COSINE_MAX and target_wins:
            return (
                {
                    "id": probe["id"],
                    "max_pair": float(pair_max),
                    "query_cosine": [float(score) for score in scores],
                    "target_argmax": True,
                },
                nonce,
            )
        if not target_wins:
            slot = max(range(len(group)), key=lambda i: scores[i + 1])
        else:
            slot = int(pair_slot)
        nonce += 1
        used = {int(item["frame"]) for i, item in enumerate(group) if i != slot}
        frame = (int(group[slot]["frame"]) + 1) % len(FRAMES)
        while frame in used:
            frame = (frame + 1) % len(FRAMES)
        name = name_for(nonce, forbidden, used_names)
        word = synth(nonce + 50000, forbidden, used_words)
        rel = f"msoftd{nonce}"
        group[slot].clear()
        group[slot].update(distractor(probe["id"], slot, frame, name, word, rel, nonce))
    keys = [key_of(probe)] + [key_of(item) for item in group]
    scores = [cosine(query, key) for key in keys]
    pair_max = 0.0
    for left in range(len(keys)):
        for right in range(left + 1, len(keys)):
            pair_max = max(pair_max, cosine(keys[left], keys[right]))
    raise RuntimeError(
        f"{probe['id']} stayed too close or lost the query "
        f"max_pair={pair_max:.4f} query={[round(s, 4) for s in scores]}"
    )


def render(payload, max_cosine: float) -> str:
    blob = json.dumps(payload, indent=2, sort_keys=True)
    if '"""' in blob:
        raise RuntimeError("payload cannot be embedded")
    header = f'''"""Fit probes and multi-slot distractors for the soft-mixture gate.

Held probes stay in ``tidal.subconscious_measure_data``. This module adds
the fit bank and the distractors that keep four slots alive. No margin has
been scored here.
"""
from __future__ import annotations

import hashlib
import json

N_FIT = {N_FIT}
N_DISTRACTORS = {N_DISTRACTORS}
COSINE_MAX = {COSINE_MAX}
MAX_KEY_COSINE = {max_cosine!r}
PROTOCOL = "subconscious-softmix-v1"

_PAYLOAD = json.loads(r"""
'''
    footer = '''
""")


def bank_sha256() -> str:
    blob = json.dumps(_PAYLOAD, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest()


def fit_items(family: str):
    return list(_PAYLOAD["fit"][family])


def distractors_for(probe_id: str):
    return list(_PAYLOAD["distractors"][probe_id])


def geometry():
    return _PAYLOAD["geometry"]
'''
    return header + blob + footer


def main() -> None:
    import torch

    from tidal.llm_memory import LLMMemory
    from tidal.subconscious_measure_data import choice_items, preference_items, tone_items

    torch.set_num_threads(os.cpu_count() or 1)
    texts = [text for text in existing_strings() if str(text).strip()]
    forbidden = set()
    for text in texts:
        forbidden |= tokens(text)
    used_words = set()
    used_names = set()
    fit = fit_items(forbidden, used_words, used_names)
    held = tone_items() + choice_items() + preference_items()
    distractors = {}
    distractors.update(assign_distractors([item for group in fit.values() for item in group], forbidden, used_words, used_names, 20000))
    distractors.update(assign_distractors(held, forbidden, used_words, used_names, 40000))
    payload = {"fit": fit, "distractors": distractors, "held_order": held}
    old_sets = [(tokens(text), text) for text in texts]
    fresh = surfaces({"fit": fit, "distractors": distractors})
    worst, pair = worst_jaccard(fresh, old_sets)
    print(f"worst jaccard before geometry {worst:.4f} {pair}", flush=True)
    if worst >= 0.5:
        raise SystemExit("fit bank is not disjoint")
    print("loading SmolLM2-135M for key geometry", flush=True)
    mem = LLMMemory(dtype=torch.float32)
    try:
        from tidal.subconscious_eval import token_ids

        for family, probes in fit.items():
            for probe in probes:
                prompt_ids = token_ids(mem.tokenizer, probe["prompt"])
                for continuation in (probe["good"], probe["bad"]):
                    full_ids = token_ids(mem.tokenizer, probe["prompt"] + continuation)
                    if full_ids[: len(prompt_ids)] != prompt_ids or len(full_ids) == len(prompt_ids):
                        raise RuntimeError(f"{probe['id']} continuation breaks the prompt boundary: {continuation!r}")
        geometry = repair_geometry(payload, mem, forbidden, used_words, used_names)
    finally:
        mem.close()
    fresh = surfaces({"fit": fit, "distractors": payload["distractors"]})
    worst, pair = worst_jaccard(fresh, old_sets)
    max_cosine = float(geometry["max_key_cosine"])
    broad = geometry["participation"]["0.40"]["4"]["mean"]
    print(f"worst jaccard {worst:.4f} {pair}", flush=True)
    print(f"max key cosine {max_cosine:.4f} target wins {geometry['target_argmax']}/{geometry['n_groups']}", flush=True)
    print(f"participation preview {geometry['participation']}", flush=True)
    if worst >= 0.5 or max_cosine >= COSINE_MAX:
        raise SystemExit("bank failed the disjointness or cosine cap")
    if geometry["target_argmax"] != geometry["n_groups"]:
        raise SystemExit("a distractor outranked its probe")
    if broad < 1.05:
        raise SystemExit("even T=0.40 leaves the bank peaked")
    # held_order is only the geometry walk. The stored module does not copy
    # held prompts or fact text.
    stored = {
        "fit": payload["fit"],
        "distractors": payload["distractors"],
        "geometry": {key: geometry[key] for key in ("max_key_cosine", "n_groups", "target_argmax", "participation", "groups")},
    }
    path = ROOT / "tidal" / "subconscious_softmix_data.py"
    path.write_text(render(stored, max_cosine), encoding="utf-8")
    print(f"wrote {path}", flush=True)


if __name__ == "__main__":
    main()
