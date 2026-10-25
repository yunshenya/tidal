"""Episodic fact slots for Continuum (native, always on).

The SmolLM study notes are in ``docs/llm_memory.md``. This module does not
import SmolLM2 or any 135M checkpoint. ``tidal.llm_memory`` reuses this store
to inject a frozen causal LM and is not called by ``Continuum.heads``.

Kept from the experiments that worked: one slot per fact ``(k, v, identity)``;
same-identity overwrite; capacity with FIFO eviction; soft top-k readout;
a write gate that drops chat-like noise. Not kept: summing several memory
matrices (the mixture diluted the value), a naïve key shield that restored
the previous key after an update (same-identity writes stopped sticking),
mean-pooling a multi-token span (digit order disappears), and L2-normalizing
values (that erases the byte code and the magnitude copy-through needs).

Keys are vectors already produced by tidal — encoder state ``h``, event
features, or a bge sentence vector run through the frozen model's own text
projection. A value is either another vector of that kind, or a positional
byte code for a string or a token list (phones, dates, multi-token spans).
Retrieval in this module is raw cosine over slot keys. The vector added by
``Continuum.heads`` is a convex combination of slot values (weights sum to 1).
Exact text is decoded from the winning slot's code, not from that blend. An
empty store returns zeros.

A residual inject MLP was measured on these byte codes and is not used
here: it pushed phone codes off the grid. ``tidal.llm_memory`` injects LM
hidden states, not these codes. The counts are in ``docs/episodic_memory.md``
and ``docs/llm_memory.md``.
"""
from __future__ import annotations

import math
import re
import zlib
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

from tidal.interface_v7 import check_fingerprint, check_modality, zero_fingerprint
from tidal.reconstruct import residue_from_answer

ArrayLike = Union[np.ndarray, Sequence[float]]
SKIP_KINDS = frozenset({"chat", "noise"})
# Default live write. Store writes may pass text|audio|image|fused|none.
# Fingerprint is stored on the slot; tidal does not compute it.
MEMORY_MODALITY = "text"
# Content-free clock. phase = sin(angle). Angle 0 is the identity.
# Write gain is 1 + CLOCK_SPAN * phase. Decay exponent rate is
# 1 - CLOCK_SPAN * phase. The grid is span in {0.25, 0.5, 1}. Span 0.25
# is the starting point; the fit script keeps the smallest span that
# clears the fit floors. CLOCK_STEP is the tick and is not on the grid.
# The angle is not read from fact text. 0 new parameters.
CLOCK_SPAN = 0.25
CLOCK_STEP = math.pi / 16
# Highest passing cell on fit bank
# 6f9530eb9a2684d258e7b2cd5714c61bb69fd6eb5950f5193e91f77d0d3130ba.
# The grid is (0.50, 0.67, 0.75, 0.85, 0.92, 1.0). A cell must collapse the
# added-token repeat (Jaccard 6/7) and keep the one-token replacement
# (Jaccard 5/7). 0.75 and 0.85 pass. The higher threshold wins. Not retuned
# on the held bank. Key cosine does not merge slots.
REPEAT_JACCARD = 0.85
_TOKEN_RE = re.compile(r"[\u4e00-\u9fff]|[A-Za-z0-9]+(?:'[A-Za-z]+)?")


def clock_write_gain(phase: float) -> float:
    """``1 + CLOCK_SPAN * phase``. Phase 0 is exactly 1."""
    return float(1.0 + float(CLOCK_SPAN) * float(phase))


def clock_decay_rate(phase: float) -> float:
    """``1 - CLOCK_SPAN * phase``. Phase 0 is exactly 1."""
    return float(1.0 - float(CLOCK_SPAN) * float(phase))


def _as1d(x: ArrayLike) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size == 0:
        raise ValueError("empty vector")
    if not np.isfinite(a).all():
        raise ValueError("non-finite vector")
    return a


def content_tokens(text: str) -> frozenset:
    """Surface tokens. CJK characters stay; other tokens are lowercased."""
    out = set()
    for match in _TOKEN_RE.finditer(str(text)):
        seg = match.group(0)
        if "\u4e00" <= seg[0] <= "\u9fff":
            out.add(seg)
        else:
            out.add(seg.lower())
    return frozenset(out)


def token_jaccard(left: str, right: str) -> float:
    """Jaccard of :func:`content_tokens`. Empty text is 0."""
    a = content_tokens(left)
    b = content_tokens(right)
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_scores(keys: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine of each row of ``keys`` with ``query``, denominator ``kn * qn + 1e-12``."""
    keys = np.asarray(keys, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64).reshape(-1)
    kn = np.linalg.norm(keys, axis=1)
    qn = float(np.linalg.norm(query))
    return (keys @ query) / (kn * qn + 1e-12)


# Positional byte code. Two payload bytes per lane, as k/2**16, which is exact
# in float32 and float64. Not L2-normalized: magnitude is the code.
_SPAN_SCALE = 65536.0
_SPAN_MAGIC = 0xE915
_SPAN_TAG_TEXT = 0x01
_SPAN_TAG_TOKENS = 0x02


def span_capacity(dim: int) -> int:
    """Bytes a value of width ``dim`` can hold, including the one-byte type tag."""
    dim = int(dim)
    if dim < 4:
        return 0
    return (dim - 4) * 2


def _encode_raw(raw: bytes, dim: int) -> np.ndarray:
    dim = int(dim)
    cap = span_capacity(dim)
    if dim < 4:
        raise ValueError(f"span value needs dim >= 4, got {dim}")
    if len(raw) < 1:
        raise ValueError("empty span")
    if len(raw) > cap:
        raise ValueError(f"span is {len(raw)} bytes, value dim {dim} holds {cap}")
    crc = zlib.crc32(raw) & 0xFFFFFFFF
    v = np.zeros(dim, dtype=np.float64)
    v[0] = _SPAN_MAGIC / _SPAN_SCALE
    v[1] = len(raw) / _SPAN_SCALE
    v[2] = (crc & 0xFFFF) / _SPAN_SCALE
    v[3] = ((crc >> 16) & 0xFFFF) / _SPAN_SCALE
    for i in range(0, len(raw), 2):
        b0 = raw[i]
        b1 = raw[i + 1] if i + 1 < len(raw) else 0
        v[4 + i // 2] = (b0 + (b1 << 8)) / _SPAN_SCALE
    return v


def encode_span(text: str, dim: int) -> np.ndarray:
    """Positional code for one string (phone, date, or a short span)."""
    if not isinstance(text, str):
        raise TypeError("span text must be a str")
    if text == "":
        raise ValueError("empty span")
    return _encode_raw(bytes([_SPAN_TAG_TEXT]) + text.encode("utf-8"), dim)


def encode_tokens(tokens: Sequence[str], dim: int) -> np.ndarray:
    """Positional code for an ordered token list. Order is part of the code."""
    if isinstance(tokens, (str, bytes, bytearray, np.ndarray)) or not isinstance(tokens, Sequence):
        raise TypeError("tokens must be a sequence of strings")
    if len(tokens) < 2:
        raise ValueError("token span needs at least two tokens")
    raw = bytearray([_SPAN_TAG_TOKENS])
    for t in tokens:
        if not isinstance(t, str) or t == "":
            raise ValueError("tokens must be non-empty strings")
        b = t.encode("utf-8")
        if len(b) > 255:
            raise ValueError("token longer than 255 bytes")
        raw.append(len(b))
        raw.extend(b)
    return _encode_raw(bytes(raw), dim)


def is_token_span(v: Any) -> bool:
    if isinstance(v, (str, bytes, bytearray, np.ndarray)):
        return False
    return isinstance(v, Sequence) and len(v) >= 1 and all(isinstance(x, str) for x in v)


def is_structured_value(v: Any) -> bool:
    """True for a string or a sequence of strings (not a numeric vector)."""
    return isinstance(v, str) or is_token_span(v)


def _lane_u16(x: float) -> Optional[int]:
    y = float(x) * _SPAN_SCALE
    if not np.isfinite(y):
        return None
    k = int(np.rint(y))
    # 0.5 is halfway between two bytes. Reject it instead of picking one.
    if abs(y - k) > 0.49 or k < 0 or k > 0xFFFF:
        return None
    return k


def _decode_raw(v: ArrayLike) -> Optional[bytes]:
    """Inverse of ``_encode_raw``. None when the lanes are not a clean code.

    A soft top-k blend of two different spans misses the uint16 grid or the
    CRC. The winning slot's stored vector still decodes.
    """
    try:
        a = np.asarray(v, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if a.size < 4 or not np.isfinite(a).all():
        return None
    if _lane_u16(a[0]) != _SPAN_MAGIC:
        return None
    n = _lane_u16(a[1])
    c0 = _lane_u16(a[2])
    c1 = _lane_u16(a[3])
    if n is None or c0 is None or c1 is None:
        return None
    cap = (a.size - 4) * 2
    if n < 1 or n > cap:
        return None
    n_lanes = (n + 1) // 2
    buf = bytearray()
    for i in range(n_lanes):
        lane = _lane_u16(a[4 + i])
        if lane is None:
            return None
        buf.append(lane & 0xFF)
        buf.append((lane >> 8) & 0xFF)
    if n % 2 == 1:
        if buf[-1] != 0:
            return None
        buf.pop()
    if len(buf) != n:
        return None
    for x in a[4 + n_lanes :]:
        if _lane_u16(x) != 0:
            return None
    crc = zlib.crc32(bytes(buf)) & 0xFFFFFFFF
    if c0 != (crc & 0xFFFF) or c1 != ((crc >> 16) & 0xFFFF):
        return None
    return bytes(buf)


def decode_span(v: ArrayLike) -> Optional[str]:
    """String from a positional code, or None if ``v`` is not one."""
    raw = _decode_raw(v)
    if raw is None or raw[0] != _SPAN_TAG_TEXT:
        return None
    try:
        text = raw[1:].decode("utf-8")
    except UnicodeDecodeError:
        return None
    if text == "":
        return None
    return text


def decode_tokens(v: ArrayLike) -> Optional[Tuple[str, ...]]:
    """Token list from a positional code, or None if ``v`` is not one."""
    raw = _decode_raw(v)
    if raw is None or raw[0] != _SPAN_TAG_TOKENS:
        return None
    body = raw[1:]
    i = 0
    out: list[str] = []
    while i < len(body):
        n = body[i]
        i += 1
        if n < 1 or i + n > len(body):
            return None
        try:
            out.append(body[i : i + n].decode("utf-8"))
        except UnicodeDecodeError:
            return None
        i += n
    if len(out) < 2:
        return None
    return tuple(out)


@dataclass
class Slot:
    k: np.ndarray
    v: np.ndarray
    name: Optional[str]
    rel: Optional[str]
    t: int
    group: Optional[str] = None
    # Sentence vector that produced a flat-row key. ask ranks these.
    # recall keeps using ``k``. None on a sparse name/relation key.
    sentence: Optional[np.ndarray] = None
    # Residual scale for the LM inject. Continuum readout ignores it.
    # 1 until replay recomputes decay, a nonzero clock phase scales the
    # write, or a later similar write scales an older neighbor. A refresh
    # sets it back to 1. McGaugh gain is consolidation_gain.
    strength: float = 1.0
    # High values are replayed first. Writers that do not stamp it leave 0.
    salience: float = 0.0
    # Consolidation cycles since the last refresh.
    age: int = 0
    # Durable surface for the reconsolidation identity check. Spelling does
    # not read this string. Continuum writes leave it empty and keep decoding
    # the byte code.
    answer: Optional[str] = None
    # Fact surface used for repeat detection. None when the writer passed none.
    # Key cosine is not this string.
    text: Optional[str] = None
    # Experience-grown preference. 0 until a stored experience adds to it.
    # Continuum readout does not read it. Replay multiplies it by the decay.
    valence: float = 0.0
    # Latest audit id. 0 before any experience, replay decay, or correction.
    episode_id: int = 0
    # Inspectable valence changes: episode id, store step, delta, value, kind.
    history: list = field(default_factory=list)
    # Write gain at stamp time. 1 until a nonzero clock phase. Not a fact.
    clock_gain: float = 1.0
    # McGaugh write / replay gain. 1 until above-baseline arousal meets a
    # nonzero |valence|. Not a cue-export field.
    consolidation_gain: float = 1.0
    # Record modality. Text-only live writes store "text".
    modality: str = MEMORY_MODALITY
    # growbrain encoder output. tidal stores and returns it. Width F=32.
    fingerprint: Optional[np.ndarray] = None
    # Phone or birthday parts. Not the raw string. None for every other answer.
    # Spelling reads this with strength, age, and valence.
    residue: Optional[tuple] = None
    # Post-retrieval labile deadline. 0 / 0.0 is closed. Not a cue-export field.
    labile_until_token: int = 0
    labile_until_wall: float = 0.0

    def __post_init__(self) -> None:
        self.modality = check_modality(str(self.modality))
        if self.fingerprint is None:
            self.fingerprint = zero_fingerprint()
        else:
            self.fingerprint = check_fingerprint(self.fingerprint)


class EpisodicStore:
    """Bounded episodic slots: append, same-identity overwrite, soft top-k read.

    Parameters
    ----------
    capacity : max slots. At capacity the oldest ``t`` is evicted (FIFO).
    overwrite_cos : accepted so existing callers keep working. Key cosine
        does not overwrite. Same-identity overwrite and the surface repeat
        rule do not read it.
    repeat_jaccard : overwrite a different identity when the surface token
        Jaccard is at least this value, the relation is the same non-empty
        string, and the group matches. ``None`` reads :data:`REPEAT_JACCARD`
        at call time. Identity match still wins over that repeat.
    soft_k, soft_temp : softmax over the top-``soft_k`` cosines (4 and 0.05).
    surprise_thr : when set, ``write(..., surprise=s)`` skips ``s < thr``.
        A missing ``surprise`` does not block the write.
    """

    def __init__(
        self,
        capacity: int = 256,
        *,
        dk: Optional[int] = None,
        dv: Optional[int] = None,
        overwrite_cos: float = 0.9,
        repeat_jaccard: Optional[float] = None,
        soft_k: int = 4,
        soft_temp: float = 0.05,
        surprise_thr: Optional[float] = None,
    ):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if not 0.0 < float(overwrite_cos) <= 1.0:
            raise ValueError("overwrite_cos must be in (0, 1]")
        if repeat_jaccard is None:
            repeat_jaccard = REPEAT_JACCARD
        if not 0.0 < float(repeat_jaccard) <= 1.0:
            raise ValueError("repeat_jaccard must be in (0, 1]")
        if int(soft_k) < 1:
            raise ValueError("soft_k must be >= 1")
        if float(soft_temp) <= 0.0:
            raise ValueError("soft_temp must be > 0")
        self.capacity = int(capacity)
        self.dk = None if dk is None else int(dk)
        self.dv = None if dv is None else int(dv)
        self.overwrite_cos = float(overwrite_cos)
        self.repeat_jaccard = float(repeat_jaccard)
        self.soft_k = int(soft_k)
        self.soft_temp = float(soft_temp)
        self.surprise_thr = surprise_thr
        self._slots: list[Slot] = []
        self._step = 0
        self._episode = 0
        self._clock_angle = 0.0
        self._clock_log: list = []

    def __len__(self) -> int:
        return len(self._slots)

    def clear(self) -> None:
        self._slots.clear()
        self._step = 0
        self._episode = 0
        self._clock_angle = 0.0
        self._clock_log = []

    def clock_phase(self) -> float:
        """``sin(angle)``. Angle 0 is 0, so the write and the decay stay put."""
        return float(math.sin(float(self._clock_angle)))

    def clock_set(self, angle: float) -> dict:
        """Set the global angle. The argument is a number, not fact text."""
        value = float(angle)
        if not math.isfinite(value):
            raise ValueError("clock angle must be finite")
        self._clock_angle = value
        return self._clock_record("set")

    def clock_tick(self, steps: int = 1) -> dict:
        """Advance the angle by ``steps * CLOCK_STEP``. No text argument."""
        count = int(steps)
        self._clock_angle = float(self._clock_angle) + count * float(CLOCK_STEP)
        return self._clock_record("tick")

    def _clock_record(self, kind: str) -> dict:
        phase = float(self.clock_phase())
        event = {
            "kind": str(kind),
            "angle": float(self._clock_angle),
            "phase": phase,
            "gain": float(clock_write_gain(phase)),
        }
        self._clock_log.append(event)
        return dict(event)

    def clock_log(self) -> tuple:
        """Phase changes only. Each row is kind, angle, phase, and gain."""
        return tuple(dict(event) for event in self._clock_log)

    @property
    def slots(self) -> Tuple[Slot, ...]:
        return tuple(self._slots)

    def gate_write(self, kind: str = "fact", surprise: Optional[float] = None) -> bool:
        """Drop chat / noise. Optional surprise threshold when both sides are set."""
        if kind in SKIP_KINDS:
            return False
        if self.surprise_thr is not None and surprise is not None and float(surprise) < float(self.surprise_thr):
            return False
        return True

    def _value_dim(self) -> int:
        if self.dv is not None:
            return int(self.dv)
        if self._slots:
            return int(self._slots[0].v.size)
        raise ValueError("span write needs a value width; construct the store with dv=")

    def _check_dims(self, k: np.ndarray, v: np.ndarray) -> None:
        if self.dk is not None and k.size != self.dk:
            raise ValueError(f"key dim {k.size} != store dk {self.dk}")
        if self.dv is not None and v.size != self.dv:
            raise ValueError(f"value dim {v.size} != store dv {self.dv}")
        if self._slots:
            if k.size != self._slots[0].k.size:
                raise ValueError("key dim mismatch with existing slots")
            if v.size != self._slots[0].v.size:
                raise ValueError("value dim mismatch with existing slots")
        if self.dk is None:
            self.dk = int(k.size)
        if self.dv is None:
            self.dv = int(v.size)

    def _content_repeat(self, slot: Slot, rel: Optional[str], group: Optional[str], text: Optional[str]) -> bool:
        """Same group, same non-empty relation, surface Jaccard at least the threshold."""
        if slot.group != group:
            return False
        if not isinstance(rel, str) or rel == "" or slot.rel != rel:
            return False
        if not isinstance(text, str) or text == "":
            return False
        if not isinstance(slot.text, str) or slot.text == "":
            return False
        return token_jaccard(text, slot.text) >= float(self.repeat_jaccard)

    def _locate(
        self,
        name: Optional[str],
        rel: Optional[str],
        group: Optional[str],
        text: Optional[str],
    ) -> tuple:
        """Identity index, then other slots whose surface repeats ``text``.

        Key cosine is not a hit. A repeat does not cross groups or relations.
        """
        hit_id = None
        for i, slot in enumerate(self._slots):
            if slot.group != group:
                continue
            if name is not None and (slot.name, slot.rel) == (name, rel):
                hit_id = i
                break
        content_ids = [
            i
            for i, slot in enumerate(self._slots)
            if i != hit_id and self._content_repeat(slot, rel, group, text)
        ]
        return hit_id, content_ids

    def write(
        self,
        k: ArrayLike,
        v: Any,
        name: Optional[str] = None,
        rel: Optional[str] = None,
        *,
        kind: str = "fact",
        surprise: Optional[float] = None,
        group: Optional[str] = None,
        sentence: Optional[np.ndarray] = None,
        answer: Optional[str] = None,
        text: Optional[str] = None,
        fingerprint: Optional[np.ndarray] = None,
        modality: Optional[str] = None,
    ) -> bool:
        """Insert or overwrite one slot. Returns True iff a write happened.

        ``v`` may be a numeric vector, a string, or a list of strings. A string
        or token list becomes a positional byte code of width ``dv`` (see
        ``encode_span``). Same-identity overwrite replaces that code; it is
        not added to the previous value. ``answer`` is stored for the
        reconsolidation identity check. Spelling does not read it. A phone or
        a birthday also stores a residue of its parts. Overwrite replaces both.
        A write that omits ``answer`` stores none. ``fingerprint`` is a
        growbrain encoder vector of width 32; omitted writes store zeros.
        ``modality`` is ``text``, ``audio``, ``image``, ``fused``, or
        ``none``; omitted writes store ``text``. A successful append then
        multiplies older near-neighbor strengths (Wixted 2004). Overwrite
        and a surface-repeat merge do not.
        """
        if not self.gate_write(kind, surprise):
            return False
        if isinstance(v, str):
            vv = encode_span(v, self._value_dim())
            kk = _as1d(k)
        elif is_token_span(v):
            vv = encode_tokens(v, self._value_dim())
            kk = _as1d(k)
        else:
            kk, vv = _as1d(k), _as1d(v)
        if float(np.linalg.norm(kk)) < 1e-12:
            return False
        self._check_dims(kk, vv)
        self._step += 1
        sent = None if sentence is None else _as1d(sentence).copy()
        stored = answer if isinstance(answer, str) and answer != "" else None
        stored_text = text if isinstance(text, str) and text != "" else None
        entry = Slot(
            k=kk.copy(),
            v=vv.copy(),
            name=name,
            rel=rel,
            t=self._step,
            group=group,
            sentence=sent,
            answer=stored,
            text=stored_text,
            modality=MEMORY_MODALITY if modality is None else modality,
            fingerprint=None if fingerprint is None else fingerprint,
            residue=residue_from_answer(stored),
        )
        hit_id, content_ids = self._locate(name, rel, group, stored_text)
        if hit_id is None and content_ids:
            hit_id = content_ids[0]
            content_ids = content_ids[1:]
        for index in reversed(content_ids):
            self._slots.pop(index)
            if hit_id is not None and index < hit_id:
                hit_id -= 1
        if hit_id is not None:
            old = self._slots[hit_id]
            # Same identity keeps the preference and its audit trail. A surface
            # repeat of a different identity starts at valence 0. The stored
            # answer is the new surface either way.
            if old.name == name and old.rel == rel and old.group == group:
                entry.valence = float(old.valence)
                entry.episode_id = int(old.episode_id)
                entry.history = list(old.history)
                entry.labile_until_token = int(old.labile_until_token)
                entry.labile_until_wall = float(old.labile_until_wall)
            self._slots[hit_id] = entry
            return True
        if len(self._slots) >= self.capacity:
            oldest = min(range(len(self._slots)), key=lambda i: (self._slots[i].t, i))
            self._slots.pop(oldest)
        self._slots.append(entry)
        # pattern_separation imports cosine_scores from this module, so the
        # write-time hit is imported here to break that cycle.
        from tidal.retroactive import apply_retroactive

        apply_retroactive(self._slots, kk, len(self._slots) - 1)
        return True

    def recompute_decay(self, decay: float) -> None:
        """Age every slot by one cycle and set ``strength = decay ** age``.

        ``decay`` is in ``(0, 1)``. Keys and values stay put. This is the
        decay replay refreshes; it is not a retrieval penalty.
        """
        factor = float(decay)
        if not 0.0 < factor < 1.0:
            raise ValueError("replay decay must be in (0, 1)")
        # Phase 0 keeps the integer power. A nonzero phase scales the exponent
        # and does not scale valence decay.
        rate = float(clock_decay_rate(self.clock_phase()))
        for slot in self._slots:
            slot.age = int(slot.age) + 1
            age = int(slot.age)
            if rate == 1.0:
                slot.strength = float(factor ** age)
            else:
                slot.strength = float(factor ** (age * rate))
        self.decay_valence(factor)

    def next_episode(self) -> int:
        """Mint one audit id. Fact writes, decay rows, and corrections each take one."""
        self._episode += 1
        return int(self._episode)

    def _find(self, name: Optional[str], rel: Optional[str], group: Optional[str]) -> Optional[Slot]:
        for slot in self._slots:
            if slot.name == name and slot.rel == rel and slot.group == group:
                return slot
        return None

    def _append_valence(
        self,
        slot: Slot,
        delta: float,
        updated: float,
        *,
        kind: str,
        source: str,
        episode_id: Optional[int] = None,
    ) -> dict:
        """Record one valence change. ``episode_id`` defaults to a new id."""
        if episode_id is None:
            self._episode += 1
            episode_id = int(self._episode)
        slot.valence = float(updated)
        slot.episode_id = int(episode_id)
        event = {
            "episode_id": int(episode_id),
            "t": int(slot.t),
            "delta": float(delta),
            "valence": float(updated),
            "kind": str(kind),
            "source": str(source),
        }
        slot.history.append(event)
        return dict(event)

    def add_experience(
        self,
        name: Optional[str],
        rel: Optional[str],
        sign: float,
        step: float,
        *,
        source: str,
        group: Optional[str] = None,
        episode_id: Optional[int] = None,
    ) -> dict:
        """Add ``sign * step`` from a stored experience. The text is not judged."""
        slot = self._find(name, rel, group)
        if slot is None:
            raise KeyError("no slot for that experience")
        direction = 1.0 if float(sign) > 0.0 else -1.0
        delta = direction * float(step)
        return self._append_valence(
            slot,
            delta,
            float(slot.valence) + delta,
            kind="experience",
            source=source,
            episode_id=episode_id,
        )

    def set_valence(
        self,
        name: Optional[str],
        rel: Optional[str],
        value: float,
        *,
        source: str,
        group: Optional[str] = None,
        kind: str = "correction",
    ) -> dict:
        """Replace the scalar and append a correction the history can show."""
        slot = self._find(name, rel, group)
        if slot is None:
            raise KeyError("no slot for that correction")
        updated = float(value)
        return self._append_valence(
            slot,
            updated - float(slot.valence),
            updated,
            kind=kind,
            source=source,
        )

    def decay_valence(self, factor: float) -> int:
        """Multiply every nonzero valence by ``factor`` and log the replay."""
        changed = 0
        for slot in self._slots:
            before = float(slot.valence)
            if before == 0.0:
                continue
            updated = before * float(factor)
            self._append_valence(slot, updated - before, updated, kind="replay", source="replay")
            changed += 1
        return changed

    def valence_history(
        self,
        name: Optional[str],
        rel: Optional[str],
        group: Optional[str] = None,
    ) -> tuple:
        slot = self._find(name, rel, group)
        if slot is None:
            raise KeyError("no slot for that history")
        return tuple(dict(event) for event in slot.history)

    def refresh_strength(self, indices: Sequence[int]) -> None:
        """Mark the given slots as just replayed. Strength returns to 1."""
        for index in indices:
            slot = self._slots[int(index)]
            slot.age = 0
            slot.strength = 1.0

    def ranked_by_salience(self) -> list:
        """Slot indices, highest salience first, then consolidation gain.

        Newer ``t`` breaks a remaining tie. When every gain is 1 the order
        matches the old ``(salience, t, -index)`` ranking.
        """
        return sorted(
            range(len(self._slots)),
            key=lambda i: (
                float(self._slots[i].salience),
                float(self._slots[i].consolidation_gain),
                int(self._slots[i].t),
                -i,
            ),
            reverse=True,
        )

    def _repeat_pair(self, left: Slot, right: Slot) -> bool:
        """Same group, same non-empty relation, surface Jaccard at the threshold."""
        return self._content_repeat(right, left.rel if isinstance(left.rel, str) else None, left.group, left.text)

    def near_duplicate_slots(self) -> int:
        """How many slots are in a same-relation surface repeat.

        The write path already collapses a new surface that repeats an
        existing one. A same-identity replace does not scan other slots
        before this count, so a leftover pair can still be here for replay.
        Key cosine is not a repeat.
        """
        hit: set[int] = set()
        slots = self._slots
        for i in range(len(slots)):
            for j in range(i + 1, len(slots)):
                if self._repeat_pair(slots[i], slots[j]):
                    hit.add(i)
                    hit.add(j)
        return len(hit)

    def merge_near_duplicates(self, *, limit: int, protect: Sequence[int] = ()) -> int:
        """Drop up to ``limit`` near-duplicate slots. Keep the stronger trace.

        A pair qualifies when the slots share a group, share a non-empty
        relation, and their surface token Jaccard is at least
        ``repeat_jaccard``. Key cosine does not qualify. The higher-salience
        slot is kept; a tie keeps the newer ``t``. A protected index is
        dropped only when its partner is protected too. Values are not
        blended. Cost is bounded by ``limit`` scans of the current slots.
        """
        budget = int(limit)
        if budget < 1 or len(self._slots) < 2:
            return 0
        protected = {int(index) for index in protect}
        merged = 0
        while merged < budget:
            pair = self._best_near_pair()
            if pair is None:
                break
            drop = self._drop_index(pair[0], pair[1], protected)
            self._slots.pop(drop)
            protected = {index - 1 if index > drop else index for index in protected if index != drop}
            merged += 1
        return merged

    def _best_near_pair(self) -> Optional[tuple]:
        best = None
        best_key = None
        slots = self._slots
        for i in range(len(slots)):
            for j in range(i + 1, len(slots)):
                left, right = slots[i], slots[j]
                if not self._repeat_pair(left, right):
                    continue
                key = (
                    max(float(left.salience), float(right.salience)),
                    max(int(left.t), int(right.t)),
                    -i,
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best = (i, j)
        return best

    def _drop_index(self, left: int, right: int, protected: set) -> int:
        left_on = left in protected
        right_on = right in protected
        if left_on and not right_on:
            return right
        if right_on and not left_on:
            return left
        a = self._slots[left]
        b = self._slots[right]
        key_a = (float(a.salience), int(a.t), -left)
        key_b = (float(b.salience), int(b.t), -right)
        if key_a == key_b:
            return max(left, right)
        return right if key_a > key_b else left

    def _visible(self, group: Optional[str]) -> list[tuple[int, Slot]]:
        if group is None:
            return list(enumerate(self._slots))
        return [(i, s) for i, s in enumerate(self._slots) if s.group is None or s.group == group]

    def _mix_visible(self, visible: list, c: np.ndarray, mode: str) -> Tuple[Optional[np.ndarray], dict]:
        """Soft top-k or top-1 over ``c``, one score per visible slot, in that order.

        ``read`` passes raw cosine. ``read_scored`` passes caller scores. The
        mix itself is the same convex combination of stored values.
        """
        info: dict[str, Any] = {"n": len(visible), "maxcos": -1.0, "top": None, "weights": None}
        if not visible:
            return None, info
        info["maxcos"] = float(c.max())
        top_local = int(np.argmax(c))
        top_i, s0 = visible[top_local]
        info["top"] = (s0.name, s0.rel)
        info["top_i"] = top_i
        info["group"] = s0.group
        if mode == "top1":
            return s0.v.copy(), info
        if mode not in ("softtopk", "softk4", "soft"):
            raise ValueError(f"unknown read mode {mode!r}")
        kk = min(self.soft_k, c.size)
        local = np.argpartition(-c, kk - 1)[:kk]
        local = local[np.argsort(-c[local])]
        tv = c[local]
        temp = max(self.soft_temp, 1e-8)
        w = np.exp((tv - tv.max()) / temp)
        w = w / w.sum()
        info["weights"] = w
        info["idx"] = np.array([visible[int(j)][0] for j in local], dtype=np.int64)
        out = np.zeros_like(visible[0][1].v)
        for j, loc in enumerate(local):
            out = out + w[j] * visible[int(loc)][1].v
        return out, info

    def read(
        self,
        q: ArrayLike,
        *,
        mode: str = "softtopk",
        group: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], dict]:
        """Value for query ``q``. ``top1`` is argmax raw cosine; ``softtopk`` mixes the top keys.

        Scores are raw cosine (``kn * qn + 1e-12``). The question head is not
        applied. ``group is None`` searches every slot. A group sees its own
        slots plus global slots (``group is None`` on the slot). Returns
        ``(None, info)`` when nothing is visible.
        """
        visible = self._visible(group)
        if not visible:
            return None, {"n": 0, "maxcos": -1.0, "top": None, "weights": None}
        qq = _as1d(q)
        if self.dk is not None and qq.size != self.dk:
            raise ValueError(f"query dim {qq.size} != store dk {self.dk}")
        K = np.stack([s.k for _, s in visible], 0)
        return self._mix_visible(visible, cosine_scores(K, qq), mode)

    def read_scored(
        self,
        scores: ArrayLike,
        *,
        mode: str = "softtopk",
        group: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], dict]:
        """Same value mix as ``read``, with one score per visible slot.

        ``read`` does not call this. ``Continuum.heads`` and ``readout`` stay
        on raw cosine. ``Continuum.query`` passes question-head scores here,
        in ``_visible(group)`` order.
        """
        visible = self._visible(group)
        if not visible:
            return None, {"n": 0, "maxcos": -1.0, "top": None, "weights": None}
        c = np.asarray(scores, dtype=np.float64).reshape(-1)
        if c.shape != (len(visible),):
            raise ValueError(f"scores length {c.size} != visible slots {len(visible)}")
        if not np.isfinite(c).all():
            raise ValueError("non-finite scores")
        return self._mix_visible(visible, c, mode)

    def read_span(
        self,
        q: ArrayLike,
        *,
        group: Optional[str] = None,
    ) -> Optional[str]:
        """Text in the winning slot, or None.

        Uses that slot's stored code. The soft top-k blend is a different
        vector: a near neighbor can push it off the byte grid, and decoding
        the blend would invent a third string.
        """
        v, _info = self.read(q, mode="top1", group=group)
        if v is None:
            return None
        return decode_span(v)

    def read_tokens(
        self,
        q: ArrayLike,
        *,
        group: Optional[str] = None,
    ) -> Optional[Tuple[str, ...]]:
        """Token list in the winning slot, or None. Same rule as ``read_span``."""
        v, _info = self.read(q, mode="top1", group=group)
        if v is None:
            return None
        return decode_tokens(v)

    def readout(
        self,
        q: ArrayLike,
        *,
        mode: str = "softtopk",
        out_dim: Optional[int] = None,
        group: Optional[str] = None,
    ) -> np.ndarray:
        """Float64 vector. Zeros when no slot is visible."""
        v, _ = self.read(q, mode=mode, group=group)
        dim = out_dim if out_dim is not None else (self.dv if self.dv is not None else (0 if v is None else int(v.size)))
        if v is None:
            return np.zeros(int(dim), dtype=np.float64)
        if out_dim is None or v.size == int(out_dim):
            return v
        out = np.zeros(int(out_dim), dtype=np.float64)
        n = min(v.size, int(out_dim))
        out[:n] = v[:n]
        return out

    def readout_torch(self, h, *, mode: str = "softtopk", group: Optional[str] = None):
        """``h`` is ``[n, d]`` → same shape. Empty store or a hidden-size mismatch is zeros.

        The store is not differentiable. Callers add the result onto ``h``;
        gradients flow through ``h``, not through the slots.
        """
        import torch

        if not isinstance(h, torch.Tensor):
            raise TypeError("readout_torch expects a torch.Tensor")
        if h.ndim != 2:
            raise ValueError("expected h with shape [n, d]")
        if (not self._slots) or self.dk is None or h.shape[-1] != self.dk or not self._visible(group):
            return torch.zeros_like(h)
        cpu = h.detach().cpu().numpy()
        rows = [self.readout(cpu[i], mode=mode, out_dim=h.shape[-1], group=group) for i in range(h.shape[0])]
        return torch.from_numpy(np.stack(rows, 0)).to(dtype=h.dtype, device=h.device)
