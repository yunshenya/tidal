"""Test-time memory for a frozen causal language model.

Facts are written into :class:`tidal.episodic.EpisodicStore` (one slot, same
identity overwrites). At generation the cue is the per-token hidden state at
the inject layer. The public name is ``cue`` and the width is ``cue_dim``.
The decoder index is not part of that interface. A memory is a candidate only
when ``cos(cue - query_center, key)`` is at least ``theta``. Theta does not
read strength. The winner is the one candidate with the greatest
``(log(strength), cosine, -index)``. One winner is added. There is no
mixture. Losers of a strict win still take the existing retrieval-induced
forgetting scale. A historical strength-weighted mixture is recorded in
``docs/llm_memory.md`` and in :func:`activation_weights`; generation does not
call it.

The add at token ``t`` uses the cue copied from the inject-layer residual
before the add, and the add is written at that same index. The cue does not
depend on the add, and no earlier position sees it. The logit at ``t``, which
is the distribution for token ``t+1``, does see it. On a trigger, ``dt`` is 0
and ``a = a_trig``. A later token that does not trigger uses
``a = a_trig * exp(-dt / tau)`` with ``dt`` the number of tokens after the
trigger. A new winner resets the clock. ``a`` below :data:`CUE_FLOOR` clears
the trace. The shipped generator re-forwards the whole prefix and recomputes
that recurrence; ``cue_state`` after the forward is the state after the last
token. A caller that forwards only the new token sets ``_cue_resume`` and
passes that state, the same way a key-value cache carries the past.

The value is an answer-span pool of that layer's hidden state. It is not
L2-normalized: magnitude is what the identity copy-through needs. A phone-like
answer (at least seven digits) is not averaged: the value concatenates the
leading coordinates of the last four answer tokens, which are the variable
digits. A birthday in the generator spelling (``March 4, 1991``) is not
averaged either: the value concatenates the leading coordinates of the month
token, the day-tens token (zeros for a one-digit day), the day-ones token,
and each of the four year-digit tokens. Every other answer pools the whole
span. The key is
not that pool. On SmolLM2-135M the first token is an attention sink (one
hidden dim near 2e4), and a mean over tokens collapses every fact onto one
direction. Keys and queries drop sink tokens, then subtract a fixed center
estimated once from a calibration set and stored with the module. On the
shipped causal LM that pool runs the decoder blocks up through the inject
layer and does not run the later blocks or the output head. The hidden state
is the one a full forward stores for that layer. Value and surprise still run
the whole model, because they need the logits. The per-token add is

    a * s * c * u * gain * (scale(v) + MLP(scale(v)))

``a`` is the decay above. ``s`` is the winner's strength frozen at the
trigger. ``c`` is the cosine-gap confidence, also frozen at the trigger: one
positive-strength slot is 1, and a cosine tie is 0 so the tie adds nothing.
``scale`` is fixed at 1. ``gain = 1 + 2 tanh(w · normalize(k))`` uses that
slot's own key. ``w = 0`` leaves ``gain = 1``. ``u = 1 + span * tanh(valence)``
uses that slot's valence. Valence stays 0 until a stored experience adds
``sign * step`` on that slot, so ``u = 1``. The sign is the experience field
on the write. The fact text is not scored, and the language model does not
supply a group prior. ``s`` stays 1 until a replay cycle recomputes decay, a
nonzero clock phase scales the write, or a same-relation retrieval scales a
competitor. A refresh sets the replayed slot back to 1. The same cycle
multiplies every nonzero valence by the replay decay and appends an audit row.
A fact is written when it is salient and, once 32 accepted surprises are
stored, its answer-token NLL is at least the quantile of that window of
numbers. Fewer stored surprises leaves the salient fact accepted. A strict
retrieval win then multiplies the strength of at most three other slots that
share the winner's relation; the closer key is scaled down more, and the
winner is left as read. ``gap_tau`` 0.80 still defines ``c``. The MLP weights
are the shipped checkpoint;
polarity does not replace them. The last layer of an untrained module is
still zeros, so that module copies ``v`` through when ``c`` is 1, ``s`` is 1,
and ``w`` is 0. Phone and birthday
spelling runs only when the winning relation is the one the prompt is
completing. The emitted string is that slot's stored answer, the full
number or the full date. It is not a trained prefix and not a digit
readout. A continuation encodes the query once. When this wrapper loads
the default SmolLM2-135M checkpoint at the default layer, the MLP is replaced
by the shipped weights in ``llm_inject_smollm2_135m.pt`` (the 1500-step MLP).
The checkpoint still holds linear readouts aimed at ``+1-415-555-dddd`` and
``Month D, YYYY``. Generation does not call them. A
caller-supplied model, another checkpoint, or another layer keeps the zero
last layer until :meth:`LLMMemory.load_inject`. There is no switch. This is
the A2s inject from
the SmolLM study (see ``docs/llm_memory.md``). It is a different representation
from the positional byte codes on ``Continuum.heads``; those codes stay on the
episodic path and do not go through this MLP.

``torch`` is imported by this module. ``transformers`` is imported only when a
checkpoint is loaded or :func:`build_tiny_lm` is called. Core ``import tidal``
does not load either one. Install the extra with ``pip install tidal[llm]``.

Default checkpoint: ``HuggingFaceTB/SmolLM2-135M``.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import math
import re

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tidal.cue_inject import CueState, trace_sequence
from tidal.episodic import EpisodicStore, clock_write_gain, cosine_scores
from tidal.llm_memory_data import _MONTHS as DATE_MONTHS
from tidal.llm_memory_data import parse_date

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_CAPACITY = 256
DEFAULT_SOFT_K = 4
DEFAULT_SOFT_TEMP = 0.05
DEFAULT_OVERWRITE_COS = 0.9
# A token is a sink when its norm is this many times the median token norm.
# SmolLM2-135M's first token is ~100x the others; ordinary tokens are not.
KEY_SINK_RATIO = 8.0
KEY_GEOM_VERSION = "sink8-qmean-v1"
_KEY_CENTER_FILE = "llm_key_center.npz"
# 1500-step inject trained after the key-center fix. ``save_inject`` dict.
_INJECT_FILE = "llm_inject_smollm2_135m.pt"
# Phone answers in ``llm_memory_data`` are ``+1-415-555-`` plus four digits.
# Seven digits leaves ``111`` and birthdays (at most six digits) on the
# full-span mean. Birthdays are detected by spelling, not by digit count,
# and use the date slice pool. Phones use four leading slices of the last
# four answer tokens. Their mean is not the value: the ones token in that
# mean is large enough to flip the tens digit.
PHONE_DIGIT_MIN = 7
PHONE_TAIL = "+1-415-555-"
# Smallest gap_tau on the grid 0.02..3.20 that passed the unrelated-store
# bound on fit bank 20b7cb3f9742709d320dc5913cf62675316673d7b457c55b6c78cf4de587e5ce
# at N=8 and N=32. Not retuned on the frozen bank or on this PR's held-out.
DEFAULT_GAP_TAU = 0.80
# Strength-weighted mixture over the top cosine neighbors. The continuum
# store still has its own soft_k and soft_temp; generate does not read those.
# Fit bank 3ba1fab4559982dcf959121224de3d0be86e03a25a056dccd874c6534dcdd6aa
# selected k=4 and temperature 0.05: the largest mean participation among
# cells whose mixture score was at least the top-1 score and whose mean
# participation was at least 1.05. k=8 tied that participation because each
# fit probe has four slots, and the smaller k wins the tie. Not retuned on
# the held-out bank.
MIX_K = 4
MIX_TEMP = 0.05
# Cue schedule. The floor is fixed and is not fit. Theta 0.32, tau 8, and
# a_trig 1 are the direction-fit cell in reports/subconscious_cue_prereg.md:
# the largest mean cued margin among the 24 cells that passed gates 1-5.
# Not retuned on the held-out bank. The interface names the signal ``cue``;
# cue_dim is the hidden width of whatever layer this wrapper injects at.
CUE_INTERFACE_VERSION = 1
CUE_FLOOR = 0.05
CUE_THETA = 0.32
CUE_TAU = 8.0
CUE_A_TRIG = 1.0
# gain = 1 + POLARITY_SPAN * tanh(w · normalize(k)). Span 2 is the smallest
# integer that lets an avoid key negate the residual (1 + 2 * -1 = -1).
# w = 0 leaves gain = 1. Not retuned on the frozen bank or the held-out bank.
POLARITY_SPAN = 2.0
# Fixed-budget replay. Every REPLAY_EVERY accepted writes, decay is
# recomputed as REPLAY_DECAY ** age, the top REPLAY_M slots by salience are
# refreshed to strength 1, and at most REPLAY_M same-group near-duplicates
# (cosine strictly above the store overwrite cosine, 0.9) are merged.
# Salience is |tanh(w · normalize(k))| on the polarity vector already in the
# checkpoint. 0 new parameters and no extra forward. The grid is N in
# {4, 8, 16, 32} and m in {1, 2, 4, 8}. 32 and 1 are the smallest budget on
# that grid (largest N, then smallest m) that clears the fit replay floors.
# Not retuned on the held-out bank. REPLAY_DECAY is not on the grid.
REPLAY_EVERY = 32
REPLAY_M = 1
REPLAY_DECAY = 0.5
# Experience valence. A write's experience field contributes its sign times
# VALENCE_STEP. The text is not scored. u = 1 + VALENCE_SPAN * tanh(valence).
# Valence 0 leaves u = 1, matching the replay add. Replay multiplies valence
# by REPLAY_DECAY. The grid is span in {1, 2} and step in {0.25, 0.5, 1}.
# Span 1 and step 0.25 are the smallest pair on that grid that clears the
# fit decision floors. Not retuned on the held-out bank. 0 new parameters.
VALENCE_SPAN = 1.0
VALENCE_STEP = 0.25
# Relative write gate. Surprise is the total answer-token NLL, in nats, from
# the logits the value forward already returned. The window keeps the last
# SURPRISE_WINDOW accepted surprises and no text. Until SURPRISE_MIN_COUNT
# accepted surprises are stored, every salient fact is written. After that,
# the fact is written when its surprise is at least the linear quantile of
# the window. A rejected attempt does not enter the window. Chat and noise
# return before any forward. The grid is the quantile in {0, 0.25, 0.5}.
# The window length is not on the grid. 0 new parameters. Not retuned on
# the held-out bank.
SURPRISE_WINDOW = 32
SURPRISE_MIN_COUNT = 32
SURPRISE_QUANTILE = 0.0
# Retrieval-induced forgetting. After a strict win, other slots that share
# the winner's relation are competitors when there are 1..RIF_MAX_SET of
# them. A larger shared relation is background (the note stores) and is not
# a cue. A slot whose salience reaches RIF_SALIENCE_EXEMPT is a polarity
# carrier and is not scaled. The keep is 1 - span * weight. Weight is 1 for
# the competitor whose key is closest to the winner and at least 1/2 for
# every other competitor in that set. The winner is not scaled. The grid is
# the span in {0.05, 0.10, 0.20}. The set size and the salience exempt are
# not on the grid. 0 new parameters. Not retuned on the held-out bank.
RIF_SPAN = 0.05
RIF_MAX_SET = 3
RIF_SALIENCE_EXEMPT = 0.5
_PHONE_WORD = re.compile(r"\bphone\b", re.IGNORECASE)
_BIRTHDAY_WORD = re.compile(r"\bbirthday\b", re.IGNORECASE)
_PHONE_KEYS = ("phone_on", "phone_mu", "phone_sd", "phone_w", "phone_b")
# Month, day-tens, day-ones, then four year digits. Absent from a checkpoint
# that was saved before the date readout; those buffers stay at init.
_DATE_KEYS = (
    "date_on",
    "date_mu",
    "date_sd",
    "date_month_w",
    "date_month_b",
    "date_digit_w",
    "date_digit_b",
)
_READOUT_KEYS = _PHONE_KEYS + _DATE_KEYS
# Absent from a checkpoint saved before polarity. Loading those tensors leaves
# ``w`` at zero, which is gain 1.
_OPTIONAL_INJECT_KEYS = ("polarity",)
# Shares of the value vector. Day-tens is the small block: it is a zero vector
# for days 1–9, and only digits 0–2 occur for days 1–27. The rest is split
# evenly across the month token, the day-ones token, and the four year digits.
_DATE_BLOCK_PARTS = (2, 1, 2, 2, 2, 2, 2)

# Held-out of the eval names in ``tidal.llm_memory_data``. One fact prefix and
# one question per name and relation. Answers are stripped from the key, so
# the center does not depend on the answer string.
_CALIB_NAMES = (
    "Sam", "Robin", "Alex", "Jordan", "Casey", "Riley",
    "Avery", "Quinn", "Parker", "Drew", "Shawn", "Logan",
)
_CALIB_ROWS = (
    ("phone", "{name}'s phone number is {a}", "What is {name}'s phone number?", "+1-202-555-0142"),
    ("birthday", "{name}'s birthday is {a}", "When is {name}'s birthday?", "March 4, 1991"),
    ("city", "{name} lives in {a}", "Which city does {name} live in?", "Tulsa"),
    ("color", "{name}'s favorite color is {a}", "What is {name}'s favorite color?", "maroon"),
    ("job", "{name}'s job is {a}", "What is {name}'s job?", "dock clerk"),
    ("team", "{name}'s team is {a}", "Which team is {name} on?", "paper kite"),
)

_LAYER_PATHS = (
    ("model", "layers"),
    ("model", "decoder", "layers"),
    ("transformer", "h"),
    ("transformer", "layers"),
    ("gpt_neox", "layers"),
    ("layers",),
)


def _require(module: str):
    from tidal.optional import require

    return require(module, "llm")


def _transformers():
    return _require("transformers")


def decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Decoder blocks of a causal LM (Llama, GPT-2, OPT, GPT-NeoX, …)."""
    for path in _LAYER_PATHS:
        obj: Any = model
        ok = True
        for name in path:
            if not hasattr(obj, name):
                ok = False
                break
            obj = getattr(obj, name)
        if ok and isinstance(obj, nn.ModuleList) and len(obj) > 0:
            return obj
    raise TypeError(
        "could not find decoder layers on this causal LM "
        "(looked for model.layers, model.decoder.layers, transformer.h, gpt_neox.layers)"
    )


def _hidden_size(model: nn.Module) -> int:
    cfg = getattr(model, "config", None)
    for name in ("hidden_size", "n_embd", "d_model"):
        value = getattr(cfg, name, None) if cfg is not None else None
        if value:
            return int(value)
    emb = model.get_input_embeddings() if hasattr(model, "get_input_embeddings") else None
    if emb is not None:
        return int(emb.embedding_dim)
    if hasattr(model, "embed"):
        return int(model.embed.embedding_dim)
    raise TypeError("could not find the model hidden size (config.hidden_size / n_embd)")


def _logits_of(out: Any) -> torch.Tensor:
    if hasattr(out, "logits"):
        return out.logits
    if isinstance(out, tuple) and out and torch.is_tensor(out[0]):
        return out[0]
    raise TypeError(f"model forward returned {type(out).__name__}, which has no logits")


def _hidden_tuple(out: Any) -> tuple:
    hs = getattr(out, "hidden_states", None)
    if hs is None and isinstance(out, tuple):
        for item in out:
            if isinstance(item, tuple) and item and torch.is_tensor(item[0]) and item[0].dim() == 3:
                hs = item
                break
    if not hs:
        raise RuntimeError("model did not return hidden states; output_hidden_states=True is required")
    return tuple(hs)


def layer_hidden(hidden_states: Sequence[torch.Tensor], layer: int, n_layers: int) -> torch.Tensor:
    """Residual stream at the output of decoder ``layer``.

    Hugging Face tuples are either one entry per block, or embeddings plus one
    entry per block. A non-final index matches that block's output.
    """
    n = len(hidden_states)
    if n == n_layers + 1:
        idx = layer + 1
    elif n == n_layers:
        idx = layer
    else:
        raise RuntimeError(f"hidden_states length {n} does not match {n_layers} decoder layers")
    h = hidden_states[idx]
    if h is None or not torch.is_tensor(h):
        raise RuntimeError(f"missing hidden state at layer {layer}")
    return h


def pool_masked(hidden: torch.Tensor, mask: Sequence[bool]) -> torch.Tensor:
    """Mean of rows selected by ``mask``. Not L2-normalized."""
    if hidden.dim() != 2:
        raise ValueError(f"pool expected [tokens, hidden], got {tuple(hidden.shape)}")
    m = torch.tensor(list(mask), dtype=hidden.dtype, device=hidden.device)
    if m.shape[0] != hidden.shape[0]:
        raise RuntimeError(f"pool mask {m.shape[0]} != tokens {hidden.shape[0]}")
    denom = m.sum()
    if float(denom) <= 0.0:
        raise ValueError("answer span produced no tokens to pool")
    return (hidden * m.unsqueeze(-1)).sum(dim=0) / denom


def phone_like(answer: str) -> bool:
    """True when ``answer`` has enough digits to be a phone number in this data."""
    return sum(ch.isdigit() for ch in str(answer)) >= PHONE_DIGIT_MIN


def date_like(answer: str) -> bool:
    """True for the generator spelling ``Month D, YYYY`` (day 1–27, year 1970–2019)."""
    return parse_date(answer) is not None


def phone_block_sizes(hidden: int) -> list:
    """Widths of the four phone digit blocks. They sum to ``hidden``.

    Order is thousands, hundreds, tens, ones: the last four answer tokens.
    Extra coordinates go to the later digits. On the shipped 576-d SmolLM2
    value this is 144, 144, 144, 144.
    """
    hidden = int(hidden)
    if hidden < 4:
        raise ValueError(f"phone pool needs at least 4 coordinates, got {hidden}")
    base, extra = divmod(hidden, 4)
    return [base + (1 if i >= 4 - extra else 0) for i in range(4)]


def assemble_phone_value(digit_rows: Sequence[torch.Tensor]) -> torch.Tensor:
    """Concatenate leading coordinates of the four phone digit tokens.

    ``digit_rows`` is thousands, hundreds, tens, ones. Averaging them is what
    flips the tens digit: on Ada's ``+1-415-555-2060`` at layer 15 the tens
    token moves L2 58.05 when 6 becomes 7, and the mean of the four tokens
    moves L2 16.63. Changing only the ones digit (2060→2061) leaves the tens
    token at L2 0 and moves that mean by L2 15.61, across the tens head
    (margin +0.057 on 2060, -0.029 on 2061). A test name moves the same mean
    by about L2 11 and is the Brigid ``2060`` → ``2070`` miss.
    """
    if len(digit_rows) != 4:
        raise ValueError("phone pool expects 4 digit tokens")
    rows = [row.reshape(-1) for row in digit_rows]
    hidden = int(rows[0].numel())
    sizes = phone_block_sizes(hidden)
    for vec in rows:
        if int(vec.numel()) != hidden:
            raise ValueError(f"phone token dim {int(vec.numel())} != {hidden}")
    return torch.cat([vec[:width] for vec, width in zip(rows, sizes)], dim=0).to(dtype=torch.float32)


def date_block_sizes(hidden: int) -> list:
    """Widths of the seven date blocks. They sum to ``hidden``.

    Order is month, day-tens, day-ones, year thousands, hundreds, tens, ones.
    On the shipped 576-d SmolLM2 value this is 88, 44, 88, 89, 89, 89, 89.
    """
    hidden = int(hidden)
    parts = _DATE_BLOCK_PARTS
    if hidden < len(parts):
        raise ValueError(f"date pool needs at least {len(parts)} coordinates, got {hidden}")
    total = sum(parts)
    sizes = [hidden * part // total for part in parts]
    remainder = hidden - sum(sizes)
    for i in range(remainder):
        sizes[3 + (i % 4)] += 1
    if sum(sizes) != hidden or any(size < 1 for size in sizes):
        raise RuntimeError(f"date block sizes {sizes} do not cover hidden {hidden}")
    return sizes


def assemble_date_value(
    month: torch.Tensor,
    day_rows: Sequence[torch.Tensor],
    year_rows: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Concatenate leading coordinates of the month, day, and year tokens.

    ``day_rows`` is one token (days 1–9) or two (the tens digit, then the ones
    digit). A one-digit day stores zeros in the tens block. ``year_rows`` is
    the four digit tokens, thousands first. Averaging those tokens is what
    makes a linear readout miss the day: the year tokens move by L2 4 when
    the day changes, and the day token moves by L2 82.
    """
    if len(year_rows) != 4 or len(day_rows) not in (1, 2):
        raise ValueError("date pool expects 1 or 2 day tokens and 4 year tokens")
    month = month.reshape(-1)
    hidden = int(month.numel())
    sizes = date_block_sizes(hidden)
    zeros = torch.zeros(hidden, dtype=month.dtype, device=month.device)
    day_tens = day_rows[0].reshape(-1) if len(day_rows) == 2 else zeros
    day_ones = day_rows[-1].reshape(-1)
    vectors = [month, day_tens, day_ones, *(row.reshape(-1) for row in year_rows)]
    for vec in vectors:
        if int(vec.numel()) != hidden:
            raise ValueError(f"date token dim {int(vec.numel())} != {hidden}")
    return torch.cat([vec[:width] for vec, width in zip(vectors, sizes)], dim=0).to(dtype=torch.float32)


def _date_components(hidden: torch.Tensor, offsets: Sequence[tuple], text: str, answer: str):
    """Month vector and day/year token rows, or None when the tokens do not line up.

    SmolLM2 splits ``March 4, 1991`` into one month token, one or two day
    digit tokens, and four year digit tokens. A tokenizer that merges the
    year returns None so the caller keeps the whole-span mean.
    """
    parsed = parse_date(answer)
    if parsed is None:
        return None
    _month_i, _day, _year, spans = parsed
    start = text.rfind(answer)
    if start < 0 or len(offsets) != hidden.shape[0]:
        return None
    groups = {"month": [], "day": [], "year": []}
    end = start + len(answer)
    for i, (a, b) in enumerate(offsets):
        a, b = int(a), int(b)
        if not (b > start and a < end and a != b):
            continue
        rel_a, rel_b = a - start, b - start
        best = None
        best_n = 0
        for name, (left, right) in spans.items():
            overlap = max(0, min(rel_b, right) - max(rel_a, left))
            if overlap > best_n:
                best, best_n = name, overlap
        if best is not None:
            groups[best].append(hidden[i])
    if not groups["month"] or len(groups["year"]) != 4 or len(groups["day"]) not in (1, 2):
        return None
    month = torch.stack(groups["month"]).mean(dim=0)
    return month, groups["day"], groups["year"]


def pool_answer(hidden: torch.Tensor, mask: Sequence[bool], answer: str) -> torch.Tensor:
    """Value pool at the inject layer. Not L2-normalized.

    Phone-like answers concatenate leading coordinates of the last four
    answer tokens (:func:`assemble_phone_value`). On SmolLM2 those tokens
    are the four variable digits. A mean of the whole ``+1-415-555-dddd``
    span moves only L2 3.93 when the last digit changes (norm 147.66). The
    mean of the last four still mixes the ones token into the tens digit;
    the slice does not. Fewer than four answer tokens falls back to their
    mean. Date answers are not pooled here; :meth:`LLMMemory.encode_value`
    concatenates field tokens instead. Every other answer uses
    :func:`pool_masked`. A one-token span is that token either way.
    """
    if hidden.dim() != 2:
        raise ValueError(f"pool expected [tokens, hidden], got {tuple(hidden.shape)}")
    idx = [i for i, flag in enumerate(mask) if flag]
    if not idx:
        raise ValueError("answer span produced no tokens to pool")
    if phone_like(answer):
        taken = idx[-4:]
        if len(taken) < 4:
            return hidden[taken].mean(dim=0)
        return assemble_phone_value([hidden[i] for i in taken])
    return pool_masked(hidden, mask)


def pool_key_tokens(rows: np.ndarray, ratio: float = KEY_SINK_RATIO) -> np.ndarray:
    """Mean of token rows, with attention-sink tokens left out.

    A sink token is much longer than the median token in the same string.
    Mean-pooling it makes every key point the same way, so cosine overwrite
    collapses a stream onto one slot. The rule is a function of this string
    only. It does not look at other facts. Values are not pooled here.
    """
    rows = np.asarray(rows, dtype=np.float64)
    if rows.ndim != 2 or rows.shape[0] < 1 or rows.shape[1] < 1:
        raise ValueError(f"key pool expected [tokens, hidden], got {tuple(rows.shape)}")
    if not np.isfinite(rows).all():
        raise ValueError("non-finite hidden state")
    if rows.shape[0] == 1 or float(ratio) <= 1.0:
        return rows.mean(axis=0)
    norms = np.linalg.norm(rows, axis=1)
    med = float(np.median(norms))
    if med <= 1e-8:
        return rows.mean(axis=0)
    keep = norms <= float(ratio) * med
    if not bool(np.any(keep)) or bool(np.all(keep)):
        return rows.mean(axis=0)
    return rows[keep].mean(axis=0)


def key_calibration() -> list:
    """Fixed ``(fact, answer, question)`` rows used to estimate key centers."""
    rows = []
    for name in _CALIB_NAMES:
        for _rel, fact_t, question_t, answer in _CALIB_ROWS:
            rows.append(
                (
                    fact_t.format(name=name, a=answer),
                    answer,
                    question_t.format(name=name),
                )
            )
    return rows


def _center_path() -> Path:
    return Path(__file__).with_name(_KEY_CENTER_FILE)


def _load_key_centers(path: Path, *, model_name: str, layer: int, hidden: int):
    """Shipped centers for the default SmolLM checkpoint. ``None`` if they do not match."""
    if not path.is_file():
        return None
    with np.load(path, allow_pickle=False) as z:
        if str(z["version"].item()) != KEY_GEOM_VERSION:
            return None
        if str(z["model"].item()) != str(model_name):
            return None
        if int(z["layer"]) != int(layer) or int(z["hidden"]) != int(hidden):
            return None
        if float(z["sink_ratio"]) != float(KEY_SINK_RATIO):
            return None
        key_mean = np.asarray(z["key_mean"], dtype=np.float64).reshape(-1)
        query_mean = np.asarray(z["query_mean"], dtype=np.float64).reshape(-1)
    if key_mean.shape != (int(hidden),) or query_mean.shape != (int(hidden),):
        return None
    if not np.isfinite(key_mean).all() or not np.isfinite(query_mean).all():
        return None
    return key_mean, query_mean


def save_key_centers(mem, path) -> None:
    """Write this module's key and query centers. Used to refresh the shipped file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        version=np.asarray(KEY_GEOM_VERSION),
        model=np.asarray(mem.model_name or ""),
        layer=np.int64(mem.layer),
        hidden=np.int64(mem.hidden),
        sink_ratio=np.float64(KEY_SINK_RATIO),
        key_mean=np.asarray(mem.key_mean, dtype=np.float32),
        query_mean=np.asarray(mem.query_mean, dtype=np.float32),
    )


def _inject_path() -> Path:
    return Path(__file__).with_name(_INJECT_FILE)


def _torch_load(path):
    try:
        return torch.load(str(path), map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(str(path), map_location="cpu")


def _ckpt_int(ckpt: Mapping[str, Any], key: str) -> Optional[int]:
    try:
        value = ckpt[key]
    except (KeyError, TypeError):
        return None
    if torch.is_tensor(value):
        if value.numel() != 1:
            return None
        value = value.detach().cpu().item()
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _inject_state_fits(module: nn.Module, state) -> bool:
    """True when ``state`` is a finite tensor dict with the module's shapes."""
    if not isinstance(state, dict):
        return False
    current = module.state_dict()
    if set(state) != set(current):
        return False
    for key, tensor in current.items():
        got = state[key]
        if not torch.is_tensor(got) or tuple(got.shape) != tuple(tensor.shape):
            return False
        if not bool(torch.isfinite(got).all()):
            return False
    return True


def _inject_core_fits(module: nn.Module, state) -> bool:
    """MLP tensors match. Phone, date, and polarity may be absent.

    A checkpoint saved before a readout only has the MLP keys. Polarity is
    optional: a file from before that vector still fits, and the load leaves
    ``w`` at zero. Unknown keys and shape mismatches do not fit.
    """
    if not isinstance(state, dict):
        return False
    current = module.state_dict()
    if not set(state) <= set(current):
        return False
    optional = set(_READOUT_KEYS) | set(_OPTIONAL_INJECT_KEYS)
    required = [key for key in current if key not in optional]
    if not set(required) <= set(state):
        return False
    for key, got in state.items():
        if not torch.is_tensor(got) or tuple(got.shape) != tuple(current[key].shape):
            return False
        if not bool(torch.isfinite(got).all()):
            return False
    return True


def _load_inject_tensors(module: nn.Module, state) -> None:
    """Load ``state`` onto ``module``. Missing readout buffers stay at init.

    A missing polarity vector is set to zero so an older checkpoint keeps
    gain 1 even if this module had a nonzero ``w``.
    """
    if not _inject_core_fits(module, state):
        raise ValueError("inject tensors do not match this MLP")
    merged = {key: tensor.detach().cpu().clone() for key, tensor in module.state_dict().items()}
    for key, tensor in state.items():
        merged[key] = tensor.detach().cpu()
    for key in _OPTIONAL_INJECT_KEYS:
        if key not in state and key in merged:
            merged[key] = torch.zeros_like(merged[key])
    module.load_state_dict(merged)


class InjectMLP(nn.Module):
    """``gain * (scaled + MLP(scaled))``. Last layer is exact zeros.

    Without a key, or with the polarity vector at zero, gain is 1 and the
    add starts as ``scaled``. A key multiplies that residual by
    ``1 + POLARITY_SPAN * tanh(w · normalize(k))``.
    """

    def __init__(self, d: int, hid: Optional[int] = None):
        super().__init__()
        d = int(d)
        h = d if hid is None else int(hid)
        if d < 1 or h < 1:
            raise ValueError("inject width must be >= 1")
        self.net = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)
        self.register_buffer("scale", torch.ones((), dtype=torch.float32))
        # Linear phone readout. ``phone_on`` stays 0 until a fit writes it, so
        # a fresh module does not spell a number. ``phone_w[p, digit]`` is the
        # weight of digit 0-9 at variable position p. ``phone_b`` is 0 for a
        # digit the fit saw and a large negative for one it did not.
        self.register_buffer("phone_on", torch.zeros((), dtype=torch.float32))
        self.register_buffer("phone_mu", torch.zeros(d, dtype=torch.float32))
        self.register_buffer("phone_sd", torch.ones(d, dtype=torch.float32))
        self.register_buffer("phone_w", torch.zeros(4, 10, d, dtype=torch.float32))
        self.register_buffer("phone_b", torch.zeros(4, 10, dtype=torch.float32))
        # Linear birthday readout. ``date_on`` stays 0 until a fit writes it.
        # ``date_month_w[m]`` is month 0–11 (January–December).
        # ``date_digit_w`` is day-tens, day-ones, then the four year digits.
        # ``date_digit_b`` is 0 for a digit the fit saw and a large negative
        # for one it did not, same as ``phone_b``.
        self.register_buffer("date_on", torch.zeros((), dtype=torch.float32))
        self.register_buffer("date_mu", torch.zeros(d, dtype=torch.float32))
        self.register_buffer("date_sd", torch.ones(d, dtype=torch.float32))
        self.register_buffer("date_month_w", torch.zeros(12, d, dtype=torch.float32))
        self.register_buffer("date_month_b", torch.zeros(12, dtype=torch.float32))
        self.register_buffer("date_digit_w", torch.zeros(6, 10, d, dtype=torch.float32))
        self.register_buffer("date_digit_b", torch.zeros(6, 10, dtype=torch.float32))
        # Zero so a fresh module, and any checkpoint saved before this vector,
        # copies the residual through. Trained only by ``fit_polarity``.
        self.polarity = nn.Parameter(torch.zeros(d, dtype=torch.float32))

    def polarity_gain(self, key: torch.Tensor) -> torch.Tensor:
        """Scalar ``1 + POLARITY_SPAN * tanh(w · normalize(key))``."""
        k = key.to(dtype=self.polarity.dtype, device=self.polarity.device).reshape(-1)
        if k.numel() != self.polarity.numel():
            raise ValueError(f"key dim {k.numel()} != polarity {self.polarity.numel()}")
        unit = k / k.norm().clamp_min(1e-6)
        raw = torch.dot(self.polarity, unit)
        return 1.0 + float(POLARITY_SPAN) * torch.tanh(raw)

    def forward(self, value: torch.Tensor, key: Optional[torch.Tensor] = None) -> torch.Tensor:
        scaled = value * self.scale
        residual = scaled + self.net(scaled)
        if key is None:
            return residual
        # w = 0 is exactly the residual. Skip the multiply on the inference
        # path so a zero vector stays bitwise identical. Training keeps the
        # graph, including at the zero initialization.
        inactive = (not torch.is_grad_enabled()) or (not self.polarity.requires_grad)
        if inactive and float(self.polarity.detach().abs().sum()) == 0.0:
            return residual
        return self.polarity_gain(key) * residual

    def phone_digits(self, value: torch.Tensor) -> Optional[list]:
        """Four digits read from a phone value, or None when the readout is off.

        Training still uses this head. Generation does not. The historical
        template those digits were appended to is :data:`PHONE_TAIL`.
        """
        if float(self.phone_on.detach().cpu()) == 0.0:
            return None
        v = value.detach().to(dtype=torch.float32, device=self.phone_mu.device).reshape(-1)
        if v.numel() != self.phone_mu.numel():
            raise ValueError(f"phone value dim {v.numel()} != {self.phone_mu.numel()}")
        z = (v - self.phone_mu) / self.phone_sd
        logits = torch.einsum("d,pcd->pc", z, self.phone_w) + self.phone_b
        return [int(i) for i in torch.argmax(logits, dim=-1).tolist()]

    def date_fields(self, value: torch.Tensor) -> Optional[tuple]:
        """``(month_index, day, year)`` from a date value, or None when the readout is off."""
        if float(self.date_on.detach().cpu()) == 0.0:
            return None
        v = value.detach().to(dtype=torch.float32, device=self.date_mu.device).reshape(-1)
        if v.numel() != self.date_mu.numel():
            raise ValueError(f"date value dim {v.numel()} != {self.date_mu.numel()}")
        z = (v - self.date_mu) / self.date_sd
        month = int(torch.argmax(self.date_month_w @ z + self.date_month_b).item())
        logits = torch.einsum("d,pcd->pc", z, self.date_digit_w) + self.date_digit_b
        digits = [int(i) for i in torch.argmax(logits, dim=-1).tolist()]
        day = digits[0] * 10 + digits[1]
        year = digits[2] * 1000 + digits[3] * 100 + digits[4] * 10 + digits[5]
        return month, day, year

    def date_spelling(self, value: torch.Tensor) -> Optional[str]:
        """``Month D, YYYY`` from :meth:`date_fields`, or None when the readout is off."""
        fields = self.date_fields(value)
        if fields is None:
            return None
        month, day, year = fields
        if not 0 <= int(month) < len(DATE_MONTHS):
            return None
        return f"{DATE_MONTHS[int(month)]} {int(day)}, {int(year)}"


class ByteTokenizer:
    """One token per code point in U+0000–U+00FF. Offsets are character indexes.

    No download. The tiny CPU training path and the unit tests use this.
    A fast Hugging Face tokenizer's ``offset_mapping`` is the same character
    coordinate, so answer-span pooling is shared.
    """

    def __init__(self):
        self.eos_token_id = 256
        self.pad_token_id = 0
        self.bos_token_id = None
        self.vocab_size = 257

    def __call__(
        self,
        text,
        return_tensors=None,
        return_offsets_mapping=False,
        add_special_tokens=True,
        **_kwargs,
    ):
        del add_special_tokens  # no bos/eos inside the span; generate stops on eos
        if isinstance(text, (list, tuple)):
            if len(text) != 1:
                raise ValueError("ByteTokenizer encodes one string at a time")
            text = text[0]
        if not isinstance(text, str):
            raise TypeError("ByteTokenizer expects a str")
        ids = []
        offsets = []
        for i, ch in enumerate(text):
            o = ord(ch)
            if o > 255:
                raise ValueError(f"ByteTokenizer only encodes U+0000–U+00FF, got U+{o:04X}")
            ids.append(o)
            offsets.append((i, i + 1))
        out = {"input_ids": ids, "attention_mask": [1] * len(ids)}
        if return_offsets_mapping:
            out["offset_mapping"] = offsets
        if return_tensors == "pt":
            out["input_ids"] = torch.tensor([ids], dtype=torch.long)
            out["attention_mask"] = torch.tensor([out["attention_mask"]], dtype=torch.long)
            if return_offsets_mapping:
                out["offset_mapping"] = torch.tensor([offsets], dtype=torch.long)
        return out

    def encode(self, text, add_special_tokens=True):
        del add_special_tokens
        return list(self(text)["input_ids"])

    def decode(self, ids, skip_special_tokens=True, **_kwargs):
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        chars = []
        for i in ids:
            i = int(i)
            if skip_special_tokens and i >= 256:
                continue
            if 0 <= i <= 255:
                chars.append(chr(i))
        return "".join(chars)


def _offset_pairs(enc: Mapping[str, Any]) -> Optional[list]:
    if "offset_mapping" not in enc:
        return None
    off = enc["offset_mapping"]
    if hasattr(off, "detach"):
        off = off.detach().cpu().tolist()
    elif hasattr(off, "tolist"):
        off = off.tolist()
    if not off:
        return []
    if isinstance(off[0], (list, tuple)) and off[0] and isinstance(off[0][0], (list, tuple)):
        off = off[0]
    return [(int(a), int(b)) for a, b in off]


def _batch_ids(enc: Mapping[str, Any]) -> torch.Tensor:
    ids = enc["input_ids"]
    if not torch.is_tensor(ids):
        ids = torch.tensor(ids, dtype=torch.long)
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    return ids


def _batch_mask(enc: Mapping[str, Any], device) -> Optional[torch.Tensor]:
    if "attention_mask" not in enc:
        return None
    mask = enc["attention_mask"]
    if not torch.is_tensor(mask):
        mask = torch.tensor(mask, dtype=torch.long)
    if mask.dim() == 1:
        mask = mask.unsqueeze(0)
    return mask.to(device=device)


def _parse_fact(fact, answer, name, rel, key, group, kind):
    if isinstance(fact, str):
        text = fact
    elif isinstance(fact, Mapping):
        text = fact.get("text", fact.get("fact"))
        if answer is None:
            answer = fact.get("a", fact.get("answer"))
        if name is None:
            name = fact.get("n", fact.get("name"))
        if rel is None:
            rel = fact.get("rel", fact.get("relation"))
        if key is None:
            key = fact.get("q", fact.get("key"))
        if group is None and "group" in fact:
            group = fact.get("group")
        if kind == "fact" and fact.get("kind"):
            kind = fact.get("kind")
    else:
        raise TypeError("fact must be a str or a mapping with 'text'")
    if not isinstance(text, str) or text == "":
        raise ValueError("fact text is empty")
    if answer is None:
        mark = " is "
        j = text.rfind(mark)
        if j < 0 or j + len(mark) >= len(text):
            raise ValueError("pass answer= so the value can be pooled on the answer span")
        answer = text[j + len(mark) :]
    if not isinstance(answer, str) or answer == "":
        raise ValueError("answer span is empty")
    if answer not in text:
        raise ValueError("answer is not a substring of the fact")
    for label, value in (("name", name), ("rel", rel), ("key", key)):
        if value is not None and not isinstance(value, str):
            raise TypeError(f"{label} must be a str")
        if value == "":
            raise ValueError(f"{label} is empty")
    return text, answer, name, rel, key, group, kind


def _experience_sign(fact) -> Optional[float]:
    """Sign of a stored experience. Missing or zero leaves valence unchanged.

    The magnitude is ignored. The update rule supplies the step. The fact
    text is not read, so the language model cannot plant a group prior.
    """
    if not isinstance(fact, Mapping) or "experience" not in fact:
        return None
    value = fact.get("experience")
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise TypeError("experience must be a number") from exc
    if number == 0.0:
        return 0.0
    if number > 0.0:
        return 1.0
    return -1.0


def _content_from_offsets(offsets: Sequence[tuple]) -> list:
    return [a != b for a, b in offsets]


def _answer_from_offsets(offsets: Sequence[tuple], text: str, answer: str) -> list:
    start = text.rfind(answer)
    if start < 0:
        raise ValueError(f"answer {answer!r} is not a substring of the fact")
    end = start + len(answer)
    return [b > start and a < end and a != b for a, b in offsets]


def answer_span(text: str) -> str:
    """First answer line of a greedy continuation.

    Leading whitespace, including the newline a base SmolLM2 emits after a
    natural question, is not part of the span. The span then ends at the next
    newline, or at sentence-ending punctuation once more than one non-space
    character has been kept. Cutting at a leading newline, which is what the
    old v22 stop did, scores that continuation as empty.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    body = text.lstrip()
    if not body:
        return ""
    kept: list[str] = []
    for ch in body:
        if ch == "\n":
            break
        kept.append(ch)
        cur = "".join(kept)
        if cur.rstrip().endswith((".", "?", "!")) and len(cur.strip()) > 1:
            break
    return "".join(kept)


def span_complete(text: str) -> bool:
    """True once another character cannot extend :func:`answer_span`.

    A string that is still only leading whitespace is not complete: the
    answer may start on a later token.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a str")
    body = text.lstrip()
    if body == "":
        return False
    line, sep, _rest = body.partition("\n")
    if sep:
        return True
    return line.rstrip().endswith((".", "?", "!")) and len(line.strip()) > 1


def decode_answer(tokenizer, ids) -> str:
    """Answer span of greedy ``ids``, using incremental detokenization.

    Stops when :func:`span_complete` flips, so a later token cannot be pulled
    into the span. ``ids`` may be longer than that cut; the extra tokens are
    ignored.
    """
    out: list[int] = []
    for nx in ids:
        out.append(int(nx))
        text = tokenizer.decode(out, skip_special_tokens=True)
        if span_complete(text):
            return answer_span(text)
    if not out:
        return ""
    return answer_span(tokenizer.decode(out, skip_special_tokens=True))


def greedy_continuation(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    eos_token_id: Optional[int],
    device,
    done: Optional[Callable[[list], bool]] = None,
) -> list:
    """Greedy token ids appended to ``prompt``. Full prefix each step (no cache).

    ``done`` sees the new ids so far and ends the loop. Generation also stops
    on ``eos_token_id`` before that id is appended.
    """
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    ids = _batch_ids(enc).to(device)
    if ids.shape[1] == 0:
        raise ValueError("prompt tokenized to an empty sequence")
    attn = _batch_mask(enc, device)
    prompt_len = int(ids.shape[1])
    new_ids: list[int] = []
    for _ in range(max_new_tokens):
        kwargs = {"input_ids": ids, "use_cache": False}
        if attn is not None:
            kwargs["attention_mask"] = attn
        logits = _logits_of(model(**kwargs))
        nxt = int(torch.argmax(logits[0, -1]).item())
        if eos_token_id is not None and nxt == int(eos_token_id):
            break
        step = torch.tensor([[nxt]], device=ids.device, dtype=ids.dtype)
        ids = torch.cat([ids, step], dim=1)
        if attn is not None:
            attn = torch.cat([attn, torch.ones((1, 1), device=attn.device, dtype=attn.dtype)], dim=1)
        new_ids.append(nxt)
        if done is not None and done(new_ids):
            break
    return new_ids


def build_tiny_lm(seed: int = 0, *, hidden: int = 32, layers: int = 4, vocab: int = 260):
    """Random LlamaForCausalLM plus :class:`ByteTokenizer`. No download."""
    tf = _transformers()
    torch.manual_seed(int(seed))
    if hidden % 4 != 0:
        raise ValueError("tiny hidden size must be divisible by 4 attention heads")
    cfg = tf.LlamaConfig(
        vocab_size=int(vocab),
        hidden_size=int(hidden),
        intermediate_size=int(hidden) * 2,
        num_hidden_layers=int(layers),
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=256,
        bos_token_id=1,
        eos_token_id=256,
        pad_token_id=259,
        tie_word_embeddings=False,
    )
    model = tf.LlamaForCausalLM(cfg)
    model.eval()
    return model, ByteTokenizer()


def completing_relation(prompt: str) -> Optional[str]:
    """Relation a prompt is completing, or None.

    ``phone`` wins when that word is present. Otherwise ``birthday``. A longer
    word such as ``telephone`` does not match.
    """
    if not isinstance(prompt, str) or prompt == "":
        return None
    if _PHONE_WORD.search(prompt):
        return "phone"
    if _BIRTHDAY_WORD.search(prompt):
        return "birthday"
    return None


def _mean_float(values) -> Optional[float]:
    if not values:
        return None
    return float(sum(float(value) for value in values) / len(values))


def factual_salience(kind: str) -> float:
    """1 for a fact, 0 for chat and noise. This is the existing write filter."""
    if kind in ("chat", "noise"):
        return 0.0
    return 1.0


def surprise_quantile(samples, q: float) -> float:
    """Linear quantile of a surprise window. ``q`` is in ``[0, 1]``.

    The position is ``q * (n - 1)`` on the sorted window, with linear
    interpolation between the surrounding samples. ``q = 0`` is the minimum.
    """
    level = float(q)
    if not 0.0 <= level <= 1.0:
        raise ValueError("surprise quantile must be in [0, 1]")
    ordered = sorted(float(value) for value in samples)
    n = len(ordered)
    if n == 0:
        raise ValueError("surprise window is empty")
    if n == 1:
        return float(ordered[0])
    pos = level * (n - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    weight = pos - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def accept_relative_write(
    kind: str,
    surprise: float,
    window,
    *,
    quantile: float,
    min_count: int,
) -> bool:
    """Write a salient fact during warmup, else when it clears the window quantile.

    Warmup is ``len(window) < min_count``. Chat and noise never write.
    """
    if factual_salience(kind) <= 0.0:
        return False
    if len(window) < int(min_count):
        return True
    bar = surprise_quantile(window, quantile)
    return float(surprise) >= float(bar)


def answer_surprise(logits: torch.Tensor, token_ids: Sequence[int], answer_mask: Sequence[bool]) -> float:
    """Total NLL, in nats, of the answer tokens under logits the forward already returned.

    Position ``i`` is predicted by ``logits[i - 1]``. The score is the sum,
    not the mean. The first token has no predictor; if the answer is only
    that token, the score is 0.
    """
    if not torch.is_tensor(logits):
        raise TypeError("logits must be a tensor")
    if logits.dim() == 3:
        if int(logits.shape[0]) != 1:
            raise ValueError("answer surprise expects one sequence")
        logits = logits[0]
    if logits.dim() != 2:
        raise ValueError(f"logits rank {logits.dim()} is not a sequence")
    ids = [int(i) for i in token_ids]
    mask = [bool(flag) for flag in answer_mask]
    n = int(logits.shape[0])
    if len(ids) != n or len(mask) != n:
        raise ValueError(f"token axis {len(ids)} / mask {len(mask)} != logits {n}")
    idxs = [i for i, flag in enumerate(mask) if flag and i >= 1]
    if not idxs:
        return 0.0
    rows = logits[[i - 1 for i in idxs]]
    target = torch.tensor([ids[i] for i in idxs], dtype=torch.long, device=rows.device)
    logp = torch.log_softmax(rows.float(), dim=-1)
    taken = logp.gather(1, target.unsqueeze(1)).squeeze(1)
    return float((-taken.sum()).detach().cpu())


def rif_weight(similarity: float, closest: float) -> float:
    """1 for the closest same-cue key, and at least 1/2 for every other one.

    ``closest`` is the highest key cosine in the competitor set. A set with
    no positive cosine still takes half a span, so the cue is not a no-op.
    """
    nearest = float(closest)
    if nearest <= 1e-8:
        return 0.5
    ratio = max(0.0, float(similarity)) / nearest
    if ratio > 1.0:
        ratio = 1.0
    return 0.5 + 0.5 * ratio


def rif_competitor_keep(similarity: float, closest: float, span: float) -> float:
    """Strength multiplier for one same-relation competitor. The winner is not passed here."""
    level = float(span)
    if not 0.0 <= level < 1.0:
        raise ValueError("rif span must be in [0, 1)")
    return 1.0 - level * rif_weight(similarity, closest)


def rival_indexes(slots, winner: int, max_set: int, salience_exempt: float) -> list:
    """Other slots that share the winner's relation, when that set is small.

    More than ``max_set`` siblings means the relation is background, not a
    cue. A slot at or above ``salience_exempt`` is left out of the list
    after that size check, so a polarity carrier inside a real cue is kept.
    """
    if winner < 0 or winner >= len(slots):
        raise IndexError("rif winner is outside the store")
    rel = slots[winner].rel
    if not isinstance(rel, str) or rel == "":
        return []
    raw = [i for i, slot in enumerate(slots) if i != winner and slot.rel == rel]
    if not raw or len(raw) > int(max_set):
        return []
    return [i for i in raw if float(slots[i].salience) < float(salience_exempt)]


def forget_rivals(slots, winner: int) -> int:
    """Scale same-cue competitors. Returns how many strengths changed.

    The span, the set cap, and the salience exempt are the module globals,
    read at call time. Keys already in the slots are the only vectors used.
    """
    rivals = rival_indexes(slots, winner, RIF_MAX_SET, RIF_SALIENCE_EXEMPT)
    if not rivals:
        return 0
    winner_key = np.asarray(slots[winner].k, dtype=np.float64)
    keys = np.stack([np.asarray(slots[i].k, dtype=np.float64) for i in rivals], 0)
    sims = cosine_scores(keys, winner_key)
    closest = float(np.max(sims))
    changed = 0
    for index, sim in zip(rivals, sims.tolist()):
        keep = rif_competitor_keep(float(sim), closest, RIF_SPAN)
        before = float(slots[index].strength)
        updated = before * float(keep)
        if updated != before:
            changed += 1
        slots[index].strength = updated
    return changed


def retrieval_confidence(n: int, gap: Optional[float], tau: float) -> float:
    """Scale for the residual. One slot is unscaled. A tie adds nothing."""
    if int(n) < 2:
        return 1.0
    if float(tau) <= 0.0:
        raise ValueError("gap_tau must be positive")
    if gap is None:
        return 0.0
    return float(min(1.0, max(0.0, float(gap) / float(tau))))


def activation_weights(cosines, strengths, *, temperature: float, k: int) -> np.ndarray:
    """Normalized weights for a strength-weighted soft mixture.

    The match score is ``log(strength) + cosine``. The weight is
    ``exp(match / temperature)`` on the ``k`` largest cosines whose strength
    is positive, then divided by the sum. This is an analogue of the ACT-R
    chunk-choice rule in Lebiere (1999): probability proportional to
    ``exp(match / t)``. Our strength is a multiplicative trace in ``(0, 1]``,
    not ACT-R base-level log odds, so ``log(strength)`` is the stand-in for
    that base level. Cosine is the stand-in for the context match. Slots
    outside the cosine neighborhood, and slots with nonpositive strength,
    get weight 0.
    """
    if float(temperature) <= 0.0:
        raise ValueError("mixture temperature must be positive")
    take = int(k)
    if take < 1:
        raise ValueError("mixture k must be >= 1")
    cosine = np.asarray(cosines, dtype=np.float64).reshape(-1)
    strength = np.asarray(strengths, dtype=np.float64).reshape(-1)
    if cosine.shape != strength.shape:
        raise ValueError("cosines and strengths must have the same shape")
    weights = np.zeros(cosine.shape[0], dtype=np.float64)
    if cosine.size == 0:
        return weights
    neighbors = min(take, int(cosine.size))
    if neighbors == int(cosine.size):
        order = np.arange(cosine.size)
    else:
        order = np.argpartition(-cosine, neighbors - 1)[:neighbors]
    chosen = []
    logits = []
    temp = float(temperature)
    for index in order:
        slot_strength = float(strength[int(index)])
        if slot_strength <= 0.0:
            continue
        logits.append((math.log(slot_strength) + float(cosine[int(index)])) / temp)
        chosen.append(int(index))
    if not chosen:
        return weights
    shifted = np.asarray(logits, dtype=np.float64)
    shifted -= shifted.max()
    raw = np.exp(shifted)
    raw /= raw.sum()
    for index, weight in zip(chosen, raw):
        weights[index] = float(weight)
    return weights


def participation_ratio(weights) -> float:
    """``1 / sum(w^2)`` over positive weights. One active slot returns 1."""
    active = np.asarray(weights, dtype=np.float64).reshape(-1)
    active = active[active > 0.0]
    if active.size == 0:
        return 0.0
    return float(1.0 / np.sum(active * active))


def valence_gain_of(valence: float) -> float:
    """``1 + VALENCE_SPAN * tanh(valence)``. Zero valence is exactly 1."""
    return float(1.0 + float(VALENCE_SPAN) * math.tanh(float(valence)))


class LLMMemory:
    """Frozen causal LM with episodic residual memory.

    ``remember`` writes one fact. ``generate`` answers from the store. The
    prompt is the only text the LM sees. Each token's cue is the inject-layer
    residual before the add. One slot whose cosine with that cue clears
    :data:`CUE_THETA` is the winner, ranked by log strength. The residual is
    that slot's composed value times activation, strength, confidence, and
    valence gain, at that token only. It then decays until the floor or until
    another token wins. Valence stays 0 until an experience is stored, so the
    gain stays 1. Strength stays 1 until replay or a nonzero clock phase. A
    fact is written when it is salient and, once 32 surprises are stored, its
    answer-token NLL is at least the window quantile. A strict win scales at
    most three other slots that share the winner's relation; the winner is
    not scaled. Every :data:`REPLAY_EVERY` accepted writes, decay is
    recomputed, the top :data:`REPLAY_M` salient slots are refreshed, and at
    most that many same-relation surface repeats are merged. Key cosine does
    not merge slots. Salience reuses the polarity vector. ``soft_k`` and
    ``soft_temp`` stay on the episodic store; ``generate`` does not read
    them, and it does not read :data:`MIX_K`. An empty store leaves the
    residual untouched, so greedy tokens match the same model with the hook
    removed. Phone and birthday spelling copies the winning slot's stored
    answer when that slot is still active on the last prompt token and the
    prompt is completing that relation. A zero polarity vector leaves the
    gain at 1.

    Loading ``HuggingFaceTB/SmolLM2-135M`` at the default layer
    (``n_layers // 2``, which is layer 15, hidden 576) installs the shipped
    inject in ``llm_inject_smollm2_135m.pt``. The cue interface still calls
    that width ``cue_dim``. A caller-supplied model, another checkpoint, or
    another layer keeps the zero-init MLP, so the add is ``v``, until
    :meth:`load_inject`. There is no switch.
    """

    def __init__(
        self,
        model: Optional[nn.Module] = None,
        tokenizer=None,
        model_name: str = DEFAULT_MODEL,
        *,
        layer: Optional[int] = None,
        capacity: int = DEFAULT_CAPACITY,
        soft_k: int = DEFAULT_SOFT_K,
        soft_temp: float = DEFAULT_SOFT_TEMP,
        overwrite_cos: float = DEFAULT_OVERWRITE_COS,
        device: Optional[Union[str, torch.device]] = None,
        dtype: Optional[torch.dtype] = None,
        inject_hidden: Optional[int] = None,
    ):
        loaded_name = None
        if model is None:
            model, tokenizer = _load_pretrained(model_name, device=device or "cpu", dtype=dtype)
            loaded_name = str(model_name)
        elif tokenizer is None:
            raise ValueError("tokenizer is required when model is passed in")
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self._device = torch.device(device)
        self.model_name = loaded_name
        self.model = model.to(self._device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.tokenizer = tokenizer
        self.layers = decoder_layers(self.model)
        n_layers = len(self.layers)
        if layer is None:
            layer = n_layers // 2
        layer = int(layer)
        if not 0 <= layer < n_layers:
            raise ValueError(f"layer {layer} is outside 0..{n_layers - 1}")
        self.layer = layer
        self.n_layers = n_layers
        self.hidden = _hidden_size(self.model)
        self.inject = InjectMLP(self.hidden, inject_hidden).to(self._device)
        self.inject.eval()
        self.store = EpisodicStore(
            capacity=capacity,
            dk=self.hidden,
            dv=self.hidden,
            overwrite_cos=overwrite_cos,
            soft_k=soft_k,
            soft_temp=soft_temp,
        )
        self._train_delta: Optional[torch.Tensor] = None
        self._cue_live = False
        self._cue_resume = False
        self.cue_state = CueState()
        self.last_cue_fields: Optional[dict] = None
        self.last_added: Optional[torch.Tensor] = None
        self.last_rif = {"n": 0}
        self._accepted = 0
        self._surprise_window: list[float] = []
        self.last_surprise: Optional[float] = None
        self.last_write_score: float = 0.0
        self.last_replay: Optional[dict] = None
        # Kept so older callers can assign it. The cue path does not read it.
        self.injection_readout = "cue"
        # Registered for the life of this wrapper. A forward that is not a
        # retrieval and is not a teacher-forced inject returns None, which
        # leaves the layer output object untouched.
        self._handle = self.layers[self.layer].register_forward_hook(self._on_layer)
        self._load_shipped_inject()
        self.key_mean, self.query_mean = self._key_centers()

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self._handle.remove()
            self._handle = None
        self._train_delta = None
        self._cue_live = False

    @property
    def cue_dim(self) -> int:
        """Width of the cue. This is the inject hidden size, not a layer index."""
        return int(self.hidden)

    @contextmanager
    def _clean_forward(self):
        """Encode a key or a value without retrieval and without a training add."""
        live = self._cue_live
        train = self._train_delta
        self._cue_live = False
        self._train_delta = None
        try:
            yield
        finally:
            self._cue_live = live
            self._train_delta = train

    def __enter__(self):
        return self

    def __exit__(self, *_exc) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _on_layer(self, _module, _args, output):
        """Add at this token after copying the cue, or broadcast a training value.

        The cue is the block output before this add. The add is applied at
        that same index, so the logit at this position sees it and the cue
        tensor does not. A training inject is the only remaining whole-sequence
        add, and only :meth:`teacher_force_loss` sets it. Retrieval never does.
        """
        train = self._train_delta
        if train is not None:
            self.last_added = train.detach()
            return _add_delta(output, train)
        if not self._cue_live or len(self.store) == 0:
            return None
        tensor = output[0] if isinstance(output, tuple) else output
        if not torch.is_tensor(tensor):
            return None
        raw = tensor.detach()
        batched = raw.dim() == 3
        if raw.dim() == 2:
            raw = raw.unsqueeze(0)
        if raw.dim() != 3:
            return None
        batch, _steps, width = raw.shape
        if int(width) != int(self.hidden):
            raise ValueError(f"cue dim {width} != hidden {self.hidden}")
        cues = raw.float().cpu().numpy()
        self._last_cue_rows = cues[0]
        resume = bool(self._cue_resume) and int(raw.shape[1]) == 1 and self.cue_state is not None
        states = self._resume_states(int(batch)) if resume else [CueState() for _ in range(int(batch))]
        packed = []
        residuals = []
        new_states = []
        for index in range(int(batch)):
            fields, residual, state = self._trace_numpy(cues[index], state=states[index])
            packed.append(fields)
            residuals.append(residual)
            new_states.append(state)
        self.cue_state = new_states[0] if int(batch) == 1 else new_states
        self.last_cue_fields = _pack_cue_batch(packed, float(self.clock_phase()))
        add = np.stack(residuals, 0)
        self.last_added = torch.tensor(np.asarray(add[0, -1], dtype=np.float32))
        if np.count_nonzero(add) == 0:
            return None
        add_t = torch.tensor(add, dtype=tensor.dtype, device=tensor.device)
        if not batched:
            add_t = add_t[0]
        updated = tensor + add_t
        if torch.is_tensor(output):
            return updated
        return (updated,) + output[1:]

    def _resume_states(self, batch: int) -> list:
        state = self.cue_state
        if int(batch) == 1 and isinstance(state, CueState):
            return [state]
        if isinstance(state, list) and len(state) == int(batch):
            return list(state)
        return [CueState() for _ in range(int(batch))]

    def _tokenize(self, text: str) -> Mapping[str, Any]:
        kwargs = {"return_tensors": "pt", "add_special_tokens": True, "return_offsets_mapping": True}
        try:
            return self.tokenizer(text, **kwargs)
        except (TypeError, ValueError):
            kwargs.pop("return_offsets_mapping")
            return self.tokenizer(text, **kwargs)

    def _key_prefix_ok(self) -> bool:
        """True when the key can run the decoder prefix instead of the whole model.

        The prefix is embeddings, the causal mask, rotary embeddings, and
        decoder blocks ``0 .. layer`` inclusive. That is the stack
        ``LlamaModel.forward`` runs before the later blocks and the final
        norm. Passed-in models that do not have that stack keep the full
        forward. Their hidden state is whatever ``output_hidden_states``
        returns.
        """
        inner = getattr(self.model, "model", None)
        layers = getattr(inner, "layers", None)
        if layers is not self.layers or not (0 <= int(self.layer) < len(self.layers)):
            return False
        if not callable(getattr(inner, "embed_tokens", None)):
            return False
        if not callable(getattr(inner, "rotary_emb", None)):
            return False
        return getattr(inner, "config", None) is not None

    def _run_key_hidden(self, text: str) -> tuple:
        """Hidden state at the inject layer, without the later blocks or the head.

        The shipped stack is ``LlamaModel.forward``. That method runs
        ``config.num_hidden_layers`` blocks and then a final norm. The key
        pool needs the inject block's output, so this sets that count to
        ``layer + 1``, runs the same forward, and copies the block output
        from a hook. The final norm still runs on that prefix. The later
        blocks and the output head do not. The count is restored before
        return. The residual is cleared first, matching :meth:`_run_hidden`.
        The copied tensor is the one :func:`layer_hidden` reads from a full
        forward. A model without this stack uses that full forward instead.
        """
        if not isinstance(text, str) or text == "":
            raise ValueError("text is empty")
        if not self._key_prefix_ok():
            hidden, enc, _logits = self._run_hidden(text)
            del _logits
            return hidden, enc
        enc = self._tokenize(text)
        ids = _batch_ids(enc).to(self._device)
        attn = _batch_mask(enc, self._device)
        inner = self.model.model
        captured: dict = {}

        def _keep(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            copied = tensor.detach()
            if copied.dim() == 3:
                copied = copied[0]
            captured["hidden"] = copied.float()

        handle = self.layers[self.layer].register_forward_hook(_keep)
        saved = inner.config.num_hidden_layers
        inner.config.num_hidden_layers = int(self.layer) + 1
        try:
            kwargs = {"input_ids": ids, "use_cache": False}
            if attn is not None:
                kwargs["attention_mask"] = attn
            with self._clean_forward():
                with torch.no_grad():
                    inner(**kwargs)
        finally:
            inner.config.num_hidden_layers = saved
            handle.remove()
        hidden = captured.get("hidden")
        if hidden is None:
            hidden, enc, _logits = self._run_hidden(text)
            del _logits
            return hidden, enc
        return hidden, enc

    def _run_hidden(self, text: str) -> tuple:
        """Hidden state at the inject layer, with memory unhooked for this forward."""
        if not isinstance(text, str) or text == "":
            raise ValueError("text is empty")
        enc = self._tokenize(text)
        ids = _batch_ids(enc).to(self._device)
        attn = _batch_mask(enc, self._device)
        kwargs = {"input_ids": ids, "use_cache": False, "output_hidden_states": True}
        if attn is not None:
            kwargs["attention_mask"] = attn
        with self._clean_forward():
            with torch.no_grad():
                out = self.model(**kwargs)
            hidden = layer_hidden(_hidden_tuple(out), self.layer, self.n_layers)
            if hidden.dim() == 3:
                hidden = hidden[0]
            # Transformers may load SmolLM2 in bfloat16. Numpy cannot hold that
            # dtype, and the key center is estimated in float32/float64.
            hidden = hidden.detach().float()
            logits = _logits_of(out).detach()
        return hidden, enc, logits

    def _answer_mask(self, text: str, enc, answer: str, n_tokens: int) -> list:
        offsets = _offset_pairs(enc)
        if offsets is not None:
            if len(offsets) != n_tokens:
                raise RuntimeError(f"offset map {len(offsets)} != tokens {n_tokens}")
            mask = _answer_from_offsets(offsets, text, answer)
            if any(mask):
                return mask
            raise ValueError(
                f"answer {answer!r} did not overlap any token offsets; "
                "a fast tokenizer with offset_mapping is required for this span"
            )
        ids = _batch_ids(enc)[0].tolist()
        return _subsequence_mask(self.tokenizer, ids, answer)

    def _content_mask(self, enc, n_tokens: int) -> list:
        offsets = _offset_pairs(enc)
        if offsets is not None:
            if len(offsets) != n_tokens:
                raise RuntimeError(f"offset map {len(offsets)} != tokens {n_tokens}")
            mask = _content_from_offsets(offsets)
            if any(mask):
                return mask
        return [True] * n_tokens

    def encode_value(self, text: str, answer: str) -> np.ndarray:
        """Answer-span pool at the inject layer. Not L2-normalized.

        Phone-like answers concatenate leading slices of the last four answer
        tokens (:func:`assemble_phone_value`). Date answers concatenate
        field-token slices (:func:`assemble_date_value`). Other answers pool
        the whole span. A date whose tokenizer does not yield 1–2 day tokens
        and 4 year tokens falls back to the whole-span mean. A phone with
        fewer than four answer tokens falls back to the mean of those tokens.
        """
        value, _surprise = self._value_and_surprise(text, answer)
        return value

    def _value_and_surprise(self, text: str, answer: str):
        """Answer-span value and its total answer-token NLL from one forward."""
        hidden, enc, logits = self._run_hidden(text)
        mask = self._answer_mask(text, enc, answer, hidden.shape[0])
        ids = _batch_ids(enc)[0].tolist()
        surprise = answer_surprise(logits, ids, mask)
        del logits
        if date_like(answer):
            offsets = _offset_pairs(enc)
            if offsets is not None:
                parts = _date_components(hidden, offsets, text, answer)
                if parts is not None:
                    month, day_rows, year_rows = parts
                    packed = assemble_date_value(month, day_rows, year_rows)
                    value = packed.detach().cpu().numpy().astype(np.float32, copy=False)
                    return value, surprise
        value = pool_answer(hidden, mask, answer).detach().cpu().numpy().astype(np.float32, copy=False)
        return value, surprise

    def _raw_key_pool(self, text: str, answer: Optional[str] = None, *, full: bool = False) -> np.ndarray:
        """Sink-dropped mean of the key tokens, before the fixed center.

        ``full`` keeps the uncut forward. The timing run uses it as the before
        column. Production leaves it false and, on the shipped causal LM,
        runs only the decoder blocks through the inject layer.
        """
        if full or not self._key_prefix_ok():
            hidden, enc, _logits = self._run_hidden(text)
            del _logits
        else:
            hidden, enc = self._run_key_hidden(text)
        content = self._content_mask(enc, hidden.shape[0])
        if answer is None:
            mask = content
        else:
            answer_m = self._answer_mask(text, enc, answer, hidden.shape[0])
            mask = [c and not a for c, a in zip(content, answer_m)]
            if not any(mask):
                mask = content
        idx = [i for i, flag in enumerate(mask) if flag]
        if not idx:
            raise ValueError("key span produced no tokens to pool")
        rows = hidden[idx].detach().cpu().numpy()
        return pool_key_tokens(rows, ratio=KEY_SINK_RATIO)

    def _key_centers(self):
        """Fixed key and query means. Shipped for the default SmolLM checkpoint.

        A passed-in model, or a different layer, is estimated from
        :func:`key_calibration` once at init. The means do not move as facts
        arrive. Fact keys and questions use different means: on SmolLM2 the
        shared mean left the 10th-percentile cosine gap at 0.021, and the
        question mean raised it to 0.072 (soft top-4 mass 0.66 to 0.83).
        """
        if self.model_name:
            loaded = _load_key_centers(
                _center_path(),
                model_name=self.model_name,
                layer=self.layer,
                hidden=self.hidden,
            )
            if loaded is not None:
                return loaded
        keys = []
        queries = []
        for text, answer, question in key_calibration():
            keys.append(self._raw_key_pool(text, answer))
            queries.append(self._raw_key_pool(question))
        return np.mean(np.stack(keys), axis=0), np.mean(np.stack(queries), axis=0)

    def _load_shipped_inject(self) -> None:
        """Install the packaged SmolLM2-135M MLP when this wrapper is that model.

        The file is the :meth:`save_inject` dict: the MLP state dict plus
        ``layer`` and ``hidden``. It is applied only when this object loaded
        :data:`DEFAULT_MODEL` and those two fields match. A missing file, any
        other model or layer, or a narrower ``inject_hidden`` keeps the zero
        last layer (the shipped tensors would not fit a narrower MLP). There
        is no switch. :meth:`load_inject` still replaces the MLP from any path.
        """
        if self.model_name != DEFAULT_MODEL:
            return
        path = _inject_path()
        if not path.is_file():
            return
        ckpt = _torch_load(path)
        if not isinstance(ckpt, dict):
            raise ValueError(f"{path.name} is not an inject checkpoint")
        layer = _ckpt_int(ckpt, "layer")
        hidden = _ckpt_int(ckpt, "hidden")
        state = ckpt.get("inject")
        if layer is None or hidden is None or not isinstance(state, dict):
            raise ValueError(f"{path.name} must contain 'inject', 'layer', and 'hidden'")
        if layer != int(self.layer) or hidden != int(self.hidden):
            return
        if not _inject_core_fits(self.inject, state):
            # Default MLP width is the hidden size. A file that names this
            # layer and hidden but does not fit that MLP is the wrong artifact.
            if int(self.inject.net[0].out_features) != int(self.hidden):
                return
            raise ValueError(
                f"{path.name} records layer {layer} and hidden {hidden}, "
                "but the tensors do not match the default inject MLP"
            )
        _load_inject_tensors(self.inject, state)
        self.inject.to(self._device)
        self.inject.eval()

    def encode_key(self, text: str, answer: Optional[str] = None) -> np.ndarray:
        """Pool used as the slot key.

        With ``answer``, the key is the fact with that span removed. Without
        it, the key is the whole string (a question or a prefix). Sink tokens
        are dropped and the fixed key center is subtracted. Queries use
        :meth:`_query_vector`, which subtracts the question center instead.
        """
        raw = self._raw_key_pool(text, answer)
        return (raw - self.key_mean).astype(np.float32, copy=False)

    def remember(
        self,
        fact: Union[str, Mapping[str, Any]],
        answer: Optional[str] = None,
        *,
        name: Optional[str] = None,
        rel: Optional[str] = None,
        key: Optional[str] = None,
        group: Optional[str] = None,
        kind: str = "fact",
        surprise: Optional[float] = None,
    ) -> bool:
        """Write one fact into the episodic store. Same ``(name, rel)`` overwrites.

        Chat and noise return before any forward. A fact's surprise is the
        total NLL of its answer tokens under the logits of the value forward.
        Until the surprise window holds 32 accepted writes, the fact is
        written. After that it is written when that NLL is at least the
        window quantile. The key forward runs only when the write is kept.
        A caller-supplied ``surprise`` does not replace the measured one.
        """
        # The measured answer NLL is the score. A caller value cannot bypass it.
        del surprise
        text, answer, name, rel, key, group, kind = _parse_fact(
            fact, answer, name, rel, key, group, kind
        )
        experience = _experience_sign(fact)
        self.last_surprise = None
        self.last_write_score = 0.0
        if len(self.store) == 0:
            self._accepted = 0
            self._surprise_window = []
        if factual_salience(kind) <= 0.0 or not self.store.gate_write(kind):
            return False
        value, measured = self._value_and_surprise(text, answer)
        self.last_surprise = float(measured)
        self.last_write_score = float(measured)
        if not accept_relative_write(
            kind,
            measured,
            self._surprise_window,
            quantile=SURPRISE_QUANTILE,
            min_count=SURPRISE_MIN_COUNT,
        ):
            return False
        if key is None:
            key_vec = self.encode_key(text, answer)
        else:
            key_vec = self.encode_key(key)
        wrote = bool(
            self.store.write(
                key_vec,
                value,
                name=name,
                rel=rel,
                kind=kind,
                group=group,
                answer=answer,
                text=text,
            )
        )
        if not wrote:
            return False
        episode_id = self.store.next_episode()
        self._stamp_salience(name, rel, group, key_vec)
        slot = self.store._find(name, rel, group)
        if slot is None:
            raise RuntimeError("accepted write did not leave a slot")
        slot.episode_id = int(episode_id)
        if experience not in (None, 0.0):
            self.store.add_experience(
                name,
                rel,
                float(experience),
                float(VALENCE_STEP),
                source=text,
                group=group,
                episode_id=episode_id,
            )
        self._surprise_window.append(float(measured))
        overflow = len(self._surprise_window) - int(SURPRISE_WINDOW)
        if overflow > 0:
            del self._surprise_window[:overflow]
        self._accepted += 1
        if self._accepted % int(REPLAY_EVERY) == 0:
            self._consolidate()
        return True

    def surprise_window(self) -> tuple:
        """Accepted write surprises. Numbers only, never fact text."""
        return tuple(float(value) for value in self._surprise_window)

    def _salience(self, key_vec) -> float:
        """``|tanh(w · normalize(k))|``. Zero when polarity is still zero.

        This is the same ``w`` the gain uses. It does not add a parameter
        and it does not run the language model.
        """
        weight = self.inject.polarity.detach().float().cpu().numpy().astype(np.float64, copy=False)
        key = np.asarray(key_vec, dtype=np.float64).reshape(-1)
        key_norm = float(np.linalg.norm(key))
        weight_norm = float(np.linalg.norm(weight))
        if key_norm < 1e-12 or weight_norm == 0.0:
            return 0.0
        raw = float(np.dot(weight, key / key_norm))
        return float(abs(np.tanh(raw)))

    def _stamp_salience(self, name, rel, group, key_vec) -> None:
        for slot in self.store._slots:
            if slot.group == group and slot.name == name and slot.rel == rel:
                slot.salience = self._salience(key_vec)
                slot.age = 0
                # Phase 0 assigns 1. A nonzero phase scales the write.
                gain = float(clock_write_gain(self.store.clock_phase()))
                if gain == 1.0:
                    slot.strength = 1.0
                    slot.clock_gain = 1.0
                else:
                    slot.strength = gain
                    slot.clock_gain = gain
                return
        raise RuntimeError("accepted write did not leave a slot")

    def _consolidate(self) -> None:
        """Recompute decay, refresh the top-m salient slots, merge surface repeats.

        The no-replay control is the store after decay and before the refresh
        and the merge. Production always continues into both. The competitive
        set is at most :data:`REPLAY_M` slots, and the merge budget is the
        same number. No language-model forward.
        """
        import time

        started = time.perf_counter()
        store = self.store
        control_n = len(store)
        control_dups = store.near_duplicate_slots()
        store.recompute_decay(REPLAY_DECAY)
        ranked = store.ranked_by_salience()
        width = min(int(REPLAY_M), len(ranked))
        chosen = list(ranked[:width])
        control_strengths = [float(store._slots[index].strength) for index in chosen]
        chosen_marks = [
            (store._slots[index].name, store._slots[index].rel, int(store._slots[index].t))
            for index in chosen
        ]
        if chosen:
            store.refresh_strength(chosen)
        merged = store.merge_near_duplicates(limit=width, protect=chosen) if width else 0
        replay_strengths = []
        for name, rel, step in chosen_marks:
            for slot in store._slots:
                if slot.name == name and slot.rel == rel and int(slot.t) == step:
                    replay_strengths.append(float(slot.strength))
                    break
        control_mean = _mean_float(control_strengths)
        replay_mean = _mean_float(replay_strengths)
        gain = None
        if control_mean is not None and replay_mean is not None:
            gain = float(replay_mean - control_mean)
        self.last_replay = {
            "every": int(REPLAY_EVERY),
            "m": int(REPLAY_M),
            "decay": float(REPLAY_DECAY),
            "control_n_slots": int(control_n),
            "replay_n_slots": int(len(store)),
            "control_near_dup_slots": int(control_dups),
            "replay_near_dup_slots": int(store.near_duplicate_slots()),
            "merged": int(merged),
            "control_salient_mean": control_mean,
            "replay_salient_mean": replay_mean,
            "salient_gain": gain,
            "consolidate_s": float(time.perf_counter() - started),
        }

    def _query_vector(self, prompt: str) -> np.ndarray:
        raw = self._raw_key_pool(prompt)
        return (raw - self.query_mean).astype(np.float64, copy=False)

    def _compose(self, value, *, grad: bool, key=None) -> torch.Tensor:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.shape != (self.hidden,):
            raise ValueError(f"value dim {arr.shape[0]} != hidden {self.hidden}")
        v = torch.tensor(arr, dtype=torch.float32, device=self._device)
        key_t = None
        if key is not None:
            arr_k = np.asarray(key, dtype=np.float32).reshape(-1)
            if arr_k.shape != (self.hidden,):
                raise ValueError(f"key dim {arr_k.shape[0]} != hidden {self.hidden}")
            key_t = torch.tensor(arr_k, dtype=torch.float32, device=self._device)
        if grad:
            return self.inject(v, key_t)
        with torch.no_grad():
            return self.inject(v, key_t).detach()

    def _slot_arrays(self):
        slots = list(self.store._slots)
        width = int(self.hidden)
        if not slots:
            return (
                np.zeros((0, width), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros(0, dtype=np.float64),
                np.zeros((0, width), dtype=np.float32),
            )
        keys = np.stack([np.asarray(slot.k, dtype=np.float64).reshape(-1) for slot in slots], 0)
        strengths = np.asarray([float(slot.strength) for slot in slots], dtype=np.float64)
        valences = np.asarray([float(slot.valence) for slot in slots], dtype=np.float64)
        values = np.stack([np.asarray(slot.v, dtype=np.float32).reshape(-1) for slot in slots], 0)
        return keys, strengths, valences, values

    def _trace_numpy(self, cues, state=None) -> tuple:
        """Per-token fields, residual ``[T, cue_dim]``, and the state after the last token."""
        rows = np.asarray(cues, dtype=np.float64)
        if rows.ndim != 2:
            raise ValueError("cues must be [tokens, cue_dim]")
        keys, strengths, valences, values = self._slot_arrays()
        fields, final = trace_sequence(
            rows,
            keys,
            strengths,
            valences,
            values,
            self.query_mean,
            theta=float(CUE_THETA),
            tau=float(CUE_TAU),
            a_trig=float(CUE_A_TRIG),
            floor=float(CUE_FLOOR),
            gap_tau=float(DEFAULT_GAP_TAU),
            valence_span=float(VALENCE_SPAN),
            state=state,
        )
        residual = self._residual_from_fields(fields, values, keys)
        return fields, residual, final

    def _residual_from_fields(self, fields, values, keys) -> np.ndarray:
        memory_id = fields["memory_id"]
        activation = fields["activation"]
        scale_sc = fields["strength_confidence"]
        valence = fields["valence"]
        steps, width = int(memory_id.shape[0]), int(self.hidden)
        out = np.zeros((steps, width), dtype=np.float32)
        cache = {}
        for index in range(steps):
            slot = int(memory_id[index])
            level = float(activation[index])
            if slot < 0 or level == 0.0:
                continue
            scale = level * float(scale_sc[index]) * (1.0 + float(valence[index]))
            if scale == 0.0:
                continue
            if slot not in cache:
                cache[slot] = (
                    self._compose(values[slot], key=keys[slot], grad=False)
                    .detach()
                    .float()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
            out[index] = cache[slot] * np.float32(scale)
        return out

    def cue_hidden(self, text: str) -> np.ndarray:
        """Pre-add cue rows, ``[T, cue_dim]``. Does not inject and does not change the store."""
        if not isinstance(text, str) or text == "":
            raise ValueError("text is empty")
        captured = {}

        def _keep(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            copied = tensor.detach()
            if copied.dim() == 3:
                if int(copied.shape[0]) != 1:
                    raise ValueError("cue encode expects one sequence")
                copied = copied[0]
            captured["hidden"] = copied.float().cpu().numpy()

        handle = self.layers[self.layer].register_forward_hook(_keep)
        try:
            if self._key_prefix_ok():
                self._run_key_hidden(text)
            else:
                self._run_hidden(text)
        finally:
            handle.remove()
        hidden = captured.get("hidden")
        if hidden is None:
            raise RuntimeError("the cue hook did not see the residual")
        return np.asarray(hidden, dtype=np.float64)

    def advance_cue(self, state: CueState, cue) -> tuple:
        """One token from ``state``. ``cue`` is the raw hidden row, not the centered match."""
        row = np.asarray(cue, dtype=np.float64).reshape(1, -1)
        fields, _residual, new_state = self._trace_numpy(row, state=state if state is not None else CueState())
        return new_state, _pack_cue_fields(fields, float(self.clock_phase()))

    def cue_fields(self, text: str) -> dict:
        """Deterministic per-token fields for ``text`` and the current store.

        Slot strength is not changed. The tensors are ``[1, T]`` or
        ``[1, T, cue_dim]``. ``memory_id`` is -1 and ``vec`` is zeros when no
        memory is active. ``interface_version`` is :data:`CUE_INTERFACE_VERSION`.
        """
        if len(self.store) == 0:
            hidden = self.cue_hidden(text)
            fields = _empty_cue_fields(hidden.shape[0], hidden.shape[1], float(self.clock_phase()))
            self.cue_state = CueState()
            self.last_cue_fields = fields
            self.last_added = torch.zeros(int(hidden.shape[1]), dtype=torch.float32)
            return fields
        hidden = self.cue_hidden(text)
        numpy_fields, residual, state = self._trace_numpy(hidden, state=CueState())
        packed = _pack_cue_fields(numpy_fields, float(self.clock_phase()))
        self.cue_state = state
        self.last_cue_fields = packed
        self._last_cue_rows = hidden
        self.last_added = torch.tensor(np.asarray(residual[-1], dtype=np.float32))
        return packed

    def _rank(self, prompt: str) -> dict:
        """Cue trace of ``prompt``. Forgetting runs after the trace, once per strict win."""
        info = {
            "n": len(self.store),
            "value": None,
            "key": None,
            "top": None,
            "maxcos": None,
            "gap": None,
            "confidence": 0.0,
            "readout": "cue",
            "weights": None,
            "parts": None,
            "participation": 0.0,
            "memory_id": -1,
        }
        if len(self.store) == 0:
            self.last_rif = {"n": 0}
            return info
        fields = self.cue_fields(prompt)
        forgotten = 0
        seen = set()
        triggered = fields["triggered"][0].tolist()
        memory = fields["memory_id"][0].tolist()
        gaps = fields["gap"][0].tolist()
        slots = self.store._slots
        for flag, slot_index, gap in zip(triggered, memory, gaps):
            if not flag:
                continue
            slot_index = int(slot_index)
            if slot_index < 0 or slot_index in seen:
                continue
            seen.add(slot_index)
            if gap == gap and float(gap) > 0.0 and len(slots) >= 2:
                forgotten += int(forget_rivals(slots, slot_index))
        self.last_rif = {"n": int(forgotten)}
        slot_index = int(fields["memory_id"][0, -1].item())
        info["memory_id"] = slot_index
        info["participation"] = 1.0 if slot_index >= 0 else 0.0
        if slot_index < 0:
            return info
        slot = slots[slot_index]
        gap_value = fields["gap"][0, -1].item()
        gap = None if gap_value != gap_value else float(gap_value)
        info.update(
            {
                "value": np.asarray(slot.v, dtype=np.float32).copy(),
                "key": np.asarray(slot.k, dtype=np.float32).copy(),
                "top": (slot.name, slot.rel),
                "maxcos": float(fields["winner_cosine"][0, -1].item()),
                "gap": gap,
                "confidence": float(fields["strength_confidence"][0, -1].item()) / float(slot.strength)
                if float(slot.strength) > 0.0
                else 0.0,
                "strength": float(slot.strength),
                "answer": slot.answer,
                "valence": float(slot.valence),
            }
        )
        return info

    def correct_valence(self, name: str, value: float, *, rel: str, group: Optional[str] = None, reason: str = "correction") -> dict:
        """Set one slot's valence and append an audit row. Does not write a fact."""
        if not isinstance(name, str) or name == "":
            raise ValueError("name is empty")
        if not isinstance(rel, str) or rel == "":
            raise ValueError("rel is empty")
        return self.store.set_valence(name, rel, float(value), source=str(reason), group=group)

    def valence_history(self, name: str, *, rel: str, group: Optional[str] = None) -> tuple:
        """Copy of the valence audit for one slot, oldest event first."""
        return self.store.valence_history(name, rel, group)

    def clock_set(self, angle: float) -> dict:
        """Set the store's clock angle. The argument is a number, not text."""
        return self.store.clock_set(angle)

    def clock_tick(self, steps: int = 1) -> dict:
        """Advance the clock. No fact text is read or stored."""
        return self.store.clock_tick(steps)

    def clock_phase(self) -> float:
        return self.store.clock_phase()

    def clock_log(self) -> tuple:
        return self.store.clock_log()

    def injection(self, prompt: str) -> np.ndarray:
        """Residual at the last prompt token. Zeros when that token is inactive."""
        if len(self.store) == 0:
            self.cue_fields(prompt)
            return np.zeros(self.hidden, dtype=np.float32)
        self.cue_fields(prompt)
        if self.last_added is None:
            return np.zeros(self.hidden, dtype=np.float32)
        return self.last_added.detach().cpu().numpy().astype(np.float32, copy=False)

    def _stored_spelling(self, prompt: str, rank: dict) -> Optional[str]:
        """Winning slot's phone or birthday surface, when the prompt completes it."""
        top = rank.get("top")
        spelled = rank.get("answer")
        if not top or not isinstance(spelled, str) or spelled == "":
            return None
        rel = top[1]
        if rel not in ("phone", "birthday"):
            return None
        if rel != completing_relation(prompt):
            return None
        return spelled

    def _spell_ids(self, prompt: str, rank: dict, max_new_tokens: int) -> Optional[list]:
        """Ids of the stored answer when the prompt is completing that relation."""
        if int(max_new_tokens) < 0:
            raise ValueError("max_new_tokens must be >= 0")
        spelled = self._stored_spelling(prompt, rank)
        if spelled is None:
            return None
        ids = _id_list(self.tokenizer, spelled, add_special_tokens=False)
        return ids[: int(max_new_tokens)]

    def scripted_continuation(self, prompt: str, max_new_tokens: int = 32) -> Optional[list]:
        """Token ids of the stored phone or birthday, or None to use the LM.

        Spelling runs only when the memory still active on the last prompt
        token has the relation the prompt is completing, and that slot stores
        an answer. The ids are that string. An empty store, a decayed trace,
        or any other prompt returns None. ``max_new_tokens`` truncates the list.
        """
        if not isinstance(prompt, str) or prompt == "":
            raise ValueError("prompt is empty")
        if len(self.store) == 0:
            return None
        return self._spell_ids(prompt, self._rank(prompt), max_new_tokens)

    def continuation_ids(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        done: Optional[Callable[[list], bool]] = None,
        *,
        arm: bool = True,
    ) -> list:
        if not isinstance(prompt, str) or prompt == "":
            raise ValueError("prompt is empty")
        if arm:
            rank = self._rank(prompt)
            spelled = self._spell_ids(prompt, rank, max_new_tokens)
            if spelled is not None:
                return spelled
        self._cue_resume = False
        self._cue_live = True
        try:
            with torch.no_grad():
                return greedy_continuation(
                    self.model,
                    self.tokenizer,
                    prompt,
                    int(max_new_tokens),
                    getattr(self.tokenizer, "eos_token_id", None),
                    self._device,
                    done=done,
                )
        finally:
            self._cue_live = False

    def generate(self, prompt: str, max_new_tokens: int = 32) -> str:
        """Greedy answer span. ``prompt`` is the only text passed to the LM.

        A phone or birthday slot copies its stored answer when it is still
        active on the last prompt token and the prompt is completing that
        relation. Any other prompt keeps the per-token cue path. The store is
        ranked once. Each later forward recomputes the decay from the start of
        the string, which matches stepping :attr:`cue_state` one token at a time.
        """
        if not isinstance(prompt, str) or prompt == "":
            raise ValueError("prompt is empty")
        if int(max_new_tokens) < 0:
            raise ValueError("max_new_tokens must be >= 0")

        def _done(new_ids: list) -> bool:
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            return span_complete(text)

        self._cue_resume = False
        if len(self.store) == 0:
            ids = self.continuation_ids(prompt, max_new_tokens=max_new_tokens, done=_done, arm=False)
            return decode_answer(self.tokenizer, ids)
        rank = self._rank(prompt)
        spelled = self._stored_spelling(prompt, rank)
        if spelled is not None:
            ids = _id_list(self.tokenizer, spelled, add_special_tokens=False)
            if len(ids) <= int(max_new_tokens):
                return spelled
            return decode_answer(self.tokenizer, ids[: int(max_new_tokens)])
        ids = self.continuation_ids(prompt, max_new_tokens=max_new_tokens, done=_done, arm=False)
        return decode_answer(self.tokenizer, ids)

    def teacher_force_loss(self, prompt: str, answer: str, value, key=None, *, armed: bool = True) -> torch.Tensor:
        """Next-token cross-entropy on ``answer`` with ``value`` injected on every token.

        This is the training add. Retrieval does not use it. ``prompt + answer``
        is the teacher-forced string. ``prompt`` alone is the generation prefix,
        so the fact text is not required in it. Used by ``tidal.llm_memory_train``.
        ``key`` is the fact key for polarity. ``armed=False`` scores the
        empty-store baseline and does not inject.
        """
        if not isinstance(prompt, str) or prompt == "":
            raise ValueError("prompt is empty")
        if not isinstance(answer, str) or answer == "":
            raise ValueError("answer is empty")
        full = prompt + answer
        enc = self._tokenize(full)
        ids = _batch_ids(enc).to(self._device)
        offsets = _offset_pairs(enc)
        if offsets is not None:
            start = len(prompt)
            end = len(full)
            mask = [b > start and a < end and a != b for a, b in offsets]
        else:
            mask = _subsequence_mask(self.tokenizer, ids[0].tolist(), answer)
        idxs = [i for i, flag in enumerate(mask) if flag]
        if not idxs or min(idxs) < 1:
            raise ValueError("answer span produced no token the model can predict")
        if armed:
            self._train_delta = self._compose(value, key=key, grad=True)
        else:
            self._train_delta = None
        try:
            attn = _batch_mask(enc, self._device)
            kwargs = {"input_ids": ids, "use_cache": False}
            if attn is not None:
                kwargs["attention_mask"] = attn
            logits = _logits_of(self.model(**kwargs))
            pred = logits[0, [i - 1 for i in idxs]]
            target = ids[0, idxs].to(device=pred.device)
            return F.cross_entropy(pred, target)
        finally:
            self._train_delta = None


    def save_inject(self, path) -> None:
        torch.save(
            {"inject": self.inject.state_dict(), "layer": self.layer, "hidden": self.hidden},
            str(path),
        )

    def load_inject(self, path) -> None:
        """Replace the inject MLP from a :meth:`save_inject` file.

        This does not check ``layer``. The tensors have to match the MLP that
        was built for this model. Use it for a checkpoint that is not the
        shipped SmolLM2-135M file, or to replace that file after init.
        """
        ckpt = _torch_load(path)
        _load_inject_tensors(self.inject, ckpt["inject"])
        self.inject.to(self._device)


def _empty_cue_fields(steps: int, cue_dim: int, phase: float) -> dict:
    steps = int(steps)
    cue_dim = int(cue_dim)
    return {
        "interface_version": int(CUE_INTERFACE_VERSION),
        "cue_dim": cue_dim,
        "memory_id": torch.full((1, steps), -1, dtype=torch.long),
        "vec": torch.zeros((1, steps, cue_dim), dtype=torch.float32),
        "activation": torch.zeros((1, steps), dtype=torch.float32),
        "strength_confidence": torch.zeros((1, steps), dtype=torch.float32),
        "valence": torch.zeros((1, steps), dtype=torch.float32),
        "circadian_phase": torch.full((1, steps), float(phase), dtype=torch.float32),
        "triggered": torch.zeros((1, steps), dtype=torch.bool),
        "gap": torch.full((1, steps), float("nan"), dtype=torch.float32),
        "winner_cosine": torch.full((1, steps), float("nan"), dtype=torch.float32),
    }


def _pack_cue_fields(fields: dict, phase: float) -> dict:
    steps = int(np.asarray(fields["memory_id"]).shape[0])
    phase_row = np.full((1, steps), float(phase), dtype=np.float32)
    return {
        "interface_version": int(CUE_INTERFACE_VERSION),
        "cue_dim": int(fields["cue_dim"]),
        "memory_id": torch.tensor(np.asarray(fields["memory_id"]), dtype=torch.long).unsqueeze(0),
        "vec": torch.tensor(np.asarray(fields["vec"]), dtype=torch.float32).unsqueeze(0),
        "activation": torch.tensor(np.asarray(fields["activation"]), dtype=torch.float32).unsqueeze(0),
        "strength_confidence": torch.tensor(np.asarray(fields["strength_confidence"]), dtype=torch.float32).unsqueeze(0),
        "valence": torch.tensor(np.asarray(fields["valence"]), dtype=torch.float32).unsqueeze(0),
        "circadian_phase": torch.tensor(phase_row),
        "triggered": torch.tensor(np.asarray(fields["triggered"]), dtype=torch.bool).unsqueeze(0),
        "gap": torch.tensor(np.asarray(fields["gap"]), dtype=torch.float32).unsqueeze(0),
        "winner_cosine": torch.tensor(np.asarray(fields["winner_cosine"]), dtype=torch.float32).unsqueeze(0),
    }


def _pack_cue_batch(rows: list, phase: float) -> dict:
    packed = [_pack_cue_fields(row, phase) for row in rows]
    if len(packed) == 1:
        return packed[0]
    merged = {
        "interface_version": packed[0]["interface_version"],
        "cue_dim": packed[0]["cue_dim"],
    }
    for key in (
        "memory_id",
        "vec",
        "activation",
        "strength_confidence",
        "valence",
        "circadian_phase",
        "triggered",
        "gap",
        "winner_cosine",
    ):
        merged[key] = torch.cat([item[key] for item in packed], 0)
    return merged


def _add_delta(output, delta: torch.Tensor):
    if torch.is_tensor(output):
        return output + _broadcast(delta, output)
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return (_add_delta(output[0], delta),) + output[1:]
    raise TypeError(f"cannot inject into layer output of type {type(output).__name__}")


def _broadcast(delta: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    add = delta.to(device=hidden.device, dtype=hidden.dtype).reshape(-1)
    if add.numel() != hidden.shape[-1]:
        raise ValueError(f"inject dim {add.numel()} != hidden {hidden.shape[-1]}")
    shape = [1] * (hidden.dim() - 1) + [add.numel()]
    return add.view(*shape)


def _id_list(tokenizer, text: str, add_special_tokens: bool) -> list:
    if hasattr(tokenizer, "encode"):
        ids = tokenizer.encode(text, add_special_tokens=add_special_tokens)
    else:
        enc = tokenizer(text, add_special_tokens=add_special_tokens)
        ids = enc["input_ids"]
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(i) for i in ids]


def _subsequence_mask(tokenizer, host: Sequence[int], answer: str) -> list:
    needle = _id_list(tokenizer, answer, add_special_tokens=False)
    if not needle:
        raise ValueError("answer tokenized to an empty id sequence")
    host = [int(i) for i in host]
    last = -1
    span = len(needle)
    for i in range(len(host) - span + 1):
        if host[i : i + span] == needle:
            last = i
    if last < 0:
        raise ValueError("answer tokens were not found inside the fact tokenization")
    mask = [False] * len(host)
    for j in range(span):
        mask[last + j] = True
    return mask


def _load_pretrained(model_name: str, *, device, dtype: Optional[torch.dtype]):
    tf = _transformers()
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = dtype
    try:
        model = tf.AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    except TypeError:
        if "dtype" in kwargs:
            kwargs["torch_dtype"] = kwargs.pop("dtype")
            model = tf.AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        else:
            raise
    tokenizer = tf.AutoTokenizer.from_pretrained(model_name)
    model.to(device)
    return model, tokenizer
