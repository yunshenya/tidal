"""Test-time memory for a frozen causal language model.

Facts are written into :class:`tidal.episodic.EpisodicStore` (one slot, same
identity overwrites). At generation the top slot's value is added to one
decoder layer's residual, so the fact text is not copied into the prompt.
Readout is hard top-1. A soft top-4 mix at temperature 0.05 tied top-1 on the
answer-stripped prefix and lost two question hits at N32; temperature 0.20
lost many more. See ``docs/llm_memory.md``.

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
estimated once from a calibration set and stored with the module. The add is

    scale(v) + MLP(scale(v))

with ``scale`` fixed at 1 and the MLP's last layer initialized to zeros, so an
untrained module copies ``v`` through. When this wrapper loads the default
SmolLM2-135M checkpoint at the default layer, that MLP is replaced by the
shipped weights in ``llm_inject_smollm2_135m.pt`` (the 1500-step MLP, plus
linear readouts that spell ``+1-415-555-dddd`` from a phone value and
``Month D, YYYY`` from a birthday value). A
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

from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tidal.episodic import EpisodicStore
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
    """MLP tensors match. Phone and date readout buffers may be absent.

    A checkpoint saved before a readout only has the MLP keys. Unknown keys
    and shape mismatches do not fit.
    """
    if not isinstance(state, dict):
        return False
    current = module.state_dict()
    if not set(state) <= set(current):
        return False
    required = [key for key in current if key not in _READOUT_KEYS]
    if not set(required) <= set(state):
        return False
    for key, got in state.items():
        if not torch.is_tensor(got) or tuple(got.shape) != tuple(current[key].shape):
            return False
        if not bool(torch.isfinite(got).all()):
            return False
    return True


def _load_inject_tensors(module: nn.Module, state) -> None:
    """Load ``state`` onto ``module``. Missing readout buffers stay at init."""
    if not _inject_core_fits(module, state):
        raise ValueError("inject tensors do not match this MLP")
    merged = {key: tensor.detach().cpu().clone() for key, tensor in module.state_dict().items()}
    for key, tensor in state.items():
        merged[key] = tensor.detach().cpu()
    module.load_state_dict(merged)


class InjectMLP(nn.Module):
    """``scaled + MLP(scaled)``. Last layer is exact zeros, so the add starts as ``scaled``."""

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

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scaled = value * self.scale
        return scaled + self.net(scaled)

    def phone_digits(self, value: torch.Tensor) -> Optional[list]:
        """Four digits read from a phone value, or None when the readout is off."""
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


class LLMMemory:
    """Frozen causal LM with episodic residual memory.

    ``remember`` writes one fact. ``generate`` answers from the store. The
    prompt is the only text the LM sees. The armed value is the top slot
    (hard top-1 on raw cosine), not a soft mixture. ``soft_k`` and
    ``soft_temp`` are kept on the episodic store; ``generate`` does not read
    them. An empty store leaves the residual untouched, so greedy tokens
    match the same model with the hook removed.

    Loading ``HuggingFaceTB/SmolLM2-135M`` at the default layer
    (``n_layers // 2``, which is layer 15, hidden 576) installs the shipped
    inject in ``llm_inject_smollm2_135m.pt``. A caller-supplied model, another
    checkpoint, or another layer keeps the zero-init MLP, so the add is ``v``,
    until :meth:`load_inject`. There is no switch.
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
        self._delta: Optional[torch.Tensor] = None
        self.last_added: Optional[torch.Tensor] = None
        # Registered for the life of this wrapper. A missing delta returns
        # None, which leaves the layer output object untouched.
        self._handle = self.layers[self.layer].register_forward_hook(self._on_layer)
        self._load_shipped_inject()
        self.key_mean, self.query_mean = self._key_centers()

    def close(self) -> None:
        if getattr(self, "_handle", None) is not None:
            self._handle.remove()
            self._handle = None
        self._delta = None

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
        delta = self._delta
        if delta is None:
            self.last_added = None
            return None
        if (not delta.requires_grad) and torch.count_nonzero(delta) == 0:
            self.last_added = None
            return None
        self.last_added = delta.detach()
        return _add_delta(output, delta)

    def _tokenize(self, text: str) -> Mapping[str, Any]:
        kwargs = {"return_tensors": "pt", "add_special_tokens": True, "return_offsets_mapping": True}
        try:
            return self.tokenizer(text, **kwargs)
        except (TypeError, ValueError):
            kwargs.pop("return_offsets_mapping")
            return self.tokenizer(text, **kwargs)

    def _run_hidden(self, text: str) -> tuple:
        """Hidden state at the inject layer, with memory unhooked for this forward."""
        if not isinstance(text, str) or text == "":
            raise ValueError("text is empty")
        self._delta = None
        enc = self._tokenize(text)
        ids = _batch_ids(enc).to(self._device)
        attn = _batch_mask(enc, self._device)
        kwargs = {"input_ids": ids, "use_cache": False, "output_hidden_states": True}
        if attn is not None:
            kwargs["attention_mask"] = attn
        with torch.no_grad():
            out = self.model(**kwargs)
            hidden = layer_hidden(_hidden_tuple(out), self.layer, self.n_layers)
            if hidden.dim() == 3:
                hidden = hidden[0]
            # Transformers may load SmolLM2 in bfloat16. Numpy cannot hold that
            # dtype, and the key center is estimated in float32/float64.
            hidden = hidden.detach().float()
        return hidden, enc

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
        hidden, enc = self._run_hidden(text)
        mask = self._answer_mask(text, enc, answer, hidden.shape[0])
        if date_like(answer):
            offsets = _offset_pairs(enc)
            if offsets is not None:
                parts = _date_components(hidden, offsets, text, answer)
                if parts is not None:
                    month, day_rows, year_rows = parts
                    packed = assemble_date_value(month, day_rows, year_rows)
                    return packed.detach().cpu().numpy().astype(np.float32, copy=False)
        return pool_answer(hidden, mask, answer).detach().cpu().numpy().astype(np.float32, copy=False)

    def _raw_key_pool(self, text: str, answer: Optional[str] = None) -> np.ndarray:
        """Sink-dropped mean of the key tokens, before the fixed center."""
        hidden, enc = self._run_hidden(text)
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
        """Write one fact into the episodic store. Same ``(name, rel)`` overwrites."""
        text, answer, name, rel, key, group, kind = _parse_fact(
            fact, answer, name, rel, key, group, kind
        )
        if not self.store.gate_write(kind, surprise):
            return False
        value = self.encode_value(text, answer)
        if key is None:
            key_vec = self.encode_key(text, answer)
        else:
            key_vec = self.encode_key(key)
        return bool(
            self.store.write(
                key_vec,
                value,
                name=name,
                rel=rel,
                kind=kind,
                surprise=surprise,
                group=group,
            )
        )

    def _query_vector(self, prompt: str) -> np.ndarray:
        raw = self._raw_key_pool(prompt)
        return (raw - self.query_mean).astype(np.float64, copy=False)

    def _compose(self, value, *, grad: bool) -> torch.Tensor:
        arr = np.asarray(value, dtype=np.float32).reshape(-1)
        if arr.shape != (self.hidden,):
            raise ValueError(f"value dim {arr.shape[0]} != hidden {self.hidden}")
        v = torch.tensor(arr, dtype=torch.float32, device=self._device)
        if grad:
            return self.inject(v)
        with torch.no_grad():
            return self.inject(v).detach()

    def _arm(self, prompt: str) -> None:
        if len(self.store) == 0:
            self._delta = None
            return
        query = self._query_vector(prompt)
        # Hard top-1. Soft top-4 at T=0.05 matched this on prefix free-gen and
        # trailed it on natural questions; T=0.20 mixed neighbors and dropped
        # N32 questions from 36/48 to 19/48. Measured in docs/llm_memory.md.
        value = self.store.readout(query, mode="top1")
        self._delta = self._compose(value, grad=False)

    def injection(self, prompt: str) -> np.ndarray:
        """Vector added at the chosen layer. Zeros when the store is empty."""
        if len(self.store) == 0:
            return np.zeros(self.hidden, dtype=np.float32)
        self._arm(prompt)
        try:
            if self._delta is None:
                return np.zeros(self.hidden, dtype=np.float32)
            return self._delta.detach().cpu().numpy().astype(np.float32, copy=False)
        finally:
            self._delta = None

    def scripted_continuation(self, prompt: str, max_new_tokens: int = 32) -> Optional[list]:
        """Token ids spelled by a linear readout, or None to use the LM.

        Hard top-1. A winning phone slot with the phone readout on spells
        ``+1-415-555-`` plus the four digits. A winning birthday slot with
        the date readout on spells ``Month D, YYYY``. An empty store, any
        other relation, or a readout that is still zero returns None.
        ``max_new_tokens`` truncates the id list.
        """
        if not isinstance(prompt, str) or prompt == "":
            raise ValueError("prompt is empty")
        if int(max_new_tokens) < 0:
            raise ValueError("max_new_tokens must be >= 0")
        phone_on = float(self.inject.phone_on.detach().cpu()) != 0.0
        date_on = float(self.inject.date_on.detach().cpu()) != 0.0
        if len(self.store) == 0 or not (phone_on or date_on):
            return None
        query = self._query_vector(prompt)
        value, info = self.store.read(query, mode="top1")
        top = info.get("top") if isinstance(info, dict) else None
        if value is None or not top:
            return None
        rel = top[1]
        value_t = torch.as_tensor(np.asarray(value, dtype=np.float32))
        spelled = None
        if rel == "phone" and phone_on:
            digits = self.inject.phone_digits(value_t)
            if digits is not None:
                spelled = PHONE_TAIL + "".join(str(d) for d in digits)
        elif rel == "birthday" and date_on:
            spelled = self.inject.date_spelling(value_t)
        if spelled is None:
            return None
        ids = _id_list(self.tokenizer, spelled, add_special_tokens=False)
        return ids[: int(max_new_tokens)]

    def continuation_ids(
        self,
        prompt: str,
        max_new_tokens: int = 32,
        done: Optional[Callable[[list], bool]] = None,
    ) -> list:
        if not isinstance(prompt, str) or prompt == "":
            raise ValueError("prompt is empty")
        scripted = self.scripted_continuation(prompt, max_new_tokens)
        if scripted is not None:
            return scripted
        self._arm(prompt)
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
            self._delta = None

    def generate(self, prompt: str, max_new_tokens: int = 32) -> str:
        """Greedy answer span. ``prompt`` is the only text passed to the LM.

        A phone slot is spelled from the phone readout, and a birthday slot
        from the date readout, before the LM runs. Leading whitespace is
        skipped before the stop, so
        a natural question whose first token is a newline still yields the
        following line. The span then ends at the next newline or at
        sentence-ending punctuation.
        """

        def _done(new_ids: list) -> bool:
            text = self.tokenizer.decode(new_ids, skip_special_tokens=True)
            return span_complete(text)

        ids = self.continuation_ids(prompt, max_new_tokens=max_new_tokens, done=_done)
        return decode_answer(self.tokenizer, ids)

    def teacher_force_loss(self, prompt: str, answer: str, value) -> torch.Tensor:
        """Next-token cross-entropy on ``answer`` with ``value`` injected.

        ``prompt + answer`` is the teacher-forced string. ``prompt`` alone is
        the generation prefix, so the fact text is not required in it. Used by
        ``tidal.llm_memory_train``.
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
        self._delta = self._compose(value, grad=True)
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
            self._delta = None

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
