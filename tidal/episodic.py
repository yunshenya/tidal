"""Episodic fact slots for Continuum (native, always on).

The deleted neuromem study is archived under ``research/neuromem/`` (design notes
only). This module does not import that package, SmolLM2, or any 135M checkpoint.

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

Ranking refines those keys with a residual query/key adapter (a 2-layer MLP
on the query and another on the key). The last layer is zero, so an untrained
adapter adds zeros and the score is raw cosine, denominator included. The
vector added by ``Continuum.heads`` is that soft top-k mixture passed through
a residual inject MLP, also zero on the last layer, so an untrained readout
equals the mixture. ``read`` returns the mixture itself (no inject); exact
text is decoded from the winning slot's code. Same-identity overwrite and the
paraphrase threshold use the raw key, not the adapted one. An empty store
returns zeros before the inject MLP runs.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

ArrayLike = Union[np.ndarray, Sequence[float]]
SKIP_KINDS = frozenset({"chat", "noise"})


def _as1d(x: ArrayLike) -> np.ndarray:
    a = np.asarray(x, dtype=np.float64).reshape(-1)
    if a.size == 0:
        raise ValueError("empty vector")
    if not np.isfinite(a).all():
        raise ValueError("non-finite vector")
    return a


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def cosine_scores(keys: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine of each row of ``keys`` with ``query``, denominator ``kn * qn + 1e-12``.

    Slot ranking uses this after the residual adapter. A zero residual leaves
    the same scores as the raw keys.
    """
    keys = np.asarray(keys, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64).reshape(-1)
    kn = np.linalg.norm(keys, axis=1)
    qn = float(np.linalg.norm(query))
    return (keys @ query) / (kn * qn + 1e-12)


def head_hidden(din: int) -> int:
    """Width of the retrieval and inject MLPs. 128-d matches the 256-d neuromem head."""
    return int(min(256, max(32, int(din) * 2)))


def episodic_head_params(din: int) -> int:
    """Parameter count of the query MLP, the key MLP, and the inject MLP."""
    din = int(din)
    hid = head_hidden(din)
    one = din * hid + hid + hid * din + din
    return 3 * one


def _residual_mlp(din: int, hid: int) -> nn.Sequential:
    """Linear → GELU → Linear. Last layer is exact zeros, so the residual adds 0."""
    net = nn.Sequential(nn.Linear(din, hid), nn.GELU(), nn.Linear(hid, din))
    net.double()
    nn.init.zeros_(net[-1].weight)
    nn.init.zeros_(net[-1].bias)
    return net


class QueryKeyAdapter(nn.Module):
    """Residual maps ``q' = q + Wq(q)``, ``k' = k + Wk(k)``. Untrained, ``q' = q`` and ``k' = k``.

    InfoNCE training lives in ``tidal.episodic_train``. The store always runs
    this module; it does not swap in raw cosine through a flag.
    """

    def __init__(self, din: int, hid: Optional[int] = None):
        super().__init__()
        din = int(din)
        if din < 1:
            raise ValueError("adapter din must be >= 1")
        self.din = din
        self.hid = head_hidden(din) if hid is None else int(hid)
        self.wq = _residual_mlp(self.din, self.hid)
        self.wk = _residual_mlp(self.din, self.hid)

    def forward(self, x: torch.Tensor, kind: str) -> torch.Tensor:
        if kind not in ("q", "k"):
            raise ValueError(f"kind must be 'q' or 'k', got {kind!r}")
        flat = x.dim() == 1
        if flat:
            x = x.unsqueeze(0)
        if x.shape[-1] != self.din:
            raise ValueError(f"adapter input {x.shape[-1]} != din {self.din}")
        net = self.wq if kind == "q" else self.wk
        y = x + net(x.to(dtype=net[-1].weight.dtype))
        return y.squeeze(0) if flat else y


class InjectResidual(nn.Module):
    """``v' = v + MLP(v)`` with a zero last layer. Untrained, ``v' = v``.

    Applied only when a readout vector exists. An empty store does not call it,
    so a later nonzero bias cannot turn emptiness into a constant.
    """

    def __init__(self, din: int, hid: Optional[int] = None):
        super().__init__()
        din = int(din)
        if din < 1:
            raise ValueError("inject din must be >= 1")
        self.din = din
        self.hid = head_hidden(din) if hid is None else int(hid)
        self.net = _residual_mlp(self.din, self.hid)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        flat = v.dim() == 1
        if flat:
            v = v.unsqueeze(0)
        if v.shape[-1] != self.din:
            raise ValueError(f"inject input {v.shape[-1]} != din {self.din}")
        y = v + self.net(v.to(dtype=self.net[-1].weight.dtype))
        return y.squeeze(0) if flat else y


def _project(module: nn.Module, rows: np.ndarray, kind: Optional[str] = None) -> np.ndarray:
    """Run ``module`` on float64 rows. ``kind`` is passed only for the query/key adapter."""
    arr = np.asarray(rows, dtype=np.float64)
    single = arr.ndim == 1
    if single:
        arr = arr.reshape(1, -1)
    x = torch.from_numpy(np.array(arr, dtype=np.float64, copy=True))
    with torch.no_grad():
        y = module(x) if kind is None else module(x, kind)
    out = np.array(y.detach().numpy(), dtype=np.float64, copy=True)
    return out.reshape(-1) if single else out


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


class EpisodicStore:
    """Bounded episodic slots: append, same-identity overwrite, soft top-k read.

    Parameters
    ----------
    capacity : max slots. At capacity the oldest ``t`` is evicted (FIFO).
    overwrite_cos : also overwrite when ``cos(k, slot.k)`` is strictly above
        this and the identity does not match (paraphrase of the same fact).
        ``1`` disables that path. Identity match always wins over cosine.
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
        soft_k: int = 4,
        soft_temp: float = 0.05,
        surprise_thr: Optional[float] = None,
    ):
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        if not 0.0 < float(overwrite_cos) <= 1.0:
            raise ValueError("overwrite_cos must be in (0, 1]")
        if int(soft_k) < 1:
            raise ValueError("soft_k must be >= 1")
        if float(soft_temp) <= 0.0:
            raise ValueError("soft_temp must be > 0")
        self.capacity = int(capacity)
        self.dk = None if dk is None else int(dk)
        self.dv = None if dv is None else int(dv)
        self.overwrite_cos = float(overwrite_cos)
        self.soft_k = int(soft_k)
        self.soft_temp = float(soft_temp)
        self.surprise_thr = surprise_thr
        self._slots: list[Slot] = []
        self._step = 0
        self.retr: Optional[QueryKeyAdapter] = None
        self.inj: Optional[InjectResidual] = None
        self._ensure_heads()

    def _ensure_heads(self) -> None:
        """Build the residual heads once the key and value widths are known.

        The store freezes them. Training uses a separate module and does not
        write into this one. Last layers start at zero.
        """
        if self.dk is not None and self.retr is None:
            self.retr = QueryKeyAdapter(self.dk)
            self.retr.eval()
            self.retr.requires_grad_(False)
        if self.dv is not None and self.inj is None:
            self.inj = InjectResidual(self.dv)
            self.inj.eval()
            self.inj.requires_grad_(False)

    def _adapt(self, rows: np.ndarray, kind: str) -> np.ndarray:
        self._ensure_heads()
        if self.retr is None:
            raise ValueError("query/key adapter needs a key width")
        return _project(self.retr, rows, kind)

    def _inject(self, rows: np.ndarray) -> np.ndarray:
        self._ensure_heads()
        if self.inj is None:
            raise ValueError("inject MLP needs a value width")
        return _project(self.inj, rows, None)

    def __len__(self) -> int:
        return len(self._slots)

    def clear(self) -> None:
        self._slots.clear()
        self._step = 0

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
        self._ensure_heads()

    def _hit(self, kk: np.ndarray, name: Optional[str], rel: Optional[str], group: Optional[str]) -> Optional[int]:
        """Identity match first, else the first in-scope near-duplicate key.

        The cosine here is the raw stored key. The retrieval adapter does not
        participate, so a trained map cannot merge two facts that are far apart
        in the original key, or split a same-identity overwrite. Cosine does
        not cross groups, and it does not override a same-identity slot that
        happens to sit later in the list. The stored key is replaced on hit —
        nothing writes the previous key back.
        """
        hit_id = None
        hit_cos = None
        for i, s in enumerate(self._slots):
            if s.group != group:
                continue
            if name is not None and (s.name, s.rel) == (name, rel):
                hit_id = i
                break
            if hit_cos is None and self.overwrite_cos < 1.0 and _cos(kk, s.k) > self.overwrite_cos:
                hit_cos = i
        return hit_id if hit_id is not None else hit_cos

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
    ) -> bool:
        """Insert or overwrite one slot. Returns True iff a write happened.

        ``v`` may be a numeric vector, a string, or a list of strings. A string
        or token list becomes a positional byte code of width ``dv`` (see
        ``encode_span``). Same-identity overwrite replaces that code; it is
        not added to the previous value.
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
        entry = Slot(k=kk.copy(), v=vv.copy(), name=name, rel=rel, t=self._step, group=group)
        hit = self._hit(kk, name, rel, group)
        if hit is not None:
            self._slots[hit] = entry
            return True
        if len(self._slots) >= self.capacity:
            oldest = min(range(len(self._slots)), key=lambda i: (self._slots[i].t, i))
            self._slots.pop(oldest)
        self._slots.append(entry)
        return True

    def _visible(self, group: Optional[str]) -> list[tuple[int, Slot]]:
        if group is None:
            return list(enumerate(self._slots))
        return [(i, s) for i, s in enumerate(self._slots) if s.group is None or s.group == group]

    def read(
        self,
        q: ArrayLike,
        *,
        mode: str = "softtopk",
        group: Optional[str] = None,
    ) -> Tuple[Optional[np.ndarray], dict]:
        """Value for query ``q``. ``top1`` is argmax score; ``softtopk`` mixes the top values.

        Scores are cosine on the residual query/key adapter. With a zero last
        layer that cosine is the raw one. The returned vector is the raw slot
        mixture: the inject MLP is not applied here.

        ``group is None`` searches every slot. A group sees its own slots plus
        global slots (``group is None`` on the slot). Returns ``(None, info)``
        when nothing is visible.
        """
        visible = self._visible(group)
        info: dict[str, Any] = {"n": len(visible), "maxcos": -1.0, "top": None, "weights": None}
        if not visible:
            return None, info
        qq = _as1d(q)
        if self.dk is not None and qq.size != self.dk:
            raise ValueError(f"query dim {qq.size} != store dk {self.dk}")
        K = np.stack([s.k for _, s in visible], 0)
        qq = self._adapt(qq, "q")
        K = self._adapt(K, "k")
        c = cosine_scores(K, qq)
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
        """Float64 vector added to ``h``. Zeros when no slot is visible.

        The zeros are produced before the inject MLP, so an empty store stays
        an exact zero even if that MLP later grows a bias. A visible mixture
        goes through the residual inject. A zero last layer leaves the mixture
        unchanged.
        """
        v, _ = self.read(q, mode=mode, group=group)
        dim = out_dim if out_dim is not None else (self.dv if self.dv is not None else (0 if v is None else int(v.size)))
        if v is None:
            return np.zeros(int(dim), dtype=np.float64)
        v = self._inject(v)
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
