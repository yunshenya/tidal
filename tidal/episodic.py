"""Episodic fact slots for Continuum (native, always on).

Design reference: https://github.com/yunshenya/neuromem (inspiration only).
This module does not import that package, SmolLM2, or any 135M checkpoint.

Kept from the experiments that worked: one slot per fact ``(k, v, identity)``;
same-identity overwrite; capacity with FIFO eviction; soft top-k readout;
a write gate that drops chat-like noise. Not kept: summing several memory
matrices (the mixture diluted the value) and a naïve key shield that restored
the previous key after an update (same-identity writes stopped sticking).

Keys and values are vectors already produced by tidal — encoder state ``h``,
event features, or a bge sentence vector run through the frozen model's own
text projection. Retrieval is cosine over slot keys. The value returned to
``Continuum.heads`` is a convex combination of slot values (weights sum to 1),
added to ``h``. An empty store returns zeros.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np

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

    def _hit(self, kk: np.ndarray, name: Optional[str], rel: Optional[str], group: Optional[str]) -> Optional[int]:
        """Identity match first, else the first in-scope near-duplicate key.

        Cosine does not cross groups, and it does not override a same-identity
        slot that happens to sit later in the list. The stored key is replaced
        on hit — nothing writes the previous key back.
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
        v: ArrayLike,
        name: Optional[str] = None,
        rel: Optional[str] = None,
        *,
        kind: str = "fact",
        surprise: Optional[float] = None,
        group: Optional[str] = None,
    ) -> bool:
        """Insert or overwrite one slot. Returns True iff a write happened."""
        if not self.gate_write(kind, surprise):
            return False
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
        """Value for query ``q``. ``top1`` is argmax cosine; ``softtopk`` mixes the top keys.

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
        kn = np.linalg.norm(K, axis=1)
        qn = float(np.linalg.norm(qq))
        c = (K @ qq) / (kn * qn + 1e-12)
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
