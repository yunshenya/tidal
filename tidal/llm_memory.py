"""Test-time memory for a frozen causal language model.

Facts are written into :class:`tidal.episodic.EpisodicStore` (one slot, same
identity overwrites, soft top-k). At generation the retrieved value is added
to one decoder layer's residual, so the fact text is not copied into the
prompt.

The value is an answer-span pool of that layer's hidden state. It is not
L2-normalized: magnitude is what the identity copy-through needs. The add is

    scale(v) + MLP(scale(v))

with ``scale`` fixed at 1 and the MLP's last layer initialized to zeros, so an
untrained module copies ``v`` through. This is the A2s inject from the SmolLM
study (see ``docs/llm_memory.md``). It is a different representation from the
positional byte codes on ``Continuum.heads``; those codes stay on the episodic
path and do not go through this MLP.

``torch`` is imported by this module. ``transformers`` is imported only when a
checkpoint is loaded or :func:`build_tiny_lm` is called. Core ``import tidal``
does not load either one. Install the extra with ``pip install tidal[llm]``.

Default checkpoint: ``HuggingFaceTB/SmolLM2-135M``.
"""
from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tidal.episodic import EpisodicStore

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M"
DEFAULT_CAPACITY = 256
DEFAULT_SOFT_K = 4
DEFAULT_SOFT_TEMP = 0.05
DEFAULT_OVERWRITE_COS = 0.9

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

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        scaled = value * self.scale
        return scaled + self.net(scaled)


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


def greedy_continuation(
    model: nn.Module,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    eos_token_id: Optional[int],
    device,
) -> list:
    """Greedy token ids appended to ``prompt``. Full prefix each step (no cache)."""
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be >= 0")
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    ids = _batch_ids(enc).to(device)
    if ids.shape[1] == 0:
        raise ValueError("prompt tokenized to an empty sequence")
    attn = _batch_mask(enc, device)
    prompt_len = int(ids.shape[1])
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
    return [int(i) for i in ids[0, prompt_len:].tolist()]


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
    prompt is the only text the LM sees. An empty store leaves the residual
    untouched, so greedy tokens match the same model with the hook removed.
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
        if model is None:
            model, tokenizer = _load_pretrained(model_name, device=device or "cpu", dtype=dtype)
        elif tokenizer is None:
            raise ValueError("tokenizer is required when model is passed in")
        if device is None:
            try:
                device = next(model.parameters()).device
            except StopIteration:
                device = torch.device("cpu")
        self._device = torch.device(device)
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
            hidden = hidden.detach()
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
        """Answer-span pool at the inject layer. Not L2-normalized."""
        hidden, enc = self._run_hidden(text)
        mask = self._answer_mask(text, enc, answer, hidden.shape[0])
        return pool_masked(hidden, mask).detach().cpu().numpy().astype(np.float32, copy=False)

    def encode_key(self, text: str, answer: Optional[str] = None) -> np.ndarray:
        """Pool used as the slot key.

        With ``answer``, the key is the fact with that span removed. Without
        it, the key is the whole string (a question or a prefix).
        """
        hidden, enc = self._run_hidden(text)
        content = self._content_mask(enc, hidden.shape[0])
        if answer is None:
            mask = content
        else:
            answer_m = self._answer_mask(text, enc, answer, hidden.shape[0])
            mask = [c and not a for c, a in zip(content, answer_m)]
            if not any(mask):
                mask = content
        return pool_masked(hidden, mask).detach().cpu().numpy().astype(np.float32, copy=False)

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
        hidden, enc = self._run_hidden(prompt)
        mask = self._content_mask(enc, hidden.shape[0])
        return pool_masked(hidden, mask).detach().cpu().numpy().astype(np.float64, copy=False)

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
        value = self.store.readout(query, mode="softtopk")
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

    def continuation_ids(self, prompt: str, max_new_tokens: int = 32) -> list:
        if not isinstance(prompt, str) or prompt == "":
            raise ValueError("prompt is empty")
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
                )
        finally:
            self._delta = None

    def generate(self, prompt: str, max_new_tokens: int = 32) -> str:
        """Greedy continuation. ``prompt`` is the only text passed to the LM."""
        ids = self.continuation_ids(prompt, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(ids, skip_special_tokens=True)

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
        try:
            ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        except TypeError:
            ckpt = torch.load(str(path), map_location="cpu")
        self.inject.load_state_dict(ckpt["inject"])
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
