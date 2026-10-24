"""Model hooks for the memory layer.

The memory layer chooses the winner, the fan, the decay, and the residual.
A backend runs the model and adds that residual. :class:`SmolLM2Backend` is
the only implementation. It is the current causal-LM path, including a
caller-supplied model. A growbrain backend is planned and is not imported.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Sequence

import numpy as np
import torch


def answer_nll(logits: torch.Tensor, token_ids: Sequence[int], answer_mask: Sequence[bool]) -> float:
    """Total NLL, in nats, of the answer tokens under logits the forward already returned.

    Position ``i`` is predicted by ``logits[i - 1]``. The score is the sum,
    not the mean. The first token has no predictor; if the answer is only
    that token, the score is 0. This does not run the model.
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


def _broadcast(delta: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
    add = delta.to(device=hidden.device, dtype=hidden.dtype).reshape(-1)
    if add.numel() != hidden.shape[-1]:
        raise ValueError(f"inject dim {add.numel()} != hidden {hidden.shape[-1]}")
    shape = [1] * (hidden.dim() - 1) + [add.numel()]
    return add.view(*shape)


def _add_delta(output, delta: torch.Tensor):
    if torch.is_tensor(output):
        return output + _broadcast(delta, output)
    if isinstance(output, tuple) and output and torch.is_tensor(output[0]):
        return (_add_delta(output[0], delta),) + output[1:]
    raise TypeError(f"cannot inject into layer output of type {type(output).__name__}")


class MemoryBackend:
    """Encode, answer NLL, and residual inject.

    ``encode_hidden(text) -> (hidden, encoding, logits)``
        One forward. ``hidden`` is float32 ``[T, H]`` at the inject layer.
        ``logits`` come from that same forward. No memory residual is added.

    ``encode_cue(text) -> cue``
        Pre-add inject-layer rows, float64 ``[T, H]``. Does not inject.

    ``encode_key_hidden(text) -> (hidden, encoding)``
        Inject-layer rows for a key. The SmolLM2 prefix must match
        ``encode_hidden`` at that layer.

    ``answer_nll(logits, token_ids, answer_mask) -> float``
        Total answer-token NLL from logits already returned. No second forward.

    ``inject_residual(output, residual, *, broadcast=False, batched=True)``
        Add a residual into a decoder-layer output. ``broadcast=False`` adds
        a numpy array ``[batch, steps, hidden]`` at the same index.
        ``broadcast=True`` adds one vector on every position. A zero retrieval
        residual is not passed in: the caller returns None so the layer output
        object stays the one the block produced.
    """

    def encode_hidden(self, text: str):
        raise NotImplementedError

    def encode_cue(self, text: str):
        raise NotImplementedError

    def encode_key_hidden(self, text: str):
        raise NotImplementedError

    def answer_nll(self, logits, token_ids, answer_mask) -> float:
        raise NotImplementedError

    def inject_residual(self, output, residual, *, broadcast: bool = False, batched: bool = True):
        raise NotImplementedError


class SmolLM2Backend(MemoryBackend):
    """Current causal-LM path. SmolLM2-135M when the host loaded that checkpoint.

    A caller-supplied model uses this same class. There is no second backend
    and no switch. A growbrain backend is planned and is not imported.
    """

    def __init__(self, model, tokenizer, *, layer: int, n_layers: int, hidden: int, layers, device):
        self.model = model
        self.tokenizer = tokenizer
        self.layer = int(layer)
        self.n_layers = int(n_layers)
        self.hidden = int(hidden)
        self.layers = layers
        self.device = device
        self._clean = nullcontext
        self._handle = None

    def bind_clean(self, clean) -> None:
        """Host context that clears retrieval and training adds during an encode."""
        self._clean = clean

    def attach_inject(self, callback):
        """Register the forward hook that adds the host's residual."""
        self._handle = self.layers[self.layer].register_forward_hook(callback)
        return self._handle

    def tokenize(self, text: str):
        kwargs = {"return_tensors": "pt", "add_special_tokens": True, "return_offsets_mapping": True}
        try:
            return self.tokenizer(text, **kwargs)
        except (TypeError, ValueError):
            kwargs.pop("return_offsets_mapping")
            return self.tokenizer(text, **kwargs)

    def key_prefix_ok(self) -> bool:
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

    def encode_hidden(self, text: str) -> tuple:
        """Hidden state at the inject layer, with memory unhooked for this forward."""
        from tidal.llm_memory import _batch_ids, _batch_mask, _hidden_tuple, _logits_of, layer_hidden

        if not isinstance(text, str) or text == "":
            raise ValueError("text is empty")
        enc = self.tokenize(text)
        ids = _batch_ids(enc).to(self.device)
        attn = _batch_mask(enc, self.device)
        kwargs = {"input_ids": ids, "use_cache": False, "output_hidden_states": True}
        if attn is not None:
            kwargs["attention_mask"] = attn
        with self._clean():
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

    def encode_key_hidden(self, text: str) -> tuple:
        """Hidden state at the inject layer, without the later blocks or the head.

        The shipped stack is ``LlamaModel.forward``. That method runs
        ``config.num_hidden_layers`` blocks and then a final norm. The key
        pool needs the inject block's output, so this sets that count to
        ``layer + 1``, runs the same forward, and copies the block output
        from a hook. The final norm still runs on that prefix. The later
        blocks and the output head do not. The count is restored before
        return. The residual is cleared first, matching :meth:`encode_hidden`.
        The copied tensor is the one :func:`tidal.llm_memory.layer_hidden`
        reads from a full forward. A model without this stack uses that full
        forward instead.
        """
        from tidal.llm_memory import _batch_ids, _batch_mask

        if not isinstance(text, str) or text == "":
            raise ValueError("text is empty")
        if not self.key_prefix_ok():
            hidden, enc, _logits = self.encode_hidden(text)
            del _logits
            return hidden, enc
        enc = self.tokenize(text)
        ids = _batch_ids(enc).to(self.device)
        attn = _batch_mask(enc, self.device)
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
            with self._clean():
                with torch.no_grad():
                    inner(**kwargs)
        finally:
            inner.config.num_hidden_layers = saved
            handle.remove()
        hidden = captured.get("hidden")
        if hidden is None:
            hidden, enc, _logits = self.encode_hidden(text)
            del _logits
            return hidden, enc
        return hidden, enc

    def encode_cue(self, text: str) -> np.ndarray:
        """Pre-add cue rows, ``[T, H]``. Does not inject and does not change the store."""
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
            if self.key_prefix_ok():
                self.encode_key_hidden(text)
            else:
                self.encode_hidden(text)
        finally:
            handle.remove()
        hidden = captured.get("hidden")
        if hidden is None:
            raise RuntimeError("the cue hook did not see the residual")
        return np.asarray(hidden, dtype=np.float64)

    def answer_nll(self, logits, token_ids, answer_mask) -> float:
        return answer_nll(logits, token_ids, answer_mask)

    def inject_residual(self, output, residual, *, broadcast: bool = False, batched: bool = True):
        """Add ``residual`` into a decoder-layer output.

        ``broadcast=True`` is the training add: one vector on every position.
        Otherwise ``residual`` is a numpy array ``[batch, steps, hidden]``
        added at the same index. The caller does not pass an all-zero
        retrieval residual; it returns None so the layer output object is
        left untouched.
        """
        if broadcast:
            return _add_delta(output, residual)
        tensor = output[0] if isinstance(output, tuple) else output
        add_t = torch.tensor(residual, dtype=tensor.dtype, device=tensor.device)
        if not batched:
            add_t = add_t[0]
        updated = tensor + add_t
        if torch.is_tensor(output):
            return updated
        return (updated,) + output[1:]
