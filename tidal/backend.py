"""Model hooks for the memory layer.

The memory layer chooses the winner, the fan, the decay, and the residual.
A backend runs the model and adds that residual.

:class:`GrowbrainBackend` is the default. It runs the growbrain model at
the inject layer. Text goes in through :meth:`MemoryBackend.encode_hidden`.
That is not :class:`growbrain.tidal_adapter.TidalAdapter`'s ``encode_hidden``,
which projects an already exported packet from ``d_model`` to ``mem_dim``.

:class:`SmolLM2Backend` stays constructible for a caller-supplied model and
for ``HuggingFaceTB/SmolLM2-135M``. It is not the default. There is no
process flag. ``import growbrain`` is tried inside
:func:`try_import_growbrain` and is not a requirement of importing this module.

Stage0 ``e1_base_rebalance`` (no ``expand_d``) pins ``d_model`` at 192,
``mem_dim`` at 64, ``mem_key_dim`` at 32, ``inject_layer`` at 2, ``n_layers``
at 4, and ``head_dim`` at 48. ``d_model`` is the packet ``vec`` width and the
episodic store width. ``mem_dim`` is the growbrain injector slot after
``TidalAdapter.proj`` and is not the store width. A later checkpoint's
``cfg.d_model`` replaces the 192 pin.
"""
from __future__ import annotations

from contextlib import nullcontext
from typing import Mapping, Sequence

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
    """Explicit causal-LM path. SmolLM2-135M when the host loaded that checkpoint.

    A caller-supplied model uses this same class. It is not the default.
    :class:`GrowbrainBackend` is the default when the host asks for one.
    There is no process flag.
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


# Stage0 e1_base_rebalance ModelConfig. No expand_d.
# Packet vec, episodic key, and episodic value use d_model.
# mem_dim is the injector slot on the growbrain side of TidalAdapter.proj.
GROWBRAIN_D_MODEL = 192
GROWBRAIN_MEM_DIM = 64
GROWBRAIN_MEM_KEY_DIM = 32
GROWBRAIN_INJECT_LAYER = 2
GROWBRAIN_N_LAYERS = 4
GROWBRAIN_HEAD_DIM = 48
GROWBRAIN_CHECKPOINT = "e1_base_rebalance"
GROWBRAIN_VOCAB_SIZE = 5135
GROWBRAIN_STAGE0_SHA256 = "1e535ca6a743b991890da4c6122ad9b643d9f78138c5b0da263426d85bf9c668"


def _cfg_get(cfg, name: str):
    """One field from a checkpoint cfg, or None when the field is absent."""
    if cfg is None:
        return None
    if isinstance(cfg, Mapping):
        if name not in cfg:
            return None
        value = cfg[name]
    else:
        value = getattr(cfg, name, None)
    return None if value is None else value


def resolve_d_model(cfg=None, model=None) -> int:
    """Packet width. A checkpoint ``d_model`` wins. Otherwise the stage0 pin.

    ``mem_dim`` is never this width. A cfg that sets ``mem_dim`` and omits
    ``d_model`` still resolves to :data:`GROWBRAIN_D_MODEL`.
    """
    from_cfg = _cfg_get(cfg, "d_model")
    from_model = None
    if model is not None:
        from_model = _cfg_get(getattr(model, "config", None), "d_model")
    if from_cfg is not None and from_model is not None and int(from_cfg) != int(from_model):
        raise ValueError(f"checkpoint cfg.d_model {int(from_cfg)} != model.config.d_model {int(from_model)}")
    if from_cfg is not None:
        return int(from_cfg)
    if from_model is not None:
        return int(from_model)
    return int(GROWBRAIN_D_MODEL)


def resolve_inject_layer(cfg, model, n_layers: int) -> int:
    """Inject block. Stage0 pins layer 2. A cfg ``inject_layer`` replaces it."""
    value = _cfg_get(cfg, "inject_layer")
    if value is None and model is not None:
        value = _cfg_get(getattr(model, "config", None), "inject_layer")
    if value is None:
        value = GROWBRAIN_INJECT_LAYER
    layer = int(value)
    if int(n_layers) < 1 or not 0 <= layer < int(n_layers):
        raise ValueError(f"inject layer {layer} is outside 0..{int(n_layers) - 1}")
    return layer


def _cfg_int(cfg, model, name: str, default: int) -> int:
    value = _cfg_get(cfg, name)
    if value is None and model is not None:
        value = _cfg_get(getattr(model, "config", None), name)
    if value is None:
        return int(default)
    return int(value)


def check_growbrain_layers(cfg, model, n_layers: int) -> None:
    """The decoder stack must match cfg.n_layers when the cfg names one."""
    expected = _cfg_get(cfg, "n_layers")
    if expected is None and model is not None:
        expected = _cfg_get(getattr(model, "config", None), "n_layers")
    if expected is not None and int(expected) != int(n_layers):
        raise ValueError(f"cfg.n_layers {int(expected)} != decoder blocks {int(n_layers)}")


def try_import_growbrain():
    """The optional package, or None. Importing this module does not import it."""
    try:
        import growbrain
    except ImportError:
        return None
    return growbrain


def _sha256_file(path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stage0_checkpoint_files():
    """``(checkpoint, vocab)`` under the repo or the working directory.

    Both files are required. A missing pair is None. This does not search
    a package tree and does not invent a path.
    """
    from pathlib import Path

    roots = []
    for root in (Path.cwd(), Path(__file__).resolve().parents[1]):
        if root not in roots:
            roots.append(root)
    for root in roots:
        ckpt = root / "checkpoints" / "e1_base_rebalance.pt"
        vocab = root / "vocab" / "charlist.txt"
        if ckpt.is_file() and vocab.is_file():
            return ckpt, vocab
    return None


class _CharTokenizerAdapter:
    """Char tokenizer with the call shape ``LLMMemory`` already uses.

    Ids are one per character. No bos or eos is added, so a fact's offsets
    stay aligned with the string. Multi-character specials are left to the
    inner tokenizer and then have no offset map.
    """

    def __init__(self, inner):
        self.inner = inner
        self.eos_token_id = None

    def __len__(self):
        return len(self.inner)

    def encode(self, text, add_special_tokens=True, **_kwargs):
        del add_special_tokens, _kwargs
        ids = self.inner.encode(text)
        if hasattr(ids, "tolist"):
            ids = ids.tolist()
        return [int(i) for i in ids]

    def decode(self, ids, skip_special_tokens=True, **_kwargs):
        del skip_special_tokens, _kwargs
        if torch.is_tensor(ids):
            ids = ids.detach().cpu().view(-1).tolist()
        elif hasattr(ids, "tolist"):
            ids = ids.tolist()
        return self.inner.decode([int(i) for i in ids])

    def __call__(self, text, return_tensors="pt", add_special_tokens=True, return_offsets_mapping=False, **_kwargs):
        del return_tensors, add_special_tokens, return_offsets_mapping, _kwargs
        ids = self.encode(text)
        if not ids:
            raise ValueError("text tokenized to an empty id sequence")
        payload = {"input_ids": torch.tensor([ids], dtype=torch.long)}
        if len(ids) == len(text):
            payload["offset_mapping"] = torch.tensor([[(i, i + 1) for i in range(len(ids))]], dtype=torch.long)
        return payload


class _GrowbrainHost(torch.nn.Module):
    """``model(idx)`` and the keyword call ``LLMMemory`` already makes.

    ``blocks`` is the growbrain decoder stack. It is not registered twice.
    """

    def __init__(self, inner):
        super().__init__()
        self.gb = inner
        object.__setattr__(self, "blocks", inner.blocks)
        object.__setattr__(self, "cfg", inner.cfg)
        object.__setattr__(self, "config", inner.cfg)

    def forward(
        self,
        idx=None,
        signals=None,
        hippo_state=None,
        return_cache=False,
        input_ids=None,
        use_cache=False,
        attention_mask=None,
        output_hidden_states=False,
    ):
        del use_cache, attention_mask, output_hidden_states
        ids = idx if idx is not None else input_ids
        if ids is None:
            raise TypeError("model(idx) requires token ids")
        if ids.dim() == 1:
            ids = ids.unsqueeze(0)
        limit = int(self.cfg.max_seq_len)
        if int(ids.shape[-1]) > limit:
            raise RuntimeError(f"sequence length {int(ids.shape[-1])} exceeds max_seq_len {limit}")
        return self.gb(ids, signals, hippo_state, return_cache)


def load_growbrain_stage0(*, device, dtype=None):
    """Load stage0 ``e1_base_rebalance`` for :class:`GrowbrainBackend`.

    The package must provide ``model.load_checkpoint`` and
    ``tokenizer.CharTokenizer``. The checkpoint and ``vocab/charlist.txt``
    must sit under the repo or the working directory. A missing package, a
    package without that loader, or a missing file raises. This does not
    construct :class:`SmolLM2Backend`.
    """
    module = try_import_growbrain()
    if module is None:
        raise ImportError(
            "LLMMemory() uses GrowbrainBackend at d_model 192 "
            "(stage0 e1_base_rebalance, inject layer 2). "
            "growbrain is not installed. "
            "Pass a model and tokenizer, or "
            "model_name='HuggingFaceTB/SmolLM2-135M', to construct SmolLM2Backend. "
            "There is no process switch."
        )
    import importlib

    try:
        model_mod = importlib.import_module("growbrain.model")
        tok_mod = importlib.import_module("growbrain.tokenizer")
    except ImportError:
        model_mod = None
        tok_mod = None
    load_checkpoint = getattr(model_mod, "load_checkpoint", None) if model_mod is not None else None
    char_tokenizer = getattr(tok_mod, "CharTokenizer", None) if tok_mod is not None else None
    if load_checkpoint is None or char_tokenizer is None:
        names = [name for name in dir(module) if not name.startswith("_")]
        raise RuntimeError(
            "growbrain imported, but e1_base_rebalance is not loaded. "
            "The package has no model.load_checkpoint or tokenizer.CharTokenizer, "
            "so the load call was not guessed. "
            f"Public names: {', '.join(names) if names else '(none)'}. "
            "Refusing a 576-d SmolLM2 shell."
        )
    found = stage0_checkpoint_files()
    if found is None:
        raise RuntimeError(
            "growbrain imported, but e1_base_rebalance is not loaded. "
            "checkpoints/e1_base_rebalance.pt and vocab/charlist.txt were not "
            "both present under the repo or the working directory. "
            "Refusing a 576-d SmolLM2 shell."
        )
    ckpt, vocab = found
    digest = _sha256_file(ckpt)
    if digest != GROWBRAIN_STAGE0_SHA256:
        raise RuntimeError(
            f"e1_base_rebalance sha256 {digest} != {GROWBRAIN_STAGE0_SHA256}. "
            "Refusing a 576-d SmolLM2 shell."
        )
    gb, _extra = load_checkpoint(str(ckpt))
    tok = char_tokenizer.from_file(vocab)
    if int(gb.cfg.vocab_size) != int(len(tok)):
        raise RuntimeError(
            f"checkpoint vocab_size {int(gb.cfg.vocab_size)} != tokenizer {len(tok)}"
        )
    if int(gb.cfg.vocab_size) != int(GROWBRAIN_VOCAB_SIZE):
        raise RuntimeError(
            f"checkpoint vocab_size {int(gb.cfg.vocab_size)} != {GROWBRAIN_VOCAB_SIZE}"
        )
    if dtype is not None:
        gb = gb.to(dtype=dtype)
    gb = gb.to(device=torch.device(device) if device is not None else "cpu")
    gb.eval()
    cfg = gb.config_dict()
    return _GrowbrainHost(gb), _CharTokenizerAdapter(tok), cfg


def _logits_from_model_call(out):
    """Logits from ``model(idx)``. A layout this cannot name is an error."""
    if torch.is_tensor(out):
        return out.detach()
    if hasattr(out, "logits") and torch.is_tensor(getattr(out, "logits")):
        return out.logits.detach()
    if isinstance(out, tuple) and out and torch.is_tensor(out[0]):
        return out[0].detach()
    raise TypeError(
        f"model(idx) returned {type(out).__name__}, which has no logits tensor. "
        "The return layout was not readable from growbrain source."
    )


def _as_id_list(ids) -> list:
    if torch.is_tensor(ids):
        ids = ids.detach().cpu().view(-1).tolist()
    elif hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], (list, tuple)):
        ids = list(ids[0])
    return [int(i) for i in ids]


class GrowbrainBackend(MemoryBackend):
    """Text forward on the growbrain model at the inject layer.

    ``encode_hidden(text)`` calls ``model(idx)`` and copies the inject-layer
    residual. It does not call ``TidalAdapter.encode_hidden``. The residual
    width is ``d_model``. ``mem_dim`` is recorded for the growbrain injector
    slot and is not the store width. A zero-width or 576-d map is not applied.
    """

    def __init__(self, model, tokenizer, *, layer: int, n_layers: int, layers, device, cfg=None):
        self.model = model
        self.tokenizer = tokenizer
        self.n_layers = int(n_layers)
        self.layers = layers
        self.device = device
        self.cfg = cfg
        self.hidden = resolve_d_model(cfg, model)
        self.layer = int(layer)
        if not 0 <= self.layer < self.n_layers:
            raise ValueError(f"inject layer {self.layer} is outside 0..{self.n_layers - 1}")
        self.mem_dim = _cfg_int(cfg, model, "mem_dim", GROWBRAIN_MEM_DIM)
        self.mem_key_dim = _cfg_int(cfg, model, "mem_key_dim", GROWBRAIN_MEM_KEY_DIM)
        self.head_dim = _cfg_int(cfg, model, "head_dim", GROWBRAIN_HEAD_DIM)
        self._clean = nullcontext
        self._handle = None

    def bind_clean(self, clean) -> None:
        self._clean = clean

    def attach_inject(self, callback):
        self._handle = self.layers[self.layer].register_forward_hook(callback)
        return self._handle

    def key_prefix_ok(self) -> bool:
        """Char stacks do not use the Llama prefix. The full ``model(idx)`` runs."""
        return False

    def tokenize(self, text: str) -> dict:
        return self._tokenize(text)

    def _tokenize(self, text: str) -> dict:
        tok = self.tokenizer
        if callable(tok):
            try:
                enc = tok(text, return_tensors="pt", add_special_tokens=True, return_offsets_mapping=True)
            except TypeError:
                enc = None
            else:
                if isinstance(enc, Mapping) or hasattr(enc, "keys"):
                    return enc
        if not hasattr(tok, "encode"):
            raise TypeError("growbrain tokenizer has no encode and is not callable as a Hugging Face tokenizer")
        ids = _as_id_list(tok.encode(text))
        if not ids:
            raise ValueError("text tokenized to an empty id sequence")
        payload = {"input_ids": torch.tensor([ids], dtype=torch.long)}
        # One id per character can name its own span. A shorter id list has
        # no offsets until the growbrain tokenizer source is available.
        if len(ids) == len(text):
            payload["offset_mapping"] = torch.tensor([[(i, i + 1) for i in range(len(ids))]], dtype=torch.long)
        return payload

    def _call_model(self, ids: torch.Tensor):
        """``model(idx)``. A rejected call is not rewritten into another signature."""
        if not callable(self.model):
            raise TypeError("growbrain model is not callable")
        try:
            return self.model(ids)
        except TypeError as exc:
            raise RuntimeError(
                "GrowbrainBackend called model(idx) and the module rejected that call. "
                "The e1_base_rebalance source was not readable here, so the signature "
                "was not guessed past model(idx). This is not a SmolLM2 forward."
            ) from exc

    def _forward_ids(self, enc) -> tuple:
        from tidal.llm_memory import _batch_ids

        ids = _batch_ids(enc).to(self.device)
        captured: dict = {}

        def _keep(_module, _args, output):
            tensor = output[0] if isinstance(output, tuple) else output
            if not torch.is_tensor(tensor):
                raise TypeError(f"inject layer returned {type(output).__name__}")
            copied = tensor.detach()
            if copied.dim() == 3:
                if int(copied.shape[0]) != 1:
                    raise ValueError("growbrain encode expects one sequence")
                copied = copied[0]
            if copied.dim() != 2:
                raise ValueError(f"inject residual rank {copied.dim()} is not [T, d_model]")
            captured["hidden"] = copied.float()

        handle = self.layers[self.layer].register_forward_hook(_keep)
        try:
            with self._clean():
                with torch.no_grad():
                    out = self._call_model(ids)
        finally:
            handle.remove()
        hidden = captured.get("hidden")
        if hidden is None:
            raise RuntimeError(
                f"inject layer {self.layer} did not run. "
                "Refusing to invent a hidden state or to reuse a 576-d vector."
            )
        if int(hidden.shape[-1]) != int(self.hidden):
            raise RuntimeError(
                f"inject residual width {int(hidden.shape[-1])} != d_model {int(self.hidden)}. "
                "Refusing to project it to mem_dim or to 576."
            )
        return hidden, _logits_from_model_call(out)

    def encode_hidden(self, text: str) -> tuple:
        if not isinstance(text, str) or text == "":
            raise ValueError("text is empty")
        enc = self._tokenize(text)
        hidden, logits = self._forward_ids(enc)
        return hidden, enc, logits

    def encode_key_hidden(self, text: str) -> tuple:
        hidden, enc, _logits = self.encode_hidden(text)
        del _logits
        return hidden, enc

    def encode_cue(self, text: str) -> np.ndarray:
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
            self.encode_hidden(text)
        finally:
            handle.remove()
        hidden = captured.get("hidden")
        if hidden is None:
            raise RuntimeError("the cue hook did not see the residual")
        if int(np.asarray(hidden).shape[-1]) != int(self.hidden):
            raise RuntimeError(
                f"cue width {int(np.asarray(hidden).shape[-1])} != d_model {int(self.hidden)}"
            )
        return np.asarray(hidden, dtype=np.float64)

    def answer_nll(self, logits, token_ids, answer_mask) -> float:
        return answer_nll(logits, token_ids, answer_mask)

    def inject_residual(self, output, residual, *, broadcast: bool = False, batched: bool = True):
        if broadcast:
            return _add_delta(output, residual)
        tensor = output[0] if isinstance(output, tuple) else output
        add_t = torch.tensor(residual, dtype=tensor.dtype, device=tensor.device)
        if int(add_t.shape[-1]) != int(tensor.shape[-1]):
            raise ValueError(f"inject dim {int(add_t.shape[-1])} != hidden {int(tensor.shape[-1])}")
        if not batched:
            add_t = add_t[0]
        updated = tensor + add_t
        if torch.is_tensor(output):
            return updated
        return (updated,) + output[1:]
