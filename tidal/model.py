"""Tiny causal temporal model over the last <=64 events with multi-task heads (CPU PyTorch)."""
import torch, torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from tidal.backbones import make_body
BODY_KINDS = ("tx_kv", "mamba3", "mamba3_siso", "m3_ablate_m2")
HEAD_DIMS = {"y_eot": 1, "y_self": 1, "y_addr": 1, "y_act": 3, "y_recheck": 7, "y_hreply": 1}

def run_gru(gru, x, valid):
    """GRU forward whose padded steps do not change the hidden state.

    `valid` is a bool mask [B, T] (True = real token). Left-padded windows (the training / shadow
    layout) and right-padded batches go through pack_padded_sequence. Any other mask falls back to
    a per-step update that copies the state through invalid steps. An all-valid batch is the plain GRU.
    """
    if valid is not None and torch.onnx.is_in_onnx_export():
        return _gru_step_mask(gru, x, valid)
    if valid is None or bool(valid.bool().all()):
        return gru(x)
    valid = valid.bool()
    B, T, _ = x.shape
    lengths = valid.sum(1).to(dtype=torch.long)
    idx = torch.arange(T, device=x.device)
    left = bool((valid == (idx >= (T - lengths).unsqueeze(1))).all())
    right = bool((valid == (idx < lengths.unsqueeze(1))).all())
    if not left and not right:
        return _gru_step_mask(gru, x, valid)
    if bool((lengths == 0).all()):
        return x.new_zeros(B, T, gru.hidden_size), x.new_zeros(gru.num_layers, B, gru.hidden_size)
    place_left = left and not bool((lengths == T).all()) and not (right and not left)
    # left-padded (and not the all-valid case): pack tokens at the front, scatter outputs back
    if left and not right:
        src = ((T - lengths).unsqueeze(1) + idx).clamp(max=T - 1)
        x_use = x.gather(1, src.unsqueeze(-1).expand_as(x))
        keep = idx.unsqueeze(0) < lengths.unsqueeze(1)
        x_use = x_use * keep.unsqueeze(-1).to(x.dtype)
        place_left = True
    else:
        x_use = x
        keep = idx.unsqueeze(0) < lengths.clamp(min=0).unsqueeze(1)
        place_left = False
    packed = pack_padded_sequence(x_use, lengths.clamp(min=1).cpu(), batch_first=True, enforce_sorted=False)
    y, h = gru(packed)
    y, _ = pad_packed_sequence(y, batch_first=True, total_length=T)
    keep = idx.unsqueeze(0) < lengths.unsqueeze(1)
    y = y * keep.unsqueeze(-1).to(y.dtype)
    nz = (lengths > 0).to(h.dtype).view(1, B, 1)
    h = h * nz
    if place_left:
        dest = torch.where(keep, (T - lengths).unsqueeze(1) + idx, torch.zeros_like(idx).expand(B, T))
        y_out = y.new_zeros(y.shape)
        y_out.scatter_(1, dest.unsqueeze(-1).expand_as(y), y)
        y = y_out
    return y, h

def _gru_step_mask(gru, x, valid):
    """Per-step mask for non-contiguous padding. Invalid steps keep the previous hidden state."""
    B, T, _ = x.shape
    h = x.new_zeros(gru.num_layers, B, gru.hidden_size)
    ys = []
    for t in range(T):
        y, h2 = gru(x[:, t:t + 1], h)
        m = valid[:, t].to(dtype=x.dtype).view(1, B, 1)
        h = h2 * m + h * (1 - m)
        ys.append(y[:, 0] * valid[:, t:t + 1].to(x.dtype))
    return torch.stack(ys, 1), h

class TurnModel(nn.Module):
    def __init__(self, n_feat, use_text, d=128, hidden=192, layers=2, kind="gru", dropout=0.2, nhead=4):
        super().__init__()
        self.use_text = use_text; self.kind = kind
        self.fproj = nn.Linear(n_feat, d)
        if use_text: self.tproj = nn.Sequential(nn.Dropout(0.1), nn.Linear(512, d))
        self.inorm = nn.LayerNorm(d); self.drop = nn.Dropout(dropout)
        if kind == "gru":
            self.body = nn.GRU(d, hidden, layers, batch_first=True, dropout=dropout); out = hidden
        elif kind in BODY_KINDS:                      # phase 4 backbones (tidal/backbones.py)
            self.body = make_body(kind, d); out = self.body.out_dim
        else:
            layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout=dropout, batch_first=True, norm_first=True)
            self.body = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
            self.pos = nn.Parameter(torch.zeros(1, 64, d)); out = d
        self.heads = nn.ModuleDict({k: nn.Sequential(nn.Linear(out, 64), nn.GELU(), nn.Linear(64, v)) for k, v in HEAD_DIMS.items()})
    def _encode(self, h, valid):
        """Causal encode. Padded steps do not update GRU / SSM state and are not attention keys."""
        if self.kind == "gru":
            h, _ = run_gru(self.body, h, valid)
        elif self.kind in BODY_KINDS:
            h = self.body(h, valid)
        else:
            B, T, _ = h.shape
            h = h + self.pos[:, -T:]
            nhead = self.body.layers[0].self_attn.num_heads
            # float mask [B * nhead, T, T]: causal, and pad keys are invisible
            mask = torch.triu(torch.full((T, T), float("-inf"), device=h.device), 1)
            mask = mask.unsqueeze(0).expand(B, T, T).clone()
            pad = ~valid.bool()
            mask = mask.masked_fill(pad[:, None, :], float("-inf"))
            eye = torch.eye(T, device=h.device, dtype=torch.bool)
            mask = mask.masked_fill(eye.unsqueeze(0) & pad[:, None, :], 0.0)
            h = self.body(h, mask=mask.repeat_interleave(nhead, 0))
        return h
    def forward(self, feats, valid, text=None):
        h = self.fproj(feats)
        if self.use_text: h = h + self.tproj(text)
        h = self.drop(self.inorm(h)) * valid.unsqueeze(-1).to(h.dtype)
        h = self._encode(h, valid)
        return {k: m(h) for k, m in self.heads.items()}
class BinHead(nn.Module):
    """Phase-5 side head on the encoder state h (y_addr / y_interrupt / y_topic). Returns a logit. Not in HEAD_DIMS."""
    def __init__(self, d=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.GELU(), nn.Linear(64, 1))
    def forward(self, h):
        return self.net(h).squeeze(-1)
def n_params(m): return sum(p.numel() for p in m.parameters())
