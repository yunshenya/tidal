"""Tiny causal temporal model over the last <=64 events with multi-task heads (CPU PyTorch)."""
import torch, torch.nn as nn
from tidal.backbones import make_body
BODY_KINDS = ("tx_kv", "mamba3", "mamba3_siso", "m3_ablate_m2")
HEAD_DIMS = {"y_eot": 1, "y_self": 1, "y_addr": 1, "y_act": 3, "y_recheck": 7, "y_hreply": 1}
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
    def forward(self, feats, valid, text=None):
        h = self.fproj(feats)
        if self.use_text: h = h + self.tproj(text)
        h = self.drop(self.inorm(h)) * valid.unsqueeze(-1).to(h.dtype)
        if self.kind == "gru":
            h, _ = self.body(h)
        elif self.kind in BODY_KINDS:
            h = self.body(h, valid)
        else:
            T = h.shape[1]; h = h + self.pos[:, -T:]
            mask = torch.triu(torch.full((T, T), float("-inf"), device=h.device), 1)
            h = self.body(h, mask=mask)
        return {k: m(h) for k, m in self.heads.items()}
def n_params(m): return sum(p.numel() for p in m.parameters())
