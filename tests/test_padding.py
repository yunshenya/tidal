"""Padded steps must not change the prediction of a short sequence."""
import torch
from tidal.backbones import make_body
from tidal.vap import VAPModel

def _pad(x, n):
    """Left-pad with NONZERO junk. The model has to ignore it via the valid mask, not via zero inputs."""
    junk = torch.randn(x.shape[0], n, x.shape[-1], dtype=x.dtype)
    return torch.cat([junk, x], 1)

def _same(a, b, n):
    assert torch.allclose(a, b[:, n:], atol=1e-5, rtol=1e-5), (a - b[:, n:]).abs().max().item()

@torch.no_grad()
def test_gru_pad_length_does_not_change_predictions():
    torch.manual_seed(0)
    m = VAPModel(8, kind="gru").eval()
    x = torch.randn(2, 5, 8)
    v0 = torch.ones(2, 5, dtype=torch.bool)
    y0 = m(x, v0)
    for n in (0, 3, 7):
        if n == 0:
            y = y0
        else:
            xp = _pad(x, n)
            v = torch.cat([torch.zeros(2, n, dtype=torch.bool), torch.ones(2, 5, dtype=torch.bool)], 1)
            y = m(xp, v)
        _same(y0["y_eot"], y["y_eot"], n)
        _same(y0["vap"], y["vap"], n)

@torch.no_grad()
def test_recurrent_backbones_ignore_pad():
    torch.manual_seed(1)
    x = torch.randn(2, 6, 128)
    for kind in ("tx_kv", "mamba3_siso", "mamba3", "m3_ablate_m2"):
        b = make_body(kind).eval()
        full = b(x, torch.ones(2, 6, dtype=torch.bool))
        n = 4
        xp = _pad(x, n)
        # bodies do not zero the input themselves; zeroing is the encoder's job. Pass zeros AND the mask,
        # and also check a streaming pad step does not move state.
        xp = torch.cat([torch.zeros(2, n, 128), x], 1)
        valid = torch.cat([torch.zeros(2, n, dtype=torch.bool), torch.ones(2, 6, dtype=torch.bool)], 1)
        got = b(xp, valid)
        assert torch.allclose(full, got[:, n:], atol=1e-5, rtol=1e-5), kind
        st = None
        y, st = b.step(torch.randn(2, 128), st, valid=torch.zeros(2, dtype=torch.bool))
        y2, st2 = b.step(x[:, 0], None)
        y3, _ = b.step(x[:, 0], st, valid=torch.ones(2, dtype=torch.bool))
        assert torch.allclose(y2, y3, atol=1e-5), kind

@torch.no_grad()
def test_learned_pos_transformer_ignores_pad():
    torch.manual_seed(2)
    m = VAPModel(8, kind="tf").eval()   # fallback encoder: learned positions, not a BODY_KIND
    x = torch.randn(2, 4, 8)
    y0 = m(x, torch.ones(2, 4, dtype=torch.bool))
    n = 3
    xp = torch.cat([torch.randn(2, n, 8), x], 1)
    v = torch.cat([torch.zeros(2, n, dtype=torch.bool), torch.ones(2, 4, dtype=torch.bool)], 1)
    y = m(xp, v)
    _same(y0["vap"], y["vap"], n)
