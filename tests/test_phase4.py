"""phase 4: backbone parity (parallel form == O(1) streaming step) and padding invariance."""
import torch, pytest
from tidal.backbones import Mamba3Mixer, make_body

def _rand_params(m, seed=0):
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for p in m.parameters(): p.add_(0.3 * torch.randn(p.shape, generator=g))

@pytest.mark.parametrize("R,trap,rope", [(1, True, True), (4, True, True), (1, False, False), (2, True, False)])
def test_mamba3_mixer_parallel_equals_step(R, trap, rope):
    torch.manual_seed(0); m = Mamba3Mixer(32, d_state=16, expand=2, headdim=16, mimo_rank=R, trapezoid=trap, rope=rope).double()
    _rand_params(m); u = torch.randn(3, 40, 32, dtype=torch.float64)
    y = m(u); m.chunk = 0; yq = m(u); m.chunk = 16
    assert torch.allclose(y, yq, atol=1e-9, rtol=1e-7)            # chunked SSD == single-block quadratic form
    st = {k: v.double() for k, v in m.init_state(3).items()}; ys = []
    for t in range(40):
        o, st = m.step(u[:, t], st); ys.append(o)
    assert torch.allclose(y, torch.stack(ys, 1), atol=1e-9, rtol=1e-7)

def test_mamba3_reference_formula_siso():
    """hand-written loop of the official mamba3_siso_step_ref equations vs our step()."""
    torch.manual_seed(1); m = Mamba3Mixer(16, d_state=8, expand=2, headdim=8).double(); _rand_params(m, 1)
    u = torch.randn(2, 12, 16, dtype=torch.float64)
    Q, K, V, Z, DT, ADT, lam, dang = m._inputs(u)
    H, P, N = m.H, m.P, m.N; S = torch.zeros(2, H, P, N, dtype=torch.float64); ang = torch.zeros(2, H, m.S, dtype=torch.float64)
    kprev = torch.zeros(2, H, N, dtype=torch.float64); vprev = torch.zeros(2, H, P, dtype=torch.float64); outs = []
    for t in range(12):
        ang = ang + dang[:, t]; q, k = Q[:, t, :, 0], K[:, t, :, 0]
        def rot(x):
            x = x.clone(); n = m.S
            x0, x1 = x[..., 0:2 * n:2].clone(), x[..., 1:2 * n:2].clone()
            x[..., 0:2 * n:2] = x0 * torch.cos(ang) - x1 * torch.sin(ang); x[..., 1:2 * n:2] = x0 * torch.sin(ang) + x1 * torch.cos(ang); return x
        q, k = rot(q), rot(k); v = V[:, t, :, 0]
        alpha = torch.exp(ADT[:, t]); beta = (1 - lam[:, t]) * DT[:, t] * alpha; gamma = lam[:, t] * DT[:, t]
        S = alpha[..., None, None] * S + beta[..., None, None] * torch.einsum("bhp,bhn->bhpn", vprev, kprev) + gamma[..., None, None] * torch.einsum("bhp,bhn->bhpn", v, k)
        y = torch.einsum("bhpn,bhn->bhp", S, q) + m.D[:, None] * v
        y = y * torch.nn.functional.silu(Z[:, t, :, 0]); outs.append(m.out_proj(y.reshape(2, -1))); kprev, vprev = k, v
    assert torch.allclose(m(u), torch.stack(outs, 1), atol=1e-9)

@pytest.mark.parametrize("kind", ["tx_kv", "mamba3", "mamba3_siso", "m3_ablate_m2"])
def test_body_step_and_padding(kind):
    torch.manual_seed(2); b = make_body(kind).double().eval(); _rand_params(b, 2)
    x = torch.randn(2, 20, 128, dtype=torch.float64)
    full = b(x, torch.ones(2, 20, dtype=torch.bool)); st = None; ys = []
    for t in range(20):
        o, st = b.step(x[:, t], st); ys.append(o)
    assert torch.allclose(full, torch.stack(ys, 1), atol=1e-8)
    xp = torch.cat([torch.randn(2, 5, 128, dtype=torch.float64), x], 1)   # left padding must not change outputs
    valid = torch.cat([torch.zeros(2, 5, dtype=torch.bool), torch.ones(2, 20, dtype=torch.bool)], 1)
    assert torch.allclose(b(xp * valid[..., None], valid)[:, 5:], full, atol=1e-8)

def test_transformer_cache_is_bounded():
    """KV cache is capped at `window` events -> O(window) memory and per-step cost. Note: with >1 layer the cached
    upper-layer keys were computed from longer contexts, so streaming past the window is (like the GRU's unbounded
    state) not bit-identical to re-running a 64-event window; within the window it is exact (test above)."""
    torch.manual_seed(3); b = make_body("tx_kv").double().eval(); b.window = 8
    x = torch.randn(1, 30, 128, dtype=torch.float64); st = None
    for t in range(30): o, st = b.step(x[:, t], st)
    assert all(k.shape[2] == 8 for k, v in st["kv"]) and torch.isfinite(o).all()
