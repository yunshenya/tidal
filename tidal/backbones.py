"""Alternative causal backbones for the ~0.5M-parameter event encoder (phase 4B), pure PyTorch, CPU.

Every body maps h [B, T, d] (+ valid mask [B, T]) -> [B, T, out_dim] and has an O(1)-per-event streaming
step(x [B, d], state) -> (y [B, out_dim], state).

* CausalTransformer: pre-norm, RoPE, causal + key-padding mask; streaming with a KV cache capped at the 64-event
  window (RoPE is relative, so a sliding cache reproduces the windowed computation exactly).
* Mamba3: faithful CPU re-implementation of the Mamba-3 mixer (Lahoti, Li, Chen, Wang, Bick, Kolter, Dao, Gu,
  "Mamba-3: Improved Sequence Modeling using State Space Principles", arXiv:2603.15569 / ICLR 2026), following the
  official reference recurrences in state-spaces/mamba (tests/ops/triton/test_mamba3_siso.py::mamba3_siso_step_ref,
  ..._fwd_ref and tests/ops/tilelang/test_mamba3_mimo.py::mamba3_MIMO_step_ref) and mamba_ssm/modules/mamba3.py:
    - exponential-trapezoidal discretisation (Prop. 1): h_t = a_t h_{t-1} + b_t B_{t-1} x_{t-1} + g_t B_t x_t with
      a_t = exp(dt_t A_t), b_t = (1 - lam_t) dt_t a_t, g_t = lam_t dt_t, lam_t = sigmoid(trap_t) (data-dependent);
    - data-dependent A_t = -heavy_tail(proj) clamped <= -A_floor; dt_t = softplus(proj + dt_bias) (official init);
    - complex-valued state via the "RoPE trick" (Prop. 4): cumulative data-dependent angles
      theta_t = sum_s tanh(proj_s) * pi * dt_s rotate B and C pairwise on the first rope_fraction*N channels;
    - BC (QK) RMSNorm, then head-wise learnable B/C biases initialised to 1; D skip; SiLU(z) output gate;
    - MIMO of rank R: B, C get rank R; x and z are up-projected per head by learnable mimo_x / mimo_z, the R outputs
      are combined by mimo_o (official init 1/R, 1, 1/R); SISO = R 1 with unit projections;
    - block layout as in the paper: Llama-style, pre-norm, alternating Mamba-3 mixer and SwiGLU MLP; no short conv.
  Parallel (training) form = the official quadratic SSD-style reference (decay mask exp(segsum(A dt)), per-key scale
  dt_s lam_s + dt_{s+1}(1 - lam_{s+1}), diagonal correction), generalised to MIMO; step form = the exact recurrence.
  tests/test_phase4.py checks parallel == step-by-step (SISO and MIMO) and padding invariance.
  CPU simplifications (documented): fp32 everywhere; one chunk (windows are <= 64 events, so the quadratic form is used
  directly instead of chunked scans); angles are NOT reduced mod 2*pi (mathematically identical, only matters for
  bf16 kernels); pairwise rotation for MIMO too (the official MIMO kernel rotates halves - an equivalent channel
  permutation); ngroups = 1 (B, C shared across heads, as in the paper's MVA layout).
  Ablation flags: trapezoid=False -> exponential-Euler (lam = 1), rope=False -> real-valued state. With both off and
  R = 1 the recurrence is the Mamba-2 (SSD) recurrence with Mamba-3's BCNorm/biases ("m3-ablate-m2")."""
import math, torch, torch.nn as nn, torch.nn.functional as F

class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-5):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps = eps
    def forward(self, x): return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.w

class SwiGLU(nn.Module):
    def __init__(self, d, hidden):
        super().__init__(); self.w12 = nn.Linear(d, 2 * hidden, bias=False); self.w3 = nn.Linear(hidden, d, bias=False)
    def forward(self, x): a, b = self.w12(x).chunk(2, -1); return self.w3(F.silu(a) * b)

def heavy_tail(x): return x.clamp_min(0) + torch.reciprocal(1 - x.clamp_max(0))

def _rotate(t, cos, sin):
    """pairwise rotation of the first 2*S channels of t[..., N] by angles (cos/sin [..., S])."""
    S = cos.shape[-1]; a, rest = t[..., :2 * S], t[..., 2 * S:]
    a = a.reshape(*a.shape[:-1], S, 2); a0, a1 = a[..., 0], a[..., 1]
    r = torch.stack([a0 * cos - a1 * sin, a0 * sin + a1 * cos], -1).reshape(*t.shape[:-1], 2 * S)
    return torch.cat([r, rest], -1)

class Mamba3Mixer(nn.Module):
    def __init__(self, d_model, d_state=32, expand=2, headdim=32, mimo_rank=1, rope_fraction=0.5, dt_min=1e-3,
                 dt_max=0.1, dt_init_floor=1e-4, A_floor=1e-4, trapezoid=True, rope=True, chunk=16):
        super().__init__(); self.chunk = chunk
        self.d_inner = expand * d_model; self.P = headdim; self.H = self.d_inner // headdim; self.N = d_state
        self.R = mimo_rank; self.A_floor = A_floor; self.trapezoid = trapezoid; self.rope = rope
        S = int(d_state * rope_fraction); S -= S % 2; self.S = S // 2 if rope else 0
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner + 2 * d_state * self.R + 3 * self.H + self.S, bias=False)
        dt = torch.exp(torch.rand(self.H) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.B_bias = nn.Parameter(torch.ones(self.H, self.R, d_state)); self.C_bias = nn.Parameter(torch.ones(self.H, self.R, d_state))
        self.B_norm = RMSNorm(d_state); self.C_norm = RMSNorm(d_state)
        if self.R > 1:
            self.mimo_x = nn.Parameter(torch.ones(self.H, self.R, headdim) / self.R)
            self.mimo_z = nn.Parameter(torch.ones(self.H, self.R, headdim))
            self.mimo_o = nn.Parameter(torch.ones(self.H, self.R, headdim) / self.R)
        self.D = nn.Parameter(torch.ones(self.H))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _inputs(self, u):
        H, P, N, R, S = self.H, self.P, self.N, self.R, self.S
        z, x, Bp, Cp, ddt, dA, trap, ang = torch.split(self.in_proj(u), [self.d_inner, self.d_inner, N * R, N * R, H, H, H, S], -1)
        A = -heavy_tail(dA).clamp(min=self.A_floor)
        DT = F.softplus(ddt + self.dt_bias); ADT = A * DT
        lam = torch.sigmoid(trap) if self.trapezoid else torch.ones_like(trap)
        lead = u.shape[:-1]
        Bn = self.B_norm(Bp.reshape(*lead, R, N)); Cn = self.C_norm(Cp.reshape(*lead, R, N))
        K = Bn.unsqueeze(-3) + self.B_bias                                # [..., H, R, N]
        Q = Cn.unsqueeze(-3) + self.C_bias
        x = x.reshape(*lead, H, P); z = z.reshape(*lead, H, P)
        if R > 1: V = torch.einsum("...hp,hrp->...hrp", x, self.mimo_x); Z = torch.einsum("...hp,hrp->...hrp", z, self.mimo_z)
        else: V = x.unsqueeze(-2); Z = z.unsqueeze(-2)
        dang = (torch.tanh(ang).unsqueeze(-2) * math.pi * DT.unsqueeze(-1)) if S else None   # [..., H, S]
        return Q, K, V, Z, DT, ADT, lam, dang

    def _out(self, y, V, Z):
        """y [..., H, R, P] -> [..., d_model]: D skip, SiLU gate, MIMO down-projection, out_proj."""
        y = y + self.D[:, None, None] * V
        y = y * F.silu(Z)
        y = torch.einsum("...hrp,hrp->...hp", y, self.mimo_o) if self.R > 1 else y[..., 0, :]
        return self.out_proj(y.reshape(*y.shape[:-2], self.d_inner))

    def forward(self, u, valid=None):
        """parallel (quadratic, single-chunk) form. u [B, T, d]."""
        Q, K, V, Z, DT, ADT, lam, dang = self._inputs(u)
        if valid is not None:                                              # padded steps carry no input and no decay
            m = valid.to(u.dtype)
            V = V * m[..., None, None, None]; DT = DT * m[..., None]; ADT = ADT * m[..., None]
            if dang is not None: dang = dang * m[..., None, None]
        if dang is not None:
            th = torch.cumsum(dang, 1); c, s = torch.cos(th), torch.sin(th)          # [B, T, H, S]
            Qr = _rotate(Q, c.unsqueeze(-2), s.unsqueeze(-2)); Kr = _rotate(K, c.unsqueeze(-2), s.unsqueeze(-2))
        else: Qr, Kr = Q, K
        T = u.shape[1]
        DTs = F.pad(DT[:, 1:], (0, 0, 0, 1)); lams = F.pad(lam[:, 1:], (0, 0, 0, 1))
        sg = DTs * (1 - lams)                                             # coefficient of k_s v_s applied at s+1
        scale = DT * lam + sg                                             # [B, T, H]
        y = (self._ssd_chunked if self.chunk else self._ssd_quadratic)(Qr, Kr, V, ADT, scale)
        qk = torch.einsum("bthrn,bthun->bthru", Q, K)                     # rotation-invariant on the diagonal
        y = y - torch.einsum("bthru,bthup->bthrp", qk * sg[..., None, None], V)
        return self._out(y, V, Z)

    def _ssd_quadratic(self, Qr, Kr, V, ADT, scale):
        """y_t = sum_{s<=t} exp(cum_t - cum_s) scale_s (q_t . k_s) v_s  (one T x T block, as the official fwd ref)."""
        Bsz, T, H, R, N = Qr.shape; P = V.shape[-1]
        cum = torch.cumsum(ADT, 1); seg = cum.unsqueeze(2) - cum.unsqueeze(1)          # [B, t, s, H]
        causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
        Lmask = torch.where(causal[None, :, :, None], torch.exp(seg.clamp(max=0)), torch.zeros(()))
        q2 = Qr.permute(0, 2, 1, 3, 4).reshape(Bsz * H, T * R, N)          # (t, r) flattened -> plain bmm
        k2 = Kr.permute(0, 2, 1, 3, 4).reshape(Bsz * H, T * R, N)
        v2 = V.permute(0, 2, 1, 3, 4).reshape(Bsz * H, T * R, P)
        w = (Lmask * scale.unsqueeze(1)).permute(0, 3, 1, 2).reshape(Bsz * H, T, T)
        if R > 1: w = w.repeat_interleave(R, 1).repeat_interleave(R, 2)
        return torch.bmm(torch.bmm(q2, k2.transpose(1, 2)) * w, v2).view(Bsz, H, T, R, P).permute(0, 2, 1, 3, 4)

    def _ssd_chunked(self, Qr, Kr, V, ADT, scale):
        """same quantity, SSD chunked form (Mamba-2/3 style): quadratic inside chunks of `chunk` steps, linear state
        passing between chunks. ~4x fewer FLOPs than one 64-step block for MIMO R=4."""
        Bsz, T, H, R, N = Qr.shape; P = V.shape[-1]; c = self.chunk; pad = (-T) % c
        if pad:                                                            # right-pad (cannot affect earlier steps)
            Qr, Kr, V = (F.pad(t, (0, 0, 0, 0, 0, 0, 0, pad)) for t in (Qr, Kr, V))
            ADT, scale = (F.pad(t, (0, 0, 0, pad)) for t in (ADT, scale))
        nc = (T + pad) // c
        f = lambda t: t.view(Bsz, nc, c, *t.shape[2:]).transpose(2, 3).contiguous() if t.dim() == 5 else t.view(Bsz, nc, c, H)
        q, k, v = f(Qr), f(Kr), f(V)                                        # [B, nc, H, c, R, *]
        a = ADT.view(Bsz, nc, c, H).permute(0, 1, 3, 2); sc = scale.view(Bsz, nc, c, H).permute(0, 1, 3, 2)   # [B, nc, H, c]
        lc = torch.cumsum(a, -1)                                           # inclusive local cumsum
        causal = torch.tril(torch.ones(c, c, dtype=torch.bool))
        w = torch.where(causal, torch.exp((lc.unsqueeze(-1) - lc.unsqueeze(-2)).clamp(max=0)), torch.zeros(())) * sc.unsqueeze(-2)
        if R > 1: w = w.repeat_interleave(R, -2).repeat_interleave(R, -1)
        q2, k2, v2 = q.flatten(3, 4), k.flatten(3, 4), v.flatten(3, 4)        # [B, nc, H, cR, *]
        y = ((q2 @ k2.transpose(-1, -2)) * w) @ v2                          # intra-chunk
        dec_end = torch.exp(lc[..., -1:] - lc) * sc                          # [B, nc, H, c]
        kd = k * dec_end[..., None, None]
        Sj = torch.einsum("bjhcrn,bjhcrp->bjhpn", kd, v)                     # chunk-end states (from zero)
        ce = torch.cumsum(lc[..., -1], 1)                                  # [B, nc, H] cum at chunk ends
        # state entering chunk j: sum_{i<j} exp(ce_{j-1} - ce_i) S_i
        M = torch.exp((ce.unsqueeze(2) - ce.unsqueeze(1)).clamp(max=0))      # [B, j', i, H] with j' = j-1
        M = torch.where(torch.tril(torch.ones(nc, nc, dtype=torch.bool))[None, :, :, None], M, torch.zeros(()))
        Hin = torch.einsum("bjih,bihpn->bjhpn", M, Sj)                     # state after chunk j' (inclusive)
        Hin = F.pad(Hin, (0, 0, 0, 0, 0, 0, 1, 0))[:, :nc]                  # shift: entering chunk j
        y = y + (torch.einsum("bjhpn,bjhxn->bjhxp", Hin, q2) * torch.exp(lc).repeat_interleave(R, -1)[..., None])
        y = y.view(Bsz, nc, H, c, R, P).transpose(2, 3).reshape(Bsz, nc * c, H, R, P)
        return y[:, :T]

    def init_state(self, b):
        z = torch.zeros
        return dict(ang=z(b, self.H, self.S), h=z(b, self.H, self.P, self.N), k=z(b, self.H, self.R, self.N), v=z(b, self.H, self.R, self.P))

    def step(self, u, st):
        """exact recurrence for one token. u [B, d]."""
        Q, K, V, Z, DT, ADT, lam, dang = self._inputs(u)
        if dang is not None:
            ang = st["ang"] + dang; c, s = torch.cos(ang).unsqueeze(-2), torch.sin(ang).unsqueeze(-2)
            Qr, Kr = _rotate(Q, c, s), _rotate(K, c, s)
        else: ang, Qr, Kr = st["ang"], Q, K
        a = torch.exp(ADT); b = (1 - lam) * DT * a; g = lam * DT
        prev = torch.einsum("bhrn,bhrp->bhpn", st["k"], st["v"]); cur = torch.einsum("bhrn,bhrp->bhpn", Kr, V)
        h = a[..., None, None] * st["h"] + b[..., None, None] * prev + g[..., None, None] * cur
        y = torch.einsum("bhpn,bhrn->bhrp", h, Qr)
        return self._out(y, V, Z), dict(ang=ang, h=h, k=Kr, v=V)

class Mamba3Body(nn.Module):
    """Llama-style stack: [x + Mamba3(RMSNorm(x)); x + SwiGLU(RMSNorm(x))] x layers, final RMSNorm."""
    def __init__(self, d=128, layers=2, mlp_hidden=192, **kw):
        super().__init__(); self.out_dim = d
        self.mix = nn.ModuleList([Mamba3Mixer(d, **kw) for _ in range(layers)])
        self.mlp = nn.ModuleList([SwiGLU(d, mlp_hidden) for _ in range(layers)])
        self.n1 = nn.ModuleList([RMSNorm(d) for _ in range(layers)]); self.n2 = nn.ModuleList([RMSNorm(d) for _ in range(layers)])
        self.nf = RMSNorm(d)
    def forward(self, h, valid=None):
        for mx, ml, a, b in zip(self.mix, self.mlp, self.n1, self.n2):
            h2 = h + mx(a(h), valid); h2 = h2 + ml(b(h2))
            if valid is None: h = h2
            else:
                m = valid.unsqueeze(-1).to(h.dtype); h = h2 * m + h * (1 - m)   # pad steps keep their hidden state
        return self.nf(h)
    def step(self, x, state=None, valid=None):
        """One token. `valid` [B] false: that row's recurrent state is left unchanged (pad)."""
        if state is None: state = [m.init_state(x.shape[0]) for m in self.mix]
        v = None
        if valid is not None:
            v = torch.as_tensor(valid, dtype=torch.bool, device=x.device).view(-1)
            if not bool(v.any()):
                return x.new_zeros(x.shape[0], self.out_dim), state
        new = []
        for mx, ml, a, b, st in zip(self.mix, self.mlp, self.n1, self.n2, state):
            y, st2 = mx.step(a(x), st); x = x + y; x = x + ml(b(x)); new.append(st2)
        y = self.nf(x)
        if v is not None and not bool(v.all()):
            # rows are independent; drop the update on pad rows so they do not enter later steps
            mk = v.to(dtype=y.dtype).view(-1, 1)
            y = y * mk
            for st, st2 in zip(state, new):
                for k in st2:
                    m = v.to(dtype=st2[k].dtype).view(-1, *([1] * (st2[k].ndim - 1)))
                    st2[k] = st2[k] * m + st[k] * (1 - m)
        return y, new

class _Attn(nn.Module):
    def __init__(self, d, nhead):
        super().__init__(); self.h = nhead; self.dh = d // nhead; self.qkv = nn.Linear(d, 3 * d); self.o = nn.Linear(d, d)
        inv = 1.0 / (10000 ** (torch.arange(0, self.dh, 2).float() / self.dh)); self.register_buffer("inv", inv, persistent=False)
    def rope(self, x, pos):                       # x [B, h, T, dh], pos [T] (absolute index)
        f = pos[:, None].float() * self.inv[None]; c, s = torch.cos(f), torch.sin(f)
        x0, x1 = x[..., 0::2], x[..., 1::2]
        return torch.stack([x0 * c - x1 * s, x0 * s + x1 * c], -1).flatten(-2)
    def split(self, x):
        q, k, v = self.qkv(x).chunk(3, -1); f = lambda t: t.view(*t.shape[:-1], self.h, self.dh).transpose(-3, -2)
        return f(q), f(k), f(v)

class CausalTransformer(nn.Module):
    def __init__(self, d=128, layers=2, nhead=4, ff=512, window=64):
        super().__init__(); self.out_dim = d; self.window = window
        self.attn = nn.ModuleList([_Attn(d, nhead) for _ in range(layers)])
        self.ff = nn.ModuleList([nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d)) for _ in range(layers)])
        self.n1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)]); self.n2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(layers)])
        self.nf = nn.LayerNorm(d)
    def forward(self, h, valid=None):
        B, T, _ = h.shape; pos = torch.arange(T)
        allow = torch.tril(torch.ones(T, T, dtype=torch.bool))[None]
        if valid is not None: allow = allow & (valid[:, None, :] | torch.eye(T, dtype=torch.bool)[None])
        bias = torch.zeros(allow.shape).masked_fill(~allow, float("-inf"))[:, None]
        for at, ff, a, b in zip(self.attn, self.ff, self.n1, self.n2):
            q, k, v = at.split(a(h)); q, k = at.rope(q, pos), at.rope(k, pos)
            w = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(at.dh) + bias, -1)
            h = h + at.o((w @ v).transpose(1, 2).reshape(B, T, -1)); h = h + ff(b(h))
        return self.nf(h)
    def step(self, x, state=None, valid=None):
        """KV cache of the last `window` real events (post-RoPE keys, absolute positions).
        A step with `valid` all false does not write the cache and does not advance the position,
        so later queries cannot attend to a pad token."""
        if state is None: state = dict(t=0, kv=[None] * len(self.attn))
        if "rows" in state:
            v = torch.ones(x.shape[0], dtype=torch.bool, device=x.device) if valid is None else torch.as_tensor(valid, dtype=torch.bool, device=x.device).view(-1)
            return self._step_rows(x, state["rows"], v)
        if valid is not None:
            v = torch.as_tensor(valid, dtype=torch.bool, device=x.device).view(-1)
            if not bool(v.any()):
                return x.new_zeros(x.shape[0], self.out_dim), state
            if not bool(v.all()):
                return self._step_mixed(x, state, v)
        t = state["t"]; pos = torch.tensor([t], device=x.device); h = x.unsqueeze(1); new = []
        for at, ff, a, b, kv in zip(self.attn, self.ff, self.n1, self.n2, state["kv"]):
            q, k, v = at.split(a(h)); q, k = at.rope(q, pos), at.rope(k, pos)
            if kv is not None: k = torch.cat([kv[0], k], 2)[:, :, -self.window:]; v = torch.cat([kv[1], v], 2)[:, :, -self.window:]
            w = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(at.dh), -1)
            h = h + at.o((w @ v).transpose(1, 2).reshape(x.shape[0], 1, -1)); h = h + ff(b(h)); new.append((k, v))
        return self.nf(h)[:, 0], dict(t=t + 1, kv=new)
    def _step_mixed(self, x, state, v):
        rows = [_tx_slice(state, b) for b in range(x.shape[0])]
        return self._step_rows(x, rows, v)
    def _step_rows(self, x, rows, v):
        ys, pieces = [], []
        for b, st_b in enumerate(rows):
            if bool(v[b]):
                yb, st_b = self.step(x[b:b + 1], st_b)
            else:
                yb = x.new_zeros(1, self.out_dim)
            ys.append(yb); pieces.append(st_b)
        return torch.cat(ys, 0), _tx_merge(pieces)

def _tx_slice(state, b):
    if "rows" in state: return state["rows"][b]
    kv = []
    for item in state["kv"]:
        kv.append(None if item is None else (item[0][b:b + 1], item[1][b:b + 1]))
    t = state["t"]
    if torch.is_tensor(t): t = int(t.view(-1)[b])
    return dict(t=int(t), kv=kv)

def _tx_merge(pieces):
    """Stack per-row caches when they have the same length; otherwise keep them split so pad rows add no keys."""
    def length(p):
        item = p["kv"][0]
        return 0 if item is None else item[0].shape[2]
    lens = [length(p) for p in pieces]
    if len(set(lens)) == 1 and len(set(int(p["t"]) for p in pieces)) == 1:
        if lens[0] == 0:
            return dict(t=int(pieces[0]["t"]), kv=[None] * len(pieces[0]["kv"]))
        kv = []
        for i in range(len(pieces[0]["kv"])):
            kv.append((torch.cat([p["kv"][i][0] for p in pieces], 0), torch.cat([p["kv"][i][1] for p in pieces], 0)))
        return dict(t=int(pieces[0]["t"]), kv=kv)
    return dict(rows=pieces)

def make_body(kind, d=128):
    if kind == "tx_kv": return CausalTransformer(d, layers=2, nhead=4, ff=512)
    if kind == "mamba3": return Mamba3Body(d, layers=2, mlp_hidden=160, d_state=32, expand=2, headdim=32, mimo_rank=4, chunk=8)
    if kind == "mamba3_siso": return Mamba3Body(d, layers=2, mlp_hidden=240, d_state=32, expand=2, headdim=32, mimo_rank=1)
    if kind == "m3_ablate_m2": return Mamba3Body(d, layers=2, mlp_hidden=240, d_state=32, expand=2, headdim=32, mimo_rank=1, trapezoid=False, rope=False)
    raise ValueError(kind)
