"""Phase 3: tiny streaming audio front end for turn-taking (CPU, causal).

Input: two audio channels at 16 kHz: "other" (the human / room mic) and "self" (her own voice, or the second speaker
when training on two-speaker recordings). Features: 40-bin log-mel, 25 ms window, 10 ms hop (computed with torch.stft;
no extra audio dependencies). Encoder: 2 stacked frames (20 ms step) -> Linear -> causal depthwise conv -> GRU(96).
Outputs per 20 ms step (both from the point of view of "self"):
  va_now[2]       current voice activity of self / other
  vap[2 x 4]      VAP-style future voice activity of self / other in bins 0-0.2, 0.2-0.6, 0.6-1.2, 1.2-2.0 s
  bc_self         self produces a backchannel onset within the next 0.5 s
Derived:   p_shift = how much "other" (vs self) is predicted to talk in 0.2-2 s  -> end-of-turn / turn-shift score.
Streaming: AudioStream.push(other_pcm, self_pcm) for each 100 ms tick -> AudioFrame-compatible dict. State is O(1)."""
import math, numpy as np, torch, torch.nn as nn

from tidal.audio_spec import (SR, HOP, WIN, NFFT, NMEL, STACK, STEP, FRAME_READY, BINS,
                              mel_fb as _numpy_mel_fb, stack_frames)

def mel_fb(n_mels=NMEL, n_fft=NFFT, sr=SR, fmin=50.0, fmax=7600.0):
    return torch.from_numpy(_numpy_mel_fb(n_mels, n_fft, sr, fmin, fmax))

_FB = mel_fb(); _WINDOW = torch.hann_window(WIN)

def logmel(x: np.ndarray) -> np.ndarray:
    """x: float32 mono [-1,1] -> [frames, NMEL] log-mel (frame i covers samples [i*HOP, i*HOP+WIN))."""
    t = torch.from_numpy(np.asarray(x, np.float32))
    if len(t) < WIN: return np.zeros((0, NMEL), np.float32)
    fr = t.unfold(0, WIN, HOP) * _WINDOW
    spec = torch.fft.rfft(fr, n=NFFT).abs() ** 2
    return torch.log(spec @ _FB.T + 1e-6).numpy().astype(np.float32)

class AudioEncoder(nn.Module):
    def __init__(self, d_in=2 * NMEL * STACK, d=96):
        super().__init__()
        self.inp = nn.Sequential(nn.Linear(d_in, d), nn.GELU())
        self.conv = nn.Conv1d(d, d, 3, groups=d)                         # causal: left pad 2
        self.gru = nn.GRU(d, d, batch_first=True)
        self.va = nn.Linear(d, 2); self.vap = nn.Linear(d, 2 * len(BINS)); self.bc = nn.Linear(d, 1)
    def forward(self, x, state=None, conv_cache=None):
        """x: [B, T, d_in] normalized features. Returns dict of logits and (gru_state, conv_cache)."""
        h = self.inp(x).transpose(1, 2)
        pad = conv_cache if conv_cache is not None else torch.zeros(h.shape[0], h.shape[1], 2)
        hc = torch.cat([pad, h], 2); new_cache = hc[:, :, -2:]
        h = torch.relu(self.conv(hc)).transpose(1, 2)
        h, st = self.gru(h, state)
        return dict(va=self.va(h), vap=self.vap(h), bc=self.bc(h)[..., 0], h=h), (st, new_cache)

def p_shift(vap_logits):
    p = torch.sigmoid(vap_logits) if torch.is_tensor(vap_logits) else 1 / (1 + np.exp(-vap_logits))
    s, o = p[..., 1:4], p[..., 5:8]                      # vap layout: [self bins 0..3, other bins 0..3]
    return (s.mean(-1) - o.mean(-1) + 1) / 2              # P(self takes the floor in 0.2-2 s) vs other keeps / resumes


class AudioStream:
    """Streaming wrapper for the duplex tick loop: push 100 ms of PCM for each channel per tick."""
    def __init__(self, model: AudioEncoder, mu, sd):
        self.m = model.eval(); self.mu = mu; self.sd = sd; self.reset()
    def reset(self):
        self.buf = {"o": np.zeros(0, np.float32), "s": np.zeros(0, np.float32)}; self.state = None; self.cache = None; self.last = None
    @torch.no_grad()
    def push(self, other_pcm, self_pcm):
        for k, x in (("o", other_pcm), ("s", self_pcm)): self.buf[k] = np.concatenate([self.buf[k], np.asarray(x, np.float32)])
        n = min(len(self.buf["o"]), len(self.buf["s"]))
        nfr = (n - WIN) // HOP + 1 if n >= WIN else 0
        nfr = nfr // STACK * STACK
        if nfr <= 0: return self.last
        mo = logmel(self.buf["o"][: (nfr - 1) * HOP + WIN]); ms = logmel(self.buf["s"][: (nfr - 1) * HOP + WIN])
        for k in self.buf: self.buf[k] = self.buf[k][nfr * HOP:]
        x = (stack_frames(mo, ms) - self.mu) / self.sd
        o, (self.state, self.cache) = self.m(torch.from_numpy(x[None].astype(np.float32)), self.state, self.cache)
        va = torch.sigmoid(o["va"][0, -1]).numpy(); vp = o["vap"][0, -1]
        self.last = dict(vad_self=float(va[0]), vad_other=float(va[1]), shift=float(p_shift(vp)), bc=float(torch.sigmoid(o["bc"][0, -1])),
                         energy=float(np.clip((mo[-STACK:].mean() + 10) / 10, 0, 1)))
        return self.last
