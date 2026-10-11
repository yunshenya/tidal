"""Dependency-free pooled audio front end.

The learned streaming speech model lives in ``tidal.audio_fe``. This interface accepts mono waveforms
(or ``(waveform, sample_rate)`` pairs) and returns a fixed 128-dim log-spectrum embedding, allowing
multimodal callers to run without a checkpoint.
"""
from __future__ import annotations
import numpy as np
from tidal.modalities import register

N_MELS = 64; SAMPLE_RATE = 16000; HOP_MS = 10

@register
class AudioFrontEnd:
    name = "audio"; dim = 128
    def available(self): return False
    def encode(self, waveforms):
        out = []
        for item in waveforms:
            x = item[0] if isinstance(item, tuple) and len(item) == 2 else item
            x = np.asarray(x, np.float32).reshape(-1)
            if x.size == 0: out.append(np.zeros(self.dim, np.float32)); continue
            if not np.isfinite(x).all(): raise ValueError("waveform contains non-finite values")
            x = np.clip(x, -1, 1); n = max(32, min(512, x.size)); win = np.hanning(n)
            frames = np.array([x[i:i+n] * win for i in range(0, max(1, x.size-n+1), max(1, n//2)) if x[i:i+n].size == n])
            if frames.size == 0: frames = (np.pad(x, (0, n-x.size)) * win)[None, :]
            spec = np.log1p(np.abs(np.fft.rfft(frames, axis=1)) ** 2).mean(0)
            bins = np.linspace(0, spec.size, N_MELS + 1).astype(int)
            mel = np.array([spec[bins[i]:max(bins[i+1], bins[i]+1)].mean() for i in range(N_MELS)], np.float32)
            stats = np.array([x.mean(), x.std(), np.abs(x).mean(), np.sqrt(np.mean(x*x)), np.max(np.abs(x))], np.float32)
            v = np.concatenate([mel, np.log1p(mel), stats, np.array([len(x)/SAMPLE_RATE], np.float32)])
            v = np.pad(v, (0, max(0, self.dim-v.size)))[:self.dim]
            out.append(v.astype(np.float32))
        return np.stack(out) if out else np.empty((0, self.dim), np.float32)
