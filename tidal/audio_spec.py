"""Dependency-light audio geometry and NumPy feature extraction for ONNX inference."""
import numpy as np

SR = 16000; HOP = 160; WIN = 400; NFFT = 512; NMEL = 40; STACK = 2
FRAME_READY = ((STACK - 1) * HOP + WIN) / SR  # stacked frame k available at k*STEP + 35ms
BINS = [(0.0, 0.2), (0.2, 0.6), (0.6, 1.2), (1.2, 2.0)]; STEP = HOP * STACK / SR   # 0.02 s

def mel_fb(n_mels=NMEL, n_fft=NFFT, sr=SR, fmin=50.0, fmax=7600.0):
    hz2m = lambda f: 2595 * np.log10(1 + f / 700.0); m2hz = lambda m: 700 * (10 ** (m / 2595.0) - 1)
    pts = m2hz(np.linspace(hz2m(fmin), hz2m(fmax), n_mels + 2)); bins = np.floor((n_fft + 1) * pts / sr).astype(int)
    fb = np.zeros((n_mels, n_fft // 2 + 1), np.float32)
    for i in range(1, n_mels + 1):
        a, b, c = bins[i - 1], bins[i], bins[i + 1]
        for k in range(a, b): fb[i - 1, k] = (k - a) / max(1, b - a)
        for k in range(b, c): fb[i - 1, k] = (c - k) / max(1, c - b)
    return fb


def stack_frames(m_other, m_self):
    """two [F, NMEL] -> [F//STACK, 2*NMEL*STACK] (20 ms steps)."""
    n = min(len(m_other), len(m_self)) // STACK * STACK
    x = np.concatenate([m_other[:n], m_self[:n]], 1)
    return x.reshape(n // STACK, -1)


_FB = mel_fb()
_WINDOW = (.5 * (1. - np.cos(2. * np.pi * np.arange(WIN) / WIN))).astype(np.float32)

def numpy_logmel(pcm):
    """Causal PCM features; same periodic Hann/mel geometry as training."""
    pcm = np.asarray(pcm, np.float32)
    if pcm.ndim != 1 or not np.isfinite(pcm).all():
        raise ValueError("Expected finite mono PCM")
    if len(pcm) < WIN:
        return np.zeros((0, NMEL), np.float32)
    frames = np.lib.stride_tricks.sliding_window_view(pcm, WIN)[::HOP] * _WINDOW
    spec = (np.abs(np.fft.rfft(frames, n=NFFT)) ** 2).astype(np.float32)
    return np.log(spec @ _FB.T + 1e-6).astype(np.float32)
