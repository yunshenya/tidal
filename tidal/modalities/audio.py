"""Audio front end (speech + music) -- PLANNED. 16 kHz log-mel (64 bins, 10 ms hop) -> small causal conv stack ->
streaming GRU/SSM block; outputs a pooled embedding plus frame-level activity, and (optionally) features from an
offline teacher (ASR transcript, emotion/event tags, acoustic end-of-turn probability) distilled into the student."""
from tidal.modalities import register

N_MELS = 64; SAMPLE_RATE = 16000; HOP_MS = 10

@register
class AudioFrontEnd:
    name = "audio"; dim = 128
    def available(self): return False
    def encode(self, waveforms):
        raise NotImplementedError("audio front end is planned; see docs/architecture.md")
