"""Text front end (implemented): frozen bge-small-zh-v1.5, int8 ONNX, CLS pooling, 512-d, CPU.
Thin wrapper over tidal.embed so all front ends share one interface."""
from tidal.modalities import register
import os

@register
class TextFrontEnd:
    name = "text"; dim = 512
    def __init__(self, threads=1, cache=None):
        self.threads = threads; self._enc = None
    def available(self):
        from tidal.embed import MODEL_DIR
        return os.path.exists(os.path.join(MODEL_DIR, "bge_q.onnx"))
    def encode(self, texts):
        if self._enc is None:
            from tidal.embed import Encoder
            self._enc = Encoder(self.threads)
        return self._enc.encode(list(texts), bs=32)
