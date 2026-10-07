"""Video front end -- PLANNED. Uniformly sample 4-8 frames, run the image front end per frame, causal temporal
pooling (mean + last + motion delta). Optional audio track goes through the audio front end."""
from tidal.modalities import register

@register
class VideoFrontEnd:
    name = "video"; dim = 128
    def available(self): return False
    def encode(self, clips):
        raise NotImplementedError("video front end is planned; see docs/architecture.md")
