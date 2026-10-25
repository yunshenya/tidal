"""McGaugh emotional-arousal consolidation gain.

McGaugh (2004) reviews evidence that the amygdala modulates the
consolidation of memories of emotionally arousing experiences. The
degree of amygdala activation by emotional arousal during encoding of
pleasant or unpleasant material correlates with later recall.
https://doi.org/10.1146/annurev.neuro.27.070203.144157

tidal already owns an LC-NE arousal scalar and a signed valence. This
module turns those existing numbers into a write-strength and sleep-replay
gain. It does not choose to speak or to wait. Those decisions belong to
the brain layer. It does not add a cue-export field.

Baseline arousal, or zero valence, leaves the gain at exactly 1. A missing
growbrain arousal (the live 0.2 stub) therefore does not invent emotional
consolidation. The span is a hypothesis. No grid was run.
"""
from __future__ import annotations

import math

from tidal.arousal import AROUSAL_BASELINE

# Write / replay span. Phase-0 clock uses the same 0.25. HYPOTHESIS.
CONSOLIDATION_SPAN = 0.25


def _arousal(level: float) -> float:
    if isinstance(level, bool) or not isinstance(level, (int, float)):
        raise TypeError("arousal must be a real number")
    value = float(level)
    if not math.isfinite(value) or value < 0.0 or value > 1.0:
        raise ValueError("arousal must be in [0, 1]")
    return value


def _valence(value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError("valence must be a real number")
    feeling = float(value)
    if not math.isfinite(feeling):
        raise ValueError("valence must be finite")
    return feeling


def consolidation_gain(arousal: float, valence: float) -> float:
    """``1 + SPAN * excess * tanh(|valence|)``.

    ``excess = max(0, (arousal - baseline) / (1 - baseline))``.
    Baseline arousal or zero valence is exactly 1.
    """
    level = _arousal(arousal)
    feeling = _valence(valence)
    if level <= float(AROUSAL_BASELINE) or feeling == 0.0:
        return 1.0
    excess = (level - float(AROUSAL_BASELINE)) / (1.0 - float(AROUSAL_BASELINE))
    return float(1.0 + float(CONSOLIDATION_SPAN) * excess * math.tanh(abs(feeling)))


def replay_priority(arousal: float, valence: float) -> float:
    """Same number as :func:`consolidation_gain`. Sleep ranking reads it."""
    return consolidation_gain(arousal, valence)
