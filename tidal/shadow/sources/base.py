"""Source interface + the unified event schema every source must produce."""
from typing import Protocol
import pandas as pd

EVENT_COLUMNS = [
    "msg_id", "conv", "conv_type", "ts", "role", "speaker", "speaker_known", "text",
    # label sources (never model inputs)
    "attention_reason", "chosen_action", "cur_addressed", "cur_reply",
    # optional pre-decision / non-text metadata
    "msg_chars", "has_question", "mentioned_bot", "reply_to_bot", "burst_len", "group_activity_1m",
    "media_refs_extra",
]

class Source(Protocol):
    name: str
    def fetch(self, store, now: float) -> pd.DataFrame:
        """Pull new data incrementally (read-only upstream) and return events that are new OR changed
        (e.g. text recovered later) as a frame with EVENT_COLUMNS. ids pseudonymous, text scrubbed."""
        ...
    def default_predict_after(self, now: float) -> float:
        """Only events after this time get predictions (earlier ones are context only)."""
        ...

def empty_events():
    return pd.DataFrame({c: pd.Series(dtype=object) for c in EVENT_COLUMNS})
