"""Per-session dialogue state for flat-row ``ask``. CPU only. No model.

The fact store already lists who has a slot. An edit of one character is the
same string decision as an absent near-duplicate, so a stored name alone
cannot license it. This object records writes, utterances, and earlier asks
with recency. ``ask`` reads the conversational slice: utterances and earlier
asks inside ``TURN_WINDOW``. A write is on the log and is not that slice.

An empty conversational slice leaves name resolution unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# Conversational turns that still count as recent. A mention older than this
# is not active. Four covers a short exchange. Fact writes do not use a turn.
# Not chosen on the frozen adversarial file.
TURN_WINDOW = 4
# Oldest events drop first. The ask gate only reads the window above.
MENTION_CAP = 32


@dataclass(frozen=True)
class Mention:
    name: str
    turn: int
    source: str  # "write" | "utterance" | "ask"
    seq: int
    surface: str


class DialogueState:
    """Ordered mentions for one session, plus an optional speaker and hearer.

    ``conversational`` is what ``ask`` consults: utterance and ask mentions
    inside ``TURN_WINDOW``, most recent first, one entry per person. Pass
    ``ignore_surface`` as the question being answered so an ask recorded from
    that same string is not prior context. Re-reading one question does not
    license a typo of itself, and it does not open another turn.
    """

    def __init__(self) -> None:
        self.turn = 0
        self.speaker: Optional[str] = None
        self.hearer: Optional[str] = None
        self._seq = 0
        self._open_ask = ""
        self._mentions: List[Mention] = []

    def mentions(self) -> Tuple[Mention, ...]:
        return tuple(self._mentions)

    def note_write(self, name: Optional[str]) -> None:
        """A fact write. Logged, not a conversational mention."""
        if isinstance(name, str) and name:
            self._push(name, "write", "")

    def note_utterance(
        self,
        names: Sequence[str],
        *,
        speaker: Optional[str] = None,
        hearer: Optional[str] = None,
        surface: str = "",
    ) -> None:
        """One conversational utterance. Opens a turn even if nobody is named."""
        self.turn += 1
        self._open_ask = ""
        if isinstance(speaker, str) and speaker:
            self.speaker = speaker
        if isinstance(hearer, str) and hearer:
            self.hearer = hearer
        for name in names:
            self._push(name, "utterance", surface)

    def note_ask(self, names: Sequence[str], surface: str) -> None:
        """People the ask resolved. A repeat of the same string stays one turn."""
        text = surface if isinstance(surface, str) else ""
        if text and self._open_ask == text:
            for name in names:
                self._push(name, "ask", text)
            return
        self.turn += 1
        self._open_ask = text
        for name in names:
            self._push(name, "ask", text)

    def conversational(self, *, ignore_surface: str = "") -> Tuple[str, ...]:
        """Active people, most recent first.

        ``ignore_surface`` drops ask-mentions recorded from that exact
        question. Utterance mentions of the same text still count.
        """
        if self.turn <= 0:
            return ()
        cutoff = self.turn - TURN_WINDOW
        best = {}
        for item in self._mentions:
            if item.source not in ("utterance", "ask"):
                continue
            if item.turn <= cutoff or item.turn > self.turn:
                continue
            if ignore_surface and item.source == "ask" and item.surface == ignore_surface:
                continue
            prev = best.get(item.name)
            if prev is None or item.seq > prev:
                best[item.name] = item.seq
        ordered = sorted(best.items(), key=lambda pair: -pair[1])
        return tuple(name for name, _seq in ordered)

    def recent(self) -> Tuple[str, ...]:
        """Every source, most recent first, including writes. Not the ask gate."""
        best = {}
        for item in self._mentions:
            prev = best.get(item.name)
            if prev is None or item.seq > prev:
                best[item.name] = item.seq
        ordered = sorted(best.items(), key=lambda pair: -pair[1])
        return tuple(name for name, _seq in ordered)

    def _push(self, name: str, source: str, surface: str) -> None:
        if not isinstance(name, str) or name == "":
            return
        self._seq += 1
        self._mentions.append(Mention(name, self.turn, source, self._seq, surface))
        if len(self._mentions) > MENTION_CAP:
            self._mentions = self._mentions[-MENTION_CAP:]
