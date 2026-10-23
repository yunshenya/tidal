"""Per-session dialogue state for flat-row ``ask``. CPU only. No model.

The fact store already lists who has a slot. An edit of one character is the
same string decision as an absent near-duplicate, so a stored name alone
cannot license it. This object records writes, utterances, and earlier asks
with recency. ``ask`` reads the conversational slice: utterances and earlier
asks inside ``TURN_WINDOW``. A write is on the log and is not that slice.

An empty conversational slice leaves name resolution unchanged. Speaker is
set only when ``note_turn`` is given one. Hearer is that argument when it
is a non-empty string, including a string that is not on the roster; when
the argument is empty, a leading exact roster name plus address punctuation
is the hearer. A third-person pronoun reads ``eligible_latest``: the people
on the newest conversational turn in the window who are neither of those
two. ``refresh`` restamps one in-window mention onto the current turn. It
does not append a name, and it does not pull a mention back into the window.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

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
        self.utterances = 0
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
        self.utterances += 1
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

    def eligible_latest(self, *, ignore_surface: str = "") -> Tuple[str, ...]:
        """People on the newest conversational turn who are not speaker or hearer.

        Same window and same ``ignore_surface`` rule as ``conversational``.
        A write is not eligible. Two names on that newest turn are both
        returned, in mention order, and the caller abstains. One name is the
        most recent third person. An older name still inside the window does
        not join this tuple.
        """
        if self.turn <= 0:
            return ()
        cutoff = self.turn - TURN_WINDOW
        blocked = set()
        if isinstance(self.speaker, str) and self.speaker:
            blocked.add(self.speaker)
        if isinstance(self.hearer, str) and self.hearer:
            blocked.add(self.hearer)
        best: Dict[str, Tuple[int, int]] = {}
        for item in self._mentions:
            if item.source not in ("utterance", "ask"):
                continue
            if item.turn <= cutoff or item.turn > self.turn:
                continue
            if ignore_surface and item.source == "ask" and item.surface == ignore_surface:
                continue
            if item.name in blocked:
                continue
            prev = best.get(item.name)
            if prev is None or (item.turn, item.seq) > prev:
                best[item.name] = (item.turn, item.seq)
        if not best:
            return ()
        top = max(turn for turn, _seq in best.values())
        tied = [(seq, name) for name, (turn, seq) in best.items() if turn == top]
        tied.sort()
        return tuple(name for _seq, name in tied)

    def refresh(self, name: str) -> bool:
        """Restamp one in-window conversational mention of ``name`` onto this turn.

        The latest utterance or ask mention still inside ``TURN_WINDOW`` moves
        to ``self.turn``. Source and surface stay, and nothing is appended.
        A mention already on this turn is left as it is. A mention outside
        the window, a write, or a name that was never said returns False and
        is not pulled back.
        """
        if not isinstance(name, str) or name == "" or self.turn <= 0:
            return False
        cutoff = self.turn - TURN_WINDOW
        best_i = -1
        best_key = (-1, -1)
        for i, item in enumerate(self._mentions):
            if item.name != name or item.source not in ("utterance", "ask"):
                continue
            if item.turn <= cutoff or item.turn > self.turn:
                continue
            key = (item.turn, item.seq)
            if key > best_key:
                best_key = key
                best_i = i
        if best_i < 0:
            return False
        item = self._mentions[best_i]
        if item.turn == self.turn:
            return True
        self._seq += 1
        self._mentions[best_i] = Mention(
            item.name, self.turn, item.source, self._seq, item.surface,
        )
        return True

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
