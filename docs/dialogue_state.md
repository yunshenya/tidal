# Dialogue state

Flat-row `ask` used to decide who a question names from that string and the fact roster alone. An edit of one character is the same string decision as an absent near-duplicate: the abstain dev bank and the frozen file (`cce20b6b…`) both label surfaces such as 黄蕾 / 周滔 / 陈靖 either way. Closed PRs #24 and #32 measured that. No string rule can accept the typo and still abstain the near-duplicate.

The fix is a small per-session record of who has been in the conversation. It is always on. There is no flag, no learned dialogue model, and no transformer on this path. `LLMMemory` is untouched.

## What is stored

One `DialogueState` per group (`tidal/dialogue_state.py`). `Continuum` keeps them. `new_session(group)` drops that group's state. `episodic.clear()` drops every group's state. Facts are not cleared by `new_session`.

Each mention is `(name, turn, source, seq, surface)`:

| source | who writes it | opens a turn | licenses an edit-1 |
|---|---|---|---|
| `write` | `remember`, when the write is kept | no | no |
| `utterance` | `note_turn` | yes | yes, inside the window |
| `ask` | `query`, after the decision | yes, unless this exact string is already the open ask | yes, inside the window, except when the question being answered is that same string |

The log is capped at 32 events. Speaker and hearer are two optional strings set only when `note_turn` is given them. The edit rule does not read them. Deixis does. `utterances` counts `note_turn` calls. A fact write and an ask do not increment it.

`conversational()` is the active list: utterance and ask mentions whose turn is inside the last `TURN_WINDOW` (4) turns, most recent first, one entry per person. `recent()` is the same log including writes, for inspection. The window is a constant, not a fit, and not chosen on the frozen file.

A question does not count as its own prior mention. `query` passes `ignore_surface` equal to the question, so ask-mentions recorded from that exact string are skipped. Asking one string twice does not change who is active, which is why a probe case that calls `ask` and then `query` stays on the empty slice. A later, different utterance or question still sees the people the first reading resolved.

## Update rules

1. `remember` appends a `write` for the slot name when the write is kept. Chat and noise writes are still dropped by the store and are not logged. A write does not open a turn.
2. `note_turn(text, speaker=, hearer=)` resolves stored people in `text` with the same `ask_people` the next question will use, including an edit the current active list already licenses, then appends those people as an `utterance`. Speaker and hearer are stored when passed. They are not inferred.
3. `query` decides with the active list as it stands, then appends the people `ask_people` returned for that question as an `ask`. An abstention that matched nobody appends nobody, and the turn still opens. A second `query` of the same string does not open another turn.

`recall`, `heads`, and `replay` do not read or write this state, except that `replay` calls `new_session`, which drops the group's mentions.

## How `ask` uses it

Name resolution is unchanged when the active list is empty. Lexical hits, the name head (nickname, pinyin, dropped surname, short honorific), the alias list, CJK verb attachment, and the longer-name rejection all stay as they are.

The one addition is in `diagnose`. For a candidate span the head refuses, if the span is an edit of distance 1 on a same-script similar-length pair (`_ambiguous_edit`) and it is not an extension (the stored name is not still inside a longer span), and exactly one active stored person is that edit, the span is that person. Zero active names, or two, and the span stays unmatched. The question then abstains `absent_person` as it does today.

Extensions stay a different person even when the shorter name is active: `王伟强`, `老王伟`, `Anne` against `Ann`, `Swithuno` against `Swithun`. That is the CJK superstring rule and the Latin longer-name rule, not the ambiguous edit.

A fact write does not put the person on the active list. The frozen near-duplicate rows write the near person and then ask the edit. If the write licensed the edit, those rows would answer. They do not. A caller who wants the written sentence to count as dialogue passes that sentence to `note_turn`.

`ABSTAIN_MARGIN` stays 0.1. `LEX_RIDGE_GAP` stays 0.15. The name-head weights are unchanged.

## Deixis

`resolve_deixis` in `question_surface.py` runs inside `matched_fact_rows` and `selected_fact_rows`. `ask_people` does not call it, so `note_turn` and the ask log still record names only. A pronoun does not become a later mention. There is no gender, no full coreference, and no flag.

The rule applies only when `utterances > 0` or a hearer string is already stored. No utterance and no hearer is the empty log: `you` / `他` stay on the ranker, and a hearer-role noun stays on the unique first-person fact. A stored name in the question, including an edit the active list already licenses, keeps the name path. `Can you read Ada's phone` stays Ada.

Otherwise:

1. `you` / `your` / `你` / `您` (not `你们`), and a hearer-role cue (`listener`, `跟前这位`, the same class as `is_indirect_hearer`), are the hearer when that string is one roster name. The cue then scopes the ranker to that person's facts. Unset after an utterance, or a hearer who is not on the roster, abstains `absent_person`. A hearer-role cue with no stored hearer does not abstain here: the unique first-person path still runs.
2. A second-person cue and a third-person pronoun in the same question, with no stored name, abstain.
3. `he` / `she` / `they` / `他` / `她` / `ta` (`them` is not in this class) are `eligible_latest`: utterance and ask mentions inside `TURN_WINDOW`, excluding the speaker and the hearer, restricted to the newest turn those mentions occupy. One name is that person. Two names on that turn abstain. Zero names abstain: nobody was said, the only record is a write, the mention has aged out of the window, or the only people said are the speaker and the hearer. An older eligible name still inside the window does not block a later one.

`ABSTAIN_MARGIN` stays 0.1. `LEX_RIDGE_GAP` stays 0.15. The dev bank and the floors are in `reports/deixis.md`.

## What this does not do

The rule does not invent a hearer the caller did not pass, and it does not read a fact write as a mention. It does not track turn-taking beyond the speaker and hearer strings, and it does not chain a pronoun to a previous pronoun. With four people in the window the list is already those four, in recency order; the third-person read keeps only the newest turn's eligible names.

The edit-1 dev bank, the with-log versus empty-log ask, and the single-turn floors for that rule are in `reports/dialogue_state.md`. A natural multi-turn measurement, with the rule left as it is, is in `reports/dialogue_real.md`.
