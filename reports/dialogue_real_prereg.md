# Real multi-turn dialogue — pre-registration

Written before the ask score. The bank and the runner are `tidal/question_dialogue_real_data.py` and `tidal/question_dialogue_real_eval.py`. Numbers land in `reports/dialogue_real.md` after one baseline run on this commit's ask path. No threshold, window, or name-head weight is fit here.

## What is being measured

Natural multi-turn episodes on the shipped ask path: `note_turn` (speaker and hearer only when the script passes them), `DialogueState` with `TURN_WINDOW` 4, edit-1 licensing, and `resolve_deixis`. The instrument is the same tied relation encoder the deixis dev uses, so a hit is the string decision (who, or which abstain), not an e5 retrieval score. Same-relation slots tie. The decoy is stored first, so an unscoped deictic ask returns that decoy.

`ABSTAIN_MARGIN` stays 0.1. `LEX_RIDGE_GAP` stays 0.15. There is no enable flag. `LLMMemory` is untouched. The frozen file `cce20b6b…` is not an input and its labels are not edited.

## Labels

**Rule.** Replayed from `DialogueState`, `ask_people`, `matched_fact_rows`, and `selected_fact_rows` before any `Continuum.query`. A fact write is not a mention. A pronoun ask does not become a later mention. Empty-log rule labels use a fresh state and the same functions: that is current main behavior (a deictic pronoun with no utterance and no hearer stays on the ranker; `ta` abstains; a stored name is that person).

**Human.** The person a listener would name, or an abstain when the line is actually ambiguous. Written with the episode, not from the ask score.

**Scope.**

| scope | meaning |
|---|---|
| gated | rule and human already agree. Ordinary talk, plus boundary controls that are inside the shipped contract (window of 3 still in, window does not eat an exact name, edit-1 while active, tie, silent write, extension, missing relation, bad hearer string). |
| gap | they disagree. Not a regression if the ask path matches the rule. |
| oos | open addressee forms left out of #34 (`other party`, `this one`, `that person`, `our guest`, `人家`, `对方`, `这位`, `那位`). Scored apart. A miss is not a regression. |
| control | no turns. Documents empty-log main behavior. Not part of the gap rate. |
| observe | short name, given-name nickname, pinyin. The head may or may not accept them. Not a gate. |

A gated row whose ask path misses the rule is a spec miss. That is the only result that would justify a code change in this PR, and only if the fix is small, always on, and rechecked on a fresh slice plus the probe floors. This bank is not that slice. Window, gender, vocative inference, and pronoun chaining are not half-implemented here.

## Overlap

Literal overlap 0 and max token Jaccard under 0.50 against probe, surface corpus, abstain dev, frozen, and every prior DEV and held-out including dialogue and deixis. Tokenizer is `tidal.question_heldout_data._tokens`. The check is `jaccard_report` in `tidal.question_dialogue_eval`. Strings are the questions, the turns, and the stored fact sentences.

## Report

State rule hit, state human hit, empty-log rule hit, false accepts (a rule abstain that returned a value), by phenomenon and by family (you, third, name, edit, abstain, indirect, window, oos, empty). Gap rows list the episode, the question, the rule outcome, and the human outcome.
