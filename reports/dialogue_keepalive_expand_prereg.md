# Keep-alive and vocative held-out expansion — pre-registration

Written before the expanded bank is scored. The rule is frozen at `1cb9e0f`. This run does not edit `question_surface.py`, `dialogue_state.py`, `continuum.py`, or `llm_memory.py`. `TURN_WINDOW` stays 4. `ABSTAIN_MARGIN` stays 0.1. `LEX_RIDGE_GAP` stays 0.15. No enable flag. The frozen file `cce20b6b…` is not an input.

The 42-ask bank in `tidal/question_keepalive_heldout_data.py` stays byte-for-byte. The expansion is `tidal/question_keepalive_expand_data.py`. Scored asks are the union: 42 + 94 = 136. English and Chinese are balanced inside the 94 (47 and 47).

Listener labels below are the gate. They follow the frozen spec: a pronoun restamps one in-window person only when `resolve_deixis` returns exactly one name; a leading exact roster name plus address punctuation is the hearer only when the caller did not pass one; gender is not a feature. A row whose replay disagrees with that label is a miss. It is reported. It is not patched, and the label is not edited after the score.

## Categories

| kind | asks | en | zh | listener |
|---|---:|---:|---:|---|
| `neg_subject` | 12 | 6 | 6 | `ABSENT`. The name is the subject, initial or after an adjunct, and the next character is not address punctuation. The question is `you` / `你`. No hearer was passed. |
| `neg_object` | 12 | 6 | 6 | `ABSENT`. The name is an object, not at position 0. The question is `you` / `你`. No hearer was passed. |
| `neg_quote` | 12 | 6 | 6 | `ABSENT`. The name sits in quotation or after a reporting verb, and the utterance does not start with that name plus address punctuation. The question is `you` / `你`. No hearer was passed. |
| `neg_ambig` | 12 | 6 | 6 | `ABSENT`. Two shapes. Two stored people share the newest turn, and `he` / `she` / `他` / `她` does not pick one, including when only one of them would match a gender reading. A leading pair (`Name, Other, …`) does not set a hearer, so `you` / `你` abstains. One subtype asks only after the ambiguous pronoun and three further turns with no name: a refresh of either person would still be inside the window, and the listener still abstains. |
| `neg_unknown` | 8 | 4 | 4 | `ABSENT`. The utterance starts with a name that is not on the roster (never stored, or a near-miss such as `Phelam` for `Phelan`), plus address punctuation. The question is `you` / `你`. No hearer was passed. |
| `neg_explicit` | 10 | 5 | 5 | The caller hearer wins. Six asks (3 en, 3 zh) name that roster hearer, who is not the leading vocative. Four asks (2 en, 2 zh) pass a hearer that is not on the roster (`Sigrid` / `沈禾` / `unlisted`), and the listener is `ABSENT`. |
| `neg_age` | 12 | 6 | 6 | `ABSENT`. The target is named once, then four turns do not mention that person and do not use `you` / `he` / `she` / `they` / `你` / `他` / `她` / `ta`. The fillers are acknowledgements, `them` / `it`, or `你们` / `它`. The question is the third-person pronoun. |
| `pos_keep` | 8 | 4 | 4 | The person named on the first turn. Four later turns use only `he` / `she` / `他` / `她` and do not repeat the name. |
| `pos_voc` | 8 | 4 | 4 | The leading exact name. The question is `you` / `你`. No hearer was passed. |

72 of the 94 asks are listener abstentions. The other 22 are listener hits: 16 keep/vocative controls and 6 explicit-hearer overrides. The weight is on the negatives.

## How a row is scored

State mode walks `note_turn` and `query`. Empty mode asks the same question on a fresh store. A false accept is a rule abstain (`absent_person` or `missing_rel`) that returned a value.

| column | meaning |
|---|---|
| state pass | the state-mode answer matches the replayed rule label |
| empty pass | the empty-mode answer matches the replayed rule label |
| listener | the state-mode answer matches the listener label in the table |
| FA | false accepts on that pass |

The replayed rule label is whatever the frozen code does. It is not rewritten to match the listener. Empty-log `you` / `he` / `你` / `他` stay on the ranker. That empty outcome is the empty-pass rule, not the listener column.

## Validity, not a rule change

Literal overlap 0 and max token Jaccard under 0.50 against every earlier bank, including the #35 real-dialogue bank and the frozen 42-ask keep-alive bank. Tokenizer is `tidal.question_heldout_data._tokens`. The new stamp is `marl docket`, not `bracken scrip`. People are Phelan, Idony, Hawise, Thurstan, Rohese, Osbern; 庞荞, 倪蒲, 滕岚, 邹蒲, 雷桐, 傅苔. Props are not the 42-ask props and not the #35 props.

A string that is not an instance of its category (the vocative fired on a subject line, a filler contained `他`, a cue collision) is rewritten so the category is true. A true instance whose listener label and rule label disagree stays in the bank.

## Floors

The ask path is unchanged, so the floors in `reports/dialogue_keepalive.md` still apply. They are re-run on this same rule and reported beside the new table. A floor that moved is reported as it measured. It is not repaired here.
