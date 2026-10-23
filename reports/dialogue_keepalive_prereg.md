# Pronoun keep-alive and leading vocatives — pre-registration

Written before the code and before either score. The #35 real-dialogue bank stays the development set for these two phenomena. A fresh held-out bank is `tidal/question_keepalive_heldout_data.py`. Numbers land in `reports/dialogue_keepalive.md` after one run. No threshold, window, or name-head weight is fit. `TURN_WINDOW` stays 4. `ABSTAIN_MARGIN` stays 0.1. `LEX_RIDGE_GAP` stays 0.15. There is no enable flag. `LLMMemory` is untouched. The frozen file `cce20b6b…` is not an input and its labels are not edited.

## What changes

Two always-on updates on the utterance path. Empty log (no `note_turn`, no stored hearer) stays the current ranker.

**Keep-alive.** When `resolve_deixis` returns exactly one person for an utterance or an ask, that person's latest conversational mention still inside `TURN_WINDOW` is restamped onto the current turn. The pronoun text is not stored, and no new name string is appended. A mention already outside the window is not pulled back. `resolve_deixis` returning `None` or `()` does not restamp anyone. That covers a stored name in the same string, a tie, a mixed second-and-third person, `them`, `你们`, and a third person who has already aged out.

**Vocative.** If `note_turn` is not given a hearer, and the utterance starts with exactly one exact roster name followed by address punctuation (`,` `，` `、` `:` `：` `!` `！`), that name is the hearer. A second name in that same position after the punctuation means the address is not unique, and the hearer stays unset. A name used as subject or object, a name later in the sentence, an edit, a nickname, or a pinyin form does not qualify. A hearer string the caller passed, including one that is not on the roster, is kept.

## Development gate (#35 bank)

Listener labels, false accepts 0.

| slice | pass |
|---|---|
| gated | 106/106 state and empty |
| ordinary talk | 52/52 state and empty |
| `pronoun_chain_age` | 2/2 listener (was the experience gap) |
| `vocative_unpassed` | 2/2 listener (was the experience gap) |
| other gap rows | still disagree with the listener, rule hit, false accepts 0 |
| open addressee | still 0/8 against the listener, not a regression |

`window_backchannel` stays a gap: four acknowledgements with no pronoun still age the name out. `edit_aged`, gender ties, recency, insertions, swaps, `老`+full name, the speaker-as-third reading, and `你们` stay gaps.

## Held-out gate

Scored apart from the development rows. Literal overlap 0 and max token Jaccard under 0.50 against every earlier bank, including the #35 real-dialogue bank. Tokenizer is `tidal.question_heldout_data._tokens`.

People: Calder, Mirren, Solway, Anwick, Berrick, Yarrow; 卞茗, 苟宁, 裴涟, 栾昭, 童屿. Props are lichen / kelp / amber / salt-wick and 青苔板 / 海带票 / 琥珀轴 / 盐芯. Not the #35 cast or props.

| kind | pass |
|---|---|
| `keep_he`, `keep_she`, `keep_ta`, `keep_zhta` | the person the pronoun chain kept alive |
| `keep_edit` | edit-1 still licensed after a pronoun chain |
| `keep_ask` | a chain of distinct pronoun asks still names that person |
| `age_quiet` | four acknowledgements, then the pronoun, abstains |
| `age_resurrect` | four acknowledgements, then one pronoun, still abstains |
| `age_tie` | two people on the newest turn, the pronoun abstains |
| `age_edit` | edit-1 after four acknowledgements and no pronoun abstains |
| `voc_lead` | leading name + address punctuation, no hearer passed, `you` is that person |
| `voc_subject`, `voc_object`, `voc_medial` | name is not a leading vocative, `you` abstains |
| `voc_two` | two leading vocatives, `you` abstains |
| `voc_explicit` | caller hearer wins over a different leading name |
| `empty` | no turns; deictic pronoun stays on the ranker (the decoy) |

State and empty each meet that row's rule label. False accepts 0. A miss stops the change.

## Floors, re-run because the ask path changes

Probe answerable at least 365/370, false abstain 0/370, unanswerable 72/72, gold lost at overwrite 0.9 is 0, co-resident cosine above 0.9 is 0. Frozen hits at least 188/216, false abstain no worse than 19/216. Dialogue dev 102/102 both passes. Dialogue held-out at window 4, gated 108/108 both passes. Deixis dev 48/48 both passes. Deixis held-out 128/136 with the turns and 136/136 on the empty log.
