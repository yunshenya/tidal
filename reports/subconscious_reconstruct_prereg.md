# Reconstructive recall pre-registration

Protocol `reconstruct-spell-v1`. Fit ids are `reconstruct-fit-000` through `reconstruct-fit-007`. Held ids are `reconstruct-held-000` through `reconstruct-held-049`. The banks are disjoint. Items are numeric schedules, not sentences. Every id runs the same probe. Bank sha256 `bad8e653287b996114e32b68920c53c35dfa142d16c6bfb85fd5b4a811f78589`.

## Constants (HYPOTHESIS)

No grid was run. The numbers below are the hypothesis. They are not taken from a figure, and they were not retuned on held.

| name | value |
|---|---|
| exact strength | 0.8 |
| exact age | 0 |
| exact absolute valence | 0.75, strict less-than |
| loss strength | 0.4, strict less-than |
| long age | 4 |
| line-loss offset | 1111 |

Exact reassembly requires strength at least 0.8, age at most 0, and absolute valence strictly below 0.75. Loss is strength below 0.4 or age at least 4. Every other finite triple is a peripheral order slip. A non-finite input is loss.

The probe phone is `+1-617-555-4826`. The probe birthday is `April 21, 1984`. The lure phone is `+1-617-555-9090`. The lure birthday is `June 9, 1972`. Prompts are `What is Ada's phone number?`, `When is Ada's birthday?`, and `Which city does Ada live in?`.

Schedules, as `(strength, age, valence)`:

| name | schedule |
|---|---|
| high | `(1.0, 0, 0.0)` |
| low | `(0.2, 0, 0.0)` |
| long | `(1.0, 4, 0.0)` |
| emotion | `(1.0, 0, 1.5)` |
| slip | `(0.5, 0, 0.0)` |

Phone gist is the `+1` country code, the area, and the exchange. The line is peripheral. An order slip swaps the last two line digits when they differ. When they match, the last digit increases by 1 modulo 10. A loss adds 1111 modulo 10000. Birthday gist is the month and the year. The day is peripheral. An order slip swaps the two day digits when the result is a different day in 1–27. Otherwise the day steps to the neighbor that stays in 1–27. A loss sets the day to 1, or to 2 when the day is already 1. This vocabulary has no synonym. The implemented errors are that order slip and that detail loss.

Age is the slot's consolidation counter. It is not wall time and it is not the circadian angle. Reconstruction does not call `advance_wall`. It does not change theta, the token half-life, lambda, fan S, the RIF span, process S, or arousal.

The residue is built at `store.write` from the answer argument. It is a tuple of a kind tag and integers. It does not contain the raw string. Spelling reads the residue, strength, age, and valence. It does not read `slot.answer`. `answer` and `text` stay on the slot. Repeat detection reads `text`. Reconsolidation checks that both strings are unchanged. Replay does not write a raw string. The audit row still has no fact text. No cue-export field is added. The live `interface_version` is 5.

## Citations

Verified on 2026-10-24 through the Crossref works API.

Bartlett and Kintsch, issued 1995-06-30. Remembering. The Crossref abstract describes the 1932 book, schema theory, subjects who changed details of a story, and memory as a process of reconstruction. https://doi.org/10.1017/CBO9780511759185

Brewer and Treyens (1981), Cognitive Psychology, issued 1981-04. Role of schemata in memory for places. The abstract field was empty, so this citation is the title. https://doi.org/10.1016/0010-0285(81)90008-6

Kensinger (2009), Emotion Review, issued 2009-04. Remembering the Details: Effects of Emotion. The abstract says emotion does not enhance memory equally for all aspects of an experience. It describes arousal's focal enhancements, more often for negative than for positive experiences, with negative affect associated with sensory processes and positive affect with conceptual processes. https://doi.org/10.1177/1754073908100432

The gist-versus-peripheral split is a hypothesis mapped onto that unequal, focal account. It is not a quotation of the abstract, and it is not a claim that the abstract says peripheral details are lost. No other DOI is cited for this mechanism.

## Gates

Locked before held. An item passes when every gate passes.

1. High. Strength 1, age 0, valence 0. The grade is exact. The phone is `+1-617-555-4826`. The birthday is `April 21, 1984`.
2. Low. Strength 0.2, age 0, valence 0. The grade is loss. The strings are not the written spellings. Gist holds: phone area 617 and exchange 555, birthday month April and year 1984. The line and the day differ. The strings are `+1-617-555-5937` and `April 1, 1984`.
3. Long. Strength 1, age 4, valence 0. The same loss strings as the low schedule. Exact rate is not the high rate.
4. Emotion. Strength 1, age 0, valence 1.5. The grade is slip, so a full-string exact match fails where the calm high schedule still matches. Gist holds. The peripheral field differs from both the written spelling and the loss spelling. The strings are `+1-617-555-4862` and `April 12, 1984`. Peripheral errors are present here and absent on the calm high schedule. That is the sense in which high absolute valence raises peripheral errors while gist stays. It is not a fitted time series.
5. Slip. Strength 0.5, age 0, valence 0. The same order-slip strings as the emotion schedule. Gist holds and the peripheral field differs from the written spelling.
6. Intrusion. None of those strings equals the lure phone or the lure birthday. The off-relation prompt's scripted continuation is None. Held intrusion events are 0/50, Wilson 95% [0.0, 0.07134759913335872]. A positive event fails the ceiling.
7. Live. On the eye model, `generate` at the high schedule equals the written spelling. At the low schedule the phone equals `+1-617-555-5937` and does not equal `slot.answer`. At the emotion schedule the phone equals `+1-617-555-4862`. At the long schedule the birthday equals `April 1, 1984`. `slot.answer` is still the written spelling. `interface_version` is 5. The cue export keys are unchanged.

Fit must be 8/8 on each gate, with intrusion events 0/8, before these floors are treated as locked by a passing fit. Held floors are 50/50 on each gate, including the item gate, and intrusion events are 0/50. A rate whose n is below 50 fails. Wilson 95% uses z = 1.959963984540054. Wilson 95% for 50/50 is [0.9286524008666414, 1.0]. Wilson 95% for 8/8 is [0.6755924351161198, 1.0]. The held canonical must match the fit canonical.

Cue-triggered floors stay the merged #57/#62 rates: prefer 46/50, avoid 37/50, avoid-stored 42/50, toward 47/50, away 47/50, coexist 50/50, RIF competitor 47/50, winner per event 50/50 items and 215/215 events, outsider 50/50, interference similarity 35/50, interference count 32/50, fan 45/50, and the direction gates from that rescore. The cue guard writes new files and must match those counts and means. The cue guard was not scored.

Sleep gates stay the #63 held record: 50/50 on wake, sleep, replay, spontaneous recall, cue source, and the item, with the canonical pressures, drop, spontaneous rate, and replay strength equal to `reports/subconscious_sleep_held.json`. The sleep guard writes a new file. The sleep guard was not scored.

Arousal gates stay the #64 held record: 50/50 on surprise, quiet, valence and triggers, export, and the item, with the canonical levels equal to `reports/subconscious_arousal_held.json`. The arousal guard writes a new file. The arousal guard was not scored.

Clock gates stay the #65 held record: 50/50 on wall, token, both, entrainment, fresh export, and the item, with the canonical levels equal to `reports/subconscious_clock_held.json`. The clock guard writes a new file. The clock guard was not scored.

Theta, the token half-life, lambda, fan S, the RIF span, the process S formulas, the arousal weights, and the wall-clock formulas are unchanged. There is no feature flag and no growbrain switch.

The held bank was not scored.

## Fit result

Report `reports/subconscious_reconstruct_fit.json`. Fit git `f73c3cb777cae3686de2d8cfa92a452d89af6ef4`. Every gate is 8/8, Wilson 95% [0.6755924351161198, 1.0]: high, low, long, emotion, slip, intrusion, live, and the item. Intrusion events are 0/8, Wilson 95% [0.0, 0.32440756488388023]. `canonical_same` is true. Elapsed 0.08174987400707323 seconds. High spelling is `+1-617-555-4826` and `April 21, 1984`. Low and long spelling is `+1-617-555-5937` and `April 1, 1984`. Emotion and slip spelling is `+1-617-555-4862` and `April 12, 1984`. The off-relation continuation is absent. `interface_version` is 5. These counts did not move a held floor. Theta, the token half-life, lambda, fan S, the RIF span, the process S formulas, the arousal weights, and the wall-clock formulas were not changed.
