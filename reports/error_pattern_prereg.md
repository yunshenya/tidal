# Error-pattern measurement preregistration

Protocol `error-pattern-v1`. Bank sha256 `b127a314367fbe273ec05b40841029748018170244c10ef84b7f72dab8fee514`.

This is a measurement. It does not change retrieval, spelling, clocks, sleep, arousal, constants, or the speak path. The brain layer (growbrain) still owns turn-taking and speaking. This measurement does not change mechanisms, constants, or the speak path, and it must not be merged.

Fit ids `error-fit-000` through `error-fit-007` are disjoint from held ids `error-held-000` through `error-held-049`. The implicit words are disjoint from each other and from the cue-direction word banks. Each scored continuation is one SmolLM token with a leading space. The template is `Hand {name} some {domain}.` and `{name} circles toward {answer} about {domain}.`

## Literature

Each DOI was read from Crossref on 2026-10-24. An empty abstract is cited by title only.

Anderson, Bjork, and Bjork (1994). Remembering can cause forgetting: Retrieval dynamics in long-term memory. Crossref recorded no abstract. https://doi.org/10.1037/0278-7393.20.5.1063

Anderson and Reder (1999). The fan effect: New results and new theories. Crossref recorded no abstract. https://doi.org/10.1037/0096-3445.128.2.186 The shipped fan factor is `max(0, (S - ln n) / S)` with `S` 1.45. This measurement does not change `S`.

Kensinger (2009). Remembering the Details: Effects of Emotion. The abstract says emotion does not enhance memory equally for all aspects of an experience, and it describes arousal's focal enhancements, more often for negative than for positive experiences. https://doi.org/10.1177/1754073908100432 Mapping that focal account onto a kept gist and a changed peripheral field is the hypothesis already shipped in reconstructive recall. It is not a quotation that peripheral details are lost.

Joordens and Becker (1997). The long and short of semantic priming effects in lexical decision. Crossref recorded no abstract. The token unit is not seconds. https://doi.org/10.1037/0278-7393.23.5.1083

Buhusi and Meck (2005). What makes us tick? Functional and neural mechanisms of interval timing. Crossref recorded no abstract. https://doi.org/10.1038/nrn1764

Allman, Teki, Griffiths, and Meck (2014). Properties of the Internal Clock: First- and Second-Order Principles of Subjective Time. Crossref recorded no abstract. https://doi.org/10.1146/annurev-psych-010213-115117

Berntsen (2020). Involuntary autobiographical memories and their relation to other forms of spontaneous thoughts. The abstract says involuntary autobiographical memories come to mind spontaneously, with no conscious initiation of the retrieval process. https://doi.org/10.1098/rstb.2019.0693 The shipped 16-tick period is unchanged.

## Shape gates

The same numeric probe runs on every id. The eye model is the stand-in. These gates are directional because they follow the shipped code. A miss is reported. It is not a license to edit production.

1. Interference. Three slots share the last-token cue. The winner and one competitor have the same relation. The outsider has another relation. Salience is 0. After `_rank`, strengths are 1, 0.95, 1. `last_rif` is `{"n": 1}`. The winner is the same-relation slot. 0.95 is `1 - 0.05`.
2. Recency. Token steps 0, 1, and 4 leave 1.0, 0.9771599684342459, and 0.9117224885582168. Wall seconds 0, 1, 4, and 300 leave 1.0, 0.9885140203528962, 0.9548416039104165, and 0.03125. Wall 300 is below the 0.05 floor and clears the carried trace. The both-clock points `(dt=1, elapsed=30)` and `(dt=8, elapsed=1)` leave 0.7071067811865476 and 0.8312378961427878, each the lower of the two factors. Live `advance_wall(4)` leaves activation 0.9548416039104165, wall 4, internal 5, and not the activation of elapsed 5.
3. Fan. Activations for 1, 2, and 4 candidates equal `a_trig * fan_factor(n)` within 1e-6, strictly decrease, and still trigger when the factor is below 0.05.
4. False trigger. A key orthogonal to the cue does not trigger. Events are 0.
5. Spontaneous recall under no cue. Wake duration 4 leaves pressure 0.2990792952357356. Sixteen offline ticks fire once. The rate is 0.0625, strictly below 0.10. The first tick's source is `spontaneous`.
6. Fresh retrieval. Wall time moved before the trigger does not change the new trace's activation. `interface_version` stays 5. Clock error is not exported.

Held requires 50/50 on each shape gate and on the item, with Wilson 95% intervals. False-trigger events are 0/50.

## Explicit spell report

Explicit phone and birthday probes are report-only. They are not inputs to Cohen's d_z. They are not a merge gate for a behavior change.

The probe is `+1-617-555-4826` and `April 21, 1984`. Lures `+1-617-555-9090` and `June 9, 1972` are not emitted. An off-relation prompt returns no scripted continuation.

Locked spellings: high `+1-617-555-4826` and `April 21, 1984`; low and long `+1-617-555-5937` and `April 1, 1984`; emotion and slip `+1-617-555-4862` and `April 12, 1984`.

Gist for a phone is the `+1` prefix, the area, and the exchange. Gist for a birthday is the month and the year. The line and the day are peripheral. High keeps the peripheral field. The other four schedules change it. Exact, gist, peripheral, intrusion, and live are reported with Wilson intervals. Intrusion events are a separate count.

## Implicit influence

Three margins on each item, length-normalized `mean logp(good) - mean logp(bad)`:

- no memory: the question prompt, empty store
- in context: the fact text in the prompt, empty store
- cue-triggered: `remember` the fact, question prompt only

The cue delta and the in-context delta each subtract the no-memory margin. Both deltas must be finite on every item. Fit n is 8. Held n is 50. Cohen's d_z is reported separately for the cue deltas and for the in-context deltas. The Wilson interval on the positive rate is descriptive. There is no sign floor and no magnitude floor. A negative d_z is still a completed measurement. Explicit spell fields are not inputs.

## Held lock

The held bank was not scored.

The fit bank is scored once before held, into `reports/error_pattern_fit.json`. Held is scored once after that, into `reports/error_pattern_held.json`, and only if fit passed and this sentence is still present.
