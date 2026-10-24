# Arousal pre-registration

Protocol `arousal-lcne-v1`. Fit ids are `arousal-fit-000` through `arousal-fit-007`. Held ids are `arousal-held-000` through `arousal-held-049`. The banks are disjoint. Items are numeric schedules, not sentences. Bank sha256 `f1b2336a04c1aa9d7ff15562347fb6c37cdf105857135b6ddf91fade401feec9`.

## Constants (HYPOTHESIS)

No grid was run. The numbers below are the hypothesis. They are not taken from a figure in the paper, and they were not retuned on held.

| name | value |
|---|---|
| baseline | 0.2 |
| tau | 8 tokens |
| surprise weight | 0.35 |
| valence weight | 0.25 |
| trigger weight | 0.20 |
| surprise window | 8 |

Range is `[0, 1]`. Quiet step: `baseline + (level - baseline) * exp(-1 / tau)`. A surprise bumps only when it is strictly above the mean of at least two prior samples in this window. The excess is `(surprise - mean) / (|mean| + 1)`. The bump is `weight * tanh(excess)`. `|valence|` adds `weight * tanh(|valence|)`, and 0 adds nothing. A positive trigger count adds `weight * tanh(count)`. A count of 0 is a no-op, not a quiet step. One triggered token uses count 1. Four triggered tokens are four of those steps. `observe_triggers(4)` is one step at count 4, which is a different number, and the monotone check scores both.

The surprise seeds are `1, 1, 1, 1`, then `4`. Quiet durations are 1 and 4, taken from the level after that `4`. Valence magnitudes are 0, 0.5, and 1.5. Trigger counts are 0, 1, and 4. This window is not the write-gate window of 32.

## Citation

Verified on 2026-10-24 through the Crossref works API.

Aston-Jones and Cohen (2005), Annual Review of Neuroscience 28:403–450. An integrative theory of locus coeruleus–norepinephrine function: adaptive gain and optimal performance. https://doi.org/10.1146/annurev.neuro.28.061604.135709

The paper treats LC-NE as an arousal signal with a phasic mode and a tonic mode. Phasic activation follows the outcome of a task decision and is proposed to facilitate the behavior that follows. When utility falls, tonic activity is associated with disengagement and search for an alternative. tidal does not choose exploitation or exploration, and it does not choose to speak or to wait. Those decisions belong to the brain layer. The scalar rises on surprise above the recent mean, on `|valence|`, and on retrieval triggers, then decays toward the baseline on quiet tokens. That decay is not the paper's tonic rise during disengagement. growbrain only consumes the scalar. There is no growbrain switch and no feature flag.

## Gates

Locked before held:

1. After four surprises of 1, arousal stays at the baseline. A later surprise of 4 raises it strictly above the baseline, and the result stays in `(baseline, 1]`. The call does not write the write-gate window and does not move process S.
2. From that same elevated level, quiet duration 4 ends lower than quiet duration 1. Both drops are positive. Both levels stay strictly above the baseline and in `(baseline, 1]`.
3. Valence 0 stays at the baseline. Valence 0.5 ends below valence 1.5, and both end above the baseline. Trigger count 0 on an elevated level does not change it. Trigger counts 1 and 4, from the baseline, satisfy `baseline < 1 < 4`. A trace of four quiet tokens stays at the baseline. A trace of four triggers ends above a trace of one trigger, which ends above the baseline.
4. Cue export `interface_version` is 5. The arousal series matches `apply_trace` within float32 absolute tolerance `1e-6`, and every value is in `[0, 1]`. A memory raised by valence 1.5 and a memory at the baseline, with the same slot, export the same activation and the same last residual. Process S is unchanged by the cue call. `wake_for` and `sleep_for` do not move arousal. An empty-store cue export quiet-steps and stays at the baseline.

An item passes when every gate passes. Fit must be 8/8 on each gate before these floors are locked. Held floors are 50/50 on each gate, including the item gate. A rate whose n is below 50 fails. Wilson 95% uses z = 1.959963984540054. The held canonical must match the fit canonical.

Cue-triggered floors stay the merged #57/#62 rates: prefer 46/50, avoid 37/50, avoid-stored 42/50, toward 47/50, away 47/50, coexist 50/50, RIF competitor 47/50, winner per event 50/50 items and 215/215 events, outsider 50/50, interference similarity 35/50, interference count 32/50, fan 45/50, and the direction gates from that rescore. The cue guard writes new files and must match those counts and means. The cue guard was not scored.

Sleep gates stay the #63 held record: 50/50 on wake, sleep, replay, spontaneous recall, cue source, and the item, with the canonical pressures, drop, spontaneous rate, and replay strength equal to `reports/subconscious_sleep_held.json`. The sleep guard writes a new file. The sleep guard was not scored.

Theta, half-life, lambda, fan S, the RIF span, and the process S formulas are unchanged. Arousal is not part of the residual and it is not an input to process S. `store.clear` does not move arousal. `_on_layer` repeats the current scalar and does not step it, because generation re-forwards the prefix. `cue_fields` and `advance_cue` are the trace that steps.

The held bank was not scored.

## Fit result

Report `reports/subconscious_arousal_fit.json`. Every gate is 8/8, Wilson 95% [0.6755924351161198, 1.0]: surprise, quiet, valence and triggers, export, and the item. `canonical_same` is true. After four surprises of 1, arousal is 0.2. After surprise 4 it is 0.5168018887757032. Quiet duration 1 ends at 0.4795766855775076 (drop 0.03722520319819561). Quiet duration 4 ends at 0.39215005859733565 (drop 0.12465183017836756). Valence 0, 0.5, and 1.5 end at 0.2, 0.31552928931500246, and 0.4262870634112166. Trigger counts 0, 1, and 4 end at 0.2, 0.352318831191153, and 0.39986585994781343. A trigger count of 0 leaves the valence-1.5 level at 0.4262870634112166. Quiet trace, one trigger, and four triggers end at 0.2, 0.352318831191153, and 0.8092753247646118. The cue export's first and last arousal values are 0.35231882333755493 and 0.3344208896160126. These counts did not move a held floor. Theta, half-life, lambda, fan S, the RIF span, and the process S formulas were not changed.
