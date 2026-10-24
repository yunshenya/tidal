# Wall-clock pre-registration

Protocol `clock-drift-v1`. Fit ids are `clock-fit-000` through `clock-fit-007`. Held ids are `clock-held-000` through `clock-held-049`. The banks are disjoint. Items are numeric schedules, not sentences. Bank sha256 `4aa7129121c21093d0e85e650aeb25726f5602f55e22d1e2e6dc9bbaa4395847`.

## Constants (HYPOTHESIS)

No grid was run. The numbers below are the hypothesis. They are not taken from a figure, and they were not retuned on held.

| name | value |
|---|---|
| wall tick | 1 second |
| wall half-life | 60 seconds |
| drift rate | 1.25 internal seconds per wall second |
| entrainment gain | 0.5 |

The token half-life stays 30 tokens. Theta stays 0.32. Lambda stays 0.3. The cue floor stays 0.05. Fan S stays 1.45. Process S formulas stay `1 - (1 - S) * exp(-t / 16)` during wake and `0.1 + (S - 0.1) * exp(-D / 4)` during sleep. Arousal baseline, tau, weights, and window stay 0.2, 8, 0.35, 0.25, 0.20, and 8. The store circadian span stays 0.25 and the circadian step stays `pi / 16`. None of those is an input to this clock, and this clock is not an input to them.

Wall tau is `60 / ln(2)`. At an update the remaining factor is the lower of `exp(-dt / token_tau)` and `exp(-elapsed / wall_tau)`. That is the stronger decay. It is not the product. When elapsed is 0 the token expression is returned directly and the minimum is not taken, so a frozen wall clock leaves the shipped token formula. A negative elapsed is 0. A continued winner never rises above the activation it already carries. On that decay path, activation below 0.05 clears the carried trace. A trigger itself may still sit below 0.05. 300 seconds is a schedule point past that floor, not a fitted constant.

## Causality

`dt` counts tokens already seen after the trigger. One utterance trace holds the wall reading fixed, so a tick that has not been applied is invisible inside that forward. A new winner resets `dt` to 0 and stamps the current wall time. A fresh `cue_fields` or a fresh generation forward starts from an empty trace, so seconds since boot do not decay that retrieval. Wall time moves only through `advance_wall`. The default tick is 1 second. The implementation does not read the operating-system clock.

The internal clock advances by `drift_rate * seconds` on the same call. `calibrate(timestamp)` adds `gain * (timestamp - internal)` and does not move wall time. Decay reads wall seconds since the trigger. It does not read the internal clock, so calibration cannot raise or lower a carried activation. This is not the store's circadian angle.

No cue-export field is added. The live `interface_version` is 5. `advance_wall` and `calibrate` return the wall reading, the internal reading, and the signed error `internal - wall` because those calls are the tick and the calibration. Those three numbers are not per-token fields.

## Citations

Verified on 2026-10-24 through the Crossref works API. Abstracts for the 2005 and 2014 papers were empty, so the claims below are the titles.

Joordens and Becker (1997), Journal of Experimental Psychology: Learning, Memory, and Cognition. The long and short of semantic priming effects in lexical decision. The token clock counts tokens. Their unit is not seconds. https://doi.org/10.1037/0278-7393.23.5.1083

Buhusi and Meck (2005), Nature Reviews Neuroscience, issued 2005-09-15. What makes us tick? Functional and neural mechanisms of interval timing. https://doi.org/10.1038/nrn1764

Allman, Teki, Griffiths, and Meck (2014), Annual Review of Psychology, issued 2014-01-03. Properties of the Internal Clock: First- and Second-Order Principles of Subjective Time. https://doi.org/10.1146/annurev-psych-010213-115117

The drift rate is not a dopamine dose. The entrainment gain is not a circadian phase shift. No other DOI is cited for this mechanism.

## Gates

Locked before held. Schedules: wall seconds 1 and 4, and 300 for the floor clear. Token steps 0, 1, and 4 at elapsed 0. Both-clock points are `(dt=1, elapsed=30)` and `(dt=8, elapsed=1)`. Entrainment advances 8 wall seconds, then calibrates to the wall reading twice.

1. With wall time advanced and no new tokens, activation falls. From peak 1 and `dt` 0, elapsed 0 is exactly 1. Elapsed 1 and elapsed 4 are strictly decreasing and both stay above the 0.05 floor. A carried trace advanced 4 wall seconds keeps `dt` 0, moves the internal clock to 5, and sets activation to the 4-second wall factor rather than the 5-second internal factor. Elapsed 300 is below the floor and the carried trace is cleared.
2. With tokens and a frozen wall clock, token decay still matches `exp(-dt / token_tau)` exactly at steps 0, 1, and 4. A non-trigger step at the trigger's own wall stamp matches the one-token factor. Theta, the token half-life, and lambda are unchanged.
3. When both clocks have advanced, remaining activation equals the lower of the two single-clock activations and is therefore no higher than either. It is strictly above the product of the two factors. At `(1, 30)` the wall factor is lower. At `(8, 1)` the token factor is lower. A wall stamp behind the trigger does not raise activation. A new winner at wall time 100, then one non-trigger token at that same reading, decays by one token and not by the 100 seconds.
4. After 8 wall seconds the signed error is 2. One calibration to the wall reading multiplies that error by `1 - gain`. A second calibration multiplies the remaining error by the same factor, so the errors are 2, then 1, then 0.5. Calibration does not change the carried activation, process S, or arousal. `advance_wall` does not change process S or arousal either.
5. A fresh cue export after `advance_wall(8)` has the same activation as a fresh cue export at wall time 0. `interface_version` is 5. The export has no `clock_error`, `wall_seconds`, or `internal_seconds` field.

An item passes when every gate passes. Fit must be 8/8 on each gate before these floors are locked. Held floors are 50/50 on each gate, including the item gate. A rate whose n is below 50 fails. Wilson 95% uses z = 1.959963984540054. The held canonical must match the fit canonical.

Cue-triggered floors stay the merged #57/#62 rates: prefer 46/50, avoid 37/50, avoid-stored 42/50, toward 47/50, away 47/50, coexist 50/50, RIF competitor 47/50, winner per event 50/50 items and 215/215 events, outsider 50/50, interference similarity 35/50, interference count 32/50, fan 45/50, and the direction gates from that rescore. The cue guard writes new files and must match those counts and means. The cue guard was not scored.

Sleep gates stay the #63 held record: 50/50 on wake, sleep, replay, spontaneous recall, cue source, and the item, with the canonical pressures, drop, spontaneous rate, and replay strength equal to `reports/subconscious_sleep_held.json`. The sleep guard writes a new file. The sleep guard was not scored.

Arousal gates stay the #64 held record: 50/50 on surprise, quiet, valence and triggers, export, and the item, with the canonical levels equal to `reports/subconscious_arousal_held.json`. The arousal guard writes a new file. The arousal guard was not scored.

Theta, the token half-life, lambda, fan S, the RIF span, the process S formulas, and the arousal weights are unchanged. `store.clear` does not move the wall clock or the internal clock. There is no feature flag and no growbrain switch.

The held bank was not scored.
