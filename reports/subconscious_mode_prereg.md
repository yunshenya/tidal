# Encode vs consolidate mode pre-registration

Protocol `encode-consolidate-mode-v0`. Fit ids are `mode-fit-000` through `mode-fit-007`. Held ids are `mode-held-000` through `mode-held-049`. The banks are disjoint. Items are numeric schedules plus the locked reconstruct phone and birthday, not free sentences. Bank sha256 `11a2828810dbe8cb41f1288772b959c9a573a940174f0b5ac3d44ff2a773b64a`.

This is human-likeness queue item #2 after retroactive interference #81 (`8e7c37a`) on main: tidal already owns `sleep_pressure`, `arousal`, `circadian`, and sleep two-phase (#79), but it lacks a hard mode that gates write vs consolidate behavior. This pull request derives one internal mode from those existing scalars. It does not add a cue-export field. Reconstruct bank `canonical_surfaces()["interface_version"]` stays 6. Live `CUE_INTERFACE_VERSION` stays 7. This pull request is a draft and is not merged.

## Constants (HYPOTHESIS)

No grid was run. The numbers below are the hypothesis. They are not taken from a figure in the papers, and they were not retuned on held.

| name | value |
|---|---|
| encode write keep | 1.0 |
| consolidate write keep | 0.75 (`1 - 0.25`, same span family as clock / McGaugh) |
| encode spontaneous keep | 0.5 |

Above-baseline arousal is encode. A positive circadian phase is encode. Otherwise process S at or above 0.2 is consolidate. The process-S floor 0.1 with baseline arousal 0.2 and a nonpositive phase is encode.

Encode multiplies a new `remember` clock strength by 1.0 and multiplies process S by 0.5 before the spontaneous floor check. After wake duration 4, that product is `0.1495396476178678`, below 0.2, so spontaneous recall does not fire. Consolidate multiplies the write by 0.75 and leaves spontaneous pressure equal to process S. Reconstructive spelling already maps strength 0.75, age 0, valence 0 onto a peripheral slip. Locked surfaces: encode phone `+1-617-555-4826`, encode birthday `April 21, 1984`; consolidate phone `+1-617-555-4862`, consolidate birthday `April 12, 1984`.

`sleep_for` still schedules SWS/REM from process S and the store circadian phase. The mode does not reimplement those operators and does not block them. An encode-mode `sleep_for(1)` after wake duration 4 still ages the slot by 1.

Dims stay 192 / 64 / 32. Sleep floor stays 0.1. Arousal baseline, tau, and window stay 0.2, 8, and 8. Theta stays 0.32. Half-life stays 30. Lambda stays 0.3. Cue interface stays 7. Reconstruct bank interface stays 6. `fusion_confidence` stays the #76 last-token setter path. `consolidation_gain` stays the #77 McGaugh path. Pattern-separation retrieve ranking stays the #78 view. Sleep two-phase operators stay the #79 SWS/REM passes. The reconsolidation window stays the #80 labile deadlines. The #81 retroactive age-hit still runs on a successful new-slot write; mode only scales the new slot's write keep.

## Citations

Verified on 2026-10-25 through the publisher pages.

Hasselmo and McGaughy (2004), Progress in Brain Research 145:207–231. High acetylcholine levels set circuit dynamics for attention and encoding and low acetylcholine levels set dynamics for consolidation. High acetylcholine enhances feedforward afferent input and reduces internal excitatory feedback; low acetylcholine permits recurrent / endogenous spread appropriate for consolidation. https://doi.org/10.1016/S0079-6123(03)45015-2

Gerstner and Yin (2010), Nature Reviews Neuroscience 11:577–588. Circadian rhythms and memory formation. The abstract says recent studies have shown time-of-day effects on neurophysiology and memory formation, and that common mechanisms underlie circadian rhythmicity and long-term memory formation. https://doi.org/10.1038/nrn2881

tidal does not simulate acetylcholine, clock genes, or the cAMP–MAPK–CREB cascade. The mode is that kind of encode vs consolidate cut on the process S, arousal, and circadian scalars tidal already owns. There is no growbrain switch and no feature flag.

## Gates

Locked before held:

1. Identity. Process S at the floor with baseline arousal and circadian phase 0 is encode. Write keep is 1. Phone and birthday stay exact. `apply_sleep_operators` is true for both modes.
2. Encode. Above-baseline arousal after wake duration 4 is encode. A positive circadian phase after that wake is encode. Encode write keep is 1. Encode spontaneous pressure is `0.1495396476178678`, below 0.2.
3. Consolidate. Wake duration 4 at baseline arousal and phase 0 is consolidate. Write keep is 0.75. Phone spelling is `+1-617-555-4862`. Birthday spelling is `April 12, 1984`. Spontaneous pressure equals process S and is at least 0.2.
4. Write. The same remember input under encode vs consolidate leaves different strength and different reconstructive spelling. Encode strength is 1. Consolidate strength is 0.75. `text` and `answer` stay. `consolidation_gain` stays 1 on a rest write. The #81 hit is unchanged: cosine below tau still keeps 1.
5. Sleep. The same `offline_ticks(16)` after wake duration 4 fires one spontaneous recall in consolidate mode and zero in encode mode (high arousal). Process S itself is unchanged by the mode.
6. Compose. `sleep_for` SWS/REM stay on `sleep_phase`. Encode-mode `sleep_for(1)` after wake still increments age by 1. Negative circadian phase at the process-S floor stays encode, so low-phase circadian writes keep the existing clock gain. `fusion_confidence` on a text-only store stays 0. Baseline `consolidation_gain` stays 1. Pattern-separation ranking, McGaugh gain, the #81 age-hit, and the labile window stay.
7. Export. Cue export `interface_version` is 7. Reconstruct bank `interface_version` is 6. The mode is not a cue key. `write_keep` is not a cue key. Sleep floor is 0.1. Dims stay 192 / 64 / 32.
8. Item. Every gate above passes.

An item passes when every gate passes. Fit must be 8/8 on each gate before these floors are locked. Held floors are 50/50 on each gate, including the item gate. A rate whose n is below 50 fails. Wilson 95% uses z = 1.959963984540054. The held canonical must match the fit canonical.

Cue-triggered floors stay the merged #57/#62 rates: prefer 46/50, avoid 37/50, avoid-stored 42/50, toward 47/50, away 47/50, coexist 50/50, RIF competitor 47/50, winner per event 50/50 items and 215/215 events, outsider 50/50, interference similarity 35/50, interference count 32/50, fan 45/50, and the direction gates from that rescore. Those floors were not moved.

Sleep gates stay the #63 held record: 50/50 on wake, sleep, replay, spontaneous recall, cue source, and the item, with the canonical pressures, drop, spontaneous rate, and replay strength equal to `reports/subconscious_sleep_held.json`. Arousal defaults stay 0.2 / 8 / 0.35 / 0.25 / 0.20 / 8. Reconstruct bank `interface_version` stays 6. The fusion scorer path is unchanged. The McGaugh consolidation path is unchanged. Pattern-separation retrieve ranking is unchanged. The sleep two-phase operators are unchanged. The reconsolidation window is unchanged. The retroactive age-hit is unchanged.

Theta, half-life, lambda, fan S, the RIF span, the process S formulas, the prediction-error blend, and the arousal weights are unchanged. The mode is not an input to process S. No cue-export field is added.

The held bank was not scored.

The cue guard was not scored.

The sleep guard was not scored.

The arousal guard was not scored.

## Fit result

Unit tests lock the canonical surfaces and keeps. Encode phone spelling is `+1-617-555-4826`. Consolidate phone spelling is `+1-617-555-4862`. Encode write keep is 1. Consolidate write keep is 0.75. Fit ids `mode-fit-000` through `mode-fit-007` all run the same fixtures and must be 8/8 on each gate before held is scored. These counts did not move a held floor.

## Held result

The held bank was not scored.
