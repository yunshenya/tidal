# Sleep two-phase consolidation pre-registration

Protocol `sleep-two-phase-v0`. Fit ids are `sleep-phase-fit-000` through `sleep-phase-fit-007`. Held ids are `sleep-phase-held-000` through `sleep-phase-held-049`. The banks are disjoint. Items are numeric schedules plus the locked reconstruct phone and birthday, not free sentences. Bank sha256 `63775a5bc64260c1996ded7bed7628e6b698d6a0175a28746c01e298774d2caf`.

This is neuroscience gap #3 after fusion scorer #76 (`6d99996`), arousal gain #77 (`604236c`), and pattern separation #78 (`a535648`) on main: sleep already has process S and prediction-error replay, but it lacks an SWS-like gist pass and a REM-like integrative / emotion-tag pass. This pull request adds those operators on the existing `sleep_for` sweep. It does not add a cue-export field. `replay_phase` is internal and is not written onto the cue wire. Reconstruct bank `canonical_surfaces()["interface_version"]` stays 6. Live `CUE_INTERFACE_VERSION` stays 7. This pull request is a draft and is not merged.

## Constants (HYPOTHESIS)

No grid was run. The numbers below are the hypothesis. They are not taken from a figure in the papers, and they were not retuned on held.

| name | value |
|---|---|
| SWS process-S floor | 0.2 (same as spontaneous recall) |
| REM valence keep | 0.9 |
| REM same-group bind | 0.05 |

High process S at sleep onset is SWS. After S has dropped below 0.2, or when circadian phase is nonzero at low S, the exit pass is REM. Sleep from the process-S floor 0.1 with a zero circadian phase is `none`: prediction-error replay only. That is the replay-only baseline.

SWS increments slot `age` by 1. Strength, keys, values, residue parts, `text`, and `answer` stay. Reconstructive spelling already maps age 1, strength 1, valence 0 onto a peripheral slip and a kept gist. Locked surfaces: phone `+1-617-555-4862`, birthday `April 12, 1984`. The no-sleep / replay-only surfaces stay `+1-617-555-4826` and `April 21, 1984`.

REM multiplies a nonzero valence by 0.9, then recomputes `consolidation_gain` with the unchanged McGaugh formula. Zero valence stays 0. Same-group value vectors blend by 0.05 and restore L2. Keys stay. Different groups do not bind. Locked numbers: valence `1.5 → 1.35`; gain at arousal 1.0 `1.2262870634112166 → 1.2185133219715016`; orthogonal unit-pair cosine `0.10497237569060777`.

Dims stay 192 / 64 / 32. Sleep floor stays 0.1. Arousal baseline, tau, and window stay 0.2, 8, and 8. Theta stays 0.32. Half-life stays 30. Lambda stays 0.3. Cue interface stays 7. Reconstruct bank interface stays 6. `fusion_confidence` stays the #76 last-token setter path. `consolidation_gain` stays the #77 McGaugh path. Pattern-separation retrieve ranking stays the #78 view. Prediction-error replay stays the #63 blend, threshold, and strength step. The operators do not simulate slow oscillations, spindles, or ripples.

## Citations

Verified on 2026-10-25 through the publisher pages.

Diekelmann and Born (2010), Nature Reviews Neuroscience 11:114–126. The memory function of sleep. The abstract says slow-wave sleep and REM sleep support system consolidation and synaptic consolidation, respectively, and that consolidation during sleep can produce qualitative changes in memory representations. https://doi.org/10.1038/nrn2762

Klinzing, Niethard, and Born (2019), Nature Neuroscience 22:1598–1610. Mechanisms of systems memory consolidation during sleep. The abstract says repeated hippocampal replay during slow-wave sleep leads to a gradual transformation and integration of representations in neocortical networks, and names qualitative transformations resulting in abstracted, gist-like representations. https://doi.org/10.1038/s41593-019-0467-3

tidal does not simulate SO–spindle–ripple coupling, cholinergic tone, or a second memory system. The SWS pass is that kind of gist transform on the residue / age machinery tidal already owns. The REM pass is that kind of weak integrative bind plus an emotion-tag refresh on the valence / arousal / `consolidation_gain` scalars tidal already owns. There is no growbrain switch and no feature flag.

## Gates

Locked before held:

1. Identity. Process S at the floor with circadian phase 0 is `none`. `sleep_for(1)` from the floor leaves phone age 0 and the exact spelling. Zero valence stays 0. `sws_age(0)` is 1 as a pure function and is not applied on that path.
2. SWS. Wake duration 4 then `sleep_for(1)` is SWS at onset. Age becomes 1. Phone spelling is `+1-617-555-4862`. Birthday spelling is `April 12, 1984`. Gist holds. The peripheral field changes. `text` and `answer` stay. Strength stays 1 on a parallel trace. Replay-only `sleep_for(1)` from the floor on the same fixture stays exact.
3. REM emotion. Floor sleep with a nonzero circadian phase is REM. Valence `1.5` becomes `1.35`. Valence `-1.5` becomes `-1.35`. `consolidation_gain(1.0, 1.35)` is `1.2185133219715016`, below the write-time `1.2262870634112166`. The McGaugh formula is unchanged. Age stays 0.
4. REM bind. Two same-group orthogonal unit values end at cosine `0.10497237569060777`. A third other-group axis is unchanged. Keys stay. `text` and `answer` stay.
5. Compose. Prediction-error replay of a parallel slot still does not change strength or the embedding. An orthogonal slot still leaves strength `exp(-0.05)` and the #63 blend. `consolidation_gain` at baseline arousal stays 1. Distinct pattern-separation keys stay orthogonal. `fusion_confidence` on a text-only store stays 0.
6. Export. Cue export `interface_version` is 7. Reconstruct bank `interface_version` is 6. `replay_phase` is not a cue key. `consolidation_gain` is not a cue key. Sleep floor is 0.1. Dims stay 192 / 64 / 32.
7. Replay. The #63 audit keys stay `index`, `error`, `updated`, `strength_before`, and `strength_after`. The raw strings are absent from the audit. `cue_fields` does not contain the audit and does not contain `replay_phase`.
8. Item. Every gate above passes.

An item passes when every gate passes. Fit must be 8/8 on each gate before these floors are locked. Held floors are 50/50 on each gate, including the item gate. A rate whose n is below 50 fails. Wilson 95% uses z = 1.959963984540054. The held canonical must match the fit canonical.

Cue-triggered floors stay the merged #57/#62 rates: prefer 46/50, avoid 37/50, avoid-stored 42/50, toward 47/50, away 47/50, coexist 50/50, RIF competitor 47/50, winner per event 50/50 items and 215/215 events, outsider 50/50, interference similarity 35/50, interference count 32/50, fan 45/50, and the direction gates from that rescore. Those floors were not moved.

Sleep gates stay the #63 held record: 50/50 on wake, sleep, replay, spontaneous recall, cue source, and the item, with the canonical pressures, drop, spontaneous rate, and replay strength equal to `reports/subconscious_sleep_held.json`. Arousal defaults stay 0.2 / 8 / 0.35 / 0.25 / 0.20 / 8. Reconstruct bank `interface_version` stays 6. The fusion scorer path is unchanged. The McGaugh consolidation path is unchanged. Pattern-separation retrieve ranking is unchanged.

Theta, half-life, lambda, fan S, the RIF span, the process S formulas, the prediction-error blend, and the arousal weights are unchanged. The phase is not an input to process S. No cue-export field is added.

The held bank was not scored.

The cue guard was not scored.

The sleep guard was not scored.

The arousal guard was not scored.

## Fit result

Unit tests lock the canonical surfaces and REM numbers. SWS phone spelling is `+1-617-555-4862`. SWS birthday spelling is `April 12, 1984`. REM valence is 1.35. REM bind cosine is 0.10497237569060777. Fit ids `sleep-phase-fit-000` through `sleep-phase-fit-007` all run the same fixtures and must be 8/8 on each gate before held is scored. These counts did not move a held floor.

## Held result

The held bank was not scored.
