# Weak-cue / spontaneous recollection pre-registration

Protocol `weak-cue-recollection-v0`. Fit ids are `weak-cue-fit-000` through `weak-cue-fit-007`. Held ids are `weak-cue-held-000` through `weak-cue-held-049`. The banks are disjoint. Items are numeric schedules plus the locked fragment cosine, not free sentences. Bank sha256 `73ecc2a57e111f2234d5c978c6117001802ae92f5eb43b237ec25654625bfc10`.

This is human-likeness queue item #3 after encode/consolidate mode #82 (`dbe6e42`) and retroactive interference #81 (`8e7c37a`) on main: the stack is still almost only explicit cue-triggered. Process S already admits a low-rate offline tick, and encode mode already lowers that pressure, but the fire does not read a weak internal overlap. Berntsen (2020) says most involuntary autobiographical memories still have a cue — a weak internal or external overlap — and cannot be equated with mind-wandering. This pull request makes a partial key, a residual, or a recent context fragment able to surface a stored slot at that same low rate, stronger in consolidate mode than encode mode. It does not add a cue-export field. It does not make a default-mode network a first-class citizen. Reconstruct bank `canonical_surfaces()["interface_version"]` stays 6. Live `CUE_INTERFACE_VERSION` stays 7. This pull request is a draft and is not merged.

## Constants (HYPOTHESIS)

No grid was run. The numbers below are the hypothesis. They are not taken from a figure in the paper, and they were not retuned on held.

| name | value |
|---|---|
| weak floor | 0.10 |
| weak overlap | 0.25 (`CLOCK_SPAN`; same family as clock / McGaugh / pattern separation) |
| weak ceiling | 0.32 (live cue theta; exclusive) |

`context_fragment(x)` mixes `x` with its deterministic complement so cosine is 0.25. That is below theta, so it is not a strong cue hit. Cosine at or above 0.10 and strictly below 0.32 is the weak band. Cosine 1 stays a cue-path hit. Cosine 0 is a miss.

`remember` plants that partial-key fragment. A cue trace plants the last cue row as a recent context fragment. `offline_ticks` still fires only on the shipped period 16 when mode-scaled process S is at least 0.2. Encode multiplies process S by 0.5 before that floor, so after wake duration 4 the product is `0.1495396476178678` and the tick is silent. Consolidate leaves pressure equal to process S. Tick 0 fires, so 16 ticks fire once. The rate is 0.0625, above 0 and below 0.10.

A weak-band fire exports `source=spontaneous`, `triggered=False`, and `winner_cosine` in the weak band. A strong cue exports `source=cue`, `triggered=True`, and `winner_cosine` at or above 0.32. No new packet field is added.

`sleep_for` still schedules SWS/REM from process S and the store circadian phase. The fragment does not reimplement those operators and does not block them. An encode-mode `sleep_for(1)` after wake duration 4 still ages the slot by 1.

Dims stay 192 / 64 / 32. Sleep floor stays 0.1. Arousal baseline, tau, and window stay 0.2, 8, and 8. Theta stays 0.32. Half-life stays 30. Lambda stays 0.3. Cue interface stays 7. Reconstruct bank interface stays 6. `fusion_confidence` stays the #76 last-token setter path. `consolidation_gain` stays the #77 McGaugh path. Pattern-separation retrieve ranking stays the #78 view. Sleep two-phase operators stay the #79 SWS/REM passes. The reconsolidation window stays the #80 labile deadlines. The #81 retroactive age-hit still runs on a successful new-slot write. Encode vs consolidate write keeps and the encode spontaneous product stay the #82 values.

## Citation

Verified on 2026-10-25 through the publisher page.

Berntsen (2020), Philosophical Transactions of the Royal Society B 376:20190693. Involuntary autobiographical memory. The paper describes memories of personal events that come to mind with no conscious initiation of retrieval. It says involuntary autobiographical memory cannot be equated with mind-wandering, and that such memories typically still have a cue — often a weak overlap with current internal or external context, not a strong deliberate probe. https://doi.org/10.1098/rstb.2019.0693

tidal does not simulate a default-mode network, mind-wandering, or cue-less magic. The fragment is that kind of weak overlap on the offline tick tidal already owns. There is no growbrain switch and no feature flag.

## Gates

Locked before held:

1. Identity. Fragment cosine is 0.25. Weak floor is 0.10. Weak ceiling is 0.32. Period is 16. Rate is 0.0625.
2. Weak. Cosine 0.25 is in the band. Cosine 0.10 is in the band. Cosine 0.32 is not. Cosine 0 is not. A low-salience slot with weak overlap beats a high-salience miss.
3. Strong. Cosine 1 is not in the band and stays a cue-path hit. A mixed `[1.0, 0.25]` list picks the weak slot for spontaneous ranking, not the strong one.
4. Consolidate. Wake duration 4 at baseline arousal and phase 0 is consolidate. Spontaneous pressure equals process S and is at least 0.2. `apply_sleep_operators` is true.
5. Encode. Above-baseline arousal after that wake is encode. Encode spontaneous pressure is `0.1495396476178678`, below 0.2. `apply_sleep_operators` is true.
6. Compose. SWS/REM stay on `sleep_phase`. Process S at the floor with baseline arousal is encode. Nobody in the band returns no weak winner, so the shipped salience ranking stays. `fusion_confidence`, McGaugh gain, pattern-separation ranking, the #81 age-hit, and the labile window stay.
7. Export. Cue export `interface_version` is 7. Reconstruct bank `interface_version` is 6. The fragment is not a cue key. Sleep floor is 0.1. Dims stay 192 / 64 / 32.
8. Item. Every gate above passes.

An item passes when every gate passes. Fit must be 8/8 on each gate before these floors are locked. Held floors are 50/50 on each gate, including the item gate. A rate whose n is below 50 fails. Wilson 95% uses z = 1.959963984540054. The held canonical must match the fit canonical.

Cue-triggered floors stay the merged #57/#62 rates: prefer 46/50, avoid 37/50, avoid-stored 42/50, toward 47/50, away 47/50, coexist 50/50, RIF competitor 47/50, winner per event 50/50 items and 215/215 events, outsider 50/50, interference similarity 35/50, interference count 32/50, fan 45/50, and the direction gates from that rescore. Those floors were not moved.

Sleep gates stay the #63 held record: 50/50 on wake, sleep, replay, spontaneous recall, cue source, and the item, with the canonical pressures, drop, spontaneous rate, and replay strength equal to `reports/subconscious_sleep_held.json`. Arousal defaults stay 0.2 / 8 / 0.35 / 0.25 / 0.20 / 8. Reconstruct bank `interface_version` stays 6. The fusion scorer path is unchanged. The McGaugh consolidation path is unchanged. Pattern-separation retrieve ranking is unchanged. The sleep two-phase operators are unchanged. The reconsolidation window is unchanged. The retroactive age-hit is unchanged. The encode vs consolidate keeps are unchanged.

Theta, half-life, lambda, fan S, the RIF span, the process S formulas, the prediction-error blend, and the arousal weights are unchanged. The fragment is not an input to process S. No cue-export field is added.

The held bank was not scored.

The cue guard was not scored.

The sleep guard was not scored.

The arousal guard was not scored.

## Fit result

Unit tests lock the canonical fragment cosine and mode-scaled pressure. Weak overlap is 0.25. Encode spontaneous pressure after wake duration 4 is `0.1495396476178678`. Fit ids `weak-cue-fit-000` through `weak-cue-fit-007` all run the same fixtures and must be 8/8 on each gate before held is scored. These counts did not move a held floor.

## Held result

The held bank was not scored.
