# Retroactive-interference v0 pre-registration

Protocol `retroactive-interference-v0`. Fit ids are `retroactive-fit-000` through `retroactive-fit-007`. Held ids are `retroactive-held-000` through `retroactive-held-049`. The banks are disjoint. Items are numeric fixtures, not sentences. Bank sha256 `6ae2a76c48853badac5ba62e21434291242c7111744387884970e2d812a251a1`.

This is neuroscience gap #6 after fusion scorer #76 (`6d99996`), arousal gain #77 (`604236c`), pattern separation #78 (`a535648`), sleep two-phase #79 (`b5cb75e`), and reconsolidation window #80 (`0f4232d`) on main: clocks already age traces, but a new similar write does not weaken the older neighbor. This pull request increments older near-neighbor `age` on a successful new-slot store write. Residual detail already reads that age. Strength is not rewritten here, so `fusion_confidence` stays the #76 last-token setter path. It does not add a cue-export field. Reconstruct bank `canonical_surfaces()["interface_version"]` stays 6. Live `CUE_INTERFACE_VERSION` stays 7. This pull request is a draft and is not merged.

## Constants (HYPOTHESIS)

No grid was run. The numbers below are the hypothesis. They are not taken from a figure in the paper, and they were not retuned on held.

| name | value |
|---|---|
| tau | 0.90 |
| span | 0.25 |

`w = 0.25 * (sim - 0.90) / 0.10` when cosine is at least 0.90, else 0. `keep = 1 - w`. Cosine 1 is keep 0.75. Cosine at or below 0.90 is keep 1. When keep is below 1, the older slot's `age` increases by 1. The new slot is skipped. Same-identity overwrite and a surface-repeat merge do not run the pass. Cosine below 0.90 does not blanket-punish the store.

The collision band reuses pattern-separation `key_cosine` and `overlap_weight`. Dims stay 192 / 64 / 32. Sleep floor stays 0.1. Arousal baseline, tau, and window stay 0.2, 8, and 8. Theta stays 0.32. Half-life stays 30. Lambda stays 0.3. Cue interface stays 7. Reconstruct bank interface stays 6. `fusion_confidence` stays the #76 last-token setter path. `consolidation_gain` stays the #77 McGaugh path. Pattern-separation retrieve ranking stays the #78 view. Sleep two-phase operators stay the #79 SWS/REM passes. The reconsolidation window stays the #80 labile deadlines.

## Formula

1. Collision. Cosine at or above 0.90 is a near-neighbor.
2. Weight. `w = SPAN * (sim - TAU) / (1 - TAU)`. Cosine 1 is 0.25.
3. Keep. `keep = 1 - w`. Cosine 1 is 0.75. At or below tau, keep is 1.
4. When `keep < 1`, `age = age + 1` on the older neighbor. Strength, salience, residue parts, keys, values, and `consolidation_gain` stay.

Replay `recompute_decay` is still `decay ** age`. After one shared cycle the hit neighbor is `0.25` and the time-matched control is `0.5`. A refresh still sets the replayed slot's age to 0 and strength to 1. RIF still multiplies same-relation competitors after a retrieve win. Reconstructive spelling already maps age 1 onto a peripheral slip.

Locked canonical scores on the 32-d fixture: similar raw cosine `0.9950371902099893`, similar keep `0.7624070244750268`, exact keep `0.75`, similar age `1`, decay treated `0.25`, decay control `0.5`. Distinct pair keep stays 1.

## Citation

Verified on 2026-10-25 through the publisher page.

Wixted (2004), Annual Review of Psychology 55:235–269. The psychology and neuroscience of forgetting. The abstract reviews interference as a cause of forgetting and distinguishes laboratory cue-overload procedures from everyday interference during consolidation. Everyday forgetting is attributed more to retroactive interference from new learning than to unused time. https://doi.org/10.1146/annurev.psych.55.090902.141555

tidal does not simulate a consolidation-period laboratory procedure and does not add a second decay clock. The hit is that kind of new-learning weakening on the age tidal already stores. It is not unused-time decay, not retrieval-induced forgetting, not a reconsolidation window, and not a sleep gist phase. There is no growbrain switch and no feature flag.

## Gates

Locked before held:

1. Identity. Cosine below tau, cosine at tau, an empty store, or a lone write leaves age 0 and strength 1. Exact cosine 1 keep is 0.75.
2. Similar. The 32-d near-neighbor pair starts at cosine `0.9950371902099893`. Encoding B after A leaves A at age 1 and B at age 0. Strength stays 1 until decay.
3. Control. A time-matched store that never encodes B leaves A at age 0.
4. Orthogonal. Encoding distinct C after A leaves A at age 0. Distinct keep is 1.
5. Write. The new slot is not aged. Same-identity overwrite does not hit the remaining neighbor. Live `remember` stores the raw key.
6. Compose. After one `0.5 ** age` cycle the hit neighbor is `0.25` and the control is `0.5`. `fusion_confidence` on a text-only write stays 0. Baseline `consolidation_gain` stays 1. Pattern-separation ranking, McGaugh gain, sleep two-phase operators, and the labile window stay.
7. Export. Cue export `interface_version` is 7. Reconstruct bank `interface_version` is 6. No new cue key. `fusion_confidence` stays the last-token setter. Sleep floor is 0.1. Dims stay 192 / 64 / 32.
8. Item. Every gate above passes.

An item passes when every gate passes. Fit must be 8/8 on each gate before these floors are locked. Held floors are 50/50 on each gate, including the item gate. A rate whose n is below 50 fails. Wilson 95% uses z = 1.959963984540054. The held canonical must match the fit canonical.

Cue-triggered floors stay the merged #57/#62 rates: prefer 46/50, avoid 37/50, avoid-stored 42/50, toward 47/50, away 47/50, coexist 50/50, RIF competitor 47/50, winner per event 50/50 items and 215/215 events, outsider 50/50, interference similarity 35/50, interference count 32/50, fan 45/50, and the direction gates from that rescore. Those floors were not moved.

Sleep gates stay the #63 held record: 50/50 on wake, sleep, replay, spontaneous recall, cue source, and the item, with the canonical pressures, drop, spontaneous rate, and replay strength equal to `reports/subconscious_sleep_held.json`. Arousal defaults stay 0.2 / 8 / 0.35 / 0.25 / 0.20 / 8. Reconstruct bank `interface_version` stays 6. The fusion scorer path is unchanged. The McGaugh consolidation path is unchanged. Pattern-separation retrieve ranking is unchanged. The sleep two-phase operators are unchanged. The reconsolidation window is unchanged.

Theta, half-life, lambda, fan S, the RIF span, the process S formulas, and the arousal weights are unchanged. The hit is not an input to process S. No cue-export field is added.

The held bank was not scored.

The cue guard was not scored.

The sleep guard was not scored.

The arousal guard was not scored.

## Fit result

Unit tests lock the canonical keep and age values. Similar raw cosine is 0.9950371902099893. Similar keep is 0.7624070244750268. Similar age is 1. Decay treated is 0.25. Decay control is 0.5. Fit ids `retroactive-fit-000` through `retroactive-fit-007` all run the same fixtures and must be 8/8 on each gate before held is scored. These counts did not move a held floor.

## Held result

The held bank was not scored.
