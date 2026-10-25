# Pattern-separation v0 pre-registration

Protocol `dg-pattern-separation-v0`. Fit ids are `separation-fit-000` through `separation-fit-007`. Held ids are `separation-held-000` through `separation-held-049`. The banks are disjoint. Items are numeric fixtures, not sentences. Bank sha256 `9a32efc8e415732cbcdfaa58d5f04138d1802c245964753fc59348768465d600`.

This is neuroscience gap #2 after fusion scorer #76 and arousal gain #77 (`604236c` on main): the stack is CA3-like pattern-completion heavy and missing DG-like orthogonalization. This pull request adds a small anti-interference ranking view on existing key geometry. It does not add a cue-export field. Reconstruct bank `canonical_surfaces()["interface_version"]` stays 6. Live `CUE_INTERFACE_VERSION` stays 7. This pull request is a draft and is not merged.

## Constants (HYPOTHESIS)

No grid was run. The numbers below are the hypothesis. They are not taken from a figure in the papers, and they were not retuned on held.

| name | value |
|---|---|
| tau | 0.90 |
| span | 0.25 |

`w = 0.25 * (sim - 0.90) / 0.10` when cosine is at least 0.90, else 0. The incoming key is pushed along its unique part (or a deterministic complement when the traces are parallel) and then restored to its original L2 norm. Cosine below 0.90 is exactly the incoming vector. Empty neighbors are exactly the incoming vector. Equal keys stay equal after a symmetric pair re-encode.

Dims stay 192 / 64 / 32. Sleep floor stays 0.1. Arousal baseline, tau, and window stay 0.2, 8, and 8. Theta stays 0.32. Half-life stays 30. Lambda stays 0.3. Cue interface stays 7. Reconstruct bank interface stays 6. `fusion_confidence` stays the #76 last-token setter path. `consolidation_gain` stays the #77 McGaugh path. Live `remember` stores the raw key.

## Formula

The residual is a contrastive push on the stored key, which already lives at `d_model` (stage0: 192). The same function accepts `mem_key_dim` / fingerprint width 32. It does not write fingerprints.

1. Collision. Cosine at or above 0.90 is a near-neighbor.
2. Weight. `w = SPAN * (sim - TAU) / (1 - TAU)`. Cosine 1 is 0.25.
3. Unique part. Subtract the neighbor projection. A parallel residual uses a deterministic complementary axis.
4. `y = x + sum w * unit(unique) * ||x||`, then restore `||x||`.

Cue retrieve keeps theta, fan, and the candidate set on raw key cosine. Ranking and the exported gap read the separated cosine. Residual inject still uses the stored key. RIF similarity still uses the stored key.

Locked canonical scores on the 32-d fixture: similar raw cosine `0.9950371902099893`, similar separated cosine `0.8506285107986749`, confusion raw gap `0.002972141040810561`, confusion separated gap `0.016305700273454904`. Distinct pair cosine stays 0. Equal pair cosine stays 1.

## Citation

Verified on 2026-10-25 through the publisher pages.

Yassa and Stark (2011), Trends in Neurosciences 34(10):515–525. Pattern separation in the hippocampus. The abstract says the ability to discriminate among similar experiences is a crucial feature of episodic memory and has long been hypothesized to require the hippocampus, with computational models suggesting it is dependent on pattern separation. https://doi.org/10.1016/j.tins.2011.06.006

Wixted (2004), Annual Review of Psychology 55:235–269. The psychology and neuroscience of forgetting. The abstract reviews interference as a cause of forgetting and distinguishes laboratory cue-overload procedures from everyday interference during consolidation. https://doi.org/10.1146/annurev.psych.55.090902.141555

tidal does not simulate granule cells, mossy fibers, or adult neurogenesis. The residual is that kind of orthogonalization on the keys tidal already stores. It is not their circuit. It is not a reconsolidation window and not a sleep gist phase. There is no growbrain switch and no feature flag.

## Gates

Locked before held:

1. Identity. No neighbors, or every neighbor below tau, leaves the vector unchanged. Distinct orthogonal keys stay orthogonal. The L2 norm is restored.
2. Similar. The 32-d near-neighbor pair starts at cosine `0.9950371902099893` and ends strictly below that after `separate_keys`.
3. Confusion. A query closer to the first of those two traces has a raw cosine gap `0.002972141040810561`. The separated gap is strictly larger, `0.016305700273454904`. The winner stays the first slot.
4. Distinct. Orthogonal keys have raw cosine 0 and separated cosine 0. `match_scores` equals raw cosine.
5. Equal. Two identical keys stay cosine 1 after a symmetric pair re-encode, so the equal-key cue tie is unchanged.
6. Write. Live `remember` stores the raw `encode_key` vector. `store.write` still stores the caller key. `fusion_confidence` on a text-only write stays 0. Baseline `consolidation_gain` stays 1. Residual `s` is unchanged.
7. Export. Cue export `interface_version` is 7. Reconstruct bank `interface_version` is 6. No new cue key. `fusion_confidence` stays the last-token setter. Sleep floor is 0.1. Dims stay 192 / 64 / 32.
8. Item. Every gate above passes.

An item passes when every gate passes. Fit must be 8/8 on each gate before these floors are locked. Held floors are 50/50 on each gate, including the item gate. A rate whose n is below 50 fails. Wilson 95% uses z = 1.959963984540054. The held canonical must match the fit canonical.

Cue-triggered floors stay the merged #57/#62 rates: prefer 46/50, avoid 37/50, avoid-stored 42/50, toward 47/50, away 47/50, coexist 50/50, RIF competitor 47/50, winner per event 50/50 items and 215/215 events, outsider 50/50, interference similarity 35/50, interference count 32/50, fan 45/50, and the direction gates from that rescore. Those floors were not moved.

Sleep gates stay the #63 held record: 50/50 on wake, sleep, replay, spontaneous recall, cue source, and the item, with the canonical pressures, drop, spontaneous rate, and replay strength equal to `reports/subconscious_sleep_held.json`. Arousal defaults stay 0.2 / 8 / 0.35 / 0.25 / 0.20 / 8. Reconstruct bank `interface_version` stays 6. The fusion scorer path is unchanged. The McGaugh consolidation path is unchanged.

Theta, half-life, lambda, fan S, the RIF span, the process S formulas, and the arousal weights are unchanged. The residual is not an input to process S. No cue-export field is added.

The held bank was not scored.

The cue guard was not scored.

The sleep guard was not scored.

The arousal guard was not scored.

## Fit result

Unit tests lock the canonical scores. Similar raw cosine is 0.9950371902099893. Similar separated cosine is 0.8506285107986749. Confusion raw gap is 0.002972141040810561. Confusion separated gap is 0.016305700273454904. Fit ids `separation-fit-000` through `separation-fit-007` all run the same fixtures and must be 8/8 on each gate before held is scored. These counts did not move a held floor.

## Held result

The held bank was not scored.
