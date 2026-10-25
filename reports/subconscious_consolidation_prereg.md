# McGaugh consolidation-gain pre-registration

Protocol `mcgaugh-consolidation-v1`. Fit ids are `consolidation-fit-000` through `consolidation-fit-007`. Held ids are `consolidation-held-000` through `consolidation-held-049`. The banks are disjoint. Items are numeric schedules, not sentences. Bank sha256 `9396a6a2cb463e4ed0f2f746dbaff3101ca913a06d56f9e437fa9668a666c66b`.

This is neuroscience gap #4 after fusion scorer #76 (`6d99996` on main): emotional arousal, not valence sign, modulates consolidation. The live LC-NE scalar and slot valence already exist. This pull request writes their product into write strength and sleep replay priority. It does not add a cue-export field. Reconstruct bank `canonical_surfaces()["interface_version"]` stays 6. Live `CUE_INTERFACE_VERSION` stays 7. This pull request is a draft and is not merged.

## Constants (HYPOTHESIS)

No grid was run. The numbers below are the hypothesis. They are not taken from a figure in the paper, and they were not retuned on held.

| name | value |
|---|---|
| baseline | 0.2 |
| span | 0.25 |

`gain = 1 + 0.25 * max(0, (arousal - 0.2) / 0.8) * tanh(|valence|)`. Baseline arousal or zero valence is exactly 1. Range of the added term is `[0, 0.25]`. The missing growbrain arousal stub is the live baseline 0.2, so a write at rest does not invent emotional consolidation.

Arousal levels are 0.2, 0.5, and 1.0. Valence magnitudes are 0, 0.5, and 1.5. Sign is checked at `±1.5` with arousal 1.0. Write checks use ambient arousal captured before `observe_valence` / `observe_surprise` of that write. Sleep ranking adds `consolidation_gain` after strength. A gain of 1 matches the old `(salience, strength, -index)` order.

Dims stay 192 / 64 / 32. Sleep floor stays 0.1. Arousal baseline, tau, and window stay 0.2, 8, and 8. Theta stays 0.32. Half-life stays 30. Lambda stays 0.3. Cue interface stays 7. Reconstruct bank interface stays 6. `fusion_confidence` stays the #76 last-token setter path.

## Citation

Verified on 2026-10-25 through the Crossref works API.

McGaugh (2004), Annual Review of Neuroscience 27:1–28. The amygdala modulates the consolidation of memories of emotionally arousing experiences. The abstract says the degree of amygdala activation by emotional arousal during encoding of emotionally arousing material, either pleasant or unpleasant, correlates highly with subsequent recall. https://doi.org/10.1146/annurev.neuro.27.070203.144157

tidal does not model BLA infusions or adrenal stress hormones. The gain is that kind of post-encoding modulator on the scalars tidal already computes. It is not their circuit. There is no growbrain switch and no feature flag.

## Gates

Locked before held:

1. Identity. Arousal 0.2 with valence 0, 0.5, or 1.5 is exactly 1. Arousal 0.5 or 1.0 with valence 0 is exactly 1. Arousal 0 with valence 1.5 is exactly 1. A rest write with a nonzero experience leaves `slot.strength` equal to the clock-only strength and `slot.consolidation_gain` equal to 1.
2. Arousal monotone. At valence 1.5, arousal 0.2 ends below 0.5, which ends below 1.0. Both raised values stay in `(1, 1.25]`.
3. Valence monotone. At arousal 1.0, valence 0 ends below 0.5, which ends below 1.5. Both raised values stay in `(1, 1.25]`.
4. Sign. Arousal 1.0 with valence +1.5 equals valence −1.5. Arousal 0.5 with valence +0.5 equals valence −0.5.
5. Write. Ambient arousal 1.0 and experience step 0.25 writes `slot.consolidation_gain` strictly above 1, equal to `consolidation_gain(1.0, 0.25)`. Residual `slot.strength` stays the clock-only value. Ambient arousal 1.0 and valence 0 leaves both the gain and the strength at 1 when the clock is 1. The call does not move process S. `fusion_confidence` on a text-only write stays 0.
6. Replay. `spontaneous_winner` with missing priority matches the old triple, including `[0.1, 0.4, 0.4]` / `[1.0, 0.2, 0.9]` → index 2. Equal salience and strength, priorities `1.0` and `1.2262870634112166`, selects the higher priority. Two gains of 1 keep the old index-0 tie.
7. Export. Cue export `interface_version` is 7. Reconstruct bank `interface_version` is 6. `consolidation_gain` is not a cue key. `fusion_confidence` stays the last-token setter. Sleep floor is 0.1. Dims stay 192 / 64 / 32.
8. Item. Every gate above passes.

An item passes when every gate passes. Fit must be 8/8 on each gate before these floors are locked. Held floors are 50/50 on each gate, including the item gate. A rate whose n is below 50 fails. Wilson 95% uses z = 1.959963984540054. The held canonical must match the fit canonical.

Cue-triggered floors stay the merged #57/#62 rates: prefer 46/50, avoid 37/50, avoid-stored 42/50, toward 47/50, away 47/50, coexist 50/50, RIF competitor 47/50, winner per event 50/50 items and 215/215 events, outsider 50/50, interference similarity 35/50, interference count 32/50, fan 45/50, and the direction gates from that rescore. Those floors were not moved.

Sleep gates stay the #63 held record: 50/50 on wake, sleep, replay, spontaneous recall, cue source, and the item, with the canonical pressures, drop, spontaneous rate, and replay strength equal to `reports/subconscious_sleep_held.json`. Arousal defaults stay 0.2 / 8 / 0.35 / 0.25 / 0.20 / 8. Reconstruct bank `interface_version` stays 6. The fusion scorer path is unchanged.

Theta, half-life, lambda, fan S, the RIF span, the process S formulas, and the arousal weights are unchanged. The gain is not an input to process S. `store.clear` does not move arousal. `_on_layer` still does not step arousal. No cue-export field is added.

The held bank was not scored.

The cue guard was not scored.

The sleep guard was not scored.

The arousal guard was not scored.

## Fit result

Unit tests lock the canonical gains. After baseline schedules the gain is 1.0. Arousal 0.5 with valence 0.5 is 1.043323483493126. Arousal 0.5 with valence 1.5 is 1.0848576487792063. Arousal 1.0 with valence 0.5 is 1.1155292893150024. Arousal 1.0 with valence 1.5 is 1.2262870634112166. Fit ids `consolidation-fit-000` through `consolidation-fit-007` all run the same fixtures and must be 8/8 on each gate before held is scored. These counts did not move a held floor.

## Held result

The held bank was not scored.
