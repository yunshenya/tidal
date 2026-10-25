# Fusion scorer v0 pre-registration

Protocol `v7-fusion-scorer-v0`. Fit ids are `fusion-fit-000` through `fusion-fit-007`. Held ids are `fusion-held-000` through `fusion-held-049`. The banks are disjoint. Items are numeric fixtures, not sentences. Bank sha256 `98bef28c3b4f687396f3405cc36d547715e85df49eaf2bd915916efb3bfd90f1`.

This is the first live writer of `fusion_confidence > 0`. growbrain parses the last-token scalar into `SignalBundle` and does not consume it yet. Outflow is allowed. The scorer does not overwrite growbrain `fusion_weights`. Reconstruct bank `canonical_surfaces()["interface_version"]` stays 6. Live `CUE_INTERFACE_VERSION` stays 7. This pull request is a draft and is not merged.

## Constants (HYPOTHESIS)

No grid was run. The numbers below are the hypothesis. They are not taken from a figure, and they were not retuned on held.

| name | value |
|---|---|
| window | 4 store steps |
| fingerprint floor | 0.25 |
| time prior | 0.55 |
| causal tau | 0.20 |
| causal floor | 0.15 |
| reliability saturation | 2.0 |

Dims stay 192 / 64 / 32. Sleep floor stays 0.1. Arousal baseline, tau, and window stay 0.2, 8, and 8. Theta stays 0.32. Half-life stays 30. Lambda stays 0.3. Cue interface stays 7. Reconstruct bank interface stays 6.

## Formula

Reliability is an Ernst and Banks inverse-noise proxy. Presence is 1 when the fingerprint is non-zero and 0 for the zero placeholder. Magnitude is not calibrated.

`r = max(strength, 0) * (1 + max(salience, 0)) * (0.25 + 0.75 * presence)`

Per source modality (`text`, `audio`, `image`) keep the max `r`. `fused` and `none` are not sources. Fewer than two source modalities yields `c = 0`.

If growbrain `fusion_weights` has positive mass on a present source, `r_eff = r * max(w, 0)`. Otherwise the suggested mix is `r` itself. Combined precision `R = sum r_eff`. The reliability term is `min(1, R / 2)`. Two unit-reliability sources saturate at 1. Suggested weights are internal. They are not written with `set_fusion_weights`.

Causal soft gate is Körding-style. Missing fingerprints use the time prior 0.55. Matching fingerprints give `P(C=1) = 1`. Opposite fingerprints give 0. If `P(C=1) < 0.15`, return 0.

`c = P(C=1) * reliability_term`, finite, in `[0, 1]`. `1.0` is allowed. `0` means no fusion.

Locked canonical scores: matching text+audio fingerprints `1.0`; time-only zero fingerprints `0.1375`; growbrain weights `[1.0, 0.25, 0.0]` `0.625`; conflict, single, empty, and outside-window `0`.

## Live hook

`tidal/fusion.py` is the pure function. `LLMMemory.refresh_fusion_confidence` reads existing store slots and calls `set_fusion_confidence`. `remember` calls that refresh after an accepted write. `cue_fields` does not rescore; it exports the last setter write on the last token, the same path as #75. Fixtures that write through `store.write` call `refresh_fusion_confidence`. The repo cannot observe growbrain audio/image encoders end-to-end here. The fixtures write labeled modalities and fingerprints into the existing store.

## Citations

Ernst and Banks (2002), Nature 415:429–433. Humans integrate visual and haptic information in a statistically optimal fashion. https://doi.org/10.1038/415429a

Körding, Beierholm, Ma, Quartz, Tenenbaum, and Shams (2007), PLoS Biology 5(9):e260. Causal inference in multisensory perception. https://doi.org/10.1371/journal.pbio.0050260

The window is a hypothesis in store-step units. It is not a millisecond superior-colliculus window. Sleep pairing, pattern completion of learned pairs, and growbrain semantic new pairings stay out of scope.

## Gates

Locked before held:

1. `v7-fusion-scorer-multi`. Matching text and audio fingerprints in the window produce `c` in `(0, 1]`. The locked value is `1.0`.
2. `v7-fusion-scorer-single`. One source modality produces `0`.
3. `v7-fusion-scorer-empty`. No observations produce `0`.
4. `v7-fusion-scorer-causal`. Opposite fingerprints produce `0`. Time-only coincidence is strictly below the matching score.
5. `v7-fusion-scorer-time`. Zero-fingerprint text+audio coincidence produces `0.1375`, which is in `(0, 1)` and below `1.0`.
6. `v7-fusion-scorer-window`. A pair separated by more than 4 store steps produces `0`.
7. `v7-fusion-scorer-range`. Matching may be `1.0`. Weighted matching stays in `(0, 1]`. NaN and Inf stay rejected by `set_fusion_confidence`.
8. `v7-fusion-scorer-weights`. growbrain weights `[1.0, 0.25, 0.0]` change `c` to `0.625` and are not overwritten.

An item passes when every gate passes. Fit must be 8/8 on each gate before these floors are locked. Held floors are 50/50 on each gate, including the item gate. A rate whose n is below 50 fails. Wilson 95% uses z = 1.959963984540054. The held canonical must match the fit canonical.

Cue-triggered floors stay the merged #57/#62 rates: prefer 46/50, avoid 37/50, avoid-stored 42/50, toward 47/50, away 47/50, coexist 50/50, RIF competitor 47/50, winner per event 50/50 items and 215/215 events, outsider 50/50, interference similarity 35/50, interference count 32/50, fan 45/50, and the direction gates from that rescore. Those floors were not moved.

Sleep gates stay the #63 held record. Arousal defaults stay 0.2 / 8 / 0.35 / 0.25 / 0.20 / 8. Reconstruct bank `interface_version` stays 6.

The held bank was not scored.

The cue guard was not scored.

The sleep guard was not scored.

## Fit result

Unit tests lock the canonical scores above. Fit ids `fusion-fit-000` through `fusion-fit-007` all run the same fixtures and must be 8/8 on each gate before held is scored. These counts did not move a held floor.

## Held result

The held bank was not scored.
