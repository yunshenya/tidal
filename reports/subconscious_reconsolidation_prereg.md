# Reconsolidation-window pre-registration

Protocol `reconsolidation-window-v0`. Fit ids are `reconsolidation-fit-000` through `reconsolidation-fit-007`. Held ids are `reconsolidation-held-000` through `reconsolidation-held-049`. The banks are disjoint. Items are numeric schedules plus the locked reconstruct phone and lure, not free sentences. Bank sha256 `abb2518688c5f51bff8d022afac5cc1970642fab503d51932abcec5d6b8df83d`.

This is neuroscience gap #5 after fusion scorer #76 (`6d99996`), arousal gain #77 (`604236c`), pattern separation #78 (`a535648`), and sleep two-phase #79 (`b5cb75e`) on main: retrieval already happens, but the store has no post-retrieval labile window. This pull request stamps that window on the matched slot after a cue hit, then lets the existing reconstructive residue path update or distort while it is open. After the window closes, the same write-back no-ops. It does not add a cue-export field. Reconstruct bank `canonical_surfaces()["interface_version"]` stays 6. Live `CUE_INTERFACE_VERSION` stays 7. This pull request is a draft and is not merged.

## Constants (HYPOTHESIS)

No grid was run. The numbers below are the hypothesis. They are not taken from a figure in the paper, and they were not retuned on held.

| name | value |
|---|---|
| labile tokens | 8 |
| labile wall | 4.0 s |

The window closes when the token budget expires or wall seconds expire. That is the stronger constraint, the same `min` composition the cue decay already uses. A frozen wall clock leaves the token budget. The spans are abstract units. They are not the paper's six hours.

A cue hit stamps `labile_until_token` and `labile_until_wall` on every slot that cleared theta on a triggered token. A miss, or a forward with no store, does not stamp. `write_back` writes a reconstructed or supplied phone/birthday surface onto `answer` and `residue` only while both clocks remain open. Keys, values, `text`, and strength stay. This is not a growbrain overwrite API. `remember` still writes new facts.

Dims stay 192 / 64 / 32. Sleep floor stays 0.1. Arousal baseline, tau, and window stay 0.2, 8, and 8. Theta stays 0.32. Half-life stays 30. Lambda stays 0.3. Cue interface stays 7. Reconstruct bank interface stays 6. `fusion_confidence` stays the #76 last-token setter path. `consolidation_gain` stays the #77 McGaugh path. Pattern-separation retrieve ranking stays the #78 view. Sleep two-phase operators stay the #79 SWS/REM passes. Prediction-error replay stays the #63 blend, threshold, and strength step.

## Citation

Verified on 2026-10-25 through the publisher page.

Nader, Schafe, and LeDoux (2000), Nature 406:722–726. Fear memories require protein synthesis in the amygdala for reconsolidation after retrieval. The abstract says consolidated fear memories, when reactivated during retrieval, return to a labile state in which infusion of anisomycin shortly after reactivation produces amnesia on later tests. The same treatment without reactivation left memory intact. Delay of the infusion until six hours after reactivation produced no amnesia. https://doi.org/10.1038/35021052

tidal does not infuse anisomycin and does not model the lateral amygdala. The window is that kind of post-retrieval lability on the residue machinery tidal already owns. There is no growbrain switch and no feature flag.

## Gates

Locked before held:

1. Identity. A never-opened stamp is closed. `write_back` on a closed slot is a no-op and leaves `+1-617-555-4826`. Exact reconstruct of that phone is itself. Deadlines from token 0 / wall 0 are 8 and 4.0.
2. Hit. A cue that clears theta opens the window on the matched slot. A triggered `cue_fields("A")` against a planted phone key leaves `labile(0)` true.
3. Update. While the window is open, `write_back(0, "+1-617-555-9090")` sticks. `answer` and residue become the lure. `text` and strength stay.
4. Token. After 8 miss tokens the window is closed. The same write-back is a no-op and the lure stays. Seven miss tokens would still be open on the pure clocks.
5. Wall. After `advance_wall(4.0)` the window is closed. The same write-back is a no-op. A 3-second tick would still be open on the pure clocks.
6. Miss. `cue_fields("B")` against the planted "A" key does not open the window. A no-cue empty-store forward, then a later plant, does not open it. A non-phone surface is refused.
7. Export. Cue export `interface_version` is 7. Reconstruct bank `interface_version` is 6. `labile` is not a cue key. `fusion_confidence` stays the last-token setter. Sleep floor is 0.1. Dims stay 192 / 64 / 32.
8. Item. Every gate above passes.

An item passes when every gate passes. Fit must be 8/8 on each gate before these floors are locked. Held floors are 50/50 on each gate, including the item gate. A rate whose n is below 50 fails. Wilson 95% uses z = 1.959963984540054. The held canonical must match the fit canonical.

Cue-triggered floors stay the merged #57/#62 rates: prefer 46/50, avoid 37/50, avoid-stored 42/50, toward 47/50, away 47/50, coexist 50/50, RIF competitor 47/50, winner per event 50/50 items and 215/215 events, outsider 50/50, interference similarity 35/50, interference count 32/50, fan 45/50, and the direction gates from that rescore. Those floors were not moved.

Sleep gates stay the #63 held record: 50/50 on wake, sleep, replay, spontaneous recall, cue source, and the item, with the canonical pressures, drop, spontaneous rate, and replay strength equal to `reports/subconscious_sleep_held.json`. Arousal defaults stay 0.2 / 8 / 0.35 / 0.25 / 0.20 / 8. Reconstruct bank `interface_version` stays 6. The fusion scorer path is unchanged. The McGaugh consolidation path is unchanged. Pattern-separation retrieve ranking is unchanged. The sleep two-phase operators are unchanged.

Theta, half-life, lambda, fan S, the RIF span, the process S formulas, the prediction-error blend, and the arousal weights are unchanged. The window is not an input to process S. No cue-export field is added.

The held bank was not scored.

The cue guard was not scored.

The sleep guard was not scored.

The arousal guard was not scored.

## Fit result

Unit tests lock the canonical window and write-back surfaces. Deadlines from token 0 / wall 0 are 8 and 4.0. Exact reconstruct is `+1-617-555-4826`. The lure is `+1-617-555-9090`. A slip write-back is `+1-617-555-4862`. Fit ids `reconsolidation-fit-000` through `reconsolidation-fit-007` all run the same fixtures and must be 8/8 on each gate before held is scored. These counts did not move a held floor.

## Held result

The held bank was not scored.
