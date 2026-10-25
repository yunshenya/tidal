# v7 e2e checklist (draft, aligned with growbrain)

Mirror of growbrain `reports/v7_e2e_checklist.md`. Growbrain v7 already landed (`c12e4e6` wire + `951ac6b` field table). Live tidal rows are this PR. Do not change backend dims (192/64/32) or the tidal sleep floor.

| Row | Growbrain (landed) | Tidal (this PR) |
|---|---|---|
| Live version | `INTERFACE_VERSION=7` | `CUE_INTERFACE_VERSION=7`. Exporters emit 7 only. No dual-version packet with empty reserved keys. |
| `fingerprint` | `from_dict` accepts float32 `(32,)` | Store + cue winner emit float32 `(32,)`. Text-only writes store `zero_fingerprint()`. tidal does not compute fingerprints. |
| `fusion_confidence` | `from_dict` accepts scalar in `[0, 1]` | Cue packet sample-level / last-token scalar. Zero-fusion path emits `0`. `LLMMemory.refresh_fusion_confidence` writes a known value in `(0, 1]` through `set_fusion_confidence`. growbrain may parse it into `SignalBundle` and does not consume it yet. |
| `modality` | `from_dict` accepts `text\|audio\|image\|fused\|none` | Store + per-token export use the same enum. Unknown is rejected (`check_modality`). Text-only writes store `text`. |
| `fusion_weights` / pairing | `TidalWrite(transport)` callable. No homemade HTTP. | `LLMMemory.set_fusion_weights(weights, *, pair=None)`. float32 `(3,)`, order text/audio/image. **Not** a `cue_fields` key. Transport on growbrain calls this write. |
| v6 compat | Kept on growbrain | Exporters emit 7. Older packets that lack v7 keys may still be read; new exports always include `fingerprint` + `fusion_confidence`. Reconstruct bank `canonical_surfaces()["interface_version"]` stays 6 (frozen measurement). |
| `sleep_pressure` missing | → `0` | Tidal floor stays `0.1`. Known cross-repo note. Do **not** change the tidal floor. |
| `arousal` missing | → `0.2` | Tidal baseline stays `0.2`. |
| `TickRuntime` / `SignalBundle` | Last-token valence / sleep_pressure / arousal / circadian | Same last-token rule for `fusion_confidence`. `fingerprint` is memory-level, not per-token. `B=1` on the single-packet path (#69). |

## Still open (cross-repo)

- [ ] Dual-checkout e2e: growbrain `from_dict` on a live tidal `cue_fields` packet (fingerprint `(32,)`, `fusion_confidence`, modality, `interface_version` 7).
- [ ] Dual-checkout e2e: growbrain `TidalWrite(transport)` → tidal `set_fusion_weights` (and optional `pair`). Confirm `fusion_weights` is absent from the cue packet.
- [ ] Confirm missing-`sleep_pressure` on growbrain stays 0 while tidal still exports ≥ 0.1.
- [ ] Confirm missing-`arousal` on growbrain stays 0.2.

## Fusion scorer v0 (this PR)

Protocol `v7-fusion-scorer-v0`. Pure function in `tidal/fusion.py`. Live hook: `remember` → `refresh_fusion_confidence` → `set_fusion_confidence`. `store.write` fixtures call refresh themselves. `fusion_weights` stay growbrain-authoritative. Reconstruct bank stays 6. Draft PR only.

Gates: `v7-fusion-scorer-multi`, `v7-fusion-scorer-single`, `v7-fusion-scorer-empty`, `v7-fusion-scorer-causal`, `v7-fusion-scorer-time`, `v7-fusion-scorer-window`, `v7-fusion-scorer-range`, `v7-fusion-scorer-weights`.

## Out of scope

Sleep pairing, growbrain semantic new pairings, inject/key_center retrain, backend dim changes, tidal `sleep_pressure` floor, feature flags, Origin/growbrain package imports.
