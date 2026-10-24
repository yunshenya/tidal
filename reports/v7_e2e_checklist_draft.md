# v7 e2e checklist (draft for 两仪式)

Cross-repo first draft after tidal live export is 7. Fill gaps on growbrain.

## Packet / adapter

- [ ] `TidalPacket.from_dict` accepts `fingerprint` float32 `(32,)` and `fusion_confidence` float in `[0, 1]`.
- [ ] Missing `fingerprint` on an older packet may still parse; new tidal exports always include both fields.
- [ ] `fusion_weights` is **not** a cue-packet key. Write via `set_fusion_weights(weights, *, pair=None)`, `(3,)` order text/audio/image.
- [ ] `modality` enum is `text | audio | image | fused | none`. Unknown is rejected (or mapped to `none` only if you choose that on growbrain; tidal rejects unknown).
- [ ] Keep `B=1` on the single-packet path. Last-token rule for gate scalars, including `fusion_confidence`. `fingerprint` is memory-level, not per-token.

## Defaults / known cross-repo limits

- [ ] tidal text-only writes store `zero_fingerprint()` and modality `text` until growbrain sends real fingerprints.
- [ ] Zero-fusion path: tidal emits `fusion_confidence=0`.
- [ ] `sleep_pressure` missing → 0 on growbrain vs tidal floor 0.1. Known e2e note; do **not** change the tidal floor.
- [ ] `arousal` missing → 0.2 (tidal baseline).
- [ ] `TickRuntime` `SignalBundle` path still consumes last-token valence / sleep_pressure / arousal / circadian; add `fusion_confidence` the same way if the gate reads it.

## Version bump

- [ ] After this tidal PR merges, bump growbrain `TIDAL_INTERFACE` live to 7.
- [ ] Do not keep a dual-version fake packet with empty reserved keys.

## Out of scope here

Full fusion / sleep pairing, inject/key_center retrain, backend dim changes (192/64/32), tidal `sleep_pressure` floor, feature flags.
