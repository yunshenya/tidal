# REPORT: Round 23 — joint CE fine-tune (retr + inj)

Held-out **test** split, n=96 free-gen match. Frozen A2s backbone `mem_ckpt_v13A2s.pt` + slot store softk4 C256. Scripts: `train_v23.py`, `run_v23.py`; weights `retr_v23.pt`, `inj_v23.pt`; results `results_v23.json` / `results_v23_train.json`. Logs: `v23train.out`, `v23.out`, `v23.log`. Smoke: `v23smoke.out`, `retr_v23_smoke.pt`, `inj_v23_smoke.pt`.

## Method

**ONE change:** joint teacher-forced next-token CE through the full read path
`retrieve → softk4 (τ=0.05) → scale → identity inj + residual MLP`, with **both** retrieval adapters and inject MLP trainable.

- Init: `retr_v22.pt` + `inj_v21.pt`. Backbone frozen.
- Cache 800 train pools (single / session N∈{0…32} / long K∈{8…60}); encodings of k/v/q frozen.
- Softk4 over slots is differentiable (top-4 softmax); 15% oracle-v mix to stabilize inject.
- AdamW: lr_retr=1e-4, lr_inj=3e-4, 1200 steps, bs=4. No capacity / RAG / key_shield / data-format changes.
- Eval configs: `slot_retr_v22_softk4_inj_C256` (frozen v22 stack) vs `slot_retr_v23_softk4_inj_C256`.

Train proxy (same 48/64 cache facts): soft gen **0.688 → 0.781**, soft ret@1 **0.917 → 1.0**, oracle gen **0.708 → 0.766**. Smoke (80 steps) also rose 0.688→0.75 — no smoke regression.

## Gen accuracy (ret@1)

| config | N0 | N4 | N16 | N32 |
|---|---|---|---|---|
| **v22** retr_v22+inj_v21 softk4 | 0.656 [1.00] | 0.573 [0.979] | **0.604** [**0.906**] | 0.594 [**0.875**] |
| **v23** joint CE softk4 | 0.656 [1.00] | **0.625** [**0.990**] | 0.583 [0.875] | **0.615** [0.865] |
| bge RAG (from SUMMARY) | 0.604 | 0.604 | 0.552 | **0.635** |

Mean gen over N0/4/16/32: v22 **0.607** → v23 **0.620** (+0.013).

## Recency (1st→last, ret@1) + updates

| config | K20 | K50 | K100 | update new/old |
|---|---|---|---|---|
| v22 | 0.492→0.617 (**0.838**) | 0.453→0.583 (**0.699**) | 0.270→0.455 (**0.590**) | **0.667**/0.000 |
| v23 | **0.533**→0.575 (0.838) | 0.448→0.594 (**0.704**) | **0.310**→0.460 (**0.586**) | 0.625/0.000 |

## Diagnosis

- **N0 flat at 0.656** (ret@1 already 1.0). Joint CE did not raise the inject ceiling on held-out test despite train-proxy oracle gen rising.
- **N4 +5.2 pts, N32 +2.1 pts** gen; N4 ret@1 also up. N32 RAG gap shrinks 0.041→**0.020**.
- **N16 regresses** −2.1 pts gen with ret@1 0.906→0.875 — CE-through-softk4 slightly hurts mid-length session retrieval.
- **K100 ret@1 essentially flat** (0.590→0.586). Early-fact gen up (first Q 0.270→0.310) — inject adaptation, not retrieval.
- **Updates slightly worse:** new 0.667→0.625; old stays 0.000. Joint CE did not improve overwrite behavior.
- Train-proxy gains (esp. soft ret@1→1.0 on small cache) **do not transfer** cleanly to held-out long streams / updates — classic overfit to cached pool distribution.

**Verdict:** Joint CE is a **mixed, weak result**. Modest mean-gen lift and better N4/N32, but N16 and update new regress and K100 ret@1 does not move. **Do not promote v23 over v22 as the default stack.** Keep artifacts for reference; next lever should not pile fixes on this round.

STOP after Round 23.
