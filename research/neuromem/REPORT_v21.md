# REPORT: Round 21 — residual inject MLP (value/injection ceiling)

Held-out **test** split, n=96 free-gen match. Frozen A2s backbone `mem_ckpt_v13A2s.pt`. Slot store + `retr_v20.pt` contrastive retrieval unchanged. Scripts: `train_v21.py`, `run_v21.py`; weights `inj_v21.pt`; results `results_v21.json`.

## Method

Primary lever: **residual inject MLP** on scaled slot values.
- `add = inj(scale(v)) + MLP(scale(v))` with `inj` frozen identity; MLP = Linear(576→576)→GELU→Linear(576→576), last layer zero-init.
- Teacher-force next-token CE on answer tokens given the query, with the correct fact value injected. Optional 20% wrong-v hinge. Frozen LM throughout.
- Eval never puts fact text in the prompt. Configs: `slot_retr_softk4_C256` (v20 baseline) vs `slot_retr_softk4_inj_C256`.

Train proxy on cached train facts: baseline gen ~0.66 → ~0.75–0.83 during training.

## Gen accuracy (ret@1)

| config | N0 | N4 | N16 | N32 |
|---|---|---|---|---|
| retr_softk4 (v20) | 0.594 [1.00] | 0.469 [0.97] | 0.458 [0.83] | 0.469 [0.82] |
| **retr_softk4 + inj** | **0.656** [1.00] | **0.573** [0.97] | **0.594** [0.83] | **0.573** [0.82] |
| fact in context | 0.604 | 0.615 | 0.562 | 0.688 |
| bge RAG | 0.604 | 0.604 | 0.552 | **0.635** |

## Recency + updates

| config | K20 1st→last (ret1) | K50 1st→last (ret1) | K100 1st→last (ret1) | update new/old |
|---|---|---|---|---|
| retr_softk4 | 0.442→0.517 (0.823) | 0.391→0.474 (0.670) | 0.225→0.410 (0.541) | **0.667**/0.000 |
| + inj | **0.483**→**0.625** (0.823) | **0.438**→**0.562** (0.670) | **0.250**→**0.455** (0.541) | 0.646/0.000 |

## Diagnosis

- **N0 ceiling moved.** With ret@1=1.00 fixed, gen rises **0.594→0.656** (+6.2 pts), meeting the ≥0.65 target and surpassing in-context N0 (0.604). Same stored answer-span `v`, better decoding via residual inject.
- **Gains follow at every N.** N4 0.469→0.573, N16 0.458→0.594, N32 0.469→**0.573**. RAG gap at N32 shrinks from 0.166 to **0.062**. Ret@1 unchanged (same retrieval) — pure inject lift.
- **Forget gen rises with the same ret@1.** K20/50/100 first +0.041/+0.047/+0.025; last-quarter also up. Retrieval remains the long-stream limit (K100 ret@1=0.54).
- **Updates stay healthy.** new/old 0.646/0.000 (was 0.667/0.000). Slight −0.02 on new; no stale answers.
- **Remaining gap vs RAG at N32 (0.062)** is mostly retrieval misses (ret@1 0.82) × residual decode errors on hard types (phones, rare names).

**Verdict:** Residual inject MLP lifts the N0 ceiling (0.594→0.656) and carries through to N32 (0.469→0.573), cutting the RAG gap roughly in half while keeping updates intact. The inject ceiling moved; retrieval remains the main long-stream bottleneck.

STOP after Round 21.
