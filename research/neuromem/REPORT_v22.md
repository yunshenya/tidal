# REPORT: Round 22 — hard long-stream retrieval adapters

Held-out **test** split, n=96 free-gen match. Frozen A2s backbone + frozen `inj_v21.pt`. Scripts: `train_v22.py`, `run_v22.py`; weights `retr_v22.pt`; results `results_v22.json`.

## Method

Retrain contrastive q/k adapters (same 256→256 GELU→128 architecture), warm-started from `retr_v20.pt`.
- Long fact-only streams K∈{40…100} with 1–3 same-(name,rel) updates (overwrite keeps NEW).
- Mixed sessions plus hard negatives: same-name/diff-rel, same-rel/diff-name.
- InfoNCE (τ=0.07) with 70% long-stream sampling + optional hard-subset CE. No inject joint fine-tune.
- Eval: v21 baseline (`retr_v20`+inj softk4) vs `retr_v22`+inj softk4 / top1.

## Gen accuracy (ret@1)

| config | N0 | N4 | N16 | N32 |
|---|---|---|---|---|
| v21 retr_v20+inj softk4 | 0.656 [1.00] | 0.573 [0.969] | 0.594 [0.833] | 0.573 [0.823] |
| **retr_v22+inj softk4** | 0.656 [1.00] | 0.573 [**0.979**] | **0.604** [**0.906**] | **0.594** [**0.875**] |
| retr_v22+inj top1 | 0.656 [1.00] | 0.573 [0.979] | 0.604 [0.906] | 0.594 [0.875] |
| bge RAG | 0.604 | 0.604 | 0.552 | **0.635** |

## Recency (1st→last, ret@1) + updates

| config | K20 | K50 | K100 | update new/old |
|---|---|---|---|---|
| v21 retr_v20+inj | 0.483→0.625 (**0.823**) | 0.438→0.562 (**0.670**) | 0.250→0.455 (**0.541**) | 0.646/0.000 |
| retr_v22+inj softk4 | 0.492→0.617 (**0.838**) | 0.453→0.583 (**0.699**) | 0.270→0.455 (**0.590**) | **0.667**/0.000 |
| retr_v22+inj top1 | 0.483→0.633 (0.838) | 0.453→0.568 (0.699) | **0.285**→0.450 (**0.590**) | **0.667**/0.000 |

## Diagnosis

- **K100 ret@1 rises above v20/v21:** 0.541→**0.590** (+4.9 pts). K50 0.670→0.699; K20 0.823→0.838. Session ret@1 also improves: N16 0.833→0.906, N32 0.823→0.875.
- **N32 gen follows modestly:** 0.573→**0.594** (+2.1 pts). Softk4 ≈ top1. RAG gap at N32 shrinks from 0.062 to **0.041**. N0 unchanged at 0.656 (inject ceiling; ret@1 already 1.0).
- **Updates intact / slightly better:** 0.667/0.000 (was 0.646). Hard update negatives in training did not hurt overwrite behavior.
- **Forget gen lift is small** vs ret@1 lift (K100 first 0.250→0.270/0.285). Remaining misses are still hard distractors; inject cannot recover a wrong top-1.
- Pure InfoNCE with harder long-stream negatives was enough; no joint CE fine-tune needed this round.

**Verdict:** Hard long-stream InfoNCE lifts K100 ret@1 0.541→0.590 and N32 gen 0.573→0.594 without hurting updates. Remaining RAG gap (~0.04) is mostly residual retrieval errors × inject ceiling.

STOP after Round 22.
