# REPORT: Round 20 — contrastive query–key retrieval head

Held-out **test** split, n=96 free-gen match. Frozen A2s backbone `mem_ckpt_v13A2s.pt`. Slot store + value/inject path unchanged from v19. Scripts: `train_v20.py`, `run_v20.py`; weights `retr_v20.pt`; results `results_v20.json`.

## Method

- Freeze backbone. Cache frozen `m.key` / `m.qvec` on train sessions (N∈{0…32}, 40% update-augmented with same-(name,rel) overwrite).
- Train 2-layer MLP adapters `Wk`, `Wq`: 256→256 GELU→128, L2-normalize, InfoNCE (τ=0.07) over in-session keys. 2000 AdamW steps, bs=32.
- Train-cache val ret@1: frozen **0.837** → adapter **0.971**.
- At eval: store original `v` for inject; rank slots with adapted embeddings when config has `retr`.

Configs: `slot_softk4_C256` (v19 baseline), `slot_retr_top1_C256`, `slot_retr_softk4_C256`.

## Gen accuracy (ret@1)

| config | N0 | N4 | N16 | N32 |
|---|---|---|---|---|
| softk4 (v19) | 0.594 [1.00] | **0.479** [0.96] | 0.458 [0.87] | 0.469 [0.82] |
| retr_top1 | 0.594 [1.00] | 0.469 [0.97] | 0.438 [0.83] | 0.469 [0.82] |
| retr_softk4 | 0.594 [1.00] | 0.469 [0.97] | 0.458 [0.83] | 0.469 [0.82] |
| bge RAG | 0.604 | 0.604 | 0.552 | **0.635** |

## Recency + updates

| config | K20 1st→last (ret1) | K50 1st→last (ret1) | K100 1st→last (ret1) | update new/old |
|---|---|---|---|---|
| softk4 | 0.442→0.475 (0.79) | 0.333→0.443 (0.60) | 0.160→0.340 (**0.451**) | 0.667/0.000 |
| retr_top1 | **0.467**→0.517 (**0.823**) | **0.406**→0.453 (**0.670**) | 0.215→0.425 (**0.541**) | 0.646/0.000 |
| retr_softk4 | 0.442→0.517 (0.823) | 0.391→**0.474** (0.670) | **0.225**→0.410 (**0.541**) | **0.667**/0.000 |

## Diagnosis

- **Success criterion met on long streams.** K100 ret@1 0.451→**0.541** (+9 pts); K50 0.60→**0.67**; K20 0.79→**0.82**. Gen follows on forget: K50 first 0.333→0.391–0.406; K100 first 0.160→0.215–0.225; K100 mean 0.24→0.31–0.32. Updates stay healthy (0.65–0.67 / 0.00).
- **No N32 gen gain.** Session ret@1 stays ~0.82–0.87 and N32 gen stays **0.469**. Mixed sessions already have high ret@1; the adapter mainly helps dense fact-only streams where distractors are harder. Train-cache gains (0.84→0.97) do not fully transfer to the test session protocol (held-out names/templates + chat distractors in the write stream but not in the slot set).
- **Inject ceiling is now the clear bottleneck for N≤32.** At N0, ret@1=1.00 for all configs yet gen=0.594 — identical to v19. Closing the remaining RAG gap at N32 (0.635−0.469=0.166) needs a better value/injection path more than more retrieval. At K100, retrieval is still leaving ~0.46 of facts unreachable, so both levers remain relevant for long streams.

**Verdict:** Contrastive adapters clearly raise long-stream retrieval (K100 ret@1 0.45→0.54) and forget gen, without hurting updates. They do not move N32 gen above 0.469; at session scale the inject/value ceiling (~0.59 at N0 with perfect retrieval) is the bottleneck.

STOP after Round 20.
