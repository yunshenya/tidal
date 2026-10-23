# Subconscious fix pre-registration

Locked before rescoring the frozen bank and before scoring the held-out bank. Those two banks have not been scored. `reports/subconscious_fix.md` is written after one run of `scripts/eval_subconscious_fix.py`. Nothing in that output changes weights, `gap_tau`, or these gates.

## Banks

| bank | role | sha256 |
|---|---|---|
| frozen | `tidal.subconscious_data.bank_payload`, protocol `subconscious-v1`, the #38 bank | `e3cdd7fd595d8a76af03a276c2ad750b25b68e1bd8460c9e22e95aa12a820e66` |
| fit | scale only. Literal string overlap with the frozen bank is 0. Max token Jaccard against the frozen bank is under 0.50. Not an influence score. | `20b7cb3f9742709d320dc5913cf62675316673d7b457c55b6c78cf4de587e5ce` |
| held | scored once, after this file. Same disjointness rule against the frozen bank and against the fit bank. | `ea0a15e5edf0dcab53fede8fad1964b61c23b52c3df55c5f8e00865e59cb053a` |

Token Jaccard uses the same word tokens as `tidal.question_heldout_data._tokens`. Training and dev strings are not part of either scored bank. Dev chose the checkpoint. The fit bank chose `gap_tau`.

## What changed on the main path

No feature flag. The add at layer `n_layers // 2` is

`c * (scale(v) + MLP(scale(v) + W k))`

`scale` stays 1. `W` is `key_proj`, zero at init, trained. `c` is 1 when the store has one slot. Otherwise `c = clip((top1 cosine - top2 cosine) / gap_tau, 0, 1)`. A tie adds nothing. Absolute cosine is not a threshold: on the fit bank a correct single fact and a wrong incidental sit in the same cosine range, so a cosine cutoff would scale the direction examples. `gap_tau` is 0.80.

Phone and birthday spelling runs only when the winning slot's relation is the relation the prompt is completing (`phone` or `birthday` as a word). One continuation encodes the query once.

Value encoding is unchanged: answer-span pool, phone slices, date slices. The linear readouts are not in the Adam step.

## Training, dev only

`fit_decision` in `tidal/llm_memory_train.py`, started from the #38 checkpoint with `W` at 0. Loss on a decision continuation, fact key passed in:

- cross-entropy of the good continuation
- relu so the good token beats the empty-store cross-entropy by 1.0 nat
- relu so the bad token is worse than the empty-store cross-entropy by 1.0 nat

An absolute margin was not enough: it went to zero while the avoid margin versus no-memory stayed negative. Each training word is stored once as a favorite and once as a forbidden answer, so the value alone cannot pick the target. 800 steps, batch 4, lr 3e-4, weight decay 0, seed 0. The checkpoint is the dev step with the highest avoid mean among steps whose prefer and avoid means are both above 0.05 with at least 6 of 8 items positive. That step is 200. Dev at that step: prefer mean 15.983 (8/8), avoid mean 8.935 (8/8). The trace is `reports/subconscious_fix_train.json`.

## gap_tau, fit bank only

Unrelated stores: the prefer prompt, salient fact absent, N=8 and N=32 incidental facts. Pass when the absolute mean margin delta is at most 0.05 and at most 2 items have absolute delta above 0.1, at both sizes. Candidates were 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60, 3.20. The smallest passing value is 0.80 (N=8 absolute mean 0.0044 with 1 large item; N=32 absolute mean 0.0155 with 0 large items). 0.40 failed N=8 with 3 large items. Incidental lines rotate the verb so eight neighbors do not collapse to one slot; a collapsed slot has no runner-up and would ignore `gap_tau`.

## Gates

Same numeric thresholds as #38 where the metric is shared. Applied to the frozen rescore and to the held-out bank. Deltas are length-normalized nats. A family has 8 items.

| gate | pass when |
|---|---|
| prefer delta, avoid delta | mean delta > 0.05 and at least 6 items have delta > 0 |
| neutral NLL | mean ΔNLL ≤ 0.05 and no item with ΔNLL > 0.5 |
| neutral choice | flips (nomem correct, inject wrong) ≤ 1 of 8 |
| leak | 0 of 8 neutral generations contain a stored incidental answer |
| unrelated store | N=8 incidental, salient absent: \|mean delta\| ≤ 0.05 and at most 2 items with \|delta\| > 0.1 |
| repeat null | 3 identical writes, one slot, max \|margin − single-write margin\| < 1e-4 |
| chat null | chat remembers accepted 0 times, one slot, same margin bound |
| update store | one slot, cosine to the new value ≥ 0.999 and greater than cosine to the old value |
| update delta | mean delta of the new word versus the old word > 0.05 and at least 6 items > 0 |
| phone unrelated, birthday unrelated | takeover rate 0 |
| phone question | takeover 1 |
| birthday question | takeover 1 on a bank that has one; the frozen bank has none |

Not gates: prefer choice lift (nomem is already high), top-1 hit rate at N=64, FIFO eviction. Those are reported. Context and RAG are reference columns, not the claim.

Prior floors stay where they are: probe 365/370, false abstain 0/370, unanswerable 72/72, frozen adversarial bank `cce20b6bf29f594992adac193ebcf05323fdd31cbe8c9f695a94d88a564723a5` at 188/216 with false abstain 19, dialogue and deixis floors, pytest. This run does not edit those paths.

## Cost

Reported from the frozen run, not gated. MLP parameter count includes `net` and `key_proj`. Per-token extra is the armed forward minus the empty forward at lengths 16, 32, and 64. Readout is hard top-1 at N = 8, 32, 64, 128, 256. Peak RSS is VmHWM. Variable-cost paths, none of them retries: empty store skips the query; spelling replaces the token loop only when the relation matches; `generate` may stop early; the prefix forward does not use a cache; `remember` is two forwards per accepted fact. The main path does not encode the query a second time.
