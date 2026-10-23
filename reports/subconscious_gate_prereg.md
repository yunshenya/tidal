# Subconscious gate pre-registration

Locked before rescoring the frozen bank and before scoring the held-out bank. Those two banks have not been scored. `reports/subconscious_gate.md` is written after one run of `scripts/eval_subconscious_gate.py`. Nothing in that output changes the inject MLP weights or `gap_tau`.

## What this PR changes

No feature flag. The #38 MLP checkpoint is unchanged (`tidal/llm_inject_smollm2_135m.pt`, sha256 `cba711e03c09faf649a82e8922d5a604e01ad6970558002e9bdc11fa37974539`). There is no `key_proj` and no decision-continuation retraining.

The add at layer `n_layers // 2` is

`c * (scale(v) + MLP(scale(v)))`

`scale` stays 1. `c` is 1 when the store has one slot. Otherwise `c = clip((top1 cosine − top2 cosine) / gap_tau, 0, 1)`. A tie adds nothing. `gap_tau` is 0.80, the smallest value on the grid 0.02, 0.05, 0.10, 0.20, 0.40, 0.80, 1.60, 3.20 that passed the unrelated-store bound on fit bank `20b7cb3f9742709d320dc5913cf62675316673d7b457c55b6c78cf4de587e5ce` at N=8 and N=32. It is not retuned here.

Phone and birthday spelling runs only when the winning slot's relation is the relation the prompt is completing (`phone` or `birthday` as a word). One continuation encodes the query once.

## Banks

| bank | role | sha256 |
|---|---|---|
| frozen | `tidal.subconscious_data.bank_payload`, protocol `subconscious-v1`, the #38 bank. Rescored, not used to choose `gap_tau`. | `e3cdd7fd595d8a76af03a276c2ad750b25b68e1bd8460c9e22e95aa12a820e66` |
| fit | #39 scale bank. Not rescored. Not used again. | `20b7cb3f9742709d320dc5913cf62675316673d7b457c55b6c78cf4de587e5ce` |
| held | this PR. Literal string overlap 0 with the frozen bank and with the #39 train, dev, fit, and held surfaces. Max token Jaccard under 0.50. | `cbca1f62c152eb928f498629e8d39a79cdcd9a144589cf2ad8484b65806dd464` |

Token Jaccard uses the same word tokens as `tidal.question_heldout_data._tokens`. The #39 surfaces are `tests/fixtures/subconscious_prior_surfaces.json`.

## Gates

Applied to the frozen rescore and to the held-out bank. These are the only gates. Deltas are length-normalized nats.

| gate | pass when |
|---|---|
| neutral NLL | absolute mean ΔNLL ≤ 0.05, and at most 1 item with ΔNLL > 0.5 |
| forced choice | flips (nomem correct, inject wrong) = 0 |
| leak | 0 neutral generations contain a stored incidental answer |
| phone unrelated, birthday unrelated | takeover rate 0 |
| phone question | takeover 1 |
| birthday question | takeover 1 on a bank that has one; the frozen bank has none |

Prefer and avoid margin deltas are reported and are not gates. Top-1 under load and FIFO eviction are reported and are not gates. Context and RAG are reference columns.

Prior floors stay where they are: probe 365/370, false abstain 0/370, unanswerable 72/72, frozen adversarial bank `cce20b6bf29f594992adac193ebcf05323fdd31cbe8c9f695a94d88a564723a5` at 188/216 with false abstain 19, dialogue and deixis floors, pytest. This run does not edit those paths.

## Cost

Reported from the frozen run, not gated. MLP parameter count is the shipped net. Per-token extra is the armed forward minus the empty forward at lengths 16, 32, and 64. Readout is hard top-1 at N = 8, 32, 64, 128, 256. Peak RSS is VmHWM. Variable-cost paths, none of them retries: empty store skips the query; spelling replaces the token loop only when the relation matches; `generate` may stop early; the prefix forward does not use a cache; `remember` is two forwards per accepted fact. The main path does not encode the query a second time.
