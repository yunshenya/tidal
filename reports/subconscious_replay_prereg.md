# Replay consolidation preregistration

Fixed-budget replay. The held-out bank has not been scored.

No feature flag. `gap_tau` stays 0.80. Polarity gain stays `1 + 2 tanh(w · normalize(k))`. Phone and birthday spelling still run only when the winning relation is the one the prompt is completing. A continuation still encodes the query once. Readout stays hard top-1. The surprise gate and the retrieval-induced-forgetting gate are not in this change.

Every N accepted writes, decay is recomputed as `0.5 ** age`, the top-m slots by salience are refreshed to strength 1, and at most m same-group near-duplicates are merged. A near-duplicate is a key cosine strictly above the store overwrite cosine, 0.9. Salience is `|tanh(w · normalize(k))|` on the polarity vector already in the checkpoint. The add is `c * s * gain * (scale(v) + MLP(scale(v)))`. `s` is 1 until that replay.

## Banks

- fit `35ae15cde3079cee5b034e2e019ef4a76324574507a8dbb58831bdb141b9f552`
- held `e684c422049c720b269a50d762ca8ec1be87b397c93068ea2d61fa20ad4127ce`
- frozen `e3cdd7fd595d8a76af03a276c2ad750b25b68e1bd8460c9e22e95aa12a820e66`
- #39 fit `20b7cb3f9742709d320dc5913cf62675316673d7b457c55b6c78cf4de587e5ce`
- #39 held `ea0a15e5edf0dcab53fede8fad1964b61c23b52c3df55c5f8e00865e59cb053a`
- #40 held `cbca1f62c152eb928f498629e8d39a79cdcd9a144589cf2ad8484b65806dd464`
- #41 train `626ec7cb9ac7824666be6a94ab8e6dc428b7e092ccfc7d8db689fde9c92be4d9`
- #41 fit `ee75f1857bfb6b7e812dfb0ef4012af1dc6efb33ca17799f1211b570ad1c0c1d`
- #41 held `e074b2ad7960bb81bbf64ae0ccd03dbeb0f358428a8a5c6c56e8f373a38d952b`
- #42 fit `4a90e639af898812b9edd13f68cf6c1edebbadaf36c761d0b2d03fdddc30bec0`
- #42 held `41ca5f243131f13deefe85800a775e8378bdeb233954a0b51119d66d93d180f1`
- #43 fit `2ad8038eefbb5bd77f7d8d3d9e52b30f2c1abf4467713c35b8eacad5bb72fe44`
- #43 held `ef0d06d320c3e8e15f1ac78bbe3e486bdd3017ca37f61b193089303fe5e3d6e9`
- literal overlap 0 and token Jaccard under 0.50 against those banks and between fit and held

## Budget

- replay every: 32
- replay m: 1
- decay: 0.50
- new parameters: 0
- MLP parameters: 664704
- polarity parameters: 576
- total parameters: 665280
- buffers: 66931
- forwards inside one remember that consolidates: 2
- forwards inside consolidate alone: 0
- forwards inside one armed query: 1
- consolidate median: 3.018e-03 s

## Fit selection

The grid is N in {4, 8, 16, 32} and m in {1, 2, 4, 8}. A point passes when the fit probe's salient-slot gain is at least 0.25 or the slot count drops by at least 1, against the decay-only prefix of that same replay. The smallest budget (smallest m/N, then the largest N, then the smallest m) is the one scored below. The full fit score then runs once. It is not repeated on a larger budget if a non-interference floor fails. The held-out bank has not been scored.

| N | m | pass | salient gain | slot drop | near-dup drop |
|---|---|---|---|---|---|
| 32 | 1 | yes | 0.500 | 1 | 2 |
| 32 | 2 | yes | 0.500 | 1 | 2 |
| 32 | 4 | yes | 0.500 | 1 | 2 |
| 32 | 8 | yes | 0.500 | 1 | 2 |
| 16 | 1 | yes | 0.500 | 1 | 2 |
| 16 | 2 | yes | 0.500 | 1 | 2 |
| 16 | 4 | yes | 0.500 | 1 | 2 |
| 16 | 8 | yes | 0.500 | 1 | 2 |
| 8 | 1 | yes | 0.500 | 1 | 2 |
| 8 | 2 | yes | 0.500 | 1 | 2 |
| 8 | 4 | yes | 0.500 | 1 | 2 |
| 8 | 8 | yes | 0.500 | 1 | 2 |
| 4 | 1 | yes | 0.500 | 1 | 2 |
| 4 | 2 | yes | 0.500 | 1 | 2 |
| 4 | 4 | yes | 0.500 | 1 | 2 |
| 4 | 8 | yes | 0.500 | 1 | 2 |

Fit bank `35ae15cde3079cee5b034e2e019ef4a76324574507a8dbb58831bdb141b9f552`.

| gate | pass | detail |
|---|---|---|
| replay consolidates | yes | salient gain 0.500, slot drop 1, near-dup drop 2 |
| prefer margin | yes | 8/8 > 0, mean 21.121 |
| avoid margin | yes | 8/8 > 0, mean 0.886 |
| forbidden logprob down | reported | 8/8 down, mean stored lift -1.121 |
| neutral NLL | yes | mean -0.035, above 0.5: 0 |
| choice flips | yes | 0/8 |
| leak | yes | 0/8 |
| phone unrelated | yes | 0.000 |
| birthday unrelated | yes | 0.000 |
| phone question | yes | 1 |
| birthday question | yes | 1 |
| incidental slots | yes | 32 slots, written 32/32 |

Fit does not require the held-out mean floors (prefer mean 8, avoid mean 0.6) or forbidden-down. Those are held-out gates.

Fit prefer mean 21.121 (8/8 > 0). Fit avoid mean 0.886 (8/8 > 0). Fit forbidden down 8/8, mean stored lift -1.121.

## Held-out gates

The held-out bank has not been scored. It is merge-critical.

- after replay, salient mean strength exceeds the decay-only control by at least 0.25, or the slot count drops by at least 1
- prefer margin delta > 0 on at least 6/8 and mean at least 8.0 (#41 held was 10.905)
- avoid margin delta > 0 on at least 6/8 and mean at least 0.6 (#41 held was 1.277)
- forbidden logprob down on at least 6/8 (#41 held mean stored lift -1.300)
- absolute mean neutral ΔNLL ≤ 0.05, and at most 1 item with ΔNLL > 0.5
- choice flips 0
- leak 0
- phone and birthday takeover 0 off-relation
- phone question takeover 1 and birthday question takeover 1
- incidental store keeps 32 slots

Report the prefer, avoid, and forbidden deltas against the #41 held numbers above. Do not retune N or m on this bank.

The stored phone's full digit string is not a gate. The readout still spells the `+1-415-555-` prefix it was trained on. A mismatch between that prefix and the stored number is out of scope here.
