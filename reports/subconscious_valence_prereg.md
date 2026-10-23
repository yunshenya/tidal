# Experience valence preregistration

Experience-grown valence. The held-out bank has not been scored.

No feature flag. `gap_tau` stays 0.80. Polarity gain stays `1 + 2 tanh(w · normalize(k))`. Phone and birthday spelling still run only when the winning relation is the one the prompt is completing. A continuation still encodes the query once. Readout stays hard top-1. Replay stays every 32 accepted writes, refreshing the top 1 slot, decay 0.50. The surprise gate and the retrieval-induced-forgetting gate are not in this change.

The add is `c * s * u * gain * (scale(v) + MLP(scale(v)))`. `u = 1 + span * tanh(valence)`. A write's experience field contributes its sign times the step. The fact text is not scored, and the language model does not supply a group prior. Valence 0 leaves `u = 1`. The same replay cycle multiplies a nonzero valence by 0.50 and appends an audit row. A correction replaces the scalar and appends its own row.

## Banks

- fit `72a1836feafaf6df89347b87f527cdc80cbe64d3709746c5ca0e57c89ad814c8`
- held `a4b5ce1da2ce61070720487e7e3e5e9d5bc8d78a9b5a1281ea69c9e2223fcbd5`
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
- #44 fit `35ae15cde3079cee5b034e2e019ef4a76324574507a8dbb58831bdb141b9f552`
- #44 held `e684c422049c720b269a50d762ca8ec1be87b397c93068ea2d61fa20ad4127ce`
- literal overlap 0 and token Jaccard under 0.50 against those banks and between fit and held

## Rule

- valence span: 1
- valence step: 0.25
- decay: 0.50
- repeats: 3
- new parameters: 0
- MLP parameters: 664704
- polarity parameters: 576
- total parameters: 665280
- buffers: 66931
- forwards inside one remember: 2
- forwards inside one armed query: 1

## Fit selection

The grid is span in {1, 2} and step in {0.25, 0.5, 1}. A point passes when at least 6/8 positive experiences move the fit margin up, at least 6/8 negative experiences move it down, each of those rows has a complete audit, and at least 6/8 contrary experiences pull the margin and the scalar back. The smallest span, then the smallest step, is the one scored below. The full fit score then runs once. It is not repeated on a larger rule if a non-interference floor fails. The held-out bank has not been scored.

| span | step | pass | toward | away | revise |
|---|---|---|---|---|---|
| 1 | 0.25 | yes | 8/8 3.037 | 8/8 -10.879 | 8/8 |
| 1 | 0.5 | yes | 7/8 3.629 | 8/8 -16.507 | 8/8 |
| 1 | 1 | yes | 7/8 3.752 | 8/8 -17.758 | 8/8 |
| 2 | 0.25 | yes | 7/8 3.987 | 8/8 -19.463 | 8/8 |
| 2 | 0.5 | yes | 7/8 4.134 | 8/8 -19.644 | 8/8 |
| 2 | 1 | yes | 7/8 4.129 | 8/8 -19.388 | 8/8 |

Fit bank `72a1836feafaf6df89347b87f527cdc80cbe64d3709746c5ca0e57c89ad814c8`.

| gate | pass | detail |
|---|---|---|
| toward experiences | yes | 8/8 > 0, mean 3.037, audit 8/8 |
| away experiences | yes | 8/8 < 0, mean -10.879, audit 8/8 |
| contrary revises | yes | margin back 8/8, valence back 8/8, peak 3.037 revised 0.000 |
| neutral NLL | yes | mean -0.011, above 0.5: 0 |
| choice flips | yes | 0/8 |
| leak | yes | 0/8 |
| phone unrelated | yes | 0.000 |
| birthday unrelated | yes | 0.000 |
| phone question | yes | 1 |
| birthday question | yes | 1 |
| incidental slots | yes | 32 slots, written 32/32 |

Fit does not require the held-out mean floors (toward mean 0.25, away mean -0.25). Those are held-out gates.

Fit toward mean 3.037 (8/8 > 0), audit 8/8.
Fit away mean -10.879 (8/8 < 0), audit 8/8.

## Held-out gates

The held-out bank has not been scored. It is merge-critical.

- repeated positive experiences move the margin up on at least 6/8, with mean at least 0.25, and a full audit
- repeated negative experiences move the margin down on at least 6/8, with mean at most -0.25, and a full audit
- a contrary experience pulls the margin and the scalar back on at least 6/8
- absolute mean neutral ΔNLL ≤ 0.05, and at most 1 item with ΔNLL > 0.5
- choice flips 0
- leak 0
- phone and birthday takeover 0 off-relation
- phone question takeover 1 and birthday question takeover 1
- incidental store keeps 32 slots
- prefer margin mean at least 8.0 on the #44 held bank (#44 held was 20.750; #41 held was 10.905)
- avoid margin mean at least 0.6 on the #44 held bank (#44 held was 1.453; #41 held was 1.277)
- forbidden logprob down on at least 6/8 on the #44 held bank (#44 held mean stored lift -1.578; #41 held mean stored lift -1.300)

Report the prefer, avoid, and forbidden deltas against the #44 and #41 held numbers above. Do not retune span or step on this bank.

The stored phone's full digit string is not a gate. The readout still spells the `+1-415-555-` prefix it was trained on. A mismatch between that prefix and the stored number is out of scope here.
