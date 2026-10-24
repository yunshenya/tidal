# Experience valence preregistration

Experience-grown valence. The held-out bank has not been scored.

No feature flag. `gap_tau` stays 0.80. Polarity gain stays `1 + 2 tanh(w · normalize(k))`. Phone and birthday spelling stays the exact stored answer, and still runs only when the winning relation is the one the prompt is completing. A continuation still encodes the query once. Readout stays hard top-1. Replay stays every 32 accepted writes, refreshing the top 1 slot, decay 0.50. The surprise gate and the retrieval-induced-forgetting gate are not in this change.

The add is `c * s * u * gain * (scale(v) + MLP(scale(v)))`. `u = 1 + span * tanh(valence)`. A write's experience field contributes its sign times the step. The fact text is not scored, and the language model does not supply a group prior. Valence 0 leaves `u = 1`. The same replay cycle multiplies a nonzero valence by 0.50 and appends an audit row. A correction replaces the scalar and appends its own row.

Prefer, avoid, and forbidden on the replay held-out bank stay in scope. They are not out of scope for this change. #46 held matched the #44 numbers.

## Banks

- fit `44af0fcf6d10c350cb8e8861ccc5489cd09b9e0807bf8795033aa33adf941d80`
- held `017d1386329cff64df4cbac7ad07c2cc69beae813b237c2f8c8ed9b069d029d5`
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
- #45 fit `72a1836feafaf6df89347b87f527cdc80cbe64d3709746c5ca0e57c89ad814c8`
- #45 held `a4b5ce1da2ce61070720487e7e3e5e9d5bc8d78a9b5a1281ea69c9e2223fcbd5`
- #46 fit `9b677681fe0b626a8a7e411bf030e53b7f167e72dcee8b4de2d0393ab4822807`
- #46 held `fd01592f5bdb8e919c113fb0f82c01a26cc42c24c14104fb77ba5bc3bf65f635`
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
| 1 | 0.25 | yes | 6/8 2.085 | 8/8 -7.341 | 8/8 |
| 1 | 0.5 | no | 4/8 2.063 | 8/8 -9.652 | 8/8 |
| 1 | 1 | no | 5/8 1.998 | 8/8 -10.289 | 8/8 |
| 2 | 0.25 | no | 5/8 1.636 | 8/8 -11.531 | 8/8 |
| 2 | 0.5 | no | 5/8 0.311 | 8/8 -12.371 | 8/8 |
| 2 | 1 | no | 5/8 -0.280 | 8/8 -12.414 | 8/8 |

Fit bank `44af0fcf6d10c350cb8e8861ccc5489cd09b9e0807bf8795033aa33adf941d80`.

| gate | pass | detail |
|---|---|---|
| toward experiences | yes | 6/8 > 0, mean 2.085, audit 8/8 |
| away experiences | yes | 8/8 < 0, mean -7.341, audit 8/8 |
| contrary revises | yes | margin back 8/8, valence back 8/8, peak 2.085 revised 0.000 |
| neutral NLL | yes | mean -0.002, above 0.5: 0 |
| choice flips | yes | 0/8 |
| leak | yes | 0/8 |
| phone off-relation takeover | yes | 0.000 |
| birthday off-relation takeover | yes | 0.000 |
| phone exact | yes | '+1-301-555-3809' vs '+1-301-555-3809' |
| birthday exact | yes | 'June 7, 2011' vs 'June 7, 2011' |
| phone off-relation not stored | yes | 0 |
| birthday off-relation not stored | yes | 0 |
| incidental slots | yes | 32 slots, written 32/32 |

Fit does not require the held-out mean floors (toward mean 0.25, away mean -0.25). Those are held-out gates.

Fit toward mean 2.085 (6/8 > 0), audit 8/8.
Fit away mean -7.341 (8/8 < 0), audit 8/8.

Fit on-relation phone '+1-301-555-3809' equals the stored '+1-301-555-3809': 1.
Fit on-relation birthday 'June 7, 2011' equals the stored 'June 7, 2011': 1.
Prefix detector on those questions: phone 0, birthday 1. It is not the gate.

## Held-out gates

The held-out bank has not been scored. It is merge-critical.

- repeated positive experiences move the margin up on at least 6/8, with mean at least 0.25, and a full audit
- repeated negative experiences move the margin down on at least 6/8, with mean at most -0.25, and a full audit
- a contrary experience pulls the margin and the scalar back on at least 6/8
- absolute mean neutral ΔNLL ≤ 0.05, and at most 1 item with ΔNLL > 0.5
- choice flips 0
- leak 0
- phone and birthday takeover 0 off-relation, including the cross question
- on-relation generated text equals the stored number and the stored date
- off-relation text is not the stored number and not the stored date
- the prefix detector is recorded and is not the on-relation gate
- incidental store keeps 32 slots
- prefer margin mean at least 8.0 on the #44 held bank (#44 and #46 held were 20.750; #41 held was 10.905)
- avoid margin mean at least 0.6 on the #44 held bank (#44 and #46 held were 1.453; #41 held was 1.277)
- forbidden logprob down on at least 6/8 on the #44 held bank (#44 and #46 held mean stored lift -1.578; #41 held mean stored lift -1.300)

Report the prefer, avoid, and forbidden deltas against the #44 and #41 held numbers above. Those floors stay in scope. Do not retune span or step on this bank.
