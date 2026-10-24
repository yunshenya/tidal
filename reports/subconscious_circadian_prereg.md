# Circadian clock preregistration

Content-free circadian clock. The held-out bank has not been scored.

There is no train bank. The fit bank selects the span. The held-out bank is the only later score.

No feature flag. `gap_tau` stays 0.80. Polarity gain stays `1 + 2 tanh(w · normalize(k))`. Valence stays `u = 1 + 1 * tanh(valence)` with step 0.25. Phone and birthday spelling stays the exact stored answer, and still runs only when the winning relation is the one the prompt is completing. A continuation still encodes the query once. Readout stays hard top-1. Replay stays every 32 accepted writes, refreshing the top 1 slot, decay 0.50. The surprise gate and the retrieval-induced-forgetting gate are not in this change. The valence formula is not in this change.

The clock is one global angle on the store. `phase = sin(angle)`. Angle 0 leaves the write strength at 1 and the decay at `0.5 ** age`. A nonzero phase scales that write's strength by `1 + span * phase` and the decay exponent by `1 - span * phase`. The angle is not read from fact text, and the log stores kind, angle, phase, and gain. Replay refresh still sets the top slot back to strength 1. The tick is `pi/16` and is not on the grid. `remember` does not tick.

Prefer, avoid, and forbidden on the replay held-out bank stay in scope. They are not out of scope for this change. #47 held matched the #44 and #46 numbers.

## Banks

- fit `ae9c1a9392fbbbf4d4dca134a7a2771a1859a1db6495658067e5292eeaf9a772`
- held `fb46f7faf4e5b995553a37f036ae77e112faeec7b9cffbe4864280df625ea7af`
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
- #47 fit `44af0fcf6d10c350cb8e8861ccc5489cd09b9e0807bf8795033aa33adf941d80`
- #47 held `017d1386329cff64df4cbac7ad07c2cc69beae813b237c2f8c8ed9b069d029d5`
- literal overlap 0 and token Jaccard under 0.50 against those banks and between fit and held

## Rule

- clock span: 0.25
- clock step: pi/16
- valence span: 1
- valence step: 0.25
- replay every 32, m 1, decay 0.50
- new parameters: 0
- MLP parameters: 664704
- polarity parameters: 576
- total parameters: 665280
- buffers: 66931
- forwards inside one remember: 2
- forwards inside one armed query: 1
- forwards inside one clock tick: 0

## Fit selection

The grid is span in {0.25, 0.5, 1}. A point passes when at least 6/8 high-phase margins move up, at least 6/8 low-phase margins move down, decay order holds on at least 6/8 with phase 0 strength exactly 0.5, every write strength matches the clock gain, and the log is content-free. The smallest span is the one scored below. The full fit score then runs once. It is not repeated on a larger span if a non-interference floor fails. Mean floors are held-out gates. The held-out bank has not been scored.

| span | pass | high | low | decay | content-free |
|---|---|---|---|---|---|
| 0.25 | yes | 8/8 2.341 | 8/8 -2.140 | 8/8 gap 0.174 | yes |
| 0.5 | yes | 8/8 4.491 | 8/8 -3.775 | 8/8 gap 0.354 | yes |
| 1 | yes | 8/8 7.496 | 8/8 -5.863 | 8/8 gap 0.750 | yes |

Fit bank `ae9c1a9392fbbbf4d4dca134a7a2771a1859a1db6495658067e5292eeaf9a772`.

| gate | pass | detail |
|---|---|---|
| high phase | yes | 8/8 > 0, mean 2.341 |
| low phase | yes | 8/8 < 0, mean -2.140 |
| decay order | yes | 8/8 high > phase0 > low, phase0 exact 1, gap 0.174 |
| write gain | yes | 8/8 match the clock gain |
| content-free | yes | equal gains 1, audit 8/8 |
| neutral NLL | yes | mean -0.009, above 0.5: 0 |
| choice flips | yes | 0/8 |
| leak | yes | 0/8 |
| phone off-relation takeover | yes | 0.000 |
| birthday off-relation takeover | yes | 0.000 |
| phone exact | yes | '+1-278-555-0989' vs '+1-278-555-0989' |
| birthday exact | yes | 'August 25, 2009' vs 'August 25, 2009' |
| phone off-relation not stored | yes | 0 |
| birthday off-relation not stored | yes | 0 |
| incidental slots | yes | 32 slots, written 32/32 |

Fit does not require the held-out mean floors (high mean 0.25, low mean -0.25, decay gap mean 0.05). Those are held-out gates.

Fit high mean 2.341 (8/8 > 0).
Fit low mean -2.140 (8/8 < 0).
Fit decay gap 0.174.

Fit on-relation phone '+1-278-555-0989' equals the stored '+1-278-555-0989': 1.
Fit on-relation birthday 'August 25, 2009' equals the stored 'August 25, 2009': 1.
Prefix detector on those questions: phone 0, birthday 1. It is not the gate.

## Held-out gates

The held-out bank has not been scored. It is merge-critical.

- high phase moves the margin up on at least 6/8, with mean at least 0.25
- low phase moves the margin down on at least 6/8, with mean at most -0.25
- decay order high > phase 0 > low on at least 6/8, phase 0 strength exactly 0.5, and mean decay gap at least 0.05
- write strength matches the content-free clock gain on every item
- the clock log has no fact text, and two facts at one phase share one gain
- absolute mean neutral ΔNLL ≤ 0.05, and at most 1 item with ΔNLL > 0.5
- choice flips 0
- leak 0
- phone and birthday takeover 0 off-relation, including the cross question
- on-relation generated text equals the stored number and the stored date
- off-relation text is not the stored number and not the stored date
- the prefix detector is recorded and is not the on-relation gate
- incidental store keeps 32 slots
- prefer margin mean at least 8.0 on the #44 held bank (#47, #44, and #46 held were 20.750; #41 held was 10.905)
- avoid margin mean at least 0.6 on the #44 held bank (#47, #44, and #46 held were 1.453; #41 held was 1.277)
- forbidden logprob down on at least 6/8 on the #44 held bank (#47, #44, and #46 held mean stored lift -1.578; #41 held mean stored lift -1.300)

Report the prefer, avoid, and forbidden deltas against the #47/#44 and #41 held numbers above. Those floors stay in scope. Do not retune the span on this bank.
