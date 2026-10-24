# Relative surprise preregistration

Surprise-gated writes, relative to the store's own window. The held-out bank has not been scored.

The train bank is the post-warmup keep floor: after 32 accepted incidental surprises, ordinary keeps must still be written. The fit bank selects the quantile. The held-out bank is the only later score.

No feature flag. `gap_tau` stays 0.80. Polarity gain stays `1 + 2 tanh(w · normalize(k))`. Valence stays `u = 1 + 1 * tanh(valence)` with step 0.25. The clock stays `phase = sin(angle)` with span 0.25 and step pi/16. Phone and birthday spelling stays the exact stored answer, and still runs only when the winning relation is the one the prompt is completing. A continuation still encodes the query once. Readout stays hard top-1. Replay stays every 32 accepted writes, refreshing the top 1 slot, decay 0.50. The retrieval-induced-forgetting gate is not in this change. The valence formula is not in this change. The circadian formula is not in this change.

A fact's surprise is the total NLL, in nats, of its answer tokens under the logits the value forward already returned. The window stores the last 32 accepted write surprises and no text. Fewer than 32 accepted writes leaves the salient fact accepted. After that, the fact is written when its surprise is at least the quantile of the last 32 accepted surprises. A rejected attempt does not enter the window. Chat and noise still return before any forward. The key forward runs only when the write is kept. There is no third forward and no new parameter. The window length is not on the grid.

On the fit and held banks the incidental facts are written first, in listed order, and that fills the window. Lows are attempted next, then highs, with no clear between them. The train keeps are attempted only after the train incidental window is full.

Prefer, avoid, and forbidden on the replay held-out bank stay in scope. They are not out of scope for this change. #48 held matched the #47, #44, and #46 numbers.

## Banks

- train `ff97715ddd8c693bf35983ca5d527a0e1ba3f107e31d7ea438926cb4bf3a6005`
- fit `87a581a35f0d8ed51e8c394c68d241fc65da9fefbf119c4a1fc99b6fcb5b641d`
- held `c68221e3a8dc696628399f2f7c605bbde8a26f314e77f822b1a2b2e90f35686c`
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
- closed valence fit `72a1836feafaf6df89347b87f527cdc80cbe64d3709746c5ca0e57c89ad814c8`
- closed valence held `a4b5ce1da2ce61070720487e7e3e5e9d5bc8d78a9b5a1281ea69c9e2223fcbd5`
- #46 fit `9b677681fe0b626a8a7e411bf030e53b7f167e72dcee8b4de2d0393ab4822807`
- #46 held `fd01592f5bdb8e919c113fb0f82c01a26cc42c24c14104fb77ba5bc3bf65f635`
- #47 fit `44af0fcf6d10c350cb8e8861ccc5489cd09b9e0807bf8795033aa33adf941d80`
- #47 held `017d1386329cff64df4cbac7ad07c2cc69beae813b237c2f8c8ed9b069d029d5`
- #48 fit `ae9c1a9392fbbbf4d4dca134a7a2771a1859a1db6495658067e5292eeaf9a772`
- #48 held `fb46f7faf4e5b995553a37f036ae77e112faeec7b9cffbe4864280df625ea7af`

## Quantile grid

| quantile | low written | high written | train keeps | bar | pass |
|---|---|---|---|---|---|
| 0 | 0/8 | 8/8 | 8/8 | 9.956 | yes |
| 0.25 | 0/8 | 8/8 | 5/8 | 11.916 | no |
| 0.5 | 0/8 | 8/8 | 4/8 | 13.200 | no |

Chosen surprise quantile: 0.
surprise quantile: 0

Fit bank `87a581a35f0d8ed51e8c394c68d241fc65da9fefbf119c4a1fc99b6fcb5b641d`. Train keeps 8/8.

| gate | pass | detail |
|---|---|---|
| low surprise not written | yes | 0/8 written, mean surprise 2.541414961218834 |
| high surprise written | yes | 8/8 written, mean surprise 16.595247268676758 |
| seed window | yes | written 32/32, slots 32, window 32, bar 9.955883026123047 |
| neutral NLL | yes | mean -0.004, above 0.5: 0 |
| choice flips | yes | 0/8 |
| leak | yes | 0/8 |
| phone off-relation takeover | yes | 0.000 |
| birthday off-relation takeover | yes | 0.000 |
| phone exact | yes | '+1-939-555-2791' vs '+1-939-555-2791' |
| birthday exact | yes | 'January 6, 2013' vs 'January 6, 2013' |
| phone off-relation not stored | yes | 0 |
| birthday off-relation not stored | yes | 0 |
| incidental slots | yes | 32 slots, written 32/32 |
| cost | yes | new 0, chat 0, below 1/0, above 2/1 |

Fit uses the same count gates as held-out for low-surprise writes (at most 2 of 8) and high-surprise writes (at least 6 of 8). There is no absolute surprise threshold and no held-out mean floor on the raw NLL, because an absolute threshold did not transfer.

Fit low mean surprise 2.541 (0/8 written).
Fit high mean surprise 16.595 (8/8 written).
Fit window bar 9.956.

Fit on-relation phone '+1-939-555-2791' equals the stored '+1-939-555-2791': 1.
Fit on-relation birthday 'January 6, 2013' equals the stored 'January 6, 2013': 1.
Prefix detector on those questions: phone 0, birthday 1. It is not the gate.

## Held-out gates

The held-out bank has not been scored. It is merge-critical.

- low-surprise facts are not written on at least 6 of 8 (written at most 2)
- high-surprise facts are written on at least 6 of 8
- the incidental seed fills 32 slots and a window of 32 numbers, and that window does not contain fact text
- absolute mean neutral ΔNLL ≤ 0.05, and at most 1 item with ΔNLL > 0.5
- choice flips 0
- leak 0
- phone and birthday takeover 0 off-relation, including the cross question
- on-relation generated text equals the stored number and the stored date
- off-relation text is not the stored number and not the stored date
- the prefix detector is recorded and is not the on-relation gate
- incidental store keeps 32 slots
- chat does 0 forwards, a below-window fact does 1 forward and is not written, an above-window fact does 2 forwards and is written, new parameters 0
- prefer margin mean at least 8.0 on the #44 held bank (#48, #47, #44, and #46 held were 20.750; #41 held was 10.905)
- avoid margin mean at least 0.6 on the #44 held bank (#48, #47, #44, and #46 held were 1.453; #41 held was 1.277)
- forbidden logprob down on at least 6/8 on the #44 held bank (#48, #47, #44, and #46 held mean stored lift -1.578; #41 held mean stored lift -1.300)
- valence fit gates already passed on the #47 fit bank; the held-out score must pass the #47 valence gates, including the held mean floors
- circadian fit gates already passed on the #48 fit bank; the held-out score must pass the #48 circadian gates, including the held mean floors

Report the prefer, avoid, and forbidden deltas against the #48/#47/#44 and #41 held numbers above. Those floors stay in scope. Do not retune the quantile on the held-out bank.

New parameters on the fit cost probe: 0. Chat forwards 0. Below-window forwards 1, written 0. Above-window forwards 2, written 1.
