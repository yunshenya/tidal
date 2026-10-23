# Surprise preregistration

Surprise-gated writes. The held-out bank has not been scored.

No feature flag. `gap_tau` stays 0.80. Polarity gain stays `1 + 2 tanh(w · normalize(k))`. Phone and birthday spelling still run only when the winning relation is the one the prompt is completing. A continuation still encodes the query once.

## Banks

- fit `4a90e639af898812b9edd13f68cf6c1edebbadaf36c761d0b2d03fdddc30bec0`
- held `41ca5f243131f13deefe85800a775e8378bdeb233954a0b51119d66d93d180f1`
- frozen `e3cdd7fd595d8a76af03a276c2ad750b25b68e1bd8460c9e22e95aa12a820e66`
- #39 fit `20b7cb3f9742709d320dc5913cf62675316673d7b457c55b6c78cf4de587e5ce`
- #39 held `ea0a15e5edf0dcab53fede8fad1964b61c23b52c3df55c5f8e00865e59cb053a`
- #40 held `cbca1f62c152eb928f498629e8d39a79cdcd9a144589cf2ad8484b65806dd464`
- #41 train `626ec7cb9ac7824666be6a94ab8e6dc428b7e092ccfc7d8db689fde9c92be4d9`
- #41 fit `ee75f1857bfb6b7e812dfb0ef4012af1dc6efb33ca17799f1211b570ad1c0c1d`
- #41 held `e074b2ad7960bb81bbf64ae0ccd03dbeb0f358428a8a5c6c56e8f373a38d952b`
- literal overlap 0 and token Jaccard under 0.50 against those banks and between fit and held

## Write score

Salience is 1 for a fact and 0 for chat and noise. Surprise is the total NLL, in nats, of the answer tokens under the logits the value forward already returned. `write_score = salience * max(0, surprise)`. A fact is written only when that score is at least `surprise_tau`. Chat and noise return before any forward. A fact below the threshold pays the value forward and skips the key forward. There is no third forward and no second memory layer.

- surprise_tau: 2.00
- green-color floor on the memory train names: min 2.092, max 2.495, n 24. Tau is at most that minimum.
- new parameters: 0
- MLP parameters: 664704
- polarity parameters: 576
- answer-token log_softmax median: 0.000054 s over 1 answer row(s) of logits already computed
- forwards: chat 0 (written False), below threshold 1 (written False, surprise 0.140), above threshold 2 (written True, surprise 17.421)

## Fit selection

The grid is `i/4` for `i` from 1 to 39. A point is eligible when it writes at most 2 fit lows, at least 6 fit highs, at least 6 prefer and 6 avoid facts, the fit phone and birthday, and all 32 neutral incidentals, and it does not exceed the green-color floor. The highest eligible point is scored. A lower point is tried only when a higher one fails because a required fact was not written. The frozen bank and the held-out bank have not been scored.

Fit bank `4a90e639af898812b9edd13f68cf6c1edebbadaf36c761d0b2d03fdddc30bec0`.

| gate | pass | detail |
|---|---|---|
| high written | yes | 8/8 |
| low written | yes | 0/8 |
| high influence | yes | high 10.727 vs low 0.000 |
| prefer margin | yes | 8/8 > 0, mean 9.913 |
| avoid margin | yes | 6/8 > 0, mean 0.734 |
| forbidden logprob down | reported | 8/8 down, mean stored lift -1.873 |
| neutral NLL | yes | mean -0.015, above 0.5: 0 |
| choice flips | yes | 0/8 |
| leak | yes | 0/8 |
| phone unrelated | yes | 0.000 |
| birthday unrelated | yes | 0.000 |
| phone question | yes | 1 |
| birthday question | yes | 1 |
| incidental written | yes | 32/32 slots 32 |

Fit does not require the held-out mean floors (prefer mean 8, avoid mean 0.6) or forbidden-down. Those are held-out gates.

## Held-out gates

The held-out bank has not been scored. It is merge-critical.

- high-surprise facts written on at least 6/8, low-surprise facts written on at most 2/8
- mean margin delta of the high facts greater than the low facts
- prefer margin delta > 0 on at least 6/8 and mean at least 8.0 (#41 held was 10.905 on 8/8)
- avoid margin delta > 0 on at least 6/8 and mean at least 0.6 (#41 held was 1.277 on 7/8)
- forbidden logprob down on at least 6/8 (#41 held was 8/8, mean stored lift -1.300)
- absolute mean neutral ΔNLL ≤ 0.05, and at most 1 item with ΔNLL > 0.5
- choice flips 0
- leak 0
- phone and birthday takeover 0 off-relation
- phone question takeover 1 and birthday question takeover 1

The stored phone's full digit string is not a gate. The readout still spells the `+1-415-555-` prefix it was trained on. A mismatch between that prefix and the stored number is out of scope here.
