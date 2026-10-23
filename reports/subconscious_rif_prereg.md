# Retrieval-induced forgetting preregistration

Retrieval-induced forgetting. The held-out bank has not been scored.

No feature flag. `gap_tau` stays 0.80. Polarity gain stays `1 + 2 tanh(w · normalize(k))`. Phone and birthday spelling still run only when the winning relation is the one the prompt is completing. A continuation still encodes the query once. Readout stays hard top-1. The surprise gate is not in this change.

When a strict top-1 wins inside a store of at least two slots, the other members of the soft top-4 have their strength multiplied by a fixed keep. The winner is not decayed by that query. A tie adds nothing and decays nothing. Strength multiplies the residual after the gap scale and the polarity gain. A same-key overwrite stores the new value and keeps none of the old one.

## Banks

- fit `2ad8038eefbb5bd77f7d8d3d9e52b30f2c1abf4467713c35b8eacad5bb72fe44`
- held `ef0d06d320c3e8e15f1ac78bbe3e486bdd3017ca37f61b193089303fe5e3d6e9`
- frozen `e3cdd7fd595d8a76af03a276c2ad750b25b68e1bd8460c9e22e95aa12a820e66`
- #39 fit `20b7cb3f9742709d320dc5913cf62675316673d7b457c55b6c78cf4de587e5ce`
- #39 held `ea0a15e5edf0dcab53fede8fad1964b61c23b52c3df55c5f8e00865e59cb053a`
- #40 held `cbca1f62c152eb928f498629e8d39a79cdcd9a144589cf2ad8484b65806dd464`
- #41 train `626ec7cb9ac7824666be6a94ab8e6dc428b7e092ccfc7d8db689fde9c92be4d9`
- #41 fit `ee75f1857bfb6b7e812dfb0ef4012af1dc6efb33ca17799f1211b570ad1c0c1d`
- #41 held `e074b2ad7960bb81bbf64ae0ccd03dbeb0f358428a8a5c6c56e8f373a38d952b`
- #42 fit `4a90e639af898812b9edd13f68cf6c1edebbadaf36c761d0b2d03fdddc30bec0`
- #42 held `41ca5f243131f13deefe85800a775e8378bdeb233954a0b51119d66d93d180f1`
- literal overlap 0 and token Jaccard under 0.50 against those banks and between fit and held

## Decay

The competitive set is the store's soft top-4. The injected value is still the hard top-1. `s` in `c * s * gain * (scale(v) + MLP(scale(v)))` is the winning slot's strength. Strength starts at 1. There is no extra forward: the top-4 are an argpartition of the cosines the gap already computed.

- competitor keep: 0.97
- overwrite keep: 0.00
- new parameters: 0
- MLP parameters: 664704
- polarity parameters: 576
- total parameters: 665280
- buffers: 66931
- forwards inside one armed query: 1
- scale_strength median: 4.780e-07 s over 3 loser slots

## Fit selection

The grid is 0.99, 0.97, 0.95, 0.93, 0.90, 0.85. The largest keep whose fit quartets have minimum competitor drop at least 0.10, mean competitor drop at least 0.20, and absolute winner drop at most 0.05 is the one scored below. The full fit score then runs once. It is not repeated at a stronger decay if a non-interference floor fails. The frozen bank and the held-out bank have not been scored.

Fit bank `2ad8038eefbb5bd77f7d8d3d9e52b30f2c1abf4467713c35b8eacad5bb72fe44`.

| gate | pass | detail |
|---|---|---|
| competitors weaken | yes | min drop 0.101, mean 0.237, winner abs 0.000, geometry yes |
| overwrite follows new | yes | 8/8 margin > 0, slots follow new 8/8 |
| prefer margin | yes | 8/8 > 0, mean 13.437 |
| avoid margin | yes | 8/8 > 0, mean 1.105 |
| forbidden logprob down | reported | 8/8 down, mean stored lift -1.513 |
| neutral NLL | yes | mean 0.007, above 0.5: 0 |
| choice flips | yes | 0/8 |
| leak | yes | 0/8 |
| phone unrelated | yes | 0.000 |
| birthday unrelated | yes | 0.000 |
| phone question | yes | 1 |
| birthday question | yes | 1 |
| incidental slots | yes | 32 slots, written 32/32 |

Fit does not require the held-out mean floors (prefer mean 8, avoid mean 0.6) or forbidden-down. Those are held-out gates.

Fit prefer mean 13.437 (8/8 > 0). Fit avoid mean 1.105 (8/8 > 0). Fit forbidden down 8/8, mean stored lift -1.513.

## Held-out gates

The held-out bank has not been scored. It is merge-critical.

- competitors weaken after one target hit, relative to a strength-1 control: minimum drop at least 0.10, mean drop at least 0.20, absolute winner drop at most 0.05
- same-key overwrite follows the new value, and the new word beats the old word on at least 6/8
- prefer margin delta > 0 on at least 6/8 and mean at least 8.0 (#41 held was 10.905)
- avoid margin delta > 0 on at least 6/8 and mean at least 0.6 (#41 held was 1.277)
- forbidden logprob down on at least 6/8 (#41 held mean stored lift -1.300)
- absolute mean neutral ΔNLL ≤ 0.05, and at most 1 item with ΔNLL > 0.5
- choice flips 0
- leak 0
- phone and birthday takeover 0 off-relation
- phone question takeover 1 and birthday question takeover 1
- incidental store keeps 32 slots

Report the prefer, avoid, and forbidden deltas against the #41 held numbers above. Do not retune the keep on this bank.

The stored phone's full digit string is not a gate. The readout still spells the `+1-415-555-` prefix it was trained on. A mismatch between that prefix and the stored number is out of scope here.

