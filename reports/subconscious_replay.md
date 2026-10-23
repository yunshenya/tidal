# Subconscious replay consolidation

Every N accepted writes, decay is recomputed, the top-m salient slots are refreshed, and at most m near-duplicates are merged. SmolLM2-135M, layer 15, hidden 576, float32. gap_tau 0.80. The add is `c * s * gain * (scale(v) + MLP(scale(v)))`, `gain = 1 + 2 tanh(w · normalize(k))`, `s` refreshed for the top 1 slot every 32 accepted writes. Decay is 0.50 per cycle. Phone and birthday spelling still runs only when the prompt is completing that relation. One query per continuation. The surprise gate and the retrieval-induced-forgetting gate are not in this model.

Fit bank `35ae15cde3079cee5b034e2e019ef4a76324574507a8dbb58831bdb141b9f552`. Held-out bank `e684c422049c720b269a50d762ca8ec1be87b397c93068ea2d61fa20ad4127ce`. N and m were chosen on the fit bank before the held-out bank was scored. Held-out was scored once.

## Gates

| bank | gate | pass | detail |
|---|---|---|---|
| fit | replay consolidates | yes | salient gain 0.500, slot drop 1, near-dup drop 2 |
| fit | prefer margin | yes | 8/8 > 0, mean 21.121 |
| fit | avoid margin | yes | 8/8 > 0, mean 0.886 |
| fit | forbidden logprob down | reported | 8/8 down, mean stored lift -1.121 |
| fit | neutral NLL | yes | mean -0.035, above 0.5: 0 |
| fit | choice flips | yes | 0/8 |
| fit | leak | yes | 0/8 |
| fit | phone unrelated | yes | 0.000 |
| fit | birthday unrelated | yes | 0.000 |
| fit | phone question | yes | 1 |
| fit | birthday question | yes | 1 |
| fit | incidental slots | yes | 32 slots, written 32/32 |
| held | replay consolidates | yes | salient gain 0.500, slot drop 1, near-dup drop 2 |
| held | prefer margin | yes | 8/8 > 0, mean 20.750 |
| held | prefer mean | yes | 20.750 >= 8.0 |
| held | avoid margin | yes | 8/8 > 0, mean 1.453 |
| held | avoid mean | yes | 1.453 >= 0.6 |
| held | forbidden logprob down | yes | 8/8 down, mean stored lift -1.578 |
| held | neutral NLL | yes | mean -0.011, above 0.5: 0 |
| held | choice flips | yes | 0/8 |
| held | leak | yes | 0/8 |
| held | phone unrelated | yes | 0.000 |
| held | birthday unrelated | yes | 0.000 |
| held | phone question | yes | 1 |
| held | birthday question | yes | 1 |
| held | incidental slots | yes | 32 slots, written 32/32 |

## Direction against #41 held

- prefer mean 20.750 (8/8 > 0), delta 9.845 against 10.905
- avoid mean 1.453 (8/8 > 0), delta 0.176 against 1.277
- forbidden mean stored lift -1.578 (8/8 down), delta -0.278 against -1.300

## Cost

- new parameters: 0
- MLP parameters: 664704
- polarity parameters: 576
- total parameters: 665280
- buffers: 66931
- forwards inside one remember that consolidates: 2
- forwards inside consolidate alone: 0
- forwards inside one armed query: 1
- consolidate median: 3.055e-03 s
- The scan is the slots already in the store. There is no second query and no feature flag.

The stored phone's full digit string is not a gate. The readout still spells the `+1-415-555-` prefix it was trained on.

Polarity span is 2. Held-out pass: yes.
