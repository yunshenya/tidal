# Subconscious retrieval-induced forgetting

When soft top-4 retrieves and top-1 wins, the other competitors in that batch are scaled by a fixed keep. A same-key overwrite keeps none of the old value. SmolLM2-135M, layer 15, hidden 576, float32. gap_tau 0.80. The add is `c * s * gain * (scale(v) + MLP(scale(v)))`, `gain = 1 + 2 tanh(w · normalize(k))`, `s` kept at 0.97 for a loser and 0.00 for a value that was corrected. Phone and birthday spelling still runs only when the prompt is completing that relation. One query per continuation. The surprise gate is not in this model.

Fit bank `2ad8038eefbb5bd77f7d8d3d9e52b30f2c1abf4467713c35b8eacad5bb72fe44`. Held-out bank `ef0d06d320c3e8e15f1ac78bbe3e486bdd3017ca37f61b193089303fe5e3d6e9`. The keep was chosen on the fit bank before the held-out bank was scored. Held-out was scored once.

## Gates

| bank | gate | pass | detail |
|---|---|---|---|
| fit | competitors weaken | yes | min drop 0.101, mean 0.237, winner abs 0.000, geometry yes |
| fit | overwrite follows new | yes | 8/8 margin > 0, slots follow new 8/8 |
| fit | prefer margin | yes | 8/8 > 0, mean 13.437 |
| fit | avoid margin | yes | 8/8 > 0, mean 1.105 |
| fit | forbidden logprob down | reported | 8/8 down, mean stored lift -1.513 |
| fit | neutral NLL | yes | mean 0.007, above 0.5: 0 |
| fit | choice flips | yes | 0/8 |
| fit | leak | yes | 0/8 |
| fit | phone unrelated | yes | 0.000 |
| fit | birthday unrelated | yes | 0.000 |
| fit | phone question | yes | 1 |
| fit | birthday question | yes | 1 |
| fit | incidental slots | yes | 32 slots, written 32/32 |
| held | competitors weaken | no | min drop 0.103, mean 0.197, winner abs 0.000, geometry yes |
| held | overwrite follows new | yes | 7/8 margin > 0, slots follow new 8/8 |
| held | prefer margin | yes | 8/8 > 0, mean 9.692 |
| held | prefer mean | yes | 9.692 >= 8.0 |
| held | avoid margin | no | 4/8 > 0, mean -0.161 |
| held | avoid mean | no | -0.161 >= 0.6 |
| held | forbidden logprob down | yes | 6/8 down, mean stored lift -0.456 |
| held | neutral NLL | yes | mean 0.003, above 0.5: 0 |
| held | choice flips | yes | 0/8 |
| held | leak | yes | 0/8 |
| held | phone unrelated | yes | 0.000 |
| held | birthday unrelated | yes | 0.000 |
| held | phone question | yes | 1 |
| held | birthday question | yes | 1 |
| held | incidental slots | yes | 32 slots, written 32/32 |

## Direction against #41 held

- prefer mean 9.692 (8/8 > 0), delta -1.213 against 10.905
- avoid mean -0.161 (4/8 > 0), delta -1.438 against 1.277
- forbidden mean stored lift -0.456 (6/8 down), delta 0.844 against -1.300

## Cost

- new parameters: 0
- MLP parameters: 664704
- polarity parameters: 576
- total parameters: 665280
- buffers: 66931
- forwards inside one armed query: 1
- scale_strength median: 5.240e-07 s over 3 loser slots
- The top-4 are the cosines already computed for the gap. There is no second query and no feature flag.

The stored phone's full digit string is not a gate. The readout still spells the `+1-415-555-` prefix it was trained on.

Polarity span is 2. Held-out pass: no.
