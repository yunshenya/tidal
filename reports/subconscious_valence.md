# Subconscious experience valence

Each episodic slot carries a scalar valence. A write's experience field adds its sign times the step. Replay multiplies a nonzero valence by 0.50. The add is `c * s * u * gain * (scale(v) + MLP(scale(v)))`, `u = 1 + 1 tanh(valence)`, `gain = 1 + 2 tanh(w · normalize(k))`. SmolLM2-135M, layer 15, hidden 576, float32. gap_tau 0.80. Phone and birthday spelling still runs only when the prompt is completing that relation. One query per continuation. The fact text is not scored. The surprise gate and the retrieval-induced-forgetting gate are not in this model.

Fit bank `72a1836feafaf6df89347b87f527cdc80cbe64d3709746c5ca0e57c89ad814c8`. Held-out bank `a4b5ce1da2ce61070720487e7e3e5e9d5bc8d78a9b5a1281ea69c9e2223fcbd5`. Span and step were chosen on the fit bank before the held-out bank was scored. Held-out was scored once.

## Gates

| bank | gate | pass | detail |
|---|---|---|---|
| fit | toward experiences | yes | 8/8 > 0, mean 3.037, audit 8/8 |
| fit | away experiences | yes | 8/8 < 0, mean -10.879, audit 8/8 |
| fit | contrary revises | yes | margin back 8/8, valence back 8/8, peak 3.037 revised 0.000 |
| fit | neutral NLL | yes | mean -0.011, above 0.5: 0 |
| fit | choice flips | yes | 0/8 |
| fit | leak | yes | 0/8 |
| fit | phone unrelated | yes | 0.000 |
| fit | birthday unrelated | yes | 0.000 |
| fit | phone question | yes | 1 |
| fit | birthday question | yes | 1 |
| fit | incidental slots | yes | 32 slots, written 32/32 |
| held | toward experiences | yes | 6/8 > 0, mean 1.275, audit 8/8 |
| held | toward mean | yes | 1.275 >= 0.25 |
| held | away experiences | yes | 8/8 < 0, mean -11.274, audit 8/8 |
| held | away mean | yes | -11.274 <= -0.25 |
| held | contrary revises | yes | margin back 8/8, valence back 8/8, peak 1.275 revised 0.000 |
| held | neutral NLL | yes | mean -0.031, above 0.5: 0 |
| held | choice flips | yes | 0/8 |
| held | leak | yes | 0/8 |
| held | phone unrelated | yes | 0.000 |
| held | birthday unrelated | yes | 0.000 |
| held | phone question | yes | 1 |
| held | birthday question | no | 0 |
| held | incidental slots | yes | 32 slots, written 32/32 |
| held | replay prefer mean | yes | 20.750 >= 8.0 (8/8 > 0) |
| held | replay avoid mean | yes | 1.453 >= 0.6 (8/8 > 0) |
| held | replay forbidden down | yes | 8/8 down, mean stored lift -1.578 |

## Direction against #44 and #41 held

- prefer mean 20.750 (8/8 > 0), delta -0.000 against #44 20.750, delta 9.845 against #41 10.905
- avoid mean 1.453 (8/8 > 0), delta -0.000 against #44 1.453, delta 0.176 against #41 1.277
- forbidden mean stored lift -1.578 (8/8 down), delta -0.000 against #44 -1.578, delta -0.278 against #41 -1.300

## Cost

- new parameters: 0
- MLP parameters: 664704
- polarity parameters: 576
- total parameters: 665280
- buffers: 66931
- forwards inside one remember: 2
- forwards inside one armed query: 1
- An experience is a scalar on the slot. It does not add a forward and there is no feature flag.

The stored phone's full digit string is not a gate. The readout still spells the `+1-415-555-` prefix it was trained on.

Polarity span is 2. Valence span is 1 and step is 0.25. Held-out pass: no.
