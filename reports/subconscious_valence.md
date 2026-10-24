# Subconscious experience valence

Each episodic slot carries a scalar valence. A write's experience field adds its sign times the step. Replay multiplies a nonzero valence by 0.50. The add is `c * s * u * gain * (scale(v) + MLP(scale(v)))`, `u = 1 + 1 tanh(valence)`, `gain = 1 + 2 tanh(w · normalize(k))`. SmolLM2-135M, layer 15, hidden 576, float32. gap_tau 0.80. Phone and birthday spelling copies the winning slot's stored answer, and only when the prompt is completing that relation. One query per continuation. The fact text is not scored. The surprise gate and the retrieval-induced-forgetting gate are not in this model.

Prefer, avoid, and forbidden stay in scope against the #44 and #46 baselines (those two held scores match) and against #41. They are not out of scope.

Fit bank `44af0fcf6d10c350cb8e8861ccc5489cd09b9e0807bf8795033aa33adf941d80`. Held-out bank `017d1386329cff64df4cbac7ad07c2cc69beae813b237c2f8c8ed9b069d029d5`. Span and step were chosen on the fit bank before the held-out bank was scored. Held-out was scored once.

## Gates

| bank | gate | pass | detail |
|---|---|---|---|
| fit | toward experiences | yes | 6/8 > 0, mean 2.085, audit 8/8 |
| fit | away experiences | yes | 8/8 < 0, mean -7.341, audit 8/8 |
| fit | contrary revises | yes | margin back 8/8, valence back 8/8, peak 2.085 revised 0.000 |
| fit | neutral NLL | yes | mean -0.002, above 0.5: 0 |
| fit | choice flips | yes | 0/8 |
| fit | leak | yes | 0/8 |
| fit | phone off-relation takeover | yes | 0.000 |
| fit | birthday off-relation takeover | yes | 0.000 |
| fit | phone exact | yes | '+1-301-555-3809' vs '+1-301-555-3809' |
| fit | birthday exact | yes | 'June 7, 2011' vs 'June 7, 2011' |
| fit | phone off-relation not stored | yes | 0 |
| fit | birthday off-relation not stored | yes | 0 |
| fit | incidental slots | yes | 32 slots, written 32/32 |
| held | toward experiences | yes | 6/8 > 0, mean 1.865, audit 8/8 |
| held | toward mean | yes | 1.865 >= 0.25 |
| held | away experiences | yes | 8/8 < 0, mean -10.242, audit 8/8 |
| held | away mean | yes | -10.242 <= -0.25 |
| held | contrary revises | yes | margin back 8/8, valence back 8/8, peak 1.865 revised 0.000 |
| held | neutral NLL | yes | mean -0.004, above 0.5: 0 |
| held | choice flips | yes | 0/8 |
| held | leak | yes | 0/8 |
| held | phone off-relation takeover | yes | 0.000 |
| held | birthday off-relation takeover | yes | 0.000 |
| held | phone exact | yes | '+1-610-555-5746' vs '+1-610-555-5746' |
| held | birthday exact | yes | 'April 4, 1996' vs 'April 4, 1996' |
| held | phone off-relation not stored | yes | 0 |
| held | birthday off-relation not stored | yes | 0 |
| held | incidental slots | yes | 32 slots, written 32/32 |
| held | replay prefer mean | yes | 20.750 >= 8.0 (8/8 > 0) |
| held | replay avoid mean | yes | 1.453 >= 0.6 (8/8 > 0) |
| held | replay forbidden down | yes | 8/8 down, mean stored lift -1.578 |

## Exact spelling

- phone question generated '+1-610-555-5746', stored '+1-610-555-5746', exact 1
- birthday question generated 'April 4, 1996', stored 'April 4, 1996', exact 1
- prefix detector on those questions: phone 0, birthday 1. It is not the gate.
- off-relation takeover phone 0.000, birthday 0.000
- off-relation equal to stored: phone 0, birthday 0

## Direction against #44 and #41 held

- prefer mean 20.750 (8/8 > 0), delta -0.000 against #44/#46 20.750, delta 9.845 against #41 10.905
- avoid mean 1.453 (8/8 > 0), delta -0.000 against #44/#46 1.453, delta 0.176 against #41 1.277
- forbidden mean stored lift -1.578 (8/8 down), delta -0.000 against #44/#46 -1.578, delta -0.278 against #41 -1.300

## Cost

- new parameters: 0
- MLP parameters: 664704
- polarity parameters: 576
- total parameters: 665280
- buffers: 66931
- forwards inside one remember: 2
- forwards inside one armed query: 1
- An experience is a scalar on the slot. It does not add a forward and there is no feature flag.

Polarity span is 2. Valence span is 1 and step is 0.25. Held-out pass: yes.
