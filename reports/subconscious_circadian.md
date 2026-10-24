# Subconscious circadian clock

A content-free clock sits under the episodic store. `phase = sin(angle)`. Angle 0 leaves write strength at 1 and decay at `0.50 ** age`. A nonzero phase scales the write by `1 + 0.25 * phase` and the decay exponent by `1 - 0.25 * phase`. The angle is not a slot and not a read of the fact text. SmolLM2-135M, layer 15, hidden 576, float32. gap_tau 0.80. Valence stays `u = 1 + 1 tanh(valence)` with step 0.25. Phone and birthday spelling copies the winning slot's stored answer, and only when the prompt is completing that relation. One query per continuation. Replay refresh still sets the top slot back to strength 1. The surprise gate and the retrieval-induced-forgetting gate are not in this model.

Prefer, avoid, and forbidden stay in scope against the #47 baselines (those match #44 and #46) and against #41. They are not out of scope.

Fit bank `ae9c1a9392fbbbf4d4dca134a7a2771a1859a1db6495658067e5292eeaf9a772`. Held-out bank `fb46f7faf4e5b995553a37f036ae77e112faeec7b9cffbe4864280df625ea7af`. The span was chosen on the fit bank before the held-out bank was scored. There is no train bank. Held-out was scored once.

## Gates

| bank | gate | pass | detail |
|---|---|---|---|
| fit | high phase | yes | 8/8 > 0, mean 2.341 |
| fit | low phase | yes | 8/8 < 0, mean -2.140 |
| fit | decay order | yes | 8/8 high > phase0 > low, phase0 exact 1, gap 0.174 |
| fit | write gain | yes | 8/8 match the clock gain |
| fit | content-free | yes | equal gains 1, audit 8/8 |
| fit | neutral NLL | yes | mean -0.009, above 0.5: 0 |
| fit | choice flips | yes | 0/8 |
| fit | leak | yes | 0/8 |
| fit | phone off-relation takeover | yes | 0.000 |
| fit | birthday off-relation takeover | yes | 0.000 |
| fit | phone exact | yes | '+1-278-555-0989' vs '+1-278-555-0989' |
| fit | birthday exact | yes | 'August 25, 2009' vs 'August 25, 2009' |
| fit | phone off-relation not stored | yes | 0 |
| fit | birthday off-relation not stored | yes | 0 |
| fit | incidental slots | yes | 32 slots, written 32/32 |
| held | high phase | yes | 8/8 > 0, mean 1.942 |
| held | high mean | yes | 1.942 >= 0.25 |
| held | low phase | yes | 8/8 < 0, mean -1.677 |
| held | low mean | yes | -1.677 <= -0.25 |
| held | decay order | yes | 8/8 high > phase0 > low, phase0 exact 1, gap 0.174 |
| held | decay gap mean | yes | 0.174 >= 0.05 |
| held | write gain | yes | 8/8 match the clock gain |
| held | content-free | yes | equal gains 1, audit 8/8 |
| held | neutral NLL | yes | mean -0.007, above 0.5: 0 |
| held | choice flips | yes | 0/8 |
| held | leak | yes | 0/8 |
| held | phone off-relation takeover | yes | 0.000 |
| held | birthday off-relation takeover | yes | 0.000 |
| held | phone exact | yes | '+1-722-555-8424' vs '+1-722-555-8424' |
| held | birthday exact | yes | 'October 23, 2013' vs 'October 23, 2013' |
| held | phone off-relation not stored | yes | 0 |
| held | birthday off-relation not stored | yes | 0 |
| held | incidental slots | yes | 32 slots, written 32/32 |
| held | replay prefer mean | yes | 20.750 >= 8.0 (8/8 > 0) |
| held | replay avoid mean | yes | 1.453 >= 0.6 (8/8 > 0) |
| held | replay forbidden down | yes | 8/8 down, mean stored lift -1.578 |

## Exact spelling

- phone question generated '+1-722-555-8424', stored '+1-722-555-8424', exact 1
- birthday question generated 'October 23, 2013', stored 'October 23, 2013', exact 1
- prefix detector on those questions: phone 0, birthday 1. It is not the gate.
- off-relation takeover phone 0.000, birthday 0.000
- off-relation equal to stored: phone 0, birthday 0

## Direction against #47/#44 and #41 held

- prefer mean 20.750 (8/8 > 0), delta -0.000 against #47/#44/#46 20.750, delta 9.845 against #41 10.905
- avoid mean 1.453 (8/8 > 0), delta -0.000 against #47/#44/#46 1.453, delta 0.176 against #41 1.277
- forbidden mean stored lift -1.578 (8/8 down), delta -0.000 against #47/#44/#46 -1.578, delta -0.278 against #41 -1.300

## Cost

- new parameters: 0
- MLP parameters: 664704
- polarity parameters: 576
- total parameters: 665280
- buffers: 66931
- forwards inside one remember: 2
- forwards inside one armed query: 1
- forwards inside one clock tick: 0
- The clock is a global scalar. It does not add a forward and there is no feature flag.

Polarity span is 2. Clock span is 0.25. Valence span is 1 and step is 0.25. Held-out pass: yes.
