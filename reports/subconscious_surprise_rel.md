# Subconscious relative surprise

A fact is written when it is salient and, once 32 accepted surprises are stored, its answer-token NLL is at least the quantile of that window. Fewer stored surprises leaves the write accepted. The window stores numbers only. Rejected attempts stay out of it. SmolLM2-135M, layer 15, hidden 576, float32. gap_tau 0.80. Quantile 0 over the last 32 accepted write surprises. Valence stays `u = 1 + 1 tanh(valence)` with step 0.25. Clock span stays 0.25. Phone and birthday spelling copies the winning slot's stored answer, and only when the prompt is completing that relation. One query per continuation. Replay stays every 32 accepted writes. The retrieval-induced-forgetting gate is not in this model.

Prefer, avoid, and forbidden stay in scope against the #48 baselines (those match #47, #44, and #46) and against #41. They are not out of scope.

Train bank `ff97715ddd8c693bf35983ca5d527a0e1ba3f107e31d7ea438926cb4bf3a6005`. Fit bank `87a581a35f0d8ed51e8c394c68d241fc65da9fefbf119c4a1fc99b6fcb5b641d`. Held-out bank `c68221e3a8dc696628399f2f7c605bbde8a26f314e77f822b1a2b2e90f35686c`. The quantile was chosen on the fit bank before the held-out bank was scored. The train bank is the post-warmup keep floor. Held-out was scored once.

## Gates

| bank | gate | pass | detail |
|---|---|---|---|
| fit | low surprise not written | yes | 0/8 written, mean surprise 2.541414961218834 |
| fit | high surprise written | yes | 8/8 written, mean surprise 16.595247268676758 |
| fit | seed window | yes | written 32/32, slots 32, window 32, bar 9.955883026123047 |
| fit | neutral NLL | yes | mean -0.004, above 0.5: 0 |
| fit | choice flips | yes | 0/8 |
| fit | leak | yes | 0/8 |
| fit | phone off-relation takeover | yes | 0.000 |
| fit | birthday off-relation takeover | yes | 0.000 |
| fit | phone exact | yes | '+1-939-555-2791' vs '+1-939-555-2791' |
| fit | birthday exact | yes | 'January 6, 2013' vs 'January 6, 2013' |
| fit | phone off-relation not stored | yes | 0 |
| fit | birthday off-relation not stored | yes | 0 |
| fit | incidental slots | yes | 32 slots, written 32/32 |
| fit | cost | yes | new 0, chat 0, below 1/0, above 2/1 |
| train | keeps written | yes | 8/8 after 32 seed slots |
| held | low surprise not written | yes | 0/8 written, mean surprise 1.7938727140426636 |
| held | high surprise written | yes | 8/8 written, mean surprise 17.729726552963257 |
| held | seed window | yes | written 32/32, slots 32, window 32, bar 10.621371269226074 |
| held | neutral NLL | yes | mean 0.000, above 0.5: 0 |
| held | choice flips | yes | 0/8 |
| held | leak | yes | 0/8 |
| held | phone off-relation takeover | yes | 0.000 |
| held | birthday off-relation takeover | yes | 0.000 |
| held | phone exact | yes | '+1-296-555-7308' vs '+1-296-555-7308' |
| held | birthday exact | yes | 'May 20, 1974' vs 'May 20, 1974' |
| held | phone off-relation not stored | yes | 0 |
| held | birthday off-relation not stored | yes | 0 |
| held | incidental slots | yes | 32 slots, written 32/32 |
| held | cost | yes | new 0, chat 0, below 1/0, above 2/1 |
| held | replay prefer mean | yes | 20.750 >= 8.0 (8/8 > 0) |
| held | replay avoid mean | yes | 1.453 >= 0.6 (8/8 > 0) |
| held | replay forbidden down | yes | 8/8 down, mean stored lift -1.578 |
| valence held | toward experiences | yes | 6/8 > 0, mean 1.865, audit 8/8 |
| valence held | toward mean | yes | 1.865 >= 0.25 |
| valence held | away experiences | yes | 8/8 < 0, mean -10.242, audit 8/8 |
| valence held | away mean | yes | -10.242 <= -0.25 |
| valence held | contrary revises | yes | margin back 8/8, valence back 8/8, peak 1.865 revised 0.000 |
| valence held | neutral NLL | yes | mean -0.004, above 0.5: 0 |
| valence held | choice flips | yes | 0/8 |
| valence held | leak | yes | 0/8 |
| valence held | phone off-relation takeover | yes | 0.000 |
| valence held | birthday off-relation takeover | yes | 0.000 |
| valence held | phone exact | yes | '+1-610-555-5746' vs '+1-610-555-5746' |
| valence held | birthday exact | yes | 'April 4, 1996' vs 'April 4, 1996' |
| valence held | phone off-relation not stored | yes | 0 |
| valence held | birthday off-relation not stored | yes | 0 |
| valence held | incidental slots | yes | 32 slots, written 32/32 |
| circadian held | high phase | yes | 8/8 > 0, mean 1.942 |
| circadian held | high mean | yes | 1.942 >= 0.25 |
| circadian held | low phase | yes | 8/8 < 0, mean -1.677 |
| circadian held | low mean | yes | -1.677 <= -0.25 |
| circadian held | decay order | yes | 8/8 high > phase0 > low, phase0 exact 1, gap 0.174 |
| circadian held | decay gap mean | yes | 0.174 >= 0.05 |
| circadian held | write gain | yes | 8/8 match the clock gain |
| circadian held | content-free | yes | equal gains 1, audit 8/8 |
| circadian held | neutral NLL | yes | mean -0.007, above 0.5: 0 |
| circadian held | choice flips | yes | 0/8 |
| circadian held | leak | yes | 0/8 |
| circadian held | phone off-relation takeover | yes | 0.000 |
| circadian held | birthday off-relation takeover | yes | 0.000 |
| circadian held | phone exact | yes | '+1-722-555-8424' vs '+1-722-555-8424' |
| circadian held | birthday exact | yes | 'October 23, 2013' vs 'October 23, 2013' |
| circadian held | phone off-relation not stored | yes | 0 |
| circadian held | birthday off-relation not stored | yes | 0 |
| circadian held | incidental slots | yes | 32 slots, written 32/32 |

## Surprise contrast

- held lows written 0/8, mean surprise 1.794
- held highs written 8/8, mean surprise 17.730
- held window bar 10.621 at quantile 0
- train keeps 8/8

## Exact spelling

- phone question generated '+1-296-555-7308', stored '+1-296-555-7308', exact 1
- birthday question generated 'May 20, 1974', stored 'May 20, 1974', exact 1
- prefix detector on those questions: phone 0, birthday 1. It is not the gate.
- off-relation takeover phone 0.000, birthday 0.000
- off-relation equal to stored: phone 0, birthday 0

## Direction against #48/#47/#44 and #41 held

- prefer mean 20.750 (8/8 > 0), delta -0.000 against #48/#47/#44/#46 20.750, delta 9.845 against #41 10.905
- avoid mean 1.453 (8/8 > 0), delta -0.000 against #48/#47/#44/#46 1.453, delta 0.176 against #41 1.277
- forbidden mean stored lift -1.578 (8/8 down), delta -0.000 against #48/#47/#44/#46 -1.578, delta -0.278 against #41 -1.300

## Cost

- new parameters: 0
- MLP parameters: 664704
- polarity parameters: 576
- total parameters: 665280
- buffers: 66931
- chat forwards: 0
- warmup forwards: 2
- below-window forwards: 1, written 0
- above-window forwards: 2, written 1
- The gate reads the value forward's logits. It does not add a third forward and there is no feature flag.

Polarity span is 2. Surprise quantile is 0. Window is 32. Valence span is 1 and step is 0.25. Clock span is 0.25. Replay decay is 0.50. Held-out pass: yes.
