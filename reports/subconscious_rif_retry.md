# Subconscious retrieval-induced forgetting retry

Retrieving a target multiplies the strength of at most three other slots that share the winner's relation. The closer key is scaled down more. The winner is left as read. A tie, a larger shared relation, a high-salience slot, and every other relation stay put. SmolLM2-135M, layer 15, hidden 576, float32. gap_tau 0.80. Span 0.05. Set cap 3. Salience exempt 0.5. Valence stays `u = 1 + 1 tanh(valence)` with step 0.25. Clock span stays 0.25. Surprise quantile stays 0 over the last 32 accepted surprises. Phone and birthday spelling copies the winning slot's stored answer, and only when the prompt is completing that relation. One query per continuation. Replay stays every 32 accepted writes.

Prefer, avoid, and forbidden stay in scope against the #49 baselines (those match #48, #47, #44, and #46) and against #41. They are not out of scope.

Fit bank `3a88dd1f8e3c98b8c2fd96fed96b0903608adc4f6ab6161b28c4daf4aaf3c0ad`. Held-out bank `c1ead9b665f56e9ccb1217c1833cae0ad5b1c4b62ca1956665f8bc6c0e94d6e8`. The span was chosen on the fit bank before the held-out bank was scored. Held-out was scored once.

## Gates

| bank | gate | pass | detail |
|---|---|---|---|
| fit | competitors weaken | yes | min drop 0.24924278259277344, mean 0.39403172334035236, winner abs 0.0, geometry yes |
| fit | noncompetitors unaffected | yes | outsider abs 0.0 |
| fit | cost | yes | new 0, forwards 1, text 0, forgotten 3 |
| held | competitors weaken | yes | min drop 0.19124603271484375, mean 0.312061071395874, winner abs 0.0, geometry yes |
| held | noncompetitors unaffected | yes | outsider abs 0.0 |
| held | cost | yes | new 0, forwards 1, text 0, forgotten 3 |
| held | replay prefer count | yes | 8/8 > 0, mean 20.750 >= 8.0 |
| held | replay avoid count | yes | 8/8 > 0, mean 1.453 >= 0.6 |
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
| surprise held | low surprise not written | yes | 0/8 written, mean surprise 1.7938727140426636 |
| surprise held | high surprise written | yes | 8/8 written, mean surprise 17.729726552963257 |
| surprise held | seed window | yes | written 32/32, slots 32, window 32, bar 10.621371269226074 |
| surprise held | neutral NLL | yes | mean 0.000, above 0.5: 0 |
| surprise held | choice flips | yes | 0/8 |
| surprise held | leak | yes | 0/8 |
| surprise held | phone off-relation takeover | yes | 0.000 |
| surprise held | birthday off-relation takeover | yes | 0.000 |
| surprise held | phone exact | yes | '+1-296-555-7308' vs '+1-296-555-7308' |
| surprise held | birthday exact | yes | 'May 20, 1974' vs 'May 20, 1974' |
| surprise held | phone off-relation not stored | yes | 0 |
| surprise held | birthday off-relation not stored | yes | 0 |
| surprise held | incidental slots | yes | 32 slots, written 32/32 |
| surprise held | cost | yes | new 0, chat 0, below 1/0, above 2/1 |

## Competitor contrast

- held min drop 0.191, mean 0.312
- held winner abs 0.000, outsider abs 0.000
- geometry yes

## Direction against #49 and #41 held

- prefer mean 20.750 (8/8 > 0), delta -0.000 against #49 20.750, delta 9.845 against #41 10.905
- avoid mean 1.453 (8/8 > 0), delta -0.000 against #49 1.453, delta 0.176 against #41 1.277
- forbidden mean stored lift -1.578 (8/8 down), delta -0.000 against #49 -1.578, delta -0.278 against #41 -1.300
- unrounded prefer mean 20.74961856333539, avoid mean 1.452641248703003, forbidden mean stored lift -1.5781219005584717, the same values as the #49 held score

## Cost

- new parameters: 0
- MLP parameters: 664704
- polarity parameters: 576
- total parameters: 665280
- buffers: 66931
- forwards inside one armed query: 1
- forgetting record contains text: 0
- Suppression multiplies stored strengths. It does not add a forward and there is no feature flag.

Polarity span is 2. Retrieval span is 0.05. Valence span is 1 and step is 0.25. Clock span is 0.25. Replay decay is 0.50. Held-out pass: yes.

## Human effect

The goal is a human-like memory, including fuzzy recall and ordinary errors, rather than a store that is as useful as possible. Retrieving a target should make the other items on that same cue a little harder to recall, and should leave them in memory. The floors already fixed below are that prediction. They are not raised after the held-out bank is scored.

Anderson, Bjork, and Bjork (1994) showed that practicing retrieval of some category members impairs later recall of the unpracticed members of those same categories. In their retrieval-practice paradigm the practiced items are Rp+, the unpracticed items that share the practiced category cue are Rp-, and items from categories that were not practiced are Nrp. Experiment 1, category-cued recall after a 20-minute delay: Rp+ 73.6%, Rp- 37.5%, Nrp 48.4%. The Rp- impairment against that Nrp baseline was reliable, F(1, 32) = 30.3. Strong exemplars were impaired by 15.7 points (Rp- 40.3% versus Nrp 56.0%). Weak exemplars were impaired by 6.3 points (Rp- 34.7% versus Nrp 41.0%). The practiced items were facilitated, not weakened. The loss is partial: the Rp- items are still recalled, only less often.

Anderson, M. C. (2003) accounts for that pattern as inhibition recruited to override competitors during selective retrieval, rather than as passive blocking caused by strengthening the practiced item. The properties reviewed there are retrieval specificity, interference dependence (strong competitors are impaired more; weak ones often are not), and strength independence (the impairment does not grow just because the practiced item was facilitated more). Cue-independent suppression is part of that account. This change does not implement an independent-cue test, and it does not claim prefrontal control.

Norman, K. A., Newman, E. L., and Detre, G. (2007) give a network account in which oscillating feedback inhibition weakens competitors that threaten the target and strengthens weak parts of the target. The consequence taken from that model is competitor punishment after the target has been selected. The mechanism here is not their oscillating k-winner-take-all rule, not contrastive Hebbian learning, and not a cortical-hippocampal network. It multiplies the strength of a small same-relation set. A competitor whose key is closer to the winner is scaled down more, which is the local stand-in for interference dependence. The winner itself is not scaled.

How the existing floors express that effect, in logprob-margin nats rather than percent recall:
- A competitor is an Rp- item. It shares the winner's relation and it is not the retrieved target. After one target hit it must sit below its own strength-1 control, which is the Nrp baseline for that item. Minimum drop at least 0.10. Mean drop at least 0.20. At span 0.05 the keep is between 0.95 and 0.975, so the item is impaired and not erased.
- The target is the Rp+ item. Human Rp+ recall rose from the Nrp baseline to 73.6%. This store does not add that facilitation. The gate is the part this mechanism can get wrong: the target is not weakened. Absolute winner drop at most 0.05.
- An outsider is an Nrp item. Its relation was not the practiced cue, so retrieval of the target should leave it alone. Absolute drop at most 0.05.
The human figures are category-cued recall percentages. These gates are not a conversion of those percentages into nats. The direction is Rp- below Nrp, Rp+ not impaired, Nrp unaffected, and the size is a partial drop of a few tenths of a nat.

Anderson, M. C., Bjork, R. A., & Bjork, E. L. (1994). Remembering can cause forgetting: Retrieval dynamics in long-term memory. Journal of Experimental Psychology: Learning, Memory, and Cognition, 20(5), 1063-1087. https://doi.org/10.1037/0278-7393.20.5.1063

Anderson, M. C. (2003). Rethinking interference theory: Executive control and the mechanisms of forgetting. Journal of Memory and Language, 49(4), 415-445. https://doi.org/10.1016/j.jml.2003.08.006

Norman, K. A., Newman, E. L., & Detre, G. (2007). A neural network model of retrieval-induced forgetting. Psychological Review, 114(4), 887-953. https://doi.org/10.1037/0033-295X.114.4.887

## This score against that effect

- Rp- competitors versus their strength-1 Nrp controls: min drop 0.191, mean drop 0.312. Anderson, Bjork, and Bjork (1994) predicted Rp- below Nrp, by about 11 recall points overall (37.5% versus 48.4%) and by 15.7 points for strong exemplars or 6.3 points for weak ones. This row is that direction in logprob margin: a partial drop, with the items still stored. The pre-registered floors are a minimum of 0.10 and a mean of 0.20. This bank clears them.
- Rp+ target absolute drop 0.000. Human Rp+ recall was facilitated (73.6% versus the 48.4% Nrp baseline). This mechanism does not add that facilitation. The pre-registered gate is that the target is not weakened (absolute drop at most 0.05). This bank clears it.
- Nrp outsider absolute drop 0.000. A different relation was not the practiced cue, so it should stay put (absolute drop at most 0.05). This bank clears it.
The human figures are category-cued recall percentages. These drops are logprob-margin nats. The comparison is the direction and a partial size, not a conversion of percent recall into nats. Fuzzy recall is the intended outcome. The floors were not retuned on this bank.

All gates and floors passed.
