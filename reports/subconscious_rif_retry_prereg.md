# Retrieval-induced forgetting retry preregistration

Retrieval-induced forgetting. Retrieving a target weakens the other slots that share the winner's relation. The aim is the human pattern of a partial, fuzzy loss of access, not a more useful memory. The held-out bank has not been scored.

No feature flag. `gap_tau` stays 0.80. Polarity gain stays `1 + 2 tanh(w · normalize(k))`. Valence stays `u = 1 + 1 * tanh(valence)` with step 0.25. The clock stays `phase = sin(angle)` with span 0.25 and step pi/16. Phone and birthday spelling stays the exact stored answer, and still runs only when the winning relation is the one the prompt is completing. A continuation still encodes the query once. Readout stays hard top-1. Replay stays every 32 accepted writes, refreshing the top 1 slot, decay 0.50. The surprise quantile stays 0 over the last 32 accepted write surprises. The valence formula is not in this change. The circadian formula is not in this change. The relative surprise gate is not in this change. The inject MLP is not in this change.

After a strict retrieval win, and after that winner's strength has been copied for the current inject, other slots that share the winner's relation are competitors when there are 1 to 3 of them. A larger shared relation is background and is not a cue. A tie adds nothing and forgets nothing. A slot whose salience reaches 0.5 is left alone. The keep is `1 - span * weight`. Weight is 1 for the competitor whose key is closest to the winner and at least 1/2 for every other competitor in that set. The winner is not scaled. There is no extra forward and no new parameter. The set size and the salience exempt are not on the grid.

Prefer, avoid, and forbidden on the replay held-out bank stay in scope. They are not out of scope for this change. Suppression does not run on a one-slot read, and a shared relation larger than three slots is not a cue, so the note stores and the polarity items are not competitors.

## Banks

- fit `3a88dd1f8e3c98b8c2fd96fed96b0903608adc4f6ab6161b28c4daf4aaf3c0ad`
- held `c1ead9b665f56e9ccb1217c1833cae0ad5b1c4b62ca1956665f8bc6c0e94d6e8`
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
- #49 train `ff97715ddd8c693bf35983ca5d527a0e1ba3f107e31d7ea438926cb4bf3a6005`
- #49 fit `87a581a35f0d8ed51e8c394c68d241fc65da9fefbf119c4a1fc99b6fcb5b641d`
- #49 held `c68221e3a8dc696628399f2f7c605bbde8a26f314e77f822b1a2b2e90f35686c`
- literal overlap 0 and token Jaccard under 0.50 against those banks and between fit and held

## Span grid

| span | min drop | mean drop | winner abs | outsider abs | geometry | pass |
|---|---|---|---|---|---|---|
| 0.05 | 0.249 | 0.394 | 0.000 | 0.000 | yes | yes |

Chosen retrieval span: 0.05.
rif span: 0.05

The grid is 0.05, 0.10, 0.20. The smallest span whose fit quartets have minimum competitor drop at least 0.10, mean competitor drop at least 0.20, absolute winner drop at most 0.05, and absolute noncompetitor drop at most 0.05 is the one scored below. The full fit score then runs once. It is not repeated at a larger span if a non-interference floor fails. The held-out bank has not been scored.

Fit bank `3a88dd1f8e3c98b8c2fd96fed96b0903608adc4f6ab6161b28c4daf4aaf3c0ad`.

| gate | pass | detail |
|---|---|---|
| competitors weaken | yes | min drop 0.24924278259277344, mean 0.39403172334035236, winner abs 0.0, geometry yes |
| noncompetitors unaffected | yes | outsider abs 0.0 |
| cost | yes | new 0, forwards 1, text 0, forgotten 3 |
| circadian fit high phase | yes | 8/8 > 0, mean 2.341 |
| circadian fit low phase | yes | 8/8 < 0, mean -2.140 |
| circadian fit decay order | yes | 8/8 high > phase0 > low, phase0 exact 1, gap 0.174 |
| circadian fit write gain | yes | 8/8 match the clock gain |
| circadian fit content-free | yes | equal gains 1, audit 8/8 |
| circadian fit neutral NLL | yes | mean -0.009, above 0.5: 0 |
| circadian fit choice flips | yes | 0/8 |
| circadian fit leak | yes | 0/8 |
| circadian fit phone off-relation takeover | yes | 0.000 |
| circadian fit birthday off-relation takeover | yes | 0.000 |
| circadian fit phone exact | yes | '+1-278-555-0989' vs '+1-278-555-0989' |
| circadian fit birthday exact | yes | 'August 25, 2009' vs 'August 25, 2009' |
| circadian fit phone off-relation not stored | yes | 0 |
| circadian fit birthday off-relation not stored | yes | 0 |
| circadian fit incidental slots | yes | 32 slots, written 32/32 |
| surprise fit low surprise not written | yes | 0/8 written, mean surprise 2.541414961218834 |
| surprise fit high surprise written | yes | 8/8 written, mean surprise 16.595247268676758 |
| surprise fit seed window | yes | written 32/32, slots 32, window 32, bar 9.955883026123047 |
| surprise fit neutral NLL | yes | mean -0.004, above 0.5: 0 |
| surprise fit choice flips | yes | 0/8 |
| surprise fit leak | yes | 0/8 |
| surprise fit phone off-relation takeover | yes | 0.000 |
| surprise fit birthday off-relation takeover | yes | 0.000 |
| surprise fit phone exact | yes | '+1-939-555-2791' vs '+1-939-555-2791' |
| surprise fit birthday exact | yes | 'January 6, 2013' vs 'January 6, 2013' |
| surprise fit phone off-relation not stored | yes | 0 |
| surprise fit birthday off-relation not stored | yes | 0 |
| surprise fit incidental slots | yes | 32 slots, written 32/32 |
| surprise fit cost | yes | new 0, chat 0, below 1/0, above 2/1 |
| valence fit toward experiences | yes | 6/8 > 0, mean 2.085, audit 8/8 |
| valence fit away experiences | yes | 8/8 < 0, mean -7.341, audit 8/8 |
| valence fit contrary revises | yes | margin back 8/8, valence back 8/8, peak 2.085 revised 0.000 |
| valence fit neutral NLL | yes | mean -0.002, above 0.5: 0 |
| valence fit choice flips | yes | 0/8 |
| valence fit leak | yes | 0/8 |
| valence fit phone off-relation takeover | yes | 0.000 |
| valence fit birthday off-relation takeover | yes | 0.000 |
| valence fit phone exact | yes | '+1-301-555-3809' vs '+1-301-555-3809' |
| valence fit birthday exact | yes | 'June 7, 2011' vs 'June 7, 2011' |
| valence fit phone off-relation not stored | yes | 0 |
| valence fit birthday off-relation not stored | yes | 0 |
| valence fit incidental slots | yes | 32 slots, written 32/32 |

Fit does not require the held-out direction mean floors (prefer mean 8.0, avoid mean 0.6) or forbidden-down. Those are held-out gates. Competitor mean drop is a fit gate.

Fit competitor min drop 0.249, mean 0.394, winner abs 0.000, outsider abs 0.000.

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

## Held-out gates

The held-out bank has not been scored. It is merge-critical. These gates are the human effect in the section above, plus the regression floors already required. The numeric floors are unchanged.

- Rp- competitors that share the winner's relation are recalled less than their strength-1 Nrp control after one target hit: minimum drop at least 0.10, mean drop at least 0.20
- the Rp+ target is not weakened: absolute winner drop at most 0.05 (human Rp+ recall was facilitated; this store does not add that facilitation)
- Nrp noncompetitors are unaffected: absolute outsider drop at most 0.05
- geometry: five slots, a strict gap, the target wins, every item hits, and exactly the three competitors are scaled
- prefer margin delta > 0 on 8/8 and mean at least 8.0 on the replay held-out bank (#49/#48/#47/#44/#46 held were 20.750; #41 held was 10.905)
- avoid margin delta > 0 on 8/8 and mean at least 0.6 on the replay held-out bank (#49/#48/#47/#44/#46 held were 1.453; #41 held was 1.277)
- forbidden logprob down on at least 6/8 on the replay held-out bank (#49/#48/#47/#44/#46 held mean stored lift -1.578; #41 held mean stored lift -1.300)
- those direction floors are bit-identical or not regressed; avoid 8/8 is required
- surprise held gates, including neutral NLL, choice flips 0, leak 0, and phone and birthday text equal to the stored answer
- valence held gates, including phone and birthday exact
- circadian held gates, including phone and birthday exact
- new parameters 0 and one forward inside the armed query

Prefer, avoid, and forbidden are not out of scope. Report their deltas against the #49 and #41 held numbers above. Do not retune the span on the held-out bank.

New parameters on the fit cost probe: 0. Query forwards 1. Forgetting record contains text: 0.

