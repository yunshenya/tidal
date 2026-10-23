# Subconscious surprise

Surprise-gated writes on the frozen SmolLM2 path. Layer 15, hidden 576, float32. gap_tau 0.80. surprise_tau 2.00. The add is `c * gain * (scale(v) + MLP(scale(v)))`, `gain = 1 + 2 tanh(w · normalize(k))`. A fact is written only when salience times the answer-token NLL of the value forward clears surprise_tau. Chat and noise stay at salience 0. Phone and birthday spelling still runs only when the prompt is completing that relation. One query per continuation.

Held-out bank `41ca5f243131f13deefe85800a775e8378bdeb233954a0b51119d66d93d180f1`. The threshold was chosen on the fit bank before this bank was scored. This is the one held-out pass. The threshold was not changed after it.

## Gates

| gate | pass | detail |
|---|---|---|
| high written | yes | 8/8 |
| low written | no | 6/8 |
| high influence | yes | high 15.140 vs low 6.253 |
| prefer margin | yes | 8/8 > 0, mean 8.842 |
| avoid margin | yes | 7/8 > 0, mean 1.444 |
| forbidden logprob down | yes | 8/8 down, mean stored lift -2.258 |
| neutral NLL | yes | mean -0.003, above 0.5: 0 |
| choice flips | yes | 0/8 |
| leak | yes | 0/8 |
| phone unrelated | yes | 0.000 |
| birthday unrelated | yes | 0.000 |
| phone question | yes | 1 |
| birthday question | yes | 1 |
| incidental written | yes | 32/32 slots 32 |

## High surprise versus low surprise

- high written 8/8, mean margin delta 15.140, surprise 11.620..17.316
- low written 6/8, mean margin delta 6.253, surprise 0.370..7.690

## Prefer and avoid

- prefer: 8/8 > 0, mean 8.842 (delta vs #41 held 10.905: -2.063)
- avoid: 7/8 > 0, mean 1.444 (delta vs #41 held 1.277: 0.167)
- forbidden logprob down: 8/8, mean stored lift -2.258 (delta vs #41 held -1.300: -0.958)
- prefer stored-word lift: 7.018

## Non-interference

- neutral NLL mean -0.003, above 0.5: 0
- choice flips 0/8
- leak 0/8
- incidental slots 32 of 32 written 32
- phone unrelated 0.000, question 1
- birthday unrelated 0.000, question 1

The phone question gate looks for the trained `+1-415-555-` prefix. The stored held-out number is a different prefix. Matching every stored digit is the known readout miss from #41 and is not fixed here.

## Cost

- new parameters: 0
- MLP parameters: 664704
- polarity parameters: 576
- total parameters: 665280
- buffers: 66931
- answer-token log_softmax, median of 30 calls on logits already returned: 0.000070 s (1 answer row(s))
- chat forwards 0, written false
- below-threshold forwards 2, written true, surprise 7.690
- above-threshold forwards 2, written true, surprise 17.316

The cost probe labels the first held low item as below-threshold. That item is "Several friends often thank you kindly." / kindly, surprise 7.690, which is above 2.00, so the row is an above-threshold write. The fit preregistration measured a real below-threshold fact: 1 forward, not written, surprise 0.140.
- No extra model forward. The key forward is skipped when the score is below tau.
- RSS after the score: VmRSS 1066380 kB, VmHWM 1273280 kB

Wall clock 23.3 s. Phase seconds: influence 6.3, neutral 4.0, leak 3.3, scripted 9.6.

## Misses

- held failed: low written. Six of eight held "low" items cleared 2.00, so they were stored. Only two stayed under the threshold.

| item | surprise | written | margin delta |
|---|---:|---|---:|
| Several friends often thank you kindly. / kindly | 7.690 | yes | 14.881 |
| Please stay very calm. / calm | 2.391 | yes | 6.228 |
| Choose either right or wrong. / wrong | 1.768 | no | 0.000 |
| Practice makes perfect. / perfect | 6.423 | yes | 14.766 |
| Better safe than sorry. / sorry | 4.699 | yes | 2.355 |
| Last but not least. / least | 0.370 | no | 0.000 |
| Better late than never. / never | 4.463 | yes | 6.251 |
| Haste makes waste. / waste | 3.498 | yes | 5.545 |

Tau stays 2.00. It was not refit on this bank.

Numbers above are read from the JSON written by this script. surprise_tau was not changed after this pass.
