# Subconscious gate

Confidence scale and relation-gated spelling on the #38 inject MLP. SmolLM2-135M, layer 15, hidden 576, float32. gap_tau 0.80. The add is `c * (scale(v) + MLP(scale(v)))`. `c` is 1 for one slot and `clip((top1-top2)/gap_tau, 0, 1)` otherwise. Phone and birthday spelling runs only when the winning relation is the one the prompt is completing. One query per continuation. The MLP weights are the #38 checkpoint.

Frozen bank `e3cdd7fd595d8a76af03a276c2ad750b25b68e1bd8460c9e22e95aa12a820e66`. Held-out bank `cbca1f62c152eb928f498629e8d39a79cdcd9a144589cf2ad8484b65806dd464`. gap_tau was not fit on either bank. Prefer and avoid deltas are measurements, not gates.

## Gates

| bank | gate | pass | detail |
|---|---|---|---|
| frozen | neutral NLL | yes | mean -0.008, |mean| 0.008, above 0.5 : 0 |
| frozen | choice flips | yes | 0/8 |
| frozen | leak | yes | 0/8 |
| frozen | phone unrelated | yes | 0.000 |
| frozen | birthday unrelated | yes | 0.000 |
| frozen | phone question | yes | 1 |
| held | neutral NLL | yes | mean -0.042, |mean| 0.042, above 0.5 : 0 |
| held | choice flips | yes | 0/8 |
| held | leak | yes | 0/8 |
| held | phone unrelated | yes | 0.000 |
| held | birthday unrelated | yes | 0.000 |
| held | phone question | yes | 1 |
| held | birthday question | yes | 1 |

## Prefer and avoid (not gated)

### frozen

- prefer inject: 13.577 (8/8 > 0), choice 8/8 vs nomem 7/8
- prefer context: 3.448 (8/8 > 0), choice 8/8 vs nomem 7/8
- prefer rag: 3.448 (8/8 > 0), choice 8/8 vs nomem 7/8
- avoid inject: -16.700 (0/8 > 0), choice 0/8 vs nomem 6/8
- avoid context: -2.638 (0/8 > 0), choice 1/8 vs nomem 6/8
- avoid rag: -2.638 (0/8 > 0), choice 1/8 vs nomem 6/8
- unrelated N=8: mean -0.010, large 0
- prefer top-1 at N=64: 1/8 (reported, not a gate)

### held

- prefer inject: 15.909 (8/8 > 0), choice 8/8 vs nomem 3/8
- prefer context: 2.317 (8/8 > 0), choice 8/8 vs nomem 3/8
- prefer rag: 2.317 (8/8 > 0), choice 8/8 vs nomem 3/8
- avoid inject: -17.237 (0/8 > 0), choice 0/8 vs nomem 6/8
- avoid context: -3.751 (0/8 > 0), choice 0/8 vs nomem 6/8
- avoid rag: -3.751 (0/8 > 0), choice 0/8 vs nomem 6/8
- unrelated N=8: mean -0.014, large 1
- prefer top-1 at N=64: 0/8 (reported, not a gate)

## Non-interference detail

- frozen: NLL mean -0.008 (0 > 0.5), flips 0/8, leak 0/8, phone unrelated 0.000, birthday unrelated 0.000, phone question 1, birthday question None
- held: NLL mean -0.042 (0 > 0.5), flips 0/8, leak 0/8, phone unrelated 0.000, birthday unrelated 0.000, phone question 1, birthday question 1

## Cost

- MLP parameters: 664704 (the #38 checkpoint; no key projection)
- buffers: 66931
- extra per token, mean over lengths 16/32/64: 0.000372 s
  - length 16: 0.000420 s
  - length 32: 0.001425 s
  - length 64: -0.000730 s
- readout median, hard top-1:
  - N=8: 0.000017 s
  - N=32: 0.000025 s
  - N=64: 0.000033 s
  - N=128: 0.000072 s
  - N=256: 0.000186 s
- query median: 0.029453 s; MLP median: 0.000048 s
- queries per continuation: empty 0, inject 1, phone slot on the decision prompt 1 (forwards 9; spelling stays off because that prompt is not completing phone)
- extra per prompt (queries × query + MLP + readout N=64): 0.029535 s
- continuation wall, 8 new tokens: empty 0.2325 s, inject 0.2672 s, phone slot on that prompt 0.2703 s
- RSS after the cost pass: VmRSS 1282064 kB, VmHWM 1424004 kB
- Variable-cost paths (no retries, no second query on the main path):
  - query encode: skipped when the store is empty
  - scripted phone or birthday: replaces the token loop; any other prompt keeps the residual path
  - span stop: stops before max_new_tokens on a newline or sentence punctuation
  - uncached prefix: each new token forwards the whole prefix; use_cache is False
  - remember: two full forwards (key, then value); chat and noise return before that

Process RSS around the run: before VmPeak 757856 kB, VmHWM 227020 kB, VmRSS 227020 kB; after load VmPeak 3313420 kB, VmHWM 1276144 kB, VmRSS 1057860 kB; after eval VmPeak 3488024 kB, VmHWM 1480836 kB, VmRSS 1294476 kB.

## Misses

- frozen met every non-interference gate.
- held met every non-interference gate.

Numbers above are read from the JSON written by this script. Weights and gap_tau were not changed after these two banks were scored.
