# Subconscious direction

Prefer versus avoid polarity on the frozen #38 MLP. SmolLM2-135M, layer 15, hidden 576, float32. gap_tau 0.80. The add is `c * gain * (scale(v) + MLP(scale(v)))`, `gain = 1 + 2 tanh(w · normalize(k))`. Phone and birthday spelling still runs only when the prompt is completing that relation. One query per continuation.

Frozen bank `e3cdd7fd595d8a76af03a276c2ad750b25b68e1bd8460c9e22e95aa12a820e66`. Held-out bank `e074b2ad7960bb81bbf64ae0ccd03dbeb0f358428a8a5c6c56e8f373a38d952b`. The checkpoint was chosen on the fit bank before either of these was scored. Held-out is the merge-critical gate. Frozen is a measurement under the same gates.

## Gates

| bank | gate | pass | detail |
|---|---|---|---|
| frozen | prefer margin | yes | 8/8 > 0, mean 15.050 |
| frozen | avoid margin | no | 1/8 > 0, mean -2.640 |
| frozen | forbidden logprob down | no | 3/8 down, mean stored lift 0.728 |
| frozen | neutral NLL | yes | mean -0.009, above 0.5: 0 |
| frozen | choice flips | yes | 0/8 |
| frozen | leak | yes | 0/8 |
| frozen | phone unrelated | yes | 0.000 |
| frozen | birthday unrelated | yes | 0.000 |
| frozen | phone question | yes | 1 |
| held | prefer margin | yes | 8/8 > 0, mean 10.905 |
| held | avoid margin | yes | 7/8 > 0, mean 1.277 |
| held | forbidden logprob down | yes | 8/8 down, mean stored lift -1.300 |
| held | neutral NLL | yes | mean -0.024, above 0.5: 0 |
| held | choice flips | yes | 0/8 |
| held | leak | yes | 0/8 |
| held | phone unrelated | yes | 0.000 |
| held | birthday unrelated | yes | 0.000 |
| held | phone question | yes | 1 |
| held | birthday question | yes | 1 |

## Prefer and avoid

### frozen

- prefer inject: 15.050 (8/8 > 0), choice 8/8 vs nomem 7/8
- prefer context: 3.448 (8/8 > 0), choice 8/8 vs nomem 7/8
- prefer rag: 3.448 (8/8 > 0), choice 8/8 vs nomem 7/8
- avoid inject: -2.640 (1/8 > 0), choice 3/8 vs nomem 6/8
- avoid context: -2.638 (0/8 > 0), choice 1/8 vs nomem 6/8
- avoid rag: -2.638 (0/8 > 0), choice 1/8 vs nomem 6/8
- forbidden logprob down: 3/8, mean stored lift 0.728
- unrelated N=8: mean 0.000, large 0
- prefer top-1 at N=64: 1/8 (reported, not a gate)
- misses: avoid-harlow delta -1.666 stored_lift 2.014; avoid-nestor delta -2.675 stored_lift 4.117; avoid-bodhi delta -8.511 stored_lift 5.421; avoid-ingrid delta -4.979 stored_lift 3.190; avoid-tomas delta -0.848 stored_lift -1.384; avoid-yara delta -1.154 stored_lift -2.074; avoid-felix delta -1.426 stored_lift 1.234

### held

- prefer inject: 10.905 (8/8 > 0), choice 8/8 vs nomem 5/8
- prefer context: 5.344 (8/8 > 0), choice 8/8 vs nomem 5/8
- prefer rag: 5.344 (8/8 > 0), choice 8/8 vs nomem 5/8
- avoid inject: 1.277 (7/8 > 0), choice 6/8 vs nomem 5/8
- avoid context: -4.809 (0/8 > 0), choice 0/8 vs nomem 5/8
- avoid rag: -4.809 (0/8 > 0), choice 0/8 vs nomem 5/8
- forbidden logprob down: 8/8, mean stored lift -1.300
- unrelated N=8: mean -0.030, large 1
- prefer top-1 at N=64: 7/8 (reported, not a gate)
- misses: dir-held-avoid-0 delta -0.514 stored_lift -0.565

## Cost

- MLP parameters (`net.*`): 664704
- polarity parameters: 576
- total parameters: 665280
- buffers: 66931
- extra per token, mean over lengths 16/32/64: 0.001201 s
  - length 16: 0.001143 s
  - length 32: 0.002238 s
  - length 64: 0.000221 s
- readout median, hard top-1, versus store size N:
  - N=8: 0.000023 s
  - N=32: 0.000031 s
  - N=64: 0.000034 s
  - N=128: 0.000069 s
  - N=256: 0.000182 s
- query median: 0.028090 s; MLP median (includes the polarity dot): 0.000082 s
- queries per continuation: empty 0, inject 1, phone slot on the decision prompt 1 (forwards 9; spelling stays off because that prompt is not completing phone)
- extra per prompt (queries × query + MLP + readout N=64): 0.028206 s
- RSS after the cost pass: VmRSS 1280204 kB, VmHWM 1423176 kB
- Variable-cost paths still in the runtime (no retries, no second query):
  - query encode: skipped when the store is empty
  - scripted phone or birthday: replaces the token loop; any other prompt keeps the residual path
  - span stop: stops before max_new_tokens on a newline or sentence punctuation
  - uncached prefix: each new token forwards the whole prefix; use_cache is False
  - remember: two full forwards (key, then value); chat and noise return before that
- Polarity is a fixed 576-d dot product inside that one MLP call. It does not add a retrieval, a second query, or a per-token branch.

Process RSS around the run: before VmPeak 757900 kB, VmHWM 227108 kB, VmRSS 227108 kB; after_load VmPeak 3311460 kB, VmHWM 1273788 kB, VmRSS 1054752 kB; after_eval VmPeak 3483156 kB, VmHWM 1477412 kB, VmRSS 1293052 kB.

## Misses

- frozen failed: avoid margin, forbidden logprob down.
- held met every direction gate.

Numbers above are read from the JSON written by this script. Weights were not changed after these two banks were scored.
