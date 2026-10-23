# Subconscious fix

Direction and interference of the soft inject on the main path. SmolLM2-135M, layer 15, hidden 576, float32. gap_tau 0.8000. The add is `c * (scale(v) + MLP(scale(v) + W k))`. `c` is 1 for one slot and `clip((top1-top2)/gap_tau, 0, 1)` otherwise. Phone and birthday spelling runs only when the winning relation is the one the prompt is completing. One query per continuation.

Frozen bank `e3cdd7fd595d8a76af03a276c2ad750b25b68e1bd8460c9e22e95aa12a820e66` (`frozen`). Held-out bank `ea0a15e5edf0dcab53fede8fad1964b61c23b52c3df55c5f8e00865e59cb053a` (`held`). Neither bank was used to train the MLP or to choose gap_tau. Train log: selected step 200, fit-bank sha `20b7cb3f9742709d320dc5913cf62675316673d7b457c55b6c78cf4de587e5ce`.

Load top-1 and FIFO eviction are reported and are not gates.

## Gates

| bank | gate | pass | detail |
|---|---|---|---|
| frozen | prefer delta | yes | mean 16.047, pos 8/8 |
| frozen | avoid delta | yes | mean 6.579, pos 8/8 |
| frozen | neutral NLL | yes | mean -0.008, bad 0 |
| frozen | choice flips | yes | 0/8 |
| frozen | leak | yes | 0/8 |
| frozen | unrelated N=8 | yes | mean 0.002, large 1 |
| frozen | repeat | yes | max abs 0.000000 |
| frozen | chat | yes | max abs 0.000000 |
| frozen | update delta | yes | mean 17.652, pos 8 |
| frozen | update store | yes | one slot, new value closer |
| frozen | phone unrelated | yes | 0.000 |
| frozen | birthday unrelated | yes | 0.000 |
| frozen | phone question | yes | 1 |
| held | prefer delta | no | mean -4.774, pos 2/8 |
| held | avoid delta | yes | mean 6.177, pos 7/8 |
| held | neutral NLL | yes | mean -0.086, bad 0 |
| held | choice flips | yes | 0/8 |
| held | leak | yes | 0/8 |
| held | unrelated N=8 | yes | mean -0.026, large 0 |
| held | repeat | yes | max abs 0.000000 |
| held | chat | yes | max abs 0.000000 |
| held | update delta | yes | mean 4.334, pos 6 |
| held | update store | yes | one slot, new value closer |
| held | phone unrelated | yes | 0.000 |
| held | birthday unrelated | yes | 0.000 |
| held | phone question | yes | 1 |
| held | birthday question | yes | 1 |

## Influence

### frozen

- prefer inject: 16.047 (8/8 > 0), choice 8/8 vs nomem 7/8, lift 0.125; stored-side lift 9.726
- prefer context: 3.448 (8/8 > 0), choice 8/8 vs nomem 7/8, lift 0.125
- prefer rag: 3.448 (8/8 > 0), choice 8/8 vs nomem 7/8, lift 0.125
- avoid inject: 6.579 (8/8 > 0), choice 8/8 vs nomem 6/8, lift 0.250; stored-side lift -2.438
- avoid context: -2.638 (0/8 > 0), choice 1/8 vs nomem 6/8, lift -0.625
- avoid rag: -2.638 (0/8 > 0), choice 1/8 vs nomem 6/8, lift -0.625
- update mean delta 17.652, new lift 11.177, old lift -6.474

### held

- prefer inject: -4.774 (2/8 > 0), choice 2/8 vs nomem 5/8, lift -0.375; stored-side lift 1.617
- prefer context: 2.429 (8/8 > 0), choice 8/8 vs nomem 5/8, lift 0.375
- prefer rag: 2.429 (8/8 > 0), choice 8/8 vs nomem 5/8, lift 0.375
- avoid inject: 6.177 (7/8 > 0), choice 8/8 vs nomem 5/8, lift 0.375; stored-side lift 0.042
- avoid context: -2.579 (0/8 > 0), choice 1/8 vs nomem 5/8, lift -0.500
- avoid rag: -2.579 (0/8 > 0), choice 1/8 vs nomem 5/8, lift -0.500
- update mean delta 4.334, new lift 7.040, old lift 2.706

## Non-interference

- frozen: NLL mean -0.008 (0 > 0.5), flips 0/8, leak 0/8, unrelated mean 0.002 (1 |delta| > 0.1), phone unrelated 0.000, birthday unrelated 0.000, phone question 1, birthday question None
- held: NLL mean -0.086 (0 > 0.5), flips 0/8, leak 0/8, unrelated mean -0.026 (0 |delta| > 0.1), phone unrelated 0.000, birthday unrelated 0.000, phone question 1, birthday question 1

## Load and persistence (not gated)

### frozen

- prefer N=8: hit 2/8, survived 8/8, slots 6.9, gap 0.0164, conf 0.020, delta 0.116
- prefer N=32: hit 1/8, survived 8/8, slots 21.9, gap 0.0138, conf 0.017, delta 0.059
- prefer N=64: hit 1/8, survived 8/8, slots 41.4, gap 0.0093, conf 0.012, delta 0.012
- avoid N=8: hit 1/8, survived 8/8, slots 6.9, gap 0.0216, conf 0.027, delta -0.093
- avoid N=32: hit 0/8, survived 8/8, slots 21.9, gap 0.0113, conf 0.014, delta 0.015
- avoid N=64: hit 0/8, survived 8/8, slots 41.4, gap 0.0087, conf 0.011, delta 0.013
- FIFO: absent 0/8, slots 138.0, delta 0.008 (single-slot delta 16.047)

### held

- prefer N=8: hit 1/8, survived 8/8, slots 8.0, gap 0.0129, conf 0.016, delta -0.023
- prefer N=32: hit 0/8, survived 8/8, slots 26.2, gap 0.0134, conf 0.017, delta -0.006
- prefer N=64: hit 0/8, survived 8/8, slots 46.0, gap 0.0110, conf 0.014, delta -0.003
- avoid N=8: hit 0/8, survived 8/8, slots 8.0, gap 0.0142, conf 0.018, delta -0.017
- avoid N=32: hit 0/8, survived 8/8, slots 26.2, gap 0.0134, conf 0.017, delta -0.006
- avoid N=64: hit 0/8, survived 8/8, slots 46.0, gap 0.0110, conf 0.014, delta -0.003
- FIFO: absent 0/8, slots 48.0, delta -0.013 (single-slot delta -4.774)

## Cost

- MLP parameters (net + key_proj): 996480 (net 664704, key_proj 331776)
- buffers: 66932
- gap_tau: 0.8000
- extra per token, mean over lengths 16/32/64: 0.000500 s
  - length 16: 0.003011 s
  - length 32: 0.003410 s
  - length 64: -0.004920 s
- readout median, hard top-1:
  - N=8: 0.000019 s
  - N=32: 0.000025 s
  - N=64: 0.000036 s
  - N=128: 0.000083 s
  - N=256: 0.000148 s
- query median: 0.028514 s; MLP median: 0.000086 s
- queries per continuation: empty 0, inject 1, phone question 1
- phone slot on the decision prompt queries 1, forwards 9 (the token loop still runs)
- extra per prompt (one query + MLP + readout N=64): 0.028635 s
- continuation wall, 8 new tokens: empty 0.2643 s, inject 0.2879 s, phone question 0.0284 s
- RSS after the cost pass: VmRSS 1284892 kB, VmHWM 1427512 kB
- Variable-cost paths (no retries, no second query on the main path):
  - query encode: skipped when the store is empty
  - scripted phone or birthday: replaces the token loop; any other prompt keeps the residual path
  - span stop: stops before max_new_tokens on a newline or sentence punctuation
  - uncached prefix: each new token forwards the whole prefix; use_cache is False
  - remember: two full forwards (key, then value); chat and noise return before that

Process RSS around the run: before VmPeak 758092 kB, VmHWM 227320 kB, VmRSS 227320 kB; after load VmPeak 3313660 kB, VmHWM 1276428 kB, VmRSS 1060704 kB; after eval VmPeak 3436460 kB, VmHWM 1427512 kB, VmRSS 1288620 kB.

## Misses

- frozen met every gate in the table above.
- frozen prefer top-1 at N=64 is 1/8. That retrieval gap is out of scope and was not tuned.
- held failed: prefer delta.
- held prefer top-1 at N=64 is 0/8. That retrieval gap is out of scope and was not tuned.

Numbers above are read from the JSON written by this script. Weights and gap_tau were not changed after these two banks were scored.
