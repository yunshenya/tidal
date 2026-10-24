# K ablation preregistration

Measurement only. `MIX_K` stays 4 and `MIX_TEMP` stays 0.05. Nothing in this run writes a new default. `_rank` cost is not in this change.

The grid is three selectors crossed with three temperatures. Fixed selection is the production rule: softmax of `log(strength) + cosine` on the 4 largest cosines. Cumulative selection, called top-p here, takes the softmax over every positive-strength slot, keeps the shortest prefix whose mass is at least 0.9, and never keeps more than `K_max`. The prefix is renormalized. With every strength equal to 1, the match score is the cosine, so a top-p prefix that happens to be the same set as fixed K=4 has the same weights. The temperatures are 0.05, 0.1, and 0.2. `K_max` is 16 and 32. That is nine cells:

| cell | selector | cap | temperature |
|---|---|---|---|
| `fixed:4:0.05` | fixed | 4 | 0.05 |
| `fixed:4:0.10` | fixed | 4 | 0.10 |
| `fixed:4:0.20` | fixed | 4 | 0.20 |
| `top_p:16:0.05` | top-p | 16 | 0.05 |
| `top_p:16:0.10` | top-p | 16 | 0.10 |
| `top_p:16:0.20` | top-p | 16 | 0.20 |
| `top_p:32:0.05` | top-p | 32 | 0.05 |
| `top_p:32:0.10` | top-p | 32 | 0.10 |
| `top_p:32:0.20` | top-p | 32 | 0.20 |

0.9 is fixed before any margin. It is not chosen on the fit split or the held split. A prefix whose summed mass is within 1e-12 of 0.9 counts as reaching it, so a binary sum such as nine times 0.1 is not forced to take another slot.

Fit is 8 probes in each of tone, choice, and preference (24). Held is the other 16 probes in each family (48). The two lists are disjoint. Fit is scored before held. A second complete held report is refused. Bank `28fe567c3e7cdd34df4a5611366fecea4dd0a98ac74ad26bd75705d84784f26b`.

Fit has not been scored.

Held has not been scored.

## Stores

Each probe has one target and 31 companions, all with the same relation and different names. The target sentence is `{name} zkablates {answer} beside {rel} {anchor}.` The prompt is `{Zkabltone|Zkablchoice|Zkablpref} {anchor}:` and does not contain the answer. Fifteen companions use that prose template with a different answer and their own nonce. Sixteen companions use `zkfn {nonce}() { zkreturn {answer}; }` with a different answer. A size-8 store is the target, the first 3 prose companions, and the first 4 code companions. Size 16 uses the first 7 and the first 8. Size 32 uses all 15 and all 16.

Writes pass the fact text, so the same-relation surface-repeat rule is the one that would collapse a copy. A companion is a different answer and a different nonce. Key cosine is not the merge rule.

The interference subsets are the target plus the first 1, 3, or 7 prose companions (high band) and the same counts of code companions (low band). The order is generation order. It is not re-sorted by a margin.

## Geometry

Keys only. No continuation margin. SmolLM2-135M, git `6a3ecf9bf9f616e3f112160c1eaa2a296b0dcab9`, 95.62339100000099 seconds. All 72 probes, both splits.

Every size-32 write kept 32 slots. The largest token Jaccard inside a probe is 0.6, under the 0.85 surface-repeat threshold. Every prompt is a tokenizer prefix of the prompt plus the good continuation and of the prompt plus the bad continuation.

Prose companions against the target key, 1080 pairs: min 0.970836841064013, median 0.9890279362758504, max 0.9931601405647932, mean 0.988626600474161. Code-frame companions, 1152 pairs: min 0.8666241170699595, median 0.8884520781182397, max 0.907919930595844, mean 0.8889420541164265. On every probe the least similar prose companion is still closer than the closest code companion. That gap's minimum is 0.07515952280613003, median 0.09284815088764714, maximum 0.10436322417178023.

The target is the unique query argmax on 72/72 probes at size 8, 16, and 32. The smallest gap from the target to the next query cosine is 0.017691708427383768 at size 8, 0.010132660389859272 at size 16, and 0.009480611304790054 at size 32.

## What is read out

The implicit score is the unweighted mean of the tone, choice, and preference `d_z` values. Each `d_z` is the mean paired difference (mixture margin minus empty-store margin) divided by the sample standard deviation (`n - 1`). The margin is the mean log-prob of the good continuation minus the mean log-prob of the bad continuation. Explicit phone and birthday recall is not in this measurement.

Participation is `1 / sum(w^2)` on the positive mixture weights. Active neighbors are the weights that are strictly positive.

Compute is counted, not assumed. `mlp_runs` is the number of `InjectMLP.forward` calls during one arm. `broadcast_tokens` is the number of sequence positions at which the summed residual was added, across the good forward and the bad forward. `mlp_runs_per_neighbor` divides the first by the active-neighbor count. `mlp_runs_per_broadcast_token` divides it by the position count.

The code path this prediction comes from: `_set_delta` calls `_compose` once for each positive-weight part, and `_compose` calls `InjectMLP` once. The hook then adds one vector. `_broadcast` views that vector across every position of the layer output. The prediction, before the count, is that an armed mixture has one `InjectMLP` run per active neighbor, an empty store has zero runs, and a one-slot store has one run. A fixed-K arm of a store with at least 4 positive-strength slots keeps 4 active neighbors. A top-p arm keeps between 1 and `K_max` inclusive.

Alignment is the cosine between the mixture residual and the residual of the target stored alone. Help on a probe is the mixture margin minus that target-only margin. The Pearson correlation of alignment with help is reported for each cell over the probes in the split. The prediction is that the correlation is positive. Pairs with a missing alignment are dropped. Fewer than 3 pairs, or a constant side, leaves the correlation unset.

The cancellation claim has no direct paper. It is a hypothesis about this broadcast: many weak memories raise the implicit score only when their residuals do not cancel in the sum that is added at every position. The operational reading, fixed before any margin, is an implication over cells. At a given store size and temperature, a top-p cell whose mean alignment is lower than the fixed-K=4 cell does not have a higher implicit score than that fixed-K cell. If either mean is missing, the implication is unset. The cell is still reported.

The interference directions are the same operational reading already written in `reports/subconscious_interfere_prereg.md`: a more similar same-relation companion lowers the target margin, and more companions lower it further. This note does not add a quotation. For each cell, the similarity contrast is the fraction of probes whose high-band margin is strictly below the low-band margin, at 1, 3, and 7 companions. The count contrast is the fraction whose 7-companion margin is strictly below their 1-companion margin. The pre-registered count prediction is the high band. The low band is reported beside it. Each fraction is reported with a Wilson 95% interval. These fractions are not a rule for changing `MIX_K`.

Strength is 1 on every slot at the arm. Retrieval-induced forgetting runs after the read and the strengths are restored before the next arm, so it does not enter these margins.

## Locked defaults

`MIX_K` 4, `MIX_TEMP` 0.05, `gap_tau` 0.80, polarity span 2, valence span 1, valence step 0.25, replay, the surprise gate, the surface-repeat threshold 0.85, and the retrieval-induced-forgetting span stay as they are. The existing tests are the floor. This measurement does not rescore those banks.
