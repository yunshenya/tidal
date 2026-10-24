# Soft-mixture injection preregistration

Strength-weighted soft mixture over several slots. Fit margins have not been scored. The held-out bank has not been scored.

This replaces the hard top-1 value that was added into the residual. With two or more slots the residual is the sum of per-slot composed values, each multiplied by its mixture weight, its strength, and its valence gain. The cosine argmax is still the winner used for phone and birthday spelling and for same-relation forgetting. `injection_readout="top1"` keeps the old arm: gap confidence times the winning value, in the same store. That arm is the baseline. The default readout is `"mixture"`. A store with fewer than two slots still uses the old one-slot add, including the unmultiplied residual when confidence, strength, and valence gain are all 1. Gap confidence is still recorded. It scales only the top-1 arm. A cosine tie splits the mixture weight. The top-1 arm still adds nothing on a tie.

`MIX_K = 4` and `MIX_TEMP = 0.05` in `tidal/llm_memory.py` are the corner of the grid below. They are not a selected point and not a held result. Continuum soft top-k, `gap_tau` 0.80, polarity span 2, valence span 1, valence step 0.25, replay, the surprise gate, and retrieval-induced forgetting are unchanged. Overwrite at cosine above 0.9 is unchanged. Cutting the cost of `_rank` is not in this change.

## Literature

Read for this note. No statistic below is taken from a paper unless the sentence says it was read there. Unpublished manuscripts named inside a paper that was read are not treated as sources.

Collins, A. M., & Loftus, E. F. (1975). A spreading-activation theory of semantic processing. *Psychological Review, 82*(6), 407–428. https://doi.org/10.1037/0033-295X.82.6.407

Read from https://bpb-us-e2.wpmucdn.com/faculty.sites.uci.edu/dist/c/571/files/2024/08/CollinsLoftus_PsychReview_75.pdf. Memory search is activation spreading from two or more concept nodes until an intersection is found. Activation can start at only one node at a time, but it continues in parallel from the nodes it reaches. Activation from different sources summates, and the intersection fires only once that sum reaches a threshold. Priming more concepts leaves less activation for each.

McClelland, J. L., & Rumelhart, D. E. (1981). An interactive activation model of context effects in letter perception: Part 1. An account of basic findings. *Psychological Review, 88*(5), 375–407. https://doi.org/10.1037/0033-295X.88.5.375

Read from https://stanford.edu/~jlmcc/papers/McClellandRumelhart81.pdf. The model processes several letter positions at once and processes feature, letter, and word levels at the same time. Nodes that suggest each other excite each other, so several detectors are active together. Connections within the word level are mutually inhibitory, because that model allows only one word at one place at one time. The citation is for graded parallel activation of more than one representation. It is not a claim that their word level blends words with a softmax.

Lebiere, C. (1999). The dynamics of cognition: An ACT-R model of cognitive arithmetic. *Kognitionswissenschaft, 8*, 5–19. https://doi.org/10.1007/s001970050071

Read from http://act-r.psy.cmu.edu/wordpress/wp-content/uploads/2012/12/459459.pdf. Chunk activation is base level plus context, `A_i = B_i + sum_j W_j S_ji`. When several chunks compete, the chunk-choice probability is `P(i) = exp(M_ip / t) / sum_j exp(M_jp / t)`, with `t = sqrt(2) * s`. When no chunk reaches the retrieval threshold, the estimation equation chooses the value `V` that minimizes `sum_i P_i (1 - Sim(V, V_i))^2`. Lebiere describes that minimum as a compromise across facts, weighted by retrieval probability. The interpolation mechanism is attributed in that paper to Lebiere and Wallach (in preparation). That manuscript was not read. The paper also points to Anderson and Lebiere (1998) for the rest of ACT-R; that book was not read, and none of the equations above are cited from it. If dissimilarity is squared Euclidean distance, the minimizer of that sum is the probability-weighted mean of the values. The code sums the composed residuals, one nonlinear compose per slot and then a weighted sum. That is the Euclidean special case of the estimation equation, not the formula as printed. The match score in the code is `log(strength) + cosine`. Strength is a multiplicative trace in `(0, 1]`, not ACT-R base-level log odds, and cosine is not ACT-R similarity. Lebiere mentions Moussouris (1974) for the Boltzmann form. That paper was not read and is not a source here.

## Banks

- fit and distractors `3ba1fab4559982dcf959121224de3d0be86e03a25a056dccd874c6534dcdd6aa` (`tidal/subconscious_softmix_data.py`, protocol `subconscious-softmix-v1`)
- held probes `0f5cbd1c3787f7cf1de93be89ee7b2b9b17a3ad5f5dc430e3d0b735d5fb97228` (`tidal/subconscious_measure_data.py`, the unmerged measurement bank, protocol `subconscious-measure-v1`)
- fit size 16 tone, 16 choice, 16 preference
- held implicit size 50 tone, 50 choice, 50 preference
- each of those 198 probes has three distractors
- surface token Jaccard against every other `tidal/*data.py` string is under 0.50
- max pairwise key cosine inside a probe group is 0.7808010045557472, under the 0.85 cap and under the 0.9 overwrite rule
- on the query vector of the probe's own prompt, the probe is the strict cosine argmax in all 198 groups

The distractors are not measurement items. They exist so the store has four slots and retrieval confidence is not trivially 1. The measurement of key cosine and of the participation preview below is geometry. It is not a log-prob margin.

Participation preview, equal strength, query cosine only. `k = 8` matches `k = 4` because each group has four slots. Mean participation is above 1.05 in every cell, so that floor alone does not reject a cell. It does not choose the cell. The margin comparison has not been run.

| temperature | k=2 mean | k=2 min | k=4 mean | k=4 min |
|---|---|---|---|---|
| 0.05 | 1.430695557755818 | 1.01716939758103 | 1.7483770531151979 | 1.0309177735082935 |
| 0.10 | 1.6965401116612635 | 1.1837366880648386 | 2.512376735830501 | 1.4164543890205508 |
| 0.20 | 1.8859227569331642 | 1.5571669995262165 | 3.335592605999829 | 2.5650926766747246 |
| 0.40 | 1.966681557931035 | 1.845941040335748 | 3.7890865078867013 | 3.5442046826927007 |

## What the mixture score is

For one probe the store holds that probe and its three distractors. The no-memory margin is the length-normalized `mean logp(good) - mean logp(bad)` with an empty store. The mixture margin and the top-1 margin use that same four-slot store and that same no-memory margin. The paired difference is armed minus no-memory. Cohen's `d_z` is the mean of those paired differences divided by their sample standard deviation (`n - 1`). A family with `n < 2` or with sample standard deviation 0 has `d_z` none. The subconscious score is the unweighted mean of the tone, choice, and preference `d_z` values. If any of the three is none, the score is none. Phone and birthday exact match are not inputs.

The single-slot measurement score on the held probes was 1.3423953633016108 (choice 1.5509790144028102, preference 1.3738399240609025, tone 1.1023671514411195), with confidence 1 because each of those trials stored one fact. That number is not the gate. The gate compares mixture with top-1 inside the four-slot store.

## Fit selection

The grid is `k` in {2, 4, 8} and temperature in {0.05, 0.10, 0.20, 0.40}. All twelve cells are scored on the 48 fit probes. Held probes are not scored in this pass.

A cell passes when every fit probe still occupies 4 slots, the mean participation ratio `1 / sum(w^2)` across the 48 mixture arms is at least 1.05, both subconscious scores are numbers, and the mixture score is greater than or equal to the top-1 score. The top-1 score does not depend on `k` or temperature. Among passing cells, take the largest mean participation, then the smallest `k`, then the smallest temperature. If no cell passes, the held bank is not scored, and the constants are not given a second grid.

| k | temperature | slots | participation mean | mixture score | top-1 score | pass |
|---|---|---|---|---|---|---|
| 2 | 0.05 | | | | | |
| 2 | 0.10 | | | | | |
| 2 | 0.20 | | | | | |
| 2 | 0.40 | | | | | |
| 4 | 0.05 | | | | | |
| 4 | 0.10 | | | | | |
| 4 | 0.20 | | | | | |
| 4 | 0.40 | | | | | |
| 8 | 0.05 | | | | | |
| 8 | 0.10 | | | | | |
| 8 | 0.20 | | | | | |
| 8 | 0.40 | | | | | |

Fit margins have not been scored.

## Held-out gates

The held-out bank has not been scored. It is scored once, after `MIX_K` and `MIX_TEMP` are set to the selected cell, and it is not scored again.

Implicit. Each of the 150 measurement probes is stored with its three distractors. The mixture subconscious score must be a number and must be greater than or equal to the top-1 subconscious score in that same store. Explicit recall is not part of either score.

Prefer, avoid, and valence. These trials store one fact, so the one-slot add is the arm. The rate is the point estimate `k / 50`. It must lie inside the measurement Wilson interval, inclusive. The intervals are the ones in the measurement report for bank `0f5cbd1c3787f7cf1de93be89ee7b2b9b17a3ad5f5dc430e3d0b735d5fb97228`.

| rate | measurement count | Wilson 95% |
|---|---|---|
| prefer margin up (`delta > 0`) | 50/50 | [0.9286524008666414, 1.0] |
| avoid margin up (`delta > 0`) | 39/50 | [0.6475845041874634, 0.8724608402978559] |
| avoid stored logprob down | 42/50 | [0.7148578393696501, 0.916625793219666] |
| toward experience (`delta > 0`) | 45/50 | [0.7863976856252035, 0.9565242350681096] |
| away experience (`delta < 0`) | 48/50 | [0.8653990931249297, 0.9889611156723802] |

An away rate of 50/50 is outside the upper end of that last interval. Means are reported and are not a gate.

Explicit. Phone and birthday exact match, 50 each, are reported and excluded from the subconscious score and from the pass rule. On these prompts `generate` copies the stored answer when the winning relation is the one the prompt is completing. A perfect count there is that copy, not a separate spelling test.

Locked floors from the earlier banks stay in the existing tests: prefer margin mean at least 8.0, avoid margin mean at least 0.6, and forbidden logprob down on at least 6/8, along with the other constants those tests pin. This change does not rescore those n=8 banks and does not retune `gap_tau`, polarity, valence, replay, surprise, or retrieval-induced forgetting. Pytest of those tests is the floor.

| gate | result |
|---|---|
| mixture score >= top-1 score on the 150 four-slot probes | |
| prefer margin up inside the measurement interval | |
| avoid margin up inside the measurement interval | |
| avoid stored logprob down inside the measurement interval | |
| toward inside the measurement interval | |
| away inside the measurement interval | |
| phone exact, excluded | |
| birthday exact, excluded | |
| locked pytest floors | |

The held-out bank has not been scored.
