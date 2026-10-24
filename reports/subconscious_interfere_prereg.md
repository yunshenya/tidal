# Surface-repeat preregistration

Similar memories stay in separate slots and compete. A cross-identity overwrite at key cosine above 0.9 is removed. A different identity is overwritten only when the group matches, the relation is the same non-empty string, and the surface token Jaccard is at least `REPEAT_JACCARD`. The same predicate is what replay merges. Same `(name, rel)` still overwrites that one slot, including when the new text is different. Key cosine does not merge slots.

`REPEAT_JACCARD` is `0.85`, the highest cell that passed the fit grid below. `MIX_K` 4, `MIX_TEMP` 0.05, `gap_tau` 0.80, polarity span 2, valence span 1, valence step 0.25, replay, the surprise gate, and the retrieval-induced-forgetting span are unchanged. Cutting the cost of `_rank` is not in this change.

The one held pass is recorded under Held result. The threshold was chosen on the fit grid before that pass.

## Literature

Read for this note. A number below is from a paper only when the sentence says it was read there. A page number attached to McGeoch is a page number printed in the secondary chapter that quotes him. The 1932 article itself was not fetched.

Underwood, B. J. (1957). Interference and forgetting. *Psychological Review, 64*(1), 49–60. https://doi.org/10.1037/h0044616

Read from https://iiif.library.cmu.edu/file/Simon_box00019_fld01304_bdl0003_doc0002/Simon_box00019_fld01304_bdl0003_doc0002.pdf. The PDF header is Psychological Review, Vol. 64, No. 1, 1957. The classical forgetting curve is treated as proactive interference from lists learned earlier in the laboratory, not as extra-laboratory learning during the retention interval. The summary says that removing that source decreases forgetting from about 75 per cent over 24 hours to about 25 per cent, and that a further methodological correction leaves 15 per cent as the 24-hour estimate. The rank-order correlation between number of previous lists and forgetting is .91 for the 14 points of Fig. 3. On similarity, the paper says amount of interference from other tasks is closely tied to similarity, both similarity among the materials and situational similarity. With low-meaningfulness material, forgetting is significantly greater with high intralist similarity than with low, and the difference in magnitude is only about 8 per cent. The text layer of that sentence garbles the total-loss figure; the summary's 15 per cent is the clear statement of the 24-hour estimate those 8 points are being compared with. The prediction below, that a more similar same-relation companion lowers the target continuation margin, is this similarity claim. It is not a claim that the margin shift will be 8 per cent.

Anderson, J. R. (1974). Retrieval of propositional information from long-term memory. *Cognitive Psychology, 6*, 451–474. https://doi.org/10.1016/0010-0285(74)90021-8

Read from the PDF whose header is Cognitive Psychology 6, 451474 (1974); the missing hyphen is the text extraction. Subjects learned propositions such as "A hippie is in the park." The experiments manipulated how many propositions involved a particular person or a particular location. In Experiment 1, averaging over true and false probes and collapsing person and location, subjects were 94 msec slower to judge a probe when the concept was in three study propositions rather than one. Of 36 pairwise comparisons, 32 increased. The person effect was 93 msec and the location effect was 96 msec. The PDF that was read does not use the word "fan." The count contrast below is this result: more same-relation propositions, lower target margin. It is not a replication of the 94 msec latency, and it does not use a model equation from a later paper.

MacLeod, C. M. Interference theory: History and current status. Chapter PDF read at https://uwaterloo.ca/memory-attention-cognition-lab/sites/default/files/uploads/files/interference_theory_4_final_revised.pdf. The file names the author as Colin M. MacLeod, University of Waterloo, and says the chapter is to appear in M. J. Kahana and A. D. Wagner (Eds.), *Handbook of human memory: Foundations and applications*. Published handbook pages were not read. That chapter quotes McGeoch, J. A. (1932), Forgetting and the law of disuse, *Psychological Review, 39*(4), 352–370, including p. 352 ("There are no warmly discussed theories of forgetting"), p. 356 (interpolated activities and changed stimulating conditions as the factors that produce forgetting), p. 364 ("Forgetting is, then, not a passive matter, but a result of an active interference from interpolated events"), and p. 369 ("Ascription of effectiveness to time violates the usage of science and is logically meaningless"). Those four quotations were read in MacLeod's chapter. They were not checked against a 1932 PDF. McGeoch is the historical source of the interference claim. The measurements below are not attributed to numbers in the 1932 article.

Norman, K. A., & O'Reilly, R. C. (preprint 29 August 2002, in press at *Psychological Review*). Modeling hippocampal and neocortical contributions to recognition memory: A complementary learning systems approach. Read from http://www.princeton.edu/~compmem/psyrev_inpress.pdf. The file says this version may differ in minor ways from the final published version, so the sentences below are not assigned published page numbers. The precis says the hippocampus assigns distinct, pattern-separated representations to stimuli, which lets it learn rapidly without catastrophic interference, whereas cortex assigns similar representations to similar stimuli so it can generalize. The citation is for keeping similar traces in separate slots instead of overwriting them. O'Reilly and Rudy (2001) and McClelland, McNaughton, and O'Reilly (1995) are named there as the complementary-learning-systems sources. Those two were not read. Marr (1971) is named there and was not read.

## Why cosine cannot make the split

Geometry only. No continuation margin was scored. Keys are `encode_key` on SmolLM2-135M for the 50 held quartets in `tidal/subconscious_measure_data.py` (bank `0f5cbd1c3787f7cf1de93be89ee7b2b9b17a3ad5f5dc430e3d0b735d5fb97228`). Surface tokens use the same regex as the other banks. Every one of the 300 same-relation pairs and the 200 cross-relation pairs has token Jaccard 0.

Same-relation key cosine, 300 pairs: min 0.8228, quartile 0.9023, median 0.9179, quartile 0.9320, max 0.9620, mean 0.9144. Cross-relation key cosine, 200 pairs: min 0.8073, median 0.9080, max 0.9521, mean 0.9066. Target-to-competitor cosine, 150 pairs: min 0.8228, median 0.9172, max 0.9539. Within a quartet the spread from the least similar competitor to the most similar one has min 0.0037, median 0.0211, max 0.0766. Twenty-eight of 50 quartets have a spread above 0.02, and 5 of 50 have a spread above 0.05.

Replaying the old first-hit rule (overwrite the first same-group key with cosine strictly above the threshold) gives these slot counts:

| threshold | 1 slot | 2 | 3 | 4 | 5 | quartets with at least 4 |
|---|---|---|---|---|---|---|
| 0.85 | 49 | 1 | 0 | 0 | 0 | 0/50 |
| 0.90 | 27 | 18 | 5 | 0 | 0 | 0/50 |
| 0.95 | 0 | 0 | 0 | 8 | 42 | 50/50 |
| 0.99 | 0 | 0 | 0 | 0 | 50 | 50/50 |

The 0.90 row is the collapse behind the earlier 0/50 geometry. On the first quartet's target sentence, encoding the same text again has key cosine 1.0000001192092896 and token Jaccard 1. Inserting a space before the period has key cosine 0.9986433982849121 and token Jaccard 1. Replacing the answer token, which the key pool drops, has key cosine 0.9997134804725647 and token Jaccard 5/7. That different answer is closer in key cosine than every quartet pair (max 0.9620), and it is closer than the punctuation variant. No cosine threshold merges the punctuation repeat and keeps the different answer. Token Jaccard does: the punctuation repeat is 1, the different answer is 5/7, and every quartet pair is 0.

## Fit

The grid is `0.50, 0.67, 0.75, 0.85, 0.92, 1.0`. Each cell writes the nine cases in `tidal/subconscious_interfere_data.py` into an empty store at that threshold. There is no language-model forward and no held quartet.

A case with rule `merge` must end at 1 slot. A case with rule `stay` or `two` must end at 2 slots. A case with rule `one` must end at 1 slot. The cases are:

| id | rule | what it is |
|---|---|---|
| exact-case | merge | same relation, case and a final period differ, token Jaccard 1 |
| extra-token | merge | same relation, one added token, token Jaccard 6/7 |
| one-word | stay | same relation, one content token replaced, token Jaccard 5/7 |
| disjoint-same-rel | two | same relation, no shared tokens |
| same-text-other-rel | two | identical text, different relations |
| close-keys-different-text | two | key cosine above 0.99, different text, same relation |
| empty-rel | two | identical text, missing relation |
| other-group | two | identical text, same relation, different groups |
| identity-update | one | same name and relation, different text |

A cell passes when every case matches its rule. Among passing cells, take the highest threshold. That is the strictest rule that still collapses the added-token repeat. If no cell passes, the held bank is not scored and the constant is not given a second grid.

The added-token pair is the near-identical repeat. The one-token replacement is a different fact and must keep its own slot. Those two Jaccard values sit on either side of 0.75 and of 0.85, and the added-token pair sits below 0.92. The grid is built that way.

Fit bank `6f9530eb9a2684d258e7b2cd5714c61bb69fd6eb5950f5193e91f77d0d3130ba`. Git `f06f636aa55fc89ff6371ce74b0f9120e3f0f812`. Elapsed 0.003436908998992294 seconds. No language-model forward.

| threshold | pass | why |
|---|---|---|
| 0.50 | no | the one-token replacement collapsed |
| 0.67 | no | the one-token replacement collapsed |
| 0.75 | yes | the added-token repeat collapsed and the one-token replacement stayed |
| 0.85 | yes | same |
| 0.92 | no | the added-token repeat stayed |
| 1.0 | no | the added-token repeat stayed |

Two cells pass. The higher threshold is 0.85, so that is the selected cell. 0.50 and 0.67 also collapsed the exact repeat, the different-relation copy stayed two slots, and the parallel keys with different text stayed two slots. They failed only because Jaccard 5/7 was treated as a repeat. 0.92 and 1.0 kept that different fact and failed because Jaccard 6/7 was not treated as a repeat.

## Held-out gates

One pass, after `REPEAT_JACCARD` is set to the selected threshold. A second complete report is refused. Explicit phone and birthday exact match is reported and is not part of the pass rule. On these prompts `generate` copies the stored answer when the prompt is completing that relation.

RIF geometry. The 50 quartets are written with `remember`. A quartet counts when the store keeps at least 4 slots. The pass bar is at least 26 of 50. More than half is the reading of "most." The count and its Wilson 95% interval are reported either way. Slots are matched to facts by `(name, rel)`. A leftover row after a collapsed write is not a trial.

RIF pattern. The denominator is the quartets that still have the target, the outsider, and all three competitors. After one retrieval of the target prompt, competitor down means every competitor strength is strictly below 1, winner unchanged means the target strength is 1, and outsider unchanged means the outsider strength is 1. Each rate is reported with a Wilson 95% interval. Each passes when its Wilson lower bound is strictly above 0.5. At 50 trials that bound needs at least 32 successes. A smaller denominator uses the same bound, recomputed. A denominator of 0 fails. Strength is the operational pattern. The span is not retuned.

Similarity. Among the three same-relation competitors, take the unique highest key cosine to the target and the unique lowest. A tie at either extreme excludes the quartet and is counted. Store the target with only the high-cosine competitor, and store the target with only the low-cosine competitor. Strength is 1. The score is the target continuation margin minus the empty-store margin on the same prompt. The prediction is that the high-cosine delta is strictly lower. The rate of that inequality, over the quartets that have a unique pair, passes when its Wilson lower bound is strictly above 0.5. The within-quartet cosine spreads above are small. There is no minimum spread. The direction was fixed from Underwood (1957), not from a margin.

Count. Store the target with the first competitor in the quartet list. Store the target with all three competitors. The outsider is not stored. The prediction is that the three-competitor delta is strictly lower than the one-competitor delta. The rate over all 50 quartets passes when its Wilson lower bound is strictly above 0.5. This is the Anderson (1974) count claim, as a margin rather than a latency.

Implicit. The 150 measurement probes are stored with their three distractors. The mixture subconscious score is the unweighted mean of the tone, choice, and preference `d_z` values, each `d_z` the mean paired difference (mixture margin minus empty-store margin) divided by the sample standard deviation (`n - 1`). #53 published the point 1.1516089064136972 and did not publish an interval. The interval used here was computed from that report's stored paired differences before this held pass: 10000 resamples within each family, `numpy.random.Generator` seed 20261024, linear percentile. The bounds are 1.0171138974369223 and 1.3856896713238247. The new point estimate must lie inside, inclusive. Every probe must keep 4 slots. The top-1 arm in the same store is reported and is not this gate.

Prefer, avoid, and valence. One fact per trial. The point estimate `k / 50` must lie inside the #53 Wilson interval, inclusive.

| rate | #53 count | Wilson 95% |
|---|---|---|
| prefer margin up (`delta > 0`) | 50/50 | [0.9286524008666414, 1.0] |
| avoid margin up (`delta > 0`) | 39/50 | [0.6475845041874634, 0.8724608402978559] |
| avoid stored logprob down | 42/50 | [0.7148578393696501, 0.916625793219666] |
| toward experience (`delta > 0`) | 45/50 | [0.7863976856252035, 0.9565242350681096] |
| away experience (`delta < 0`) | 48/50 | [0.8653990931249297, 0.9889611156723802] |

An away rate of 50/50 is outside that last interval. Means are reported and are not a gate.

Locked floors from the earlier banks stay in the existing tests. This change does not rescore those n=8 banks. Pytest of the suite is the floor.

## Held result

One pass. Git `3b20da0aaead34a136f3ba5e51a28a80afd7d55b`. Elapsed 290.51303531399753 seconds. Model HuggingFaceTB/SmolLM2-135M. `REPEAT_JACCARD` stayed 0.85. The report is `reports/subconscious_interfere_held.json`. Overall pass is true.

| gate | result |
|---|---|
| at least 4 slots on at least 26/50 quartets | yes. 50/50 kept 5 slots, rate 1.0, Wilson [0.9286524008666414, 1.0] |
| competitor strength below 1 | yes. 50/50, rate 1.0, Wilson [0.9286524008666414, 1.0] |
| winner strength unchanged | yes. 50/50, rate 1.0, Wilson [0.9286524008666414, 1.0] |
| outsider strength unchanged | yes. 50/50, rate 1.0, Wilson [0.9286524008666414, 1.0] |
| more similar companion lowers the target margin | yes. 42/50, rate 0.84, Wilson [0.7148578393696501, 0.916625793219666]. Ties excluded: 0 |
| three companions lower the target margin more than one | yes. 48/50, rate 0.96, Wilson [0.8653990931249297, 0.9889611156723802] |
| mixture score inside [1.0171138974369223, 1.3856896713238247] and 4 slots | yes. 1.1516089064136972. All 150 probes kept 4 slots. Probe won 150/150 |
| prefer margin up inside the #53 interval | yes. 50/50, rate 1.0, Wilson [0.9286524008666414, 1.0] |
| avoid margin up inside the #53 interval | yes. 39/50, rate 0.78, Wilson [0.6475845041874634, 0.8724608402978559] |
| avoid stored logprob down inside the #53 interval | yes. 42/50, rate 0.84, Wilson [0.7148578393696501, 0.916625793219666] |
| toward inside the #53 interval | yes. 45/50, rate 0.9, Wilson [0.7863976856252035, 0.9565242350681096] |
| away inside the #53 interval | yes. 48/50, rate 0.96, Wilson [0.8653990931249297, 0.9889611156723802] |
| phone exact, excluded | 50/50, rate 1.0, Wilson [0.9286524008666414, 1.0]. Not part of the score |
| birthday exact, excluded | 50/50, rate 1.0, Wilson [0.9286524008666414, 1.0]. Not part of the score |

The pattern denominator was 50. Each pattern Wilson lower bound is 0.9286524008666414. The similarity Wilson lower bound is 0.7148578393696501. The count Wilson lower bound is 0.8653990931249297. Each of those is above 0.5.

Four-slot family `d_z`, mixture: tone 0.8377903630705432, choice 1.2943307338186498, preference 1.322705622351898. Top-1 in the same store is 0.598030495653073 and is not the gate. Top-1 family `d_z`: tone 0.3815099176111003, choice 0.5295369079844059, preference 0.883044661363713. Mean participation was 1.9403741947075708. Mean top-1 gap was 0.06502326433924588, so mean top-1 confidence was 0.08127908042405735. All 150 probes kept 4 slots. The mixture point equals the #53 four-slot point 1.1516089064136972, which lies inside the bootstrap interval. The softmix fit bank records a max pairwise key cosine of 0.7808010045557472 inside a probe group, under the old 0.9 overwrite.

One fact at a time, the prefer, avoid, and valence counts and means match the #53 report. Prefer mean delta 10.267230100784607, stored lift 7.149745581532674, stored down 0/50. Avoid mean delta 0.29070510550153755, stored lift -0.4041234005987644. Toward mean delta 1.9166707942753174, audit 50/50, repeats 3. Away mean delta -5.041761126883793, audit 50/50.

Locked pytest floors on this revision: 313 passed, 14 warnings, 40.93 seconds. The warnings are the existing ONNX deprecation warnings. The suite does not rescore the held bank.

#52 stays open and unmerged. It is the measurement record. #53 is the code those measurements were taken against before this change.
