# Cue-triggered injection

The whole-residual broadcast is removed. There is no flag that turns it back on. A training step still adds one value on every token inside `teacher_force_loss`. Retrieval does not.

The cue is the per-token hidden state copied at the inject block, before the add. The public name is `cue` and the width is `cue_dim`. This preregistration does not name a decoder index. On the shipped SmolLM2-135M checkpoint the existing inject block is the one this wrapper already uses, and `cue_dim` is that block's hidden width.

## Causality

The cue at index `t` is the inject-block output before the add. The add is written at that same index, after the copy. The cue tensor does not depend on the add. No earlier index sees it. The logit at `t` is the distribution for token `t+1`, and that logit does see the add. Injection is therefore not delayed until `t+1`.

On a trigger, `dt` is 0 and `a = a_trig`. A later token that does not trigger uses `a = a_trig * exp(-dt / tau)`, where `dt` is the number of tokens after the trigger. A new trigger resets `dt` and `a`. If `a` is below `CUE_FLOOR` (0.05), the trace clears: memory id -1, vector zeros, activation 0. The floor is not fit.

The shipped generator still re-forwards the whole string and recomputes the recurrence. `cue_state` after a forward is the state after the last token. A caller that forwards only the new token sets `_cue_resume` and passes that state, the same way a cache carries past keys. `advance_cue` on the raw cue rows matches the full trace.

## Competition

A candidate is a slot with positive finite strength and `cos(cue - query_center, key) >= theta`. Theta does not read strength. The winner is the one candidate with the greatest `(log(strength), cosine, -index)`. One winner is injected. There is no average across neighbors. A cosine tie (equal log strength and equal cosine) records the lower index and adds nothing, because its confidence is 0.

Losers of a strict win still take the existing retrieval-induced forgetting scale. The hook does not change strengths. `_rank` applies `forget_rivals` once per distinct triggered winner whose cosine gap is finite and positive.

The per-token fields are `[B, T]` (or `[B, T, cue_dim]` for the vector): `memory_id` int64, -1 if none; `vec` float32, the stored value, zeros if none; `activation` in `[0, 1]`; `strength_confidence`; `valence` as `u - 1`; `circadian_phase` copied from the store clock onto every token; `triggered` bool, true only on a token that selects a winner; `interface_version` 1. `cue_fields(text)` emits those fields and does not change slot strength.

## Literature

Verified before this file was committed. No other citation is included.

- Tulving, E., & Thomson, D. M. (1973). Encoding specificity and retrieval processes in episodic memory. Psychological Review, 80(5), 352–373. https://doi.org/10.1037/h0020071
- Teyler, T. J., & DiScenna, P. (1986). The hippocampal memory indexing theory. Behavioral Neuroscience, 100(2), 147–154. https://doi.org/10.1037/0735-7044.100.2.147
- Vaz, A. P., Inati, S. K., Brunel, N., & Zaghloul, K. A. (2019). Coupled ripple oscillations between the medial temporal lobe and neocortex retrieve human memory. Science, 363(6430), 975–978. https://doi.org/10.1126/science.aau8956
- Lu, Q., Hasson, U., & Norman, K. A. (2022). A neural network model of when to retrieve and encode episodic memories. eLife. https://doi.org/10.7554/eLife.74445

## Bracket, not a selected cell

`reports/subconscious_cue_histogram.json` is the direction fit bank only: 8 prefer facts, each stored alone, cosine of `cue - query_center` with that fact's key. No margin was scored. No cell was selected.

| quantity | n | min | max |
| --- | --- | --- | --- |
| max cosine on the fact text | 8 | 0.516 | 0.593 |
| max cosine on the cued prompt | 8 | 0.436 | 0.582 |
| max cosine on an unrelated prompt | 64 | 0.078 | 0.303 |

The cued prompts are 10 tokens. At every theta in the grid below, the last token that clears theta is 4 tokens before the end, and the last token itself is below 0.11. The decision logit therefore sees the decayed trace, not a fresh trigger.

The filler is `fact text + "\nLater the clerk filed the spare form and left the room quietly today."` The max cosine on the filler, after the fact tokens, is 0.294. At theta 0.32 and above, the filler does not trigger, and 15 filler tokens follow the last fact trigger.

The grid is the open-closed band that sits above the unrelated max (0.303) and the filler max (0.294) and at or below the lowest cued-prompt max (0.436). Tau and `a_trig` are not taken from a margin.

`theta` in {0.32, 0.35, 0.40, 0.43}, `tau` in {2, 4, 8}, `a_trig` in {0.5, 1.0}. 24 cells. The floor stays 0.05. The module constants at commit time are a placeholder (`theta` 0.15, `tau` 4, `a_trig` 1). They are not a selected cell.

## Selection

Fit uses the direction fit bank only. The script's `histogram` and `fit` functions do not import a held bank. Among cells that pass gates 1–5, pick the largest mean cued margin delta. Ties go to the higher theta, then the larger tau, then the smaller `a_trig`. If no cell passes, held is not scored.

## Gates 1–5, fit bank

1. Eight prefer facts, one at a time. The cued prompt is that fact's prompt. The effect is the continuation margin minus the empty-store margin. Pass when at least 6 of 8 deltas are positive and the mean delta is positive. Unrelated prompts are the 8 `NEUTRAL_CHOICE_FIT` strings, scored with all 8 facts stored. Pass when the mean absolute margin delta is at most 0.05.
2. A false trigger is an unrelated prompt with any token `triggered`. The ceiling is 0.10 on the point estimate. On this n=8 bank, 1/8 is 0.125 and fails, so the fit bank needs 0/8. Wilson is reported and is not the pass rule.
3. Text is the fact plus the filler above. Distance 0 is the last triggering token. Distances 0, 1, 2, 4, and 8 must exist. No token in between may trigger. Activation is non-increasing on those distances, the distance-8 activation is strictly below distance 0, and distance 0 is positive. The L1 distance between the armed logits and the no-memory logits is strictly smaller at distance 8 than at distance 0. At least 6 of 8 facts must pass. The logit comparison is only those two distances.
4. The same 8 prefer facts. Distractors are the first rows of `fit_incidental`, disjoint by construction. Store sizes 8, 16, and 32 are the target plus that many distractors minus one, and the smaller store is a prefix of the larger one. Pass when the mean cued margin delta at 16 and at 32 is positive and is at least the size-8 mean minus 0.05 nats.
5. `NEUTRAL_NLL_FIT`, with the 8 prefer facts stored. For each continuation token whose predicting position has activation 0, take `NLL(memory) - NLL(empty)`. Pass when at least one such token exists and the mean is at most 0.02 nats. If every predicting position is active, the gate fails.

Each prompt starts from the strengths just written. `_rank` still runs, and strengths are restored afterward so one prompt does not suppress the next.

## Gate 6, held, not used to select

One pass, after a cell is selected. n=50. Not worse than the #54 point counts. A rate with n below 50 fails. Wilson is reported for each count.

| rate | floor |
| --- | --- |
| prefer margin up | 50/50 |
| avoid margin up | 39/50 |
| avoid stored logprob down | 42/50 |
| toward experience | 45/50 |
| away experience | 48/50 |
| at least 4 slots | 50/50 |
| competitor strength below 1 | 50/50 |
| winner strength unchanged | 50/50 |
| outsider strength unchanged | 50/50 |
| more similar companion lowers the target margin | 42/50 |
| three companions lower the target margin more than one | 48/50 |

Phone and birthday exact strings are not in this gate. The removed mixture score is not in this gate.

## Cost

At the selected cell, or at the placeholder if none is selected, the inject cost is the median extra seconds of a cue-live forward over a hook no-op, divided by the token count. Store sizes 8, 16, and 32. Warmup 2, repeats 7. This is not a pass rule.

## Fit result

One pass of the 24-cell grid. Git of the scorer before this result was written: `d25d74279c1b49969192bc1764af8501dc8f44a0`. Elapsed 83.7 seconds. All 24 cells passed gates 1–5. The selected cell is theta 0.32, tau 8, a_trig 1, mean cued margin 1.381. It is the unique maximum. The module constants now match that cell. Held was scored once after this selection and was not used to change it.

Selected cell, direction fit bank, n=8:

| gate | result |
| --- | --- |
| cued margin up, mean positive | 6/8 positive, mean 1.381 |
| unrelated mean absolute margin ≤ 0.05 | 0.0 |
| false triggers ≤ 0.10 | 0/8, Wilson [0.0, 0.324] |
| decay after the last trigger | 6/8. Two facts failed the logit comparison |
| store 16 and 32 within 0.05 of size 8, and positive | size 8: 0.164, size 16: 0.148, size 32: 0.136 |
| quiet-token NLL increase ≤ 0.02 | 0.0 on 9 tokens |

Inject cost at this cell, median extra seconds per token of the cue-live forward over the hook no-op, 10 tokens, warmup 2, repeats 7: store 8 is 0.000125, store 16 is 0.000084, store 32 is 0.000159. The full armed forward is about 0.0029 seconds per token.

## Held result

One pass. Git `9f4ea6184ff3018317bd019b9dd20d0b61acba8c`. Elapsed 120.0 seconds. The selected cell was theta 0.32, tau 8, a_trig 1. The pass is false. The floors were not moved. The pull request stays a draft.

| rate | count | Wilson 95% | floor | pass |
| --- | --- | --- | --- | --- |
| prefer margin up | 50/50 | [0.9286524008666414, 1.0] | 50/50 | yes |
| avoid margin up | 36/50 | [0.5833487630404344, 0.8252582933408878] | 39/50 | no |
| avoid stored logprob down | 42/50 | [0.7148578393696501, 0.916625793219666] | 42/50 | yes |
| toward experience | 45/50 | [0.7863976856252035, 0.9565242350681096] | 45/50 | yes |
| away experience | 48/50 | [0.8653990931249297, 0.9889611156723802] | 48/50 | yes |
| at least 4 slots | 50/50 | [0.9286524008666414, 1.0] | 50/50 | yes |
| competitor strength below 1 | 50/50 | [0.9286524008666414, 1.0] | 50/50 | yes |
| winner strength unchanged | 15/50 | [0.19103553500880954, 0.437503504644534] | 50/50 | no |
| outsider strength unchanged | 50/50 | [0.9286524008666414, 1.0] | 50/50 | yes |
| more similar companion lowers the target margin | 41/50 | [0.6920394632569922, 0.9022980732976582] | 42/50 | no |
| three companions lower the target margin more than one | 19/50 | [0.25862789310193596, 0.5184955306900703] | 48/50 | no |

Means, not a gate: prefer delta 9.892, avoid delta 0.262, avoid stored lift -0.356, toward delta 1.935, away delta -4.943.

The report is `reports/subconscious_cue_held.json`.

## Amendment

This section supersedes the activation formula, the grid, and the selection rule above. The fit and held numbers above are the first schedule. They are not used to choose theta, half-life, lambda, or the fan constant.

### Fan

`n` is the number of candidates with `cos(cue, key) >= theta`. Theta still does not read strength. The winner is unchanged: greatest `(log strength, cosine, -index)`. One winner. No average.

`fan_factor(n) = max(0, (S - ln n) / S)`. `S` is 1.45, the value Anderson and Reder estimated for the 1974 fan experiment. It is not fit. Dividing by `S` makes fan 1 a factor of 1. The factor decreases in `n`. Fan 4 is `(1.45 - ln 4) / 1.45`, which is below the 0.05 floor, so a fan-4 trigger clears.

Exported activation is `a_trig * fan_factor * decay`. `decay` is `exp(-dt / tau)`. The fan factor is frozen at the trigger and kept while the trace decays. A new trigger recomputes it. `interface_version` is 2.

### Two timescales

The fast timescale is the per-token decay above. The slow timescale is the slot strength and the existing replay update. No new state is added.

The half-life sweep is 10, 20, and 30 tokens. `tau = half_life / ln 2`, so the activation is half of `a_trig * fan_factor` after that many tokens. This range is a HYPOTHESIS. Joordens and Becker measured items, not tokens, and there is no direct conversion. The value is fit on the direction fit bank only.

`a_trig` stays 1. It is not swept again.

### Lambda

Lambda is a global scale on the residual this wrapper adds to SmolLM2. The sweep is 0.10, 0.15, 0.20, 0.25, 0.30. The center 0.25 is the reference point of that range. Lambda is not part of the exported activation, the exported value, or the training broadcast in `teacher_force_loss`.

### Fit order

Theta is frozen before any half-life or lambda cell is scored. The freeze rule uses the cosine histogram only: the lowest value in {0.32, 0.35, 0.40, 0.43} that is above both the unrelated max (0.303) and the filler max (0.294), and at or below the lowest cued-prompt max (0.436). That value is theta 0.32. Fan tests use this theta and do not co-sweep it. `n` is discontinuous in theta.

The fit-bank fan manipulation is not a margin. Companions have strength 0.5, relation `fan-norm`, and a key mixed from the target key toward the match vector at the fan-1 last trigger. Alpha is tried in {0.25, 0.125, 0.0625, 0.03125, 0.015625}. The largest alpha that, on all 8 fit prefer items, keeps the fan-2 clean trace on that same last trigger with confidence 1, and clears the fan-4 trace from that trigger on, is frozen. If no alpha does that, cells are not scored and held is not scored. The same alpha is used for every cell and for the held fan gate. It is not chosen again on held.

Among cells that pass gates 1–5 and the fit fan check, pick the largest mean cued margin. Ties go to the half-life closer to 20, then the lambda closer to 0.25. The fit fan check is: mean effect at fan 1, 2, and 4 is strictly decreasing, and at least 6 of 8 items are strictly decreasing. The effect is the continuation margin minus the empty-store margin. If no cell passes, held is not scored.

### Gate 7, held

With the frozen theta and the frozen alpha, fan 1, 2, and 4 on the 50 held prefer items. Pass when at least 38/50 items are strictly decreasing and the three means are strictly decreasing. n below 50 fails. Wilson is reported on the item count. This gate is not used to select the cell.

Gates 1–6 are otherwise unchanged. Gate 6 still uses the #54 floors.

### Literature added with this amendment

Verified before this section was used to score a cell. No other citation is included.

- Anderson, J. R., & Reder, L. M. (1999). The fan effect: New results and new theories. Journal of Experimental Psychology: General, 128(2), 186–197. https://doi.org/10.1037/0096-3445.128.2.186
- Anderson, J. R. (1974). Retrieval of propositional information from long-term memory. Cognitive Psychology, 6(4), 451–474. https://doi.org/10.1016/0010-0285(74)90021-8
- Joordens, S., & Becker, S. (1997). The long and short of semantic priming effects in lexical decision. Journal of Experimental Psychology: Learning, Memory, and Cognition, 23(5), 1083–1105. https://doi.org/10.1037/0278-7393.23.5.1083
- Tulving, E., Schacter, D. L., & Stark, H. A. (1982). Priming effects in word-fragment completion are independent of recognition memory. Journal of Experimental Psychology: Learning, Memory, and Cognition, 8(4), 336–342. https://doi.org/10.1037/0278-7393.8.4.336

Joordens and Becker report that semantic priming is reduced to non-significant levels if as few as one or two items intervene between a related prime and the target. Tulving, Schacter, and Stark report that word-fragment priming was unchanged from one hour to seven days while recognition fell. That slow effect is the strength and replay state already in the store.

## Amendment result

One pass of the 15-cell schedule. Git of the scorer before this result was written: `1f1c6968ea923173f6c93a5b69d99c93a7a9a3d6`. Elapsed 66.3 seconds. Alpha 0.25 is the largest value that kept the clean fan trace on all 8 fit items; every smaller candidate did too. All 15 cells passed gates 1–5 and the fit fan check. The selected cell is theta 0.32, half-life 30, lambda 0.3, mean cued margin 0.479. It is the unique maximum. The module constants now match that cell. The held pass was not used to choose it.

Selected cell, direction fit bank, n=8:

| gate | result |
| --- | --- |
| cued margin up, mean positive | 7/8 positive, mean 0.479 |
| unrelated mean absolute margin ≤ 0.05 | 0.0 |
| false triggers ≤ 0.10 | 0/8, Wilson [0.0, 0.324] |
| decay after the last trigger | 6/8 |
| store 16 and 32 within 0.05 of size 8, and positive | size 8: 0.0144, size 16: 0.0132, size 32: 0.0129 |
| quiet-token NLL increase ≤ 0.02 | 0.0 on 9 tokens |
| fan 1 / 2 / 4 mean margin, at least 6/8 strict | 0.479 / 0.203 / 0.0, 6/8 |

Fan 4 is 0 because `(1.45 - ln 4) / 1.45` is below the 0.05 floor, so that trigger clears. Inject cost at this cell, median extra seconds per token, 10 tokens, warmup 2, repeats 7: store 8 is 0.000159, store 16 is 0.000173, store 32 is 0.000104. The armed forward is about 0.0026 seconds per token.

The report is `reports/subconscious_cue_schedule_fit.json`.

## Amendment held result

One pass. Git `6f7644d344b0d8a563ae07982407196b87295694`. Elapsed 115.4 seconds. The selected cell was theta 0.32, half-life 30, lambda 0.3. The amendment pass is false. The floors were not moved. The pull request stays a draft.

| rate | count | Wilson 95% | floor | pass |
| --- | --- | --- | --- | --- |
| prefer margin up | 46/50 | [0.8116175308165717, 0.968450485911407] | 50/50 | no |
| avoid margin up | 37/50 | [0.6044684273515117, 0.8412847250644762] | 39/50 | no |
| avoid stored logprob down | 42/50 | [0.7148578393696501, 0.916625793219666] | 42/50 | yes |
| toward experience | 47/50 | [0.8378290831116182, 0.979385029651026] | 45/50 | yes |
| away experience | 47/50 | [0.8378290831116182, 0.979385029651026] | 48/50 | no |
| at least 4 slots | 50/50 | [0.9286524008666414, 1.0] | 50/50 | yes |
| competitor strength below 1 | 15/50 | [0.19103553500880954, 0.437503504644534] | 50/50 | no |
| winner strength unchanged | 46/50 | [0.8116175308165717, 0.968450485911407] | 50/50 | no |
| outsider strength unchanged | 50/50 | [0.9286524008666414, 1.0] | 50/50 | yes |
| more similar companion lowers the target margin | 39/50 | [0.6475845041874634, 0.8724608402978559] | 42/50 | no |
| three companions lower the target margin more than one | 29/50 | [0.4423344176857823, 0.7062499664528802] | 48/50 | no |
| fan 1/2/4 strictly decreasing | 44/50 | [0.7619518261679701, 0.9438239984906773] | 38/50 | yes |

Fan means, which did decrease: fan 1 is 3.035, fan 2 is 0.992, fan 4 is 0.00018. Means, not a gate: prefer delta 3.035, avoid delta 0.097, avoid stored lift -0.129, toward delta 1.788, away delta -1.094.

The report is `reports/subconscious_cue_schedule_held.json`.

## Widened half-life

This section supersedes the half-life sweep and the tie-break above. The 15-cell fit and its held pass stay the record of half-life in {10, 20, 30}. They are not used to choose among the cells in this section.

The sweep is half-life in {2, 4, 8, 16, 30} and lambda in {0.10, 0.15, 0.20, 0.25, 0.30}. Theta stays 0.32. `a_trig` stays 1. That is 25 cells.

The item-to-token conversion is a HYPOTHESIS. Joordens and Becker found semantic priming reduced to non-significant levels after one or two intervening items. A Chinese word is about one to two tokens, so a half-life of about 2–4 tokens is a plausible reading of that lag. It is not a measured conversion. The short values are on the grid because of that reading. The selected cell is still the passing cell with the largest mean cued margin on the direction fit bank. Ties go to the half-life closer to 4, then the lambda closer to 0.25. Held banks are not used to choose.

## Widened half-life result

One pass of the 25-cell grid. Git of the scorer before this result was written: `f07de9b0b90613b7d76246bb236cd773ff6576a2`. Elapsed 106.0 seconds. Alpha 0.25 again. All 25 cells passed gates 1–5 and the fit fan check. Mean cued margin on the direction fit bank:

| half-life | lambda 0.10 | 0.15 | 0.20 | 0.25 | 0.30 |
| --- | --- | --- | --- | --- | --- |
| 2 | 0.099 | 0.152 | 0.208 | 0.265 | 0.323 |
| 4 | 0.116 | 0.180 | 0.245 | 0.313 | 0.383 |
| 8 | 0.130 | 0.201 | 0.275 | 0.352 | 0.431 |
| 16 | 0.138 | 0.214 | 0.294 | 0.377 | 0.462 |
| 30 | 0.142 | 0.221 | 0.304 | 0.390 | 0.479 |

The widened half-life fit selected half-life 30 and lambda 0.3. Mean cued margin 0.479. It is the unique maximum. The 2–4 token reading did not override that rule. The module constants were already that cell, so they were not changed. The existing held pass for that cell still applies. Held was not scored again. The amendment pass remains false, and the pull request stays a draft.

Selected cell, same direction fit bank, n=8: 7/8 cued margins positive, unrelated absolute margin 0.0, false triggers 0/8 with Wilson [0.0, 0.324], decay 6/8, store means 0.0144 / 0.0132 / 0.0129, quiet NLL 0.0 on 9 tokens, fan means 0.479 / 0.203 / 0.0 with 6/8 strict. Inject cost on this run, median extra seconds per token, 10 tokens, warmup 2, repeats 7: store 8 is 0.000120, store 16 is 0.000084, store 32 is 0.000108. The armed forward is about 0.0025 seconds per token.

The report is `reports/subconscious_cue_halflife_fit.json`.

## Modality

This section does not rescore a cell and does not move theta, half-life, lambda, or the floors.

Every stored slot has `modality`. Every write sets it to `text`. A slot constructed with any other value is rejected. Each exported token carries `modality`, and the value is `text` on every token, including a cleared trace. `interface_version` is 3. Version 2 was `a_trig * fan_factor * decay`. That formula is unchanged. Version 3 adds this field. There is no perceptual-fingerprint field.

## Single-winner diagnostic

[#53](https://github.com/yunshenya/tidal/pull/53) reported a four-slot mixture score of 1.152 against a top-1 score of 0.598. [#55](https://github.com/yunshenya/tidal/pull/55) kept `MIX_K` 4. One winner may miss the floors. The floors stay the counts already stated in this file: the #54 held counts, plus fan 1/2/4 strictly decreasing on at least 38/50. The held pass is already false. This pull request does not merge, and this diagnostic does not move a floor.

The measurement is the direction fit bank only. `anisotropy` and `compete` do not import a held bank. They do not change theta, half-life, lambda, or fan S. The shipped path stays one winner.

The shipped column is `k=1`, the residual the code injects. The competitive columns are `k=2`, `k=3`, and `k=4`. While the shipped trace is active, those columns replace the injected vector with a k-winner-take-all mix. Rank is `(log strength, cosine, -index)` over positive-strength slots. The threshold is the cosine of the first excluded slot. A kept slot's raw weight is `max(0, cosine - threshold)`, then the weights sum to 1. A raw-weight sum of 0 shares the weight equally across the kept slots. When every positive slot is kept, the threshold is `min(0, lowest kept cosine)`. Each kept slot is composed with its own key and scaled by its own strength and valence. The shipped activation and the shipped winner's confidence multiply the whole mix, so the time gate stays the one-winner trace. `k=1` does not use this mix. It is the shipped residual.

Norman, Newman, and Detre (2007) describe oscillating k-winner-take-all feedback inhibition. This diagnostic is that kind of competition. It is not their oscillator and not their learning rule. https://doi.org/10.1037/0033-295X.114.4.887

The scored gates are gates 1–5 and the fit fan check, at the shipped cell theta 0.32, half-life 30, lambda 0.3. The comparison does not select a k.

## Cue cosine anisotropy

Cue-key cosine is computed after the shipped centers in `tidal/llm_key_center.npz`. The stored key has `key_mean` subtracted. The cue row has `query_mean` subtracted. No other center is estimated. Related cues are the tokens of each direction-fit prefer prompt against that fact's key. Unrelated cues are the tokens of each direction-fit neutral choice prompt against that same key. AUC is the probability a related cosine exceeds an unrelated cosine. A tie counts half. The same AUC is also reported on the per-prompt maximum. This report does not change theta. Theta stays 0.32.

## Single-winner diagnostic result

One pass. Git `8e085f31e5841405832fd374e931ad31f0378918`. Elapsed 20.5 seconds. Alpha 0.25. All four columns passed gates 1–5 and the fit fan check. No k was selected. Held was not scored. The floors were not moved. The pull request stays a draft.

Mean cued margin is 0.479 at k=1, 2, 3, and 4, because that gate stores one slot. Unrelated absolute margin is 0.0. False triggers are 0/8. Decay is 6/8. Quiet NLL is 0.0 on 9 tokens.

Store-size mean cued margin at 8 / 16 / 32:

| k | 8 | 16 | 32 | fan 1 / 2 / 4 | strict | mean positive slots |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | 0.0144 | 0.0132 | 0.0129 | 0.479 / 0.203 / 0.0 | 6/8 | shipped one winner |
| 2 | 0.0177 | 0.0126 | 0.0125 | 0.479 / 0.097 / 0.0 | 7/8 | 1.332 |
| 3 | 0.0162 | 0.0123 | 0.0125 | 0.479 / 0.097 / 0.0 | 7/8 | 1.510 |
| 4 | 0.0163 | 0.0125 | 0.0125 | 0.479 / 0.097 / 0.0 | 7/8 | 1.697 |

k=2 is higher at store size 8. At 16 and 32 every k above 1 is slightly below k=1. The fit bank does not show a k that should replace the shipped winner. The report is `reports/subconscious_cue_kwta_fit.json`.

## Cue cosine anisotropy result

One pass. Git `8e085f31e5841405832fd374e931ad31f0378918`. Elapsed 0.495 seconds. Centers were `key_mean` and `query_mean` from `tidal/llm_key_center.npz`. Theta before and after was 0.32. Theta was not changed.

Related prompt tokens, n=80: min −0.191, median 0.223, mean 0.216, max 0.582. Unrelated prompt tokens, n=368: min −0.192, median 0.036, mean 0.032, max 0.303. Token AUC is 0.797. Per-prompt maximum AUC is 1.0. The lowest related maximum is 0.436 and the highest unrelated maximum is 0.303. The report is `reports/subconscious_cue_anisotropy.json`.

## Direction held mechanism gates

This section does not choose a cell and does not move theta, half-life, lambda, fan S, or the floors. The cell is the one already selected: theta 0.32, half-life 30, lambda 0.3, `a_trig` 1. Alpha stays 0.25. The n=50 floor table is not rescored.

The bank is the direction held bank. It has 8 prefer items, 8 neutral-choice prompts, and 12 neutral-NLL items. Distractors are the held incidental facts. One pass.

The point rules are gates 1–5 and the fit fan check. Wilson 95% intervals are reported on the cued-positive count, the false-trigger count, the decay-pass count, and the strict fan count. A rate whose n is below 50 fails. The prefer count and the neutral-choice count are 8, so those rates fail that rule before the point estimate is read. The point estimates are still reported. Quiet-token NLL is a mean. Its token count fails the same rule when it is below 50. Store size is the number of scored targets, which is 8, so it fails too.

This measurement does not change theta.

## Citation check

On 2026-10-24 each DOI in this file and in `docs/architecture.md` was resolved through `https://doi.org` as CSL JSON. Each record returned the paper named beside that DOI. The McClelland and Rumelhart record titles the article with "I." The architecture prose says "Part 1." It is that article. The fusion references in the architecture file still have no DOIs and were not part of this check.

## Direction held mechanism result

One pass. Git `15943c570169113a67d665724362b6a7edfd3709`. Elapsed 7.1 seconds. Theta 0.32, half-life 30, lambda 0.3, alpha 0.25. Theta was not changed. The floors were not rescored.

The point pass is false because decay is 5/8. The n≥50 pass is false.

| gate | count | Wilson 95% | point rule | n≥50 |
| --- | --- | --- | --- | --- |
| cued margin positive | 7/8 | [0.529, 0.978] | pass, mean 1.191 | fail |
| unrelated absolute margin | n=8, mean 0.0 |  | pass, mean ≤ 0.05 | fail |
| false triggers | 0/8 | [0.0, 0.324] | pass | fail |
| decay after the last trigger | 5/8 | [0.306, 0.863] | fail, floor 6/8 | fail |
| store 8 / 16 / 32 | n=8 |  | pass, 0.0154 / 0.0119 / 0.0125 | fail |
| quiet-token NLL | 0.0 on 11 tokens |  | pass, mean ≤ 0.02 | fail |
| fan 1 / 2 / 4 strict | 7/8 | [0.529, 0.978] | pass, means 1.191 / 0.461 / 0.000034 | fail |

The three decay failures are the logit comparison, on `dir-held-prefer-2`, `dir-held-prefer-4`, and `dir-held-prefer-7`. Activation on those items still fell with distance. The report is `reports/subconscious_cue_direction_held.json`.

Inject cost at the selected cell, from the widened half-life fit, median extra seconds per token, 10 tokens, warmup 2, repeats 7: store 8 is 0.000120, store 16 is 0.000084, store 32 is 0.000108. The armed forward is about 0.0025 seconds per token.

## Competitive retrieval

This section supersedes the rule that `forget_rivals` scales same-relation rivals of the winner whether or not those rivals cleared theta.

Theta still gates on centered cosine. Strength still ranks the candidates that clear theta. One winner is injected. Decay is unchanged. Lambda still scales only the residual.

Before that injection, every other candidate that cleared theta participates. `compete_retrieved` multiplies its strength by the existing retrieval-induced-forgetting keep. Similarity is key cosine to the winner, so a closer companion is scaled more. That is the similar-companion interference. The winner is not scaled. A slot that missed theta is not scaled. A slot whose salience is at least 0.5 is exempt. The span stays 0.05. `cue_fields` does not change strength. The scoring forward reads the updated strengths and still adds only the winner.

Norman, Newman, and Detre (2007) describe competitive retrieval in which co-active competitors are punished. This is that consequence. It is not their oscillator and not their learning rule. https://doi.org/10.1037/0033-295X.114.4.887

No constant was retuned. The already-scored n=50 floors were not used to choose this rule. The direction fit bank is the design bank.

One cell. Git `077f440eb6ef9360c0f67e332fa5747c603f5b9d`. Elapsed 7.1 seconds. The direction fit gates pass. Mean cued margin 0.479, 7/8 positive, unrelated absolute margin 0.0, false triggers 0/8, decay 6/8, store means 0.0144 / 0.0132 / 0.0129, quiet NLL 0.0 on 9 tokens, fan means 0.479 / 0.203 / 0.0 with 6/8 strict. The fresh bank was not scored. The report is `reports/subconscious_cue_competition_fit.json`.

## Fresh direction gates

Protocol `cue-direction-fresh-v1`. The bank is `tidal/subconscious_cue_fresh_data.py`: 50 prefer items, 50 neutral-choice prompts, 50 neutral-NLL items, and 40 incidental facts. It is disjoint from the direction fit bank and the direction held bank. It is not used to choose theta, half-life, lambda, alpha, fan S, or the RIF span.

One pass, after this section is committed. Wilson 95% intervals are reported. A rate with n below 50 fails.

| gate | rule |
| --- | --- |
| cued margin positive, mean positive | 38/50 and mean > 0 |
| unrelated absolute margin | n=50 and mean ≤ 0.05 |
| false triggers | n=50 and rate ≤ 0.10 |
| decay after the last trigger | 38/50 |
| store 8 / 16 / 32 | n=50, means at 16 and 32 positive and within 0.05 of size 8 |
| quiet-token NLL | n≥50 tokens and mean ≤ 0.02 |
| fan 1 / 2 / 4 strictly decreasing | 38/50 and the three means decreasing |

38/50 is the fit point rule of 6/8 on a bank of 50. It is not a number from the already-scored floors.

The same pass also scores the previous n=50 floors: prefer 50/50, avoid up 39/50, avoid stored 42/50, toward 45/50, away 48/50, at least 4 slots 50/50, competitor below 1 is 50/50, winner unchanged 50/50, outsider unchanged 50/50, similarity 42/50, count 48/50, and fan strictly decreasing 38/50. Those floors are not moved. The previous schedule-held file is not rewritten. The report is `reports/subconscious_cue_revision_held.json`. Inject cost per token is reported on this pass at store sizes 8, 16, and 32.
