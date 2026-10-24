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

One pass of the 24-cell grid. Git of the scorer before this result was written: `d25d74279c1b49969192bc1764af8501dc8f44a0`. Elapsed 83.7 seconds. All 24 cells passed gates 1–5. The selected cell is theta 0.32, tau 8, a_trig 1, mean cued margin 1.381. It is the unique maximum. The module constants now match that cell. Held has not been scored.

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

Held has not been scored.
