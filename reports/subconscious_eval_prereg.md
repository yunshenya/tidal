# Subconscious eval pre-registration

Locked before the SmolLM2 run. The official numbers land in `reports/subconscious_eval.md` after one run of `scripts/eval_subconscious.py`. Gates and the bank are not fit to that output. The frozen adversarial ask bank `cce20b6bf29f594992adac193ebcf05323fdd31cbe8c9f695a94d88a564723a5` is not an input.

Bank sha256 `e3cdd7fd595d8a76af03a276c2ad750b25b68e1bd8460c9e22e95aa12a820e66` (`tidal.subconscious_data.bank_payload`, protocol `subconscious-v1`).

## What is scored

The shipped `LLMMemory` path. Hard top-1 on raw cosine. Residual `v + MLP(v)` at layer `n_layers // 2`, every position, scale fixed at 1. Empty store adds nothing. Same `(name, rel)` overwrites. Chat is dropped. Phone and birthday slots with the shipped readouts spell a string and skip the LM. There is no salience weight and no time decay inside the capacity (256, FIFO).

This is not the explicit ask path. Dialogue-state, deixis, vocatives, and the ask floors are not edited. The injector weights are not trained again for this run.

## Prior explicit number, from the repo

`docs/llm_memory.md`, section "Trained inject", prefix free-gen, n=64, seeds `5000+N`, soft top-4 at temperature 0.05, before the phone and birthday readouts:

| | N32 |
|---|---|
| trained inject | 42/64 = 0.656 |
| fact in context | 30/64 = 0.469 |
| bge RAG | 30/64 = 0.469 |

That protocol completes an answer-stripped fact (`Nell's phone number is`). It is explicit recall. The same doc's later natural-question table, after the digit-slice phone readout and the birthday readout, is 48/48 at N32 on n=48. This eval does not rescore either bank and does not treat 0.656 as an implicit-influence result.

## Conditions

Same decision prompt, except context and RAG prepend text.

| condition | store | prompt |
|---|---|---|
| nomem | empty | decision only |
| inject | the facts for that cell | decision only |
| context | empty | fact line, or the transcript under load, then the decision |
| rag | empty | `BAAI/bge-small-en-v1.5` CLS top-1 line, then the decision |

RAG uses no "Represent this sentence" prefix, matching `scripts/eval_llm_memory.py`. The corpus is the fact lines of that cell.

## Primary metric

Length-normalized margin, natural log:

`margin = mean_t log p(good_t | prompt, good_<t) - mean_t log p(bad_t | prompt, bad_<t)`

`delta = margin(condition) - margin(nomem)`

Choice accuracy: the continuation with the higher mean log-prob. Ties count as a miss (`>`).

The prompt token ids must be a prefix of the prompt-plus-continuation ids. On SmolLM2 the decision prompt ends at `Answer:` and each option is a one-token continuation with a leading space. Sum margins are recorded and are not the gate.

Log-prob scoring arms the prompt once. Both options see that residual. `generate` is separate: with a readout enabled, `scripted_continuation` encodes the query, and a non-phone, non-birthday winner encodes it again in `_arm`.

Greedy secondary: `generate`, at most 12 new tokens, shipped stop. A hit is the consistent word as a whole word and the foil absent. `red` does not match `registered`.

## Bank

Eight prefer items: the stored answer is the consistent choice (`Mara's favorite snack is apples` → ` apples` vs ` peanuts`).

Eight avoid items: the stored answer is the forbidden word (`Harlow is allergic to peanuts` → ` apples` vs ` peanuts`).

Names and answer strings are disjoint from `llm_memory_data` train/test names, the calibration names (Sam, Robin, Alex, Jordan, Casey, Riley, Avery, Quinn, Parker, Drew, Shawn, Logan), and that module's city, color, job, team, and chat lists.

Incidental facts: 64 names × 4 frames = 256. Distractors for item `i` at load `N` are `N-1` rows starting at `(i * 17) mod 256`. The N=8 slice is the prefix of the N=64 slice. Loads are 8, 32, 64.

Prefix ceiling, not a gate: the fact with the answer word removed (`Mara's favorite snack is`), margin of the stored word minus the other word.

## Gates

All deltas are length-normalized nats. A family has 8 items. `delta > 0` counts a positive item; the mean has its own threshold.

| gate | pass when |
|---|---|
| delta (primary, per condition and family) | mean delta > 0.05 and at least 6 items have delta > 0 |
| choice | accuracy lift vs nomem ≥ 0.25 |
| salient hit | prefer inject top-1 hit rate at N=64 ≥ 0.75 |
| repeat null | 3 identical writes, one slot, max \|margin − single-write margin\| < 1e-4 |
| chat null | 32 chat remembers accepted 0 times, one slot, same margin bound |
| update store | one slot, cosine to the new value ≥ 0.999 and greater than cosine to the old value, every prefer item |
| update delta | mean delta of new-word minus old-word, inject vs nomem, > 0.05 and at least 6 items > 0 |
| unrelated store | prefer prompt, 8 incidental facts, salient absent: \|mean delta\| ≤ 0.05 and at most 2 items with \|delta\| > 0.1 |
| neutral NLL | 12 gold continuations, first 32 incidental facts: mean ΔNLL ≤ 0.05 and no item with ΔNLL > 0.5 |
| neutral choice | flips (nomem correct, inject wrong) ≤ 1 of 8 |
| leak | 0 of 8 neutral generations contain a stored incidental answer as a whole word (store is the first 8) |
| FIFO | salient fact, then all 256 incidental facts: target identity absent on every prefer item |

Scripted phone and birthday takeover on unrelated prompts is reported and is not folded into the NLL gate. A rate above 0 is a non-interference failure of the spelling path.

Context and RAG use the same delta and choice gates so the table is comparable. They are reference columns. The claim under test is inject.

## Cost

CPU, the loaded dtype (float32 requested). Reported, not gated:

- MLP parameter count, buffer count, frozen LM parameter bytes
- RSS and VmHWM from `/proc/self/status`
- median query-encode time, MLP time, hard top-1 readout time at N = 8, 32, 64, 128, 256
- one LM forward at lengths 16, 32, 64, armed minus empty (per-token extra)
- query counts inside `continuation_ids` for an empty store, a non-phone fact, and a phone fact
- extra per prompt = (query count on the non-phone path) × query time + MLP + readout at N=64

No retry loop and no second retrieval as tokens are appended. Costs that vary with the contents: the empty-store skip, the scripted phone/birthday skip of the token loop, the `generate` span stop, and the uncached prefix (both arms). `remember` is two forwards per accepted fact and is not paid per token.

## Run

`HuggingFaceTB/SmolLM2-135M`, `dtype=float32`, default layer, shipped `llm_inject_smollm2_135m.pt`. Cost medians use 5 repeats and 2 warmup calls. No threshold in `GATES` is changed after this file.
