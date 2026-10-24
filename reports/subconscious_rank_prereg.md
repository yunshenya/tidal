# Rank cost

Performance only. `MIX_K` stays 4 and `MIX_TEMP` stays 0.05. No gate, bank, mixture weight, or spelling rule changes. The production readout stays the strength-weighted mixture.

`_rank` was about 0.014 s per continuation token in #52, against about 0.0018 s for the in-context extra forward. That `_rank` time is one query forward plus the slot read. The query forward was running every decoder layer and the output head. The pool only needs the inject layer.

On the shipped causal LM the key and the query run that model's decoder forward with `num_hidden_layers` set to the inject layer plus one, and copy that block's output. The count is restored before the method returns. Later blocks and the output head do not run. Value and surprise still run the whole model, because they need the logits. A passed-in model keeps the full forward: its hidden state is whatever `output_hidden_states` returns.

## Identity

No absolute tolerance. Recorded before the measurement run.

Identity runs with one CPU thread. On the default thread pool this bfloat16 checkpoint is not bitwise stable: a repeated full forward moved one coordinate by one bfloat16 step, 128 at the sink magnitude, and moved a pooled key by as much as 1.49. One thread makes the full forward and the prefix bitwise, so `torch.equal` is required there.

The float32 hidden state at the key layer is `torch.equal` to the full forward. The pooled key is `np.array_equal`. The mixture value and the weight vector are `np.array_equal`. The injection residual is `torch.equal`. A second forward with that same residual is the logit control. When the control is bitwise, the armed last-token logits are required to be `torch.equal` too.

The strings are every fact text and every prompt in the n=50 measure banks (tone, choice, preference, and their distractors; the 50 RIF quartets; phone; birthday; prefer; avoid; toward; away; surprise) and every held bank of the locked gates (frozen, gate, direction, replay, spell, valence, circadian, relative surprise, retrieval-induced forgetting). `_rank` and the residual are compared on each of the 150 implicit probes (50 tone, 50 choice, 50 preference).

## Cost

Before is `_rank` with the uncut query forward (`full=True`). After is `_rank` with the layer stop. Same process, same prompt, same slots.

The prompt is `Weigh unlabeled cadence.\nTone mqtag0:`. The good continuation is ` warm` and the bad continuation is ` harsh`. Continuation tokens are `token_ids(prompt + good)` minus `token_ids(prompt)`, plus the same difference for the bad continuation. `token_ids` is `tidal.subconscious_eval.token_ids`. Per continuation token is the median seconds divided by that count.

Store sizes are 8, 16, 32, 128. Each size is filled from `numpy` Generator seed 0 with float32 normal keys and values, unique names, relation `rankcost`, and strength 1, through `store.write`. There is no language-model write. warmup 2, repeats 7, median of `perf_counter`. The scan column replaces the query vector with a cached one, so it is the slot read without the forward.

Threads are `os.cpu_count()`. The model is `HuggingFaceTB/SmolLM2-135M` as loaded. This table is this CPU. It is not a reason to change `MIX_K` or `MIX_TEMP`.

At the cost thread count the report also records the max absolute difference among 7 repeated pooled keys of the cost prompt: full against full, prefix against prefix, and prefix against full. That row is the noise floor. It is not a pass rule.

## Result

The measurement has not been recorded.
