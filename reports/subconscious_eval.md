# Subconscious eval

Measurement of implicit influence on the shipped `LLMMemory` path (`HuggingFaceTB/SmolLM2-135M`, layer 15, hidden 576, float path as loaded). Bank `subconscious-v1` sha256 `e3cdd7fd595d8a76af03a276c2ad750b25b68e1bd8460c9e22e95aa12a820e66`. Gates were locked in `reports/subconscious_eval_prereg.md` before this run.

## Prior explicit number

`docs/llm_memory.md` (Trained inject) reports prefix free-gen at N32 of **42/64 = 0.656** inject vs **30/64 = 0.469** fact-in-context and **30/64** bge RAG. Protocol: answer-stripped prefix free-gen, n=64, seeds 5000+N, soft top-4 at temperature 0.05, before the phone and birthday readouts. That table is explicit completion of an answer-stripped fact, not this eval. Later in the same doc, natural-question free-gen with the phone digit slices and the birthday readout is 48/48 at N32 on n=48. This run does not rescore that bank.

## Setup

Conditions, same decision prompt where the condition allows it:

- **nomem**: empty store, prompt only.
- **inject**: facts in `EpisodicStore`, prompt only, hard top-1 residual (`v + MLP(v)` at every position).
- **context**: empty store, the fact line (or the whole transcript under load) prepended.
- **rag**: empty store, `BAAI/bge-small-en-v1.5` CLS top-1 line prepended. No query-instruction prefix, matching `scripts/eval_llm_memory.py`.

Primary score is the length-normalized margin: mean token log-prob of the memory-consistent continuation minus the mean token log-prob of the foil. `delta` subtracts the nomem margin. Choice accuracy is which continuation has the higher mean. Log-prob scoring arms once per prompt. `generate` uses at most 12 new tokens and the shipped stop.

Phone readout on: True. Date readout on: True. Capacity 256, overwrite cosine 0.9. Eval wall clock 78.8 s, model load 3.1 s. Phase seconds: axis1 19.7, controls 2.9, cost 6.3, encode 15.0, fifo 1.0, load 29.4, neutral 4.1, scripted 0.3.

Prefer items store the option that should be chosen. Avoid items store the option that should not (allergy, fear, refusal). A single fact is always the top slot, so axis 1 is the effect of the injected vector when retrieval cannot miss.

## Axis 1 — implicit choice

| family | condition | mean delta | items delta>0 | choice acc | nomem acc | delta gate | choice gate |
|---|---|---:|---:|---:|---:|---|---|
| prefer | inject | 13.577 | 8/8 | 8/8 = 1.000 | 7/8 = 0.875 | pass | fail |
| prefer | context | 3.448 | 8/8 | 8/8 = 1.000 | 7/8 = 0.875 | pass | fail |
| prefer | rag | 3.448 | 8/8 | 8/8 = 1.000 | 7/8 = 0.875 | pass | fail |
| avoid | inject | -16.700 | 0/8 | 0/8 = 0.000 | 6/8 = 0.750 | fail | fail |
| avoid | context | -2.638 | 0/8 | 1/8 = 0.125 | 6/8 = 0.750 | fail | fail |
| avoid | rag | -2.638 | 0/8 | 1/8 = 0.125 | 6/8 = 0.750 | fail | fail |

Stored-string lift (mean log-prob of the stored word under inject, minus nomem): prefer 9.229, avoid 9.945. On avoid items the stored word is the foil. A positive avoid lift means injection makes the forbidden word more likely.

Prefix ceiling (diagnostic): prompt is the fact with the answer removed (`Mara's favorite snack is`), margin is stored word minus the other word. Not an implicit prompt.

| id | family | prefix delta | inject chooses stored | decision inject delta | decision context delta |
|---|---|---:|---:|---:|---:|
| prefer-mara | prefer | 12.970 | 1 | 16.520 | 5.481 |
| prefer-helio | prefer | 13.725 | 1 | 16.448 | 1.658 |
| prefer-nia | prefer | 7.313 | 1 | 9.573 | 1.750 |
| prefer-colm | prefer | 21.653 | 1 | 19.149 | 4.925 |
| prefer-petra | prefer | 11.548 | 1 | 9.911 | 3.620 |
| prefer-idris | prefer | 15.114 | 1 | 15.883 | 4.373 |
| prefer-sable | prefer | 12.686 | 1 | 14.184 | 2.358 |
| prefer-juniper | prefer | 9.439 | 1 | 6.945 | 3.419 |
| avoid-harlow | avoid | 10.968 | 1 | -17.803 | -2.817 |
| avoid-nestor | avoid | 15.821 | 1 | -14.038 | -0.922 |
| avoid-cleo | avoid | 11.161 | 1 | -13.890 | -1.224 |
| avoid-bodhi | avoid | 12.215 | 1 | -16.499 | -0.941 |
| avoid-ingrid | avoid | 21.422 | 1 | -22.007 | -2.291 |
| avoid-tomas | avoid | 16.254 | 1 | -14.709 | -3.758 |
| avoid-yara | avoid | 24.410 | 1 | -18.996 | -6.248 |
| avoid-felix | avoid | 14.171 | 1 | -15.659 | -2.906 |

Greedy whole-word hit (consistent word present, foil absent):

| family | nomem | inject | context | rag |
|---|---:|---:|---:|---:|
| prefer | 0/8 | 8/8 | 4/8 | 4/8 |
| avoid | 0/8 | 0/8 | 0/8 | 0/8 |

## Axis 2 and 3 — load, transcript length, repeat, update, FIFO

Under load the store is the salient fact plus N−1 incidental facts (nested slice, stride 17). Context is the full transcript plus the decision prompt. RAG retrieves over those lines. Inject's prompt stays the short decision. Slot counts below capacity mean a cosine overwrite merged two identities.

| family | N | slots | survived | inject hit | inject delta | context delta | rag delta | rag line hit |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| prefer | 8 | 6.9 | 8/8 | 2/8 | 2.888 | 2.079 | 3.448 | 1.000 |
| prefer | 32 | 21.9 | 8/8 | 1/8 | -0.289 | 1.649 | 3.448 | 1.000 |
| prefer | 64 | 41.4 | 8/8 | 1/8 | 0.973 | 0.964 | 3.448 | 1.000 |
| avoid | 8 | 6.9 | 8/8 | 1/8 | -2.260 | -1.444 | -2.638 | 1.000 |
| avoid | 32 | 21.9 | 8/8 | 0/8 | -0.441 | -0.911 | -2.638 | 1.000 |
| avoid | 64 | 41.4 | 8/8 | 0/8 | -1.589 | -0.669 | -2.638 | 1.000 |

Salient-hit gate at N=64 on prefer (hit rate ≥ 0.75): fail (1/8 = 0.125).

Identical fact written 3 times (same identity overwrites). Max |margin − single-write margin| = 0.000000. Repeat-null gate (< 0.0001): pass.

Prefer prompt with 8 incidental facts and the salient fact absent. Mean delta -0.652, items with |delta| > 0.1: 7. Unrelated-store gate: fail.

Thirty-two chat lines after the salient fact (chat is dropped by the write gate). Accepted writes 0. Max |margin change| 0.000000. Gate: pass.

Same-key update on prefer items (apples overwritten by pears, and the other pairs). The scored pair is the new word versus the old word.

Store holds the new value (cosine ≥ 0.999 and above the old value) on every item: pass. Mean inject delta 15.084 (8/8 items > 0). Delta gate: pass. Mean lift of the new word 10.168, of the old word -4.916.

FIFO: salient fact, then 256 incidental facts (capacity 256). Target absent on 0/8. Mean slots 138.0. Gate (every target gone): fail. Mean decision delta after the fill 1.114 versus single-fact delta 13.577.

## Axis 4 — non-interference

Neutral gold continuations, store = first 32 incidental facts. Mean ΔNLL (inject − nomem) 4.405. Items with ΔNLL > 0.5: 9. Gate: fail.

| id | ΔNLL | top slot |
|---|---:|---|
| nll-mat | 5.781 | ['Kestrel', 'keeps'] |
| nll-there | 8.104 | ['Faye', 'keeps'] |
| nll-much | 4.397 | ['Honora', 'keeps'] |
| nll-day | 3.473 | ['Eamon', 'keeps'] |
| nll-begin | 10.775 | ['Gideon', 'keeps'] |
| nll-rose | -1.267 | ['Vesper', 'keeps'] |
| nll-east | 6.126 | ['Gideon', 'keeps'] |
| nll-four | -1.079 | ['Vesper', 'keeps'] |
| nll-everyone | 9.359 | ['Kestrel', 'keeps'] |
| nll-door | -1.106 | ['Vesper', 'keeps'] |
| nll-book | 7.192 | ['Nadine', 'keeps'] |
| nll-night | 1.107 | ['Eamon', 'keeps'] |

Neutral forced choices. Nomem acc 1.000, inject acc 0.625, flips (nomem right, inject wrong) 3/8. Gate (flips ≤ 1): fail.

Leak: greedy on 8 neutral prompts with 8 rare incidental answers in the store. Prompts that emit one of those words: 3. Gate: fail.

Scripted readout. A phone slot spells `+1-415-555-` plus four digits and never calls the LM. A birthday slot spells `Month D, YYYY`. Unrelated prompts are the first four neutral generations. The phone question is a positive control that the readout is on. Takeover on an unrelated prompt is reported and is not part of the NLL gate.

Phone unrelated takeover rate 1.000. Phone question takeover 1. Birthday unrelated takeover rate 1.000.

| kind | prompt | takeover | text |
|---|---|---:|---|
| phone | `What should we cook tonight?` | 1 | +1-415-555-4242 |
| phone | `Tell a very short joke.` | 1 | +1-415-555-4242 |
| phone | `How does a bedtime story start?` | 1 | +1-415-555-4242 |
| phone | `Name something seen in the sky.` | 1 | +1-415-555-4242 |
| phone | `What is Quill's phone number?` | 1 | +1-415-555-4242 |
| birthday | `What should we cook tonight?` | 1 | March 4, 1991 |
| birthday | `Tell a very short joke.` | 1 | March 4, 1991 |
| birthday | `How does a bedtime story start?` | 1 | March 4, 1991 |
| birthday | `Name something seen in the sky.` | 1 | March 4, 1991 |
| birthday | `What is Quill's phone number?` | 1 | March 4, 1991 |

## Cost

Inject MLP parameters 664704 (module parameters 664704; buffers 66931, the phone and date readout tensors). Frozen LM parameter bytes 538060032. Torch threads 4.

RSS before the cost section: VmHWM 1428824 kB, VmPeak 3438360 kB, VmRSS 1285292 kB. After: VmHWM 1428824 kB, VmPeak 3438360 kB, VmRSS 1285868 kB. VmHWM is the process high-water mark and includes the model load. CPUs 4. Model-load RSS: after_eval (VmHWM 1428824 kB, VmPeak 3438360 kB, VmRSS 1286048 kB); after_load (VmHWM 1275720 kB, VmPeak 3312932 kB, VmRSS 1057284 kB); before (VmHWM 226428 kB, VmPeak 756352 kB, VmRSS 226428 kB).

Median query encode 0.0257 s. Median MLP forward 0.000052 s. Median hard top-1 readout (store only, no LM forward):

| N | median s |
|---:|---:|
| 8 | 0.000017 |
| 32 | 0.000024 |
| 64 | 0.000032 |
| 128 | 0.000076 |
| 256 | 0.000164 |

One LM forward, residual armed minus residual absent. This is the per-token extra cost of the add, at a fixed length. Generation itself recomputes the prefix each token for both conditions.

| length | empty s | armed s | extra s |
|---:|---:|---:|---:|
| 16 | 0.0277 | 0.0274 | -0.000280 |
| 32 | 0.0340 | 0.0343 | 0.000341 |
| 64 | 0.0491 | 0.0520 | 0.002904 |

Mean extra per token across those lengths: 0.000989 s.

Shipped `continuation_ids` query count on 8 forced tokens: empty 0 (forwards 8, tokens 8), non-phone inject 2 (forwards 10, tokens 8), phone slot 1 (forwards 1, tokens 8).

Extra per prompt versus nomem, from the pieces: 2 × query (0.0514 s) + MLP (0.000052 s) + readout at N=64 (0.000032 s) = 0.0515 s. The token loop's extra is the armed-minus-empty column above, paid again on every new token.

Median `continuation_ids` wall time for the same prompt:

| store | median s |
|---|---:|
| empty | 0.2422 |
| inject | 0.2822 |
| scripted_phone | 0.0256 |

Variable-cost paths (no retries, no re-query inside the token loop):

- **query encode**. scripted_continuation and _arm, before any new token. skipped when the store is empty; runs twice when a readout is on and the winning relation is not phone or birthday.
- **scripted phone or birthday**. winning slot rel is phone or birthday and that readout buffer is on. replaces the token loop.
- **span stop**. generate only. stops before max_new_tokens on a newline or sentence punctuation.
- **uncached prefix**. every LM token, with or without a residual. each new token forwards the whole prefix; use_cache is False.
- **remember**. each accepted fact. two full forwards (key, then value); chat and noise return before that.

## Failure analysis

Single-fact inject moves every prefer item toward the stored word and every avoid item toward the forbidden word. Prefer deltas run from 6.945 to 19.149. Avoid deltas run from -22.007 to -13.890.

Nomem already chooses the prefer word on 7/8 items. The choice gate wants a +0.25 accuracy lift, and that base rate leaves no room for it.

Prefix completion of the fact, with the answer stripped, chooses the stored word on 16/16 and the margin delta is positive on 16/16. The same vector that completes `allergic to` with `peanuts` is the vector added during `Pick one snack`.

Greedy hits, prefer: nomem 0/8, inject 8/8.

Greedy hits, avoid: nomem 0/8, inject 0/8.

On those avoid misses the greedy span contains the stored word and not the safe word: peanuts, milk, dogs, steak, honey, planes, drums, grapes.

N=64 prefer: 1/8 retrieval hits, salient slot still stored on 8/8, slots 23–51 after 64 writes. Mean inject delta on hits 6.945, on misses 0.119. top cosine mean 0.399, top1−top2 gap mean 0.009 (min 0.002, max 0.021).

N=64 avoid: 0/8 retrieval hits, salient slot still stored on 8/8, slots 23–51 after 64 writes. Mean inject delta on hits —, on misses -1.589. top cosine mean 0.407, top1−top2 gap mean 0.009 (min 0.001, max 0.022).

The salient fact plus 256 later facts (257 writes) into capacity 256 left 138 slots. Cosine overwrite merged the repeated incidental frames, so the store never filled and no salient identity was evicted (0/8 absent). The decision delta still fell from mean 13.577 to 1.114, because a different slot won the query.

Leak on 3/8 neutral generations. The stored word that appears is ['bodkin']. Samples: 'Bodkin napheadkin bodkin napheadkin'; 'bodkin needle.'; 'BodPax bodkin needle, bodkin needle, bod'.

bge returns the salient line on prefer at N=64 with rate 1.000. The decision prompt contains the person's name and the fact line does too, so lexical retrieval succeeds where the hidden-state key does not. The RAG delta therefore matches the single-fact RAG delta at every load. The transcript condition is the one that gets longer, and its prefer delta falls as N grows.

## Recommendation

Single-fact prefer inject mean delta is 13.577 (gate pass). Avoid inject mean delta is -16.700 (gate fail). Prefix mean delta is 14.429. Prefer stored-word lift is 9.229; avoid stored-word lift is 9.945.

Prefer moves toward the stored word and avoid does not apply the constraint. The residual is doing what it was trained to do, which is to emit the value. The next training objective should score the decision continuation, with the forbidden word as a negative, instead of next-token loss on the fact's answer span alone.

In-context text also moves avoid the wrong way (mean delta -2.638). Inject moves it further (-16.700). Both are lexical priming of the stored word. The context column shows the 135M model does not treat `allergic to peanuts` as a reason to refuse peanuts. The residual is a stronger version of that prime, not a weaker copy of a constraint the text already applied.

Prefer retrieval at N=64 is 0.125, under the 0.75 gate. The query is a sink-dropped mean of the decision prompt, and the key is the fact with the answer removed. A salience-weighted write does not change that geometry. The next retrieval step is a query that is built for the decision, or a margin gate that refuses a weak top-1, still computed on every call.

Writing the same fact three times leaves one slot and the same margin. There is no repetition weight on this path.

An incidental store moves prefer decisions with the salient fact absent. Hard top-1 always adds some value when the store is non-empty. A gated add, scaled by the top-1 cosine or by the gap to the second slot, is the non-interference mechanism. It stays on the main path: the scale is a function of the scores, not a switch.

Neutral NLL moves past the pre-registered band. The same cosine gate that would drop a weak top-1 is the candidate fix. It should be fit on a later bank, not on this one.

Unrelated prompts with a phone slot are spelled as a phone number. `scripted_continuation` keys off the winning relation, not off the prompt. The spelling path should run only when that relation is the one the prompt is completing, or the residual path should be the one that runs for every other prompt. Collapsing the scripted probe and `_arm` into one query removes the second prompt forward on ordinary facts; that is a cost fix, and it does not change the armed vector.

Overwrite replaces the slot, and the decision margin moves toward the new word.

The salient fact plus 256 later facts (257 writes) leave 138.0 slots, and every salient identity is still there. Similar incidental frames collapse under cosine 0.9, so FIFO never runs. Influence does not stay with the surviving slot: mean decision delta falls from 13.577 to 1.114. There is no age-based fade. There is also no guarantee that a stored fact is the one added at the next prompt.
