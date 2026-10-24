# tidal architecture

tidal is only the subconscious and below-subconscious memory layer. Turn-taking, speaking decisions, and multimodal perception belong to the brain layer (growbrain). tidal only supplies memory signals.

The principle is human-likeness over usefulness. A mechanism stays when it reproduces a human memory effect, including a partial or fuzzy one. It is not tuned to maximize usefulness after the held-out bank is seen. Each mechanism names a neuroscience paper. Where a preregistration read that paper, the sentence says so. The other papers are the sources of the effect. Their measurements are not the fit numbers.

Memory vectors are text-only and modality-agnostic. `LLMMemory` encodes a text fact with SmolLM2-135M and stores the residuals. It does not read image, audio, video, or metadata features. Each stored slot has `modality`, and every write sets it to `text`.

## Store and inject

`EpisodicStore` (`tidal/episodic.py`) keeps one slot per fact: key, value, identity. Same-identity overwrite replaces the slot. Capacity 256 evicts FIFO. Key cosine does not merge a different identity. A different identity merges only when the group matches, the relation is the same non-empty string, and the surface token Jaccard is at least 0.85 (`REPEAT_JACCARD`). Chat and noise are not written.

`LLMMemory` (`tidal/llm_memory.py`) reuses that store. The default checkpoint is `HuggingFaceTB/SmolLM2-135M`. The inject layer is `n_layers // 2` (layer 15, hidden 576) and loads `tidal/llm_inject_smollm2_135m.pt`. The prompt does not contain the fact. The value is an answer-span pool at the inject layer, not L2-normalized. A phone of at least seven digits concatenates the leading coordinates of the last four answer tokens. A birthday in the spelling `Month D, YYYY` concatenates the month token, the day-tens token, the day-ones token, and the four year-digit tokens. The key drops sink tokens (norm above 8 times the median token norm of that string) and subtracts the fixed center in `tidal/llm_key_center.npz`.

With one slot the residual is

`s * u * gain * (scale(v) + MLP(scale(v)))`

`scale` is 1. `gain = 1 + 2 tanh(w · normalize(k))`. `u = 1 + tanh(valence)` at span 1. Generation does not sum a mixture onto every position. The cue is the per-token hidden state at layer 15. One slot with cosine at least theta 0.32 wins, ranked by `(log strength, cosine, -index)`. The residual on that token is `lambda * a * s * c * u * gain * (scale(v) + MLP(scale(v)))`, with `a = a_trig * fan_factor * decay` and `lambda` 0.3. `lambda` is not part of the exported activation. `fan_factor` is `max(0, (S - ln n) / S)` with `S` 1.45, and the half-life is 30 tokens. `activation_weights` still computes the earlier top-`MIX_K = 4` mixture at temperature 0.05. Generation does not call it. Spelling and retrieval-induced forgetting follow the triggered winner. `generate` does not read the store's `soft_k` or `soft_temp`.

Teyler and DiScenna (1986) describe the hippocampus as an index that reinstates a cortical pattern. The residual added into the language model is that idea: the stored pattern comes back into the network that is about to continue. https://doi.org/10.1037/0735-7044.100.2.147

## Mechanisms

Constants were chosen on a fit bank and were not retuned after the held-out score.

**Mixture.** `activation_weights` can leave several slots partly active. The weight is the analogue of Lebiere's chunk-choice softmax on `log(strength) + cosine`, and the sum of composed residuals is the Euclidean case of his estimation equation. Generation does not use that sum. It injects one cue winner. The preregistration read Collins and Loftus (1975) on spreading activation that summates from more than one node, McClelland and Rumelhart (1981) on parallel activation, and Lebiere (1999). https://doi.org/10.1037/0033-295X.82.6.407 https://doi.org/10.1037/0033-295X.88.5.375 https://doi.org/10.1007/s001970050071

**Coexist.** Similar memories stay in separate slots and then interfere. The preregistration read Norman and O'Reilly's preprint of the complementary-learning-systems model for pattern-separated traces, Underwood (1957) for similarity-dependent interference, and Anderson (1974) for a larger same-relation set lowering the target. The published Norman and O'Reilly paper is Psychological Review 110(4):611–646. https://doi.org/10.1037/0033-295X.110.4.611 https://doi.org/10.1037/h0044616 https://doi.org/10.1016/0010-0285(74)90021-8

**Retrieval-induced forgetting.** After a strict win, at most three other slots that share the winner's relation are scaled by `1 - 0.05 * weight`. The closer key is scaled more. The winner is unchanged. A larger shared relation, a tie, and a slot whose salience reaches 0.5 are left alone. The preregistration read Anderson, Bjork, and Bjork (1994), Anderson (2003), and Norman, Newman, and Detre (2007). The human result is a partial drop in recall. The gate is that partial drop in logprob margin. Cue-independent suppression is not implemented. https://doi.org/10.1037/0278-7393.20.5.1063 https://doi.org/10.1016/j.jml.2003.08.006 https://doi.org/10.1037/0033-295X.114.4.887

**Valence.** A write's experience field adds its sign times 0.25 to a scalar on the slot. The text is not scored. `u = 1 + tanh(valence)`. Replay multiplies a nonzero valence by 0.5. Paton, Belova, Morrison, and Salzman (2006) show amygdala neurons that represent the positive and negative value of stimuli during learning. The signed gain is that distinction. The fit did not use their firing rates. https://doi.org/10.1038/nature04490

**Circadian clock.** One global angle sits under the store. `phase = sin(angle)`. Angle 0 leaves write strength at 1 and decay at `0.5 ** age`. A nonzero phase scales the write by `1 + 0.25 * phase` and the decay exponent by `1 - 0.25 * phase`. The angle is not read from fact text. `remember` does not tick. The tick is `pi/16`. Hastings, Maywood, and Brancaccio (2018) review the suprachiasmatic nucleus as a content-free circadian generator. This clock is that kind of oscillator. It is not a model of the nucleus. https://doi.org/10.1038/s41583-018-0026-z

**Relative surprise.** Surprise is the answer-token NLL already returned by the value forward. After 32 accepted surprises are stored, a salient fact is written when its surprise is at least the quantile of that window. Quantile 0, the window minimum, is the shipped point. Fewer than 32 accepted surprises leaves the salient fact accepted. A rejection does not enter the window. Lisman and Grace (2005) describe a hippocampal–VTA loop that uses novelty to control entry into long-term memory. The gate is that kind of write filter. It is not their circuit. https://doi.org/10.1016/j.neuron.2005.05.002

**Replay.** Every 32 accepted writes, strength is recomputed as `0.5 ** age`, the single highest-salience slot is refreshed to strength 1, and at most one same-relation surface repeat is merged. Salience reuses the polarity vector. Wilson and McNaughton (1994) recorded hippocampal ensemble reactivation during sleep. The refresh is that kind of reactivation on a fixed budget. It is not a sleep model. https://doi.org/10.1126/science.8036517

**Spell.** When the prompt is completing the winning slot's relation, phone and birthday generation copies that slot's stored string, the full number or the full date. It does not spell a trained prefix and it does not spell a digit readout. Eichenbaum, Yonelinas, and Ranganath (2007) separate recollection of a specific episode from familiarity. The copy is that specific trace. Their recognition rates are not a gate. https://doi.org/10.1146/annurev.neuro.30.051606.094328

## Held record, #46–#55

The score is only implicit influence: the continuation margin against an empty store, summarized as Cohen's `d_z` across tone, choice, and preference where a mixture is scored. Explicit recall, an exact phone or birthday string, is reported separately and is not the score. Full intervals are in the linked reports and PR bodies. [#56](https://github.com/yunshenya/tidal/pull/56) later cut a key forward inside `_rank` with a locked identical readout, so these held numbers still describe the behavior. The cue schedule in `reports/subconscious_cue_prereg.md` selected theta 0.32, half-life 30, and lambda 0.3 on the direction fit bank (mean cued margin 0.479). Its held pass is false: prefer 46/50, avoid up 37/50, away 47/50, competitor below 1 is 15/50, winner unchanged 46/50, similarity 39/50, count 29/50, and fan 1/2/4 decreasing 44/50. The floors were not moved. A fit-only k-winner-take-all comparison (Norman, Newman, and Detre 2007, https://doi.org/10.1037/0033-295X.114.4.887) did not select a k. Store-size means at 8/16/32 were 0.0144/0.0132/0.0129 for one winner and 0.0163/0.0125/0.0125 for k=4. Cue-key cosine after `tidal/llm_key_center.npz` has token AUC 0.797 and per-prompt maximum AUC 1.0. Theta stays 0.32. Before that winner is injected, every other candidate that cleared theta takes the existing retrieval-induced-forgetting keep, and a closer companion is scaled more (Norman, Newman, and Detre 2007, https://doi.org/10.1037/0033-295X.114.4.887). On the direction held bank the same cell has cued mean margin 1.191 (7/8), unrelated absolute margin 0.0, false triggers 0/8, decay 5/8, store means 0.0154/0.0119/0.0125, quiet NLL 0.0 on 11 tokens, and fan means 1.191/0.461/0.000034 (7/8 strict). Prefer n is 8, so the n≥50 rule fails, and the decay point rule fails too. Inject cost is 0.000120, 0.000084, and 0.000108 extra seconds per token at store sizes 8, 16, and 32. On 2026-10-24 every DOI in this file resolved to the paper named beside it.

| PR | what | state | held |
|---|---|---|---|
| [#46](https://github.com/yunshenya/tidal/pull/46) | spell | merged | implicit: choice flips 1/8, held pass no; prefer / avoid / forbidden 20.750 / 1.453 / −1.578 (8/8). Explicit, reported separately: phone `+1-802-555-3145`, birthday `July 26, 1990`. Report `reports/subconscious_spell.md` |
| [#47](https://github.com/yunshenya/tidal/pull/47) | valence | merged | pass. toward 6/8 mean 1.865, away 8/8 mean −10.242. Report `reports/subconscious_valence.md` |
| [#48](https://github.com/yunshenya/tidal/pull/48) | circadian | merged | pass. high phase 8/8 mean 1.942, low phase 8/8 mean −1.677, decay gap 0.174. Report `reports/subconscious_circadian.md` |
| [#49](https://github.com/yunshenya/tidal/pull/49) | relative surprise | merged | pass. lows written 0/8, highs 8/8, bar 10.621. Report `reports/subconscious_surprise_rel.md` |
| [#50](https://github.com/yunshenya/tidal/pull/50) | RIF | merged | pass. competitor min drop 0.191, mean 0.312, winner drop 0, outsider drop 0. Report `reports/subconscious_rif_retry.md` |
| [#51](https://github.com/yunshenya/tidal/pull/51) | delete unused question dialogue | merged | no held bank. Behavior unchanged |
| [#52](https://github.com/yunshenya/tidal/pull/52) | implicit-influence measurement | draft | n=50 Wilson rates: prefer 50/50, avoid up 39/50, avoid stored down 42/50, toward 45/50, away 48/50. One-slot `d_z` 1.342. RIF geometry 0/50 because cosine overwrite collapsed the quartets |
| [#53](https://github.com/yunshenya/tidal/pull/53) | mixture K=4, T=0.05 | merged | pass. Mixture score 1.152, four-slot top-1 0.598, 150/150 probes kept 4 slots. Same n=50 implicit rates. Phone 50/50 and birthday 50/50 reported separately. `reports/subconscious_softmix_prereg.md` |
| [#54](https://github.com/yunshenya/tidal/pull/54) | coexist | merged | pass. Slots 50/50, competitor 50/50, winner 50/50, outsider 50/50, similarity 42/50, count 48/50. Same n=50 implicit rates. Phone 50/50 and birthday 50/50 reported separately. `reports/subconscious_interfere_prereg.md` |
| [#55](https://github.com/yunshenya/tidal/pull/55) | K=4 versus top-p | draft | constants unchanged. Shipped cell `d_z` 0.291 / 0.558 / 0.665 at store size 8 / 16 / 32. Help, mixture margin minus target-only margin, is negative on every cell |

## Planned tidal to growbrain interface

The per-token cue export is what generation injects. The repository still has no growbrain package and no encode hook, NLL hook, or inject hook for that backend. Those hooks stay **planned**. Each exported token carries `memory_id`, `vec`, `activation = a_trig * fan_factor * decay`, `strength*confidence`, `valence` as `u - 1`, `circadian`, `triggered`, `modality`, and `interface_version`. `modality` is `text` on every token, including a cleared trace. `interface_version` is 3. Version 2 was the activation formula. Version 3 adds `modality`. There is no perceptual-fingerprint field. `u - 1` is the offset in `u = 1 + tanh(valence)`, so a valence of 0 exports 0.

`fan_factor` decreases with the number of memories the cue retrieves. Anderson and Reder (1999) write source activation as `S - ln(fan)`. https://doi.org/10.1037/0096-3445.128.2.186 The fast `decay` is a per-token timescale. Joordens and Becker (1997) found semantic priming reduced to non-significant levels after one or two intervening items. Their unit is items. A conversion from items to tokens is a hypothesis. https://doi.org/10.1037/0278-7393.23.5.1083 The slow timescale is the strength and replay state already in the store. Tulving, Schacter, and Stark (1982) found word-fragment priming unchanged from one hour to seven days while recognition fell. https://doi.org/10.1037/0278-7393.8.4.336

The cue itself is the planned reason a memory is retrieved. Tulving and Thomson (1973) state encoding specificity: retrieval depends on the cue that was encoded with the episode. https://doi.org/10.1037/h0020071 Vaz, Inati, Brunel, and Zaghloul (2019) recorded coupled ripples between the medial temporal lobe and neocortex during human memory retrieval. https://doi.org/10.1126/science.aau8956 Lu, Hasson, and Norman (2022) model when an episodic memory is retrieved and when a new one is encoded. https://doi.org/10.7554/eLife.74445

Later, and not in the first signal list: a `source` field whose values are `cue`, `spontaneous`, or `none`, and a per-tick output. tidal only supplies the signals. Turn-taking, speaking decisions, and multimodal perception belong to the brain layer (growbrain).

## Not part of tidal: belongs to the brain layer (growbrain); kept in repo for now, pending handoff or removal

`duplex*` (`tidal/duplex.py`, `tidal/duplex_sim.py`), `turn_*`, `audio_*`, `tidal/modalities/`, the six `HEAD_DIMS` in `tidal/model.py`, `head_*`, `phase5_*`, and `tidal/shadow/` belong to the brain layer (growbrain). They cover turn-taking, speaking decisions, and multimodal perception. They stay in this repository for now, pending handoff or removal. This change keeps the files. Phase writeups remain `reports/phase1.md` through `reports/phase5.md` and `shadow/README.md`. Shadow mode loads `m3_ablate_m2`, logs, and does not call an external API. The 2026-10-08 bake-off recorded head-sum real validation 3.9061 for that ablation and 3.9276 for `mamba3_siso`. Continuum `heads` there still reads a raw-cosine soft top-4 at temperature 0.05. The memory inject is the cue trace in the store section.

## Multimodal fusion division (PLANNED, after text stage-1; no code now)

This split is planned. It comes after text stage-1. There is no code for it, and it adds no interface fields. The references below have not yet been re-verified by tidal.

tidal, the subconscious layer, would do only low-level automatic fusion: a temporal window, spatial or source matching, and reliability weighting. Cross-sensory events that occur together would raise salience and trigger a direct report upward (Meredith and Stein 1983; Van der Burg 2008). Already-learned pairs would complete through concept vectors (Faivre 2014). Cross-modal pairs would replay during sleep (Rudoy 2009). The fusion window and the weights would be learned from co-occurrence (Wallace and Stein 2007).

The brain layer (growbrain) would form new pairings and semantic-level fusion, including lip with voice and text with image. That needs attention (Palmer and Ramsey 2012; Alsius 2005; Talsma 2010). Pairings it learns would be written into tidal. growbrain may modulate tidal's fusion weights, in the way cortex modulates the superior colliculus.

Hypothesis, on weak evidence: the amygdala does fast cross-modal affect tagging.

The modules in the previous section stay in the repository pending handoff or removal. This planned split leaves them there.
