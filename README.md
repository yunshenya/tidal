# tidal 潮汐

tidal is only the subconscious and below-subconscious memory layer. It stores specific text facts and adds a residual into a frozen causal language model. Turn-taking, speaking decisions, and multimodal perception belong to the brain layer (growbrain). tidal only supplies memory signals.

The design prefers a human-like memory, including partial and fuzzy effects, over a store that is as useful as possible. Each mechanism below is tied to a neuroscience paper. The fit numbers are not the paper's numbers.

Memory vectors are text-only and modality-agnostic. A key and a value are residuals of text from SmolLM2-135M. They do not read image, audio, video, or other modality features. Each stored slot has `modality`, and every write sets it to `text`.

The longer account is [docs/architecture.md](docs/architecture.md).

Any pull request that changes behavior, mechanisms, constants, or the interface must update `README.md` and `docs/architecture.md` in that same pull request (what changed, the citations, and the held numbers). A pull request with stale docs is not mergeable.

## Implemented memory

`tidal/episodic.py` `EpisodicStore` is the slot list. One fact is one slot. The same identity overwrites that slot. Capacity is 256, then FIFO. Key cosine does not merge slots. A different identity is merged only when the group matches, the relation is the same non-empty string, and the surface token Jaccard is at least `REPEAT_JACCARD` 0.85.

`tidal/llm_memory.py` `LLMMemory` writes those slots from a frozen `HuggingFaceTB/SmolLM2-135M`. The inject layer is `n_layers // 2`, layer 15, hidden 576. That default loads `tidal/llm_inject_smollm2_135m.pt`. The fact text is not copied into the prompt. Generation does not broadcast a mixture. The cue is the per-token hidden state at the inject layer. One slot whose cosine with that cue clears theta 0.32 is the winner, ranked by `(log strength, cosine, -index)`. The residual on that token is `lambda * a * s * c * u * gain * (scale(v) + MLP(scale(v)))`, with `a = a_trig * fan_factor * decay`, `scale` fixed at 1, `gain = 1 + 2 tanh(w · normalize(k))`, and `u = 1 + tanh(valence)`. `lambda` is 0.3 and is not part of the exported activation. `fan_factor` is `max(0, (S - ln n) / S)` with `S` 1.45. The half-life is 30 tokens. `activation_weights` still implements the earlier mixture (`MIX_K` 4, `MIX_TEMP` 0.05). Generation does not call it. Spelling and retrieval-induced forgetting follow the triggered winner. The value is an answer-span pool. A phone or a birthday is concatenated slices of that span, and when the prompt is completing that relation the emitted string is the stored answer.

Shipped constants, fit on a fit bank and not retuned on held-out: polarity span 2, valence span 1 and step 0.25, clock span 0.25 with tick `pi/16`, surprise window 32 at quantile 0, replay every 32 accepted writes with strength `0.5 ** age` and one refresh, RIF span 0.05 on the other same-relation candidates that cleared theta, with a closer companion scaled more, once per cue retrieval event and not once per token. A different relation is not scaled. The fan factor scales activation down and does not clear a trigger.

The score is only implicit influence: the continuation margin against no memory, summarized as Cohen's `d_z`. Explicit recall, an exact phone or birthday string, is reported separately and is not the score.

| PR | mechanism | state | held |
|---|---|---|---|
| [#46](https://github.com/yunshenya/tidal/pull/46) | spell the stored phone and birthday | merged | implicit: choice flips 1/8, held pass no; prefer / avoid / forbidden means 20.750 / 1.453 / −1.578 (8/8). Explicit, reported separately: `+1-802-555-3145` and `July 26, 1990` |
| [#47](https://github.com/yunshenya/tidal/pull/47) | valence | merged | pass. toward 6/8, mean 1.865; away 8/8, mean −10.242 |
| [#48](https://github.com/yunshenya/tidal/pull/48) | circadian clock | merged | pass. high phase 8/8, mean 1.942; low phase 8/8, mean −1.677; decay gap 0.174 |
| [#49](https://github.com/yunshenya/tidal/pull/49) | relative surprise gate | merged | pass. lows written 0/8, highs 8/8, window bar 10.621 |
| [#50](https://github.com/yunshenya/tidal/pull/50) | retrieval-induced forgetting | merged | pass. competitor min drop 0.191, mean 0.312; winner and outsider drop 0 |
| [#51](https://github.com/yunshenya/tidal/pull/51) | delete unused question-dialogue code | merged | no held bank |
| [#52](https://github.com/yunshenya/tidal/pull/52) | measure implicit influence | draft | n=50: prefer 50/50, avoid up 39/50, avoid stored down 42/50, toward 45/50, away 48/50. One-slot score 1.342. Under the old cosine overwrite, RIF geometry was 0/50 |
| [#53](https://github.com/yunshenya/tidal/pull/53) | mixture K=4, T=0.05 | merged | pass. four-slot mixture score 1.152 against top-1 0.598 (150/150 probes kept 4 slots). Same n=50 implicit rates. Phone and birthday exact 50/50, reported separately |
| [#54](https://github.com/yunshenya/tidal/pull/54) | coexist, then interfere | merged | pass. slots 50/50, competitor down 50/50, winner unchanged 50/50, outsider unchanged 50/50, similarity 42/50, count 48/50. Same n=50 implicit rates. Phone and birthday exact 50/50, reported separately |
| [#55](https://github.com/yunshenya/tidal/pull/55) | fixed K=4 against cumulative top-p | draft | `MIX_K` and `MIX_TEMP` unchanged. Shipped cell `d_z` 0.291 / 0.558 / 0.665 at store size 8 / 16 / 32. Help is negative on every cell |

[#56](https://github.com/yunshenya/tidal/pull/56) is merged. It removes a key forward inside `_rank` and locks the readout as identical, so it adds no held behavior.

The cue schedule is in `reports/subconscious_cue_prereg.md`. On the direction fit bank the selected cell is theta 0.32, half-life 30, lambda 0.3, mean cued margin 0.479. The schedule held, scored before competitive retrieval, is false: prefer 46/50, avoid up 37/50, away 47/50, competitor below 1 is 15/50, winner unchanged 46/50, similarity 39/50, count 29/50, and fan 1/2/4 decreasing 44/50. The floors were not moved. `interface_version` is 3. Version 2 was the activation `a_trig * fan_factor * decay`. Version 3 adds `modality` on every stored slot and every exported token. The value is `text`. There is no perceptual-fingerprint field.

A fit-only diagnostic compared that one winner with a k-winner-take-all mix at k=2, 3, and 4 (Norman, Newman, and Detre 2007). It did not select a k. Mean cued margin stayed 0.479. Store-size means at 8/16/32 were 0.0144/0.0132/0.0129 for k=1 and 0.0163/0.0125/0.0125 for k=4. Held was not rescored.

Cue-key cosine after `tidal/llm_key_center.npz`, on the direction fit bank: token AUC 0.797 (related n=80, unrelated n=368). The per-prompt maximum AUC is 1.0. Theta stays 0.32.

Before the winner is injected, every other same-relation candidate that cleared theta is scaled by the existing retrieval-induced-forgetting keep, once per cue retrieval event and not once per token. A retrieval event is a trigger onset: the first token where a winner fires after no open retrieval for that memory, or a winner change. Retriggers of the same winner within its active decay window do not rerun competition. The retrieval stays open on a retrigger of that winner, including when activation is 0, and it stays open on a later token only while the carried state has memory_id >= 0 and activation > 0. Anderson, Bjork, and Bjork (1994) treat retrieval-induced forgetting as a consequence of a retrieval attempt. A different relation is never scaled. A closer companion, measured by key cosine to the winner, is scaled more (Norman, Newman, and Detre 2007). The winner is not scaled, and a slot that missed theta is not scaled. The fan factor only scales activation. A token with candidates past theta still triggers when that activation is below 0.05, including when the factor is 0. On a later token, activation below 0.05 still clears the carried trace.

The direction held bank was scored once at that cell and was not used to choose it. Prefer n=8, so every rate fails the n≥50 rule. The point pass is also false: decay is 5/8, below the 6/8 point rule, on the logit comparison. Cued mean margin is 1.191 (7/8 positive). Unrelated absolute margin is 0.0. False triggers are 0/8. Store-size means at 8/16/32 are 0.0154/0.0119/0.0125. Quiet NLL is 0.0 on 11 tokens. Fan means are 1.191/0.461/0.000034 with 7/8 strict. Inject cost on the widened half-life fit is 0.000120, 0.000084, and 0.000108 extra seconds per token at store sizes 8, 16, and 32. The citations in `docs/architecture.md` were resolved on 2026-10-24; each DOI returned the named paper.

A fresh direction bank, protocol `cue-direction-fresh-v1`, with 50 prefer items and disjoint from the fit bank and that direction held bank, was scored once after those gates were pre-registered. The fresh pass is false. Cued margin is positive on 1/50 (Wilson 95% [0.004, 0.105]), mean −0.147. Unrelated absolute margin is 0.00000344. False triggers are 0/50 (Wilson [0.000, 0.071]). Decay is 0/50 (Wilson [0.000, 0.071]); all 50 failures are the logit comparison, and activation still fell with distance. Store means at 8/16/32 are 0.0393/0.0362/0.0356. Quiet NLL is 0.0 on 150 tokens. Fan means are −0.147/0.093/−0.000031 with 0/50 strict (Wilson [0.000, 0.071]). The same pass rescored the previous floors. The floor pass is false: prefer 46/50 (Wilson [0.812, 0.968]), avoid up 37/50 (Wilson [0.604, 0.841]), avoid stored 42/50 (Wilson [0.715, 0.917]), toward 47/50 (Wilson [0.838, 0.979]), away 47/50 (Wilson [0.838, 0.979]), at least 4 slots 50/50 (Wilson [0.929, 1.0]), competitor below 1 is 0/50 (Wilson [0.000, 0.071]), winner unchanged 48/50 (Wilson [0.865, 0.989]), outsider unchanged 46/50 (Wilson [0.812, 0.968]), similarity 35/50 (Wilson [0.562, 0.809]), count 32/50 (Wilson [0.501, 0.759]), and fan strictly decreasing 44/50 (Wilson [0.762, 0.944]). Fan means on that bank are 3.035/0.992/0.000184. Inject cost on this pass is 0.000057, 0.000072, and 0.000080 extra seconds per token at store sizes 8, 16, and 32 (15 tokens). Theta, half-life, lambda, and the floors stay at the pre-registered values. The pull request stays a draft.

A diagnosis pass did not change the mechanism. On the floor bank, replaying `compete_retrieved` matches the stored strengths on 50/50. A strict win fires on 19/50. All 150 competitors clear theta somewhere, and 131 of them are only in candidate lists that do not trigger, because four or five slots past theta put the fan factor under the 0.05 floor. Competitor-below-1 stays 0/50. The 4 outsider changes are outsiders that entered a triggered three-candidate set. A 50-item fit bank in the n=8 template, disjoint from held, scores cued margin positive 41/50, mean 0.247, Wilson 95% [0.692, 0.902]. Decay on that bank is 4/50. Fan is 38/50 with means 0.247/0.059/−0.000011. On the fresh bank the intended slot is the winner on 50/50, and activation at the first answer predictor is about 0.933, three tokens after the cue. The mean margin there remains −0.147.

The nonce bank is not a valid direction held bank: distribution mismatch: injection cannot steer toward unknown nonce tokens. The fan factor now only scales activation down, so a token with candidates past theta still triggers when that product is below 0.05. Retrieval-induced forgetting scales only same-relation candidates in the theta set. Decay is scored on activation from the last trigger, with no retrigger on the tokens in between. Logit L1 is not part of that gate. A new held bank, protocol `cue-direction-word-v1`, uses the same real-word template as fit and is disjoint from the fit banks and the earlier held banks. The gates stay the pre-registered 38/50 direction rates and the previous n=50 floors. The word bank was not scored. A fit-only check at the shipped cell, alpha still 0.25, scores the n=8 bank at cued mean 0.479 (7/8) and decay 8/8, and the 50-item fit bank at cued mean 0.247 (41/50) and decay 50/50. Those fit counts did not move a floor.

The word bank was scored once after that pre-registration, at git `b1884217b6d3ada635acc351bc70d8cc581858ca`, sha256 `7a8f62d244ec98b52349bf441bcc3262c9c5ff10f7148d3d79fb824e7d109110`, elapsed 197.2 seconds. Theta, half-life, lambda, alpha, fan S, the RIF span, and the floors were not changed. The direction pass is true. The floor pass is false. The overall pass is false. Cued margin is positive on 43/50, mean 0.574797535019073, Wilson 95% [0.7381380628941446, 0.9304916657298371]. Unrelated absolute margin is 0.0007146249711513519 on n=50. False triggers are 0/50, Wilson [0.0, 0.07134759913335872]. Decay is 50/50, Wilson [0.9286524008666414, 1.0]. Store means at 8/16/32 are 0.054827355566279226 / 0.04411715648680307 / 0.03872253612925597. Quiet NLL is −2.347582425826635e-05 on 117 tokens. Fan is strict on 40/50, Wilson [0.6696289406777459, 0.887562499842239], means 0.574797535019073 / 0.19655592603958205 / 0.008063986022801456. The floor re-score is prefer 46/50 (Wilson [0.8116175308165717, 0.968450485911407]), avoid up 37/50 (Wilson [0.6044684273515117, 0.8412847250644762]), avoid stored 42/50 (Wilson [0.7148578393696501, 0.916625793219666]), toward 47/50 (Wilson [0.8378290831116182, 0.979385029651026]), away 47/50 (Wilson [0.8378290831116182, 0.979385029651026]), at least 4 slots 50/50 (Wilson [0.9286524008666414, 1.0]), competitor below 1 is 44/50 (Wilson [0.7619518261679701, 0.9438239984906773]), winner unchanged 16/50 (Wilson [0.20758216260656845, 0.45810297308144077]), outsider unchanged 50/50 (Wilson [0.9286524008666414, 1.0]), similarity 35/50 (Wilson [0.562496495355466, 0.8089644649911906]), count 35/50 (Wilson [0.562496495355466, 0.8089644649911906]), and fan strictly decreasing 45/50 (Wilson [0.7863976856252035, 0.9565242350681096]). Floor-bank fan means are 3.0352763720974325 / 0.9923983175990482 / 0.03559282800182707. Inject cost on this pass is 0.00011272154436235063, 0.0002204299089498818, and 0.00020730981883719903 extra seconds per token at store sizes 8, 16, and 32 (11 tokens, warmup 2, repeats 7). Armed seconds per token are 0.003010886090289039, 0.003038886090507731, and 0.0029256822731853886. The pull request stays a draft.

RIF now runs once per cue retrieval event, not per token. A retrieval event is a trigger onset: the first token where a winner fires after no open retrieval for that memory, or a winner change. Retriggers of the same winner within its active decay window do not rerun competition. Anderson, Bjork, and Bjork (1994) treat retrieval-induced forgetting as a consequence of a retrieval attempt. Theta, half-life, lambda, fan S, the RIF span, alpha, and the floors are unchanged. A fit-only check at the shipped cell scores the n=8 bank at cued mean 0.479 (7/8) and decay 8/8, and the 50-item fit bank at cued mean 0.247 (41/50) and decay 50/50. Those fit counts did not move a floor. The event bank was not scored.

## Planned interface to growbrain

The per-token cue export is what generation injects. There is still no growbrain package and no encode, NLL, or inject hook for that backend. Those hooks stay **planned**. Each exported token carries:

- `memory_id`
- `vec`
- `activation = a_trig * fan_factor * decay`
- `strength*confidence`
- `valence`, exported as `u - 1`
- `circadian`
- `triggered`
- `modality`, currently always `text`
- `interface_version` 3

A `source` field, with values `cue`, `spontaneous`, or `none`, and a per-tick output are later. They are not in that first signal list. tidal only supplies these signals. Turn-taking, speaking decisions, and multimodal perception belong to the brain layer (growbrain).

Low-level cross-sensory fusion is planned after text stage-1. The account is in [docs/architecture.md](docs/architecture.md). It has no code and adds no interface fields.

## Not part of tidal: belongs to the brain layer (growbrain); kept in repo for now, pending handoff or removal

`duplex*` (`tidal/duplex.py`, `tidal/duplex_sim.py`), `turn_*`, `audio_*`, `tidal/modalities/`, the six `HEAD_DIMS` in `tidal/model.py`, `head_*`, `phase5_*`, and `tidal/shadow/` belong to the brain layer (growbrain). They cover turn-taking, speaking decisions, and multimodal perception. They stay in this repository for now, pending handoff or removal. This change keeps the files. Writeups remain `reports/phase1.md` through `reports/phase5.md` and `shadow/README.md`. Shadow mode loads `m3_ablate_m2`, logs, and does not call an external API. The 2026-10-08 bake-off recorded head-sum real validation 3.9061 for that ablation and 3.9276 for `mamba3_siso`. Continuum `heads` there still reads a raw-cosine soft top-4 at temperature 0.05. The memory inject is the cue trace above.

## Tests

```bash
python -m venv .venv && . .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
pytest -q
```

The tests use synthetic data. `pip install tidal[llm]` is the extra for `LLMMemory`.

## Privacy

The repository has no real chat, no weights trained on real chat, and no redistributed public datasets. IDs are salted HMACs. Secrets stay in `.secrets/` and `config/local.json`, both gitignored.

## License

[MIT](LICENSE)
