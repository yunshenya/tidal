# tidal 潮汐

tidal is only the subconscious and below-subconscious memory layer. It stores specific text facts and adds a residual into a frozen causal language model. Turn-taking, speaking decisions, and multimodal perception belong to the brain layer (growbrain). tidal only supplies memory signals.

The design prefers a human-like memory, including partial and fuzzy effects, over a store that is as useful as possible. Each mechanism below is tied to a neuroscience paper. The fit numbers are not the paper's numbers.

Memory vectors are text-only and modality-agnostic. A key and a value are residuals of text from SmolLM2-135M. They do not read image, audio, video, or other modality features.

The longer account is [docs/architecture.md](docs/architecture.md).

Any pull request that changes behavior, mechanisms, constants, or the interface must update `README.md` and `docs/architecture.md` in that same pull request (what changed, the citations, and the held numbers). A pull request with stale docs is not mergeable.

## Implemented memory

`tidal/episodic.py` `EpisodicStore` is the slot list. One fact is one slot. The same identity overwrites that slot. Capacity is 256, then FIFO. Key cosine does not merge slots. A different identity is merged only when the group matches, the relation is the same non-empty string, and the surface token Jaccard is at least `REPEAT_JACCARD` 0.85.

`tidal/llm_memory.py` `LLMMemory` writes those slots from a frozen `HuggingFaceTB/SmolLM2-135M`. The inject layer is `n_layers // 2`, layer 15, hidden 576. That default loads `tidal/llm_inject_smollm2_135m.pt`. The fact text is not copied into the prompt. With one slot the residual is `s * u * gain * (scale(v) + MLP(scale(v)))`, with `scale` fixed at 1, `gain = 1 + 2 tanh(w · normalize(k))`, and `u = 1 + tanh(valence)`. With more slots the same term is summed over the top `MIX_K = 4` cosine neighbors, weighted by `exp((log(strength) + cosine) / MIX_TEMP)` at `MIX_TEMP = 0.05`. `_broadcast` adds that one vector at every position of the layer. Spelling and retrieval-induced forgetting follow the cosine argmax. The value is an answer-span pool. A phone or a birthday is concatenated slices of that span, and when the prompt is completing that relation the emitted string is the stored answer.

Shipped constants, fit on a fit bank and not retuned on held-out: polarity span 2, valence span 1 and step 0.25, clock span 0.25 with tick `pi/16`, surprise window 32 at quantile 0, replay every 32 accepted writes with strength `0.5 ** age` and one refresh, RIF span 0.05 over at most 3 same-relation competitors.

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

## Planned interface to growbrain

**Planned.** This tree has no growbrain backend and no per-token cue export. The implemented inject is the broadcast mixture above.

The planned backend is growbrain behind an encode hook, an NLL hook, and an inject hook. Injection is cue-triggered and per token. Each token carries:

- `memory_id`
- `vec`
- `activation = a_trig * fan_factor * decay`
- `strength*confidence`
- `valence`, exported as `u - 1`
- `circadian`
- `triggered`
- `interface_version`

A `source` field, with values `cue`, `spontaneous`, or `none`, and a per-tick output are later. They are not in that first signal list. tidal only supplies these signals. Turn-taking, speaking decisions, and multimodal perception belong to the brain layer (growbrain).

Low-level cross-sensory fusion is planned after text stage-1. The account is in [docs/architecture.md](docs/architecture.md). It has no code and adds no interface fields.

## Not part of tidal: belongs to the brain layer (growbrain); kept in repo for now, pending handoff or removal

`duplex*` (`tidal/duplex.py`, `tidal/duplex_sim.py`), `turn_*`, `audio_*`, `tidal/modalities/`, the six `HEAD_DIMS` in `tidal/model.py`, `head_*`, `phase5_*`, and `tidal/shadow/` belong to the brain layer (growbrain). They cover turn-taking, speaking decisions, and multimodal perception. They stay in this repository for now, pending handoff or removal. This change keeps the files. Writeups remain `reports/phase1.md` through `reports/phase5.md` and `shadow/README.md`. Shadow mode loads `m3_ablate_m2`, logs, and does not call an external API. The 2026-10-08 bake-off recorded head-sum real validation 3.9061 for that ablation and 3.9276 for `mamba3_siso`. Continuum `heads` there still reads a raw-cosine soft top-4 at temperature 0.05. The memory inject uses the mixture above.

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
