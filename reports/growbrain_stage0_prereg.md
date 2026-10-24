# Growbrain stage0 backend

Protocol `growbrain-stage0-v1`. This is the behavior change. The live `interface_version` is 6.

## Locked design

1. Thin wrapper. tidal keeps `encode_hidden(text) -> (hidden, encoding, logits)`, `encode_cue(text)`, `encode_key_hidden(text)`, `answer_nll(logits, token_ids, answer_mask)`, and `inject_residual(output, residual, *, broadcast=False, batched=True)`. Winner, fan, decay, and residual selection stay in tidal. `GrowbrainBackend.encode_hidden(text)` is the growbrain text forward. `TidalAdapter.encode_hidden(vec)` is a different line: it projects an exported packet from `d_model` to `mem_dim`. tidal does not call that projection.

2. growbrain is optional. `import growbrain` is tried when `LLMMemory()` runs. When the import works, the default backend is `GrowbrainBackend`. When it does not, that constructor raises. `SmolLM2Backend` stays constructible by passing a model and tokenizer, or `model_name="HuggingFaceTB/SmolLM2-135M"`. There is no environment flag. `pip install tidal[growbrain]` uses `git+https://origin.cursor.com/shen-yun/growbrain.git`. A local dual-checkout is `pip install -e ../growbrain`.

3. True cut. The vector space changed, so the old `matches_pr57` floors are not gates. This is not a SmolLM2-weight run under a new name.

## Path B

Migrate the packet and the episodic store onto growbrain `d_model`. Stage0 `e1_base_rebalance` has no `expand_d`. The pinned ModelConfig is `d_model` 192, `mem_dim` 64, `mem_key_dim` 32, `inject_layer` 2, `n_layers` 4, `head_dim` 48. `d_model` is the text-hidden width and the packet `vec` width. `mem_dim` is the injector slot only. The episodic store and the key center are not 64. A later checkpoint's `cfg.d_model` wins over 192.

Retire `TIDAL_DIM` 576 as the live packet width. `GrowbrainBackend` does not load `tidal/llm_key_center.npz` or `tidal/llm_inject_smollm2_135m.pt` and does not map 192 back to 576. Those two files stay on the explicit SmolLM2 constructor so that path can still be built. They are not a retrain. The 192-d key center and inject MLP are not in this pull request. Origin `shen-yun/growbrain` does not store the weights: `checkpoints/` is gitignored. The file is `/workspace/checkpoints/e1_base_rebalance.pt` on the growbrain VM, with a vocab sidecar, vocab 5135. Training, fit, and held stay paused until that file is attached. A zero MLP is not a trained substitute. `scripts/train_growbrain_inject.py` exits before writing a file.

`interface_version` was 5. It is 6. The field set is unchanged. Version 6 records the width. growbrain should set `TIDAL_DIM` to 192 and point `TidalAdapter.proj` at 192 → `mem_dim` 64.

## What was readable

No `e1_base_rebalance` checkpoint and no growbrain tree were on this machine. Origin `shen-yun/growbrain` has no `.pt` in git. The real file is `/workspace/checkpoints/e1_base_rebalance.pt` on the growbrain VM, plus a vocab sidecar, vocab 5135. It is being exported as a cloud-agent artifact or a Hugging Face file. This protocol does not score until that file is attached. No cfg file was found, so these pins are the stage0 numbers above, not a cfg read from disk. `model(idx)` is the text call. A return value that does not expose a logits tensor raises. The loader does not guess a checkpoint constructor.

## Gates

Fit ids `growbrain-fit-000` through `growbrain-fit-007` are disjoint from held ids `growbrain-held-000` through `growbrain-held-049`. Neither bank was scored. The held bank was not scored.

Pre-registered checks, for the run that can load the checkpoint:

- Default `LLMMemory()` is `GrowbrainBackend` with store width `cfg.d_model`, which is 192 on this stage0 pin, and not `mem_dim` 64.
- A passed-in model is `SmolLM2Backend`. `LLMMemory()` without the package raises `ImportError` and does not construct SmolLM2.
- Packet `vec` last dimension equals that store width. `interface_version` is 6.
- Inject layer is `cfg.inject_layer` (2 on the pin). `n_layers` is 4, `head_dim` is 48, and `mem_key_dim` is 32 on the pin. `mem_key_dim` is not the episodic key width.
- Theta 0.32, half-life 30, lambda 0.3, fan S 1.45, and RIF span 0.05 are unchanged.
- Implicit influence is Cohen's `d_z` on tone, choice, and preference, n≥50 on held, Wilson 95%. Explicit phone and birthday strings are reported and are not part of `d_z`.
- The old `matches_pr57` floors are not gates.

No fit number and no held number are in this file. Scoring them on SmolLM2 would not be this protocol.

Constants were not changed. The held bank was not scored.
