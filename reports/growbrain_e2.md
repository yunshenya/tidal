# Growbrain E2 results

Two finished curriculum runs. Budget 100000 steps, eval every 1000 steps, CUDA on a CompShare RTX 4090. The scratch_final report records `gpu_name` `NVIDIA GeForce RTX 4090`, torch `2.11.0+cu128`, CUDA `12.8`. This page is that training record. Stage0 memory scores stay in `reports/growbrain_stage0_fit.json` and `reports/growbrain_stage0_held.json`.

## Files

Committed summaries, copied from the run artifacts:

- `reports/growbrain_e2_multiseed_12.csv`
- `reports/growbrain_e2_multiseed_12.json`
- `reports/growbrain_e2_scratch_final_012.csv`
- `reports/growbrain_e2_scratch_final_012.json` (`"reconstructed": true`)
- `reports/growbrain_e2_scratch_final_012.sha256` (box-archive checksums)

External archives, on the bot box, not in git:

- `/workspace/scratch/e2_multiseed_12/`
- `/workspace/scratch/e2_scratch_final_012/`

`curriculum.pt` checkpoints stay in those archives.

## Multi-seed, seeds 1 and 2

Arms `coupled`, `decoupled`, `random_order`, `growth_only`, `graduation_only`, each at seeds 1 and 2. The matrix ran serially after a parallel launch ran out of memory.

Growth moves 2819058 parameters to 18406322. `graduation_only` never grows, so both counts stay 2819058. `growth_only` applies growth and stays `final_stage` 0. The other four arms end at `final_stage` 3.

`graduation_only` seed 1 and seed 2 are verified reconstructions. The combined `e2_multiseed_12.json` was never written, so those two rows were rebuilt from `graduation_only/seed1/evals.jsonl` and `graduation_only/seed2/evals.jsonl`. The summary JSON says so in `source`. `elapsed_s` is null on both rows.

`random_order` probe0–probe3 are shuffled pseudo-stages. The run note already says they must not be read as a true stage ladder.

`growth_only` leaves `probe2_acc` and `probe3_acc` null.

### Run counts

| arm | seed | n_graduation | n_grow_attempted | n_grow_applied | n_grow_skipped | params_before | params_after | final_stage | elapsed_s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| coupled | 1 | 3 | 3 | 3 | 0 | 2819058 | 18406322 | 3 | 23118.1 |
| coupled | 2 | 3 | 3 | 3 | 0 | 2819058 | 18406322 | 3 | 22958.0 |
| decoupled | 1 | 3 | 10 | 3 | 7 | 2819058 | 18406322 | 3 | 7043.2 |
| decoupled | 2 | 3 | 12 | 3 | 9 | 2819058 | 18406322 | 3 | 6760.1 |
| graduation_only | 1 | 3 | 0 | 0 | 0 | 2819058 | 2819058 | 3 | null |
| graduation_only | 2 | 3 | 0 | 0 | 0 | 2819058 | 2819058 | 3 | null |
| growth_only | 1 | 0 | 11 | 3 | 8 | 2819058 | 18406322 | 0 | 8711.7 |
| growth_only | 2 | 0 | 11 | 3 | 8 | 2819058 | 18406322 | 0 | 10286.8 |
| random_order | 1 | 3 | 9 | 3 | 6 | 2819058 | 18406322 | 3 | 7738.8 |
| random_order | 2 | 3 | 10 | 3 | 7 | 2819058 | 18406322 | 3 | 7849.4 |

`coupled` applied every grow attempt (3 attempted, 3 applied, 0 skipped) and took 23118.1 s and 22958.0 s. `decoupled` applied 3 grows as well, skipped 7 and 9 attempts, and took 7043.2 s and 6760.1 s. Same final width on both arms.

### Probe and handwritten accuracy

| arm | seed | probe0_acc | probe1_acc | probe2_acc | probe3_acc | hw0_acc | hw2_acc | hw3_acc | source |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| coupled | 1 | 0.633141328011611 | 0.4703068951773615 | 0.5216368289416734 | 0.4355225638845222 | 0.45251396648044695 | 0.29303547963206306 | 0.3037974683544304 | e2_ms_coupled_s1.json |
| coupled | 2 | 0.6398993105950653 | 0.479473893981666 | 0.5207305134916094 | 0.4332172210281595 | 0.4720670391061452 | 0.27989487516425754 | 0.27250351617440227 | e2_ms_coupled_s2.json |
| decoupled | 1 | 0.6309869375907112 | 0.47668393782383417 | 0.521811120374378 | 0.4336541049535616 | 0.4581005586592179 | 0.28843626806833117 | 0.2879746835443038 | e2_ms_decoupled_s1.json |
| decoupled | 2 | 0.6242969883889695 | 0.47150259067357514 | 0.5278183650882641 | 0.4327959401000932 | 0.441340782122905 | 0.2792378449408673 | 0.2858649789029536 | e2_ms_decoupled_s2.json |
| graduation_only | 1 | 0.5837633981738786 | 0.4575528098844161 | 0.553586626929058 | 0.4227885812230906 | 0.4720670391061452 | 0.2463863337713535 | 0.25584024114544085 | graduation_only/seed1/evals.jsonl (reconstructed) |
| graduation_only | 2 | 0.5623594018790525 | 0.46552411319250697 | 0.5521181049430623 | 0.42314771359193587 | 0.43575418994413406 | 0.26346911957950064 | 0.26375282592313487 | graduation_only/seed2/evals.jsonl (reconstructed) |
| growth_only | 1 | 0.6803791727140783 | 0.4830609804703069 | null | null | 0.46089385474860334 | 0.30749014454664914 | 0.2893811533052039 | e2_ms_growth_only_s1.json |
| growth_only | 2 | 0.6754354136429608 | 0.4906337186129932 | null | null | 0.49162011173184356 | 0.2943495400788436 | 0.2883263009845288 | e2_ms_growth_only_s2.json |
| random_order | 1 | 0.5460860315567231 | 0.5521018940143866 | 0.5404972941026684 | 0.5446413076314927 | 0.4553072625698324 | 0.3068331143232589 | 0.30520393811533053 | e2_ms_random_order_s1.json |
| random_order | 2 | 0.536184091973907 | 0.5411234889784309 | 0.5408373774798961 | 0.5420818889963135 | 0.4748603351955307 | 0.3114323258869908 | 0.30133614627285515 | e2_ms_random_order_s2.json |

## scratch_final, seeds 0, 1, and 2

Full width from scratch, 18406322 parameters, about 18.4M. The seed-2 report config and every result `cfg_after` are `d_model` 384, `n_layers` 10, `n_heads` 6, `d_ff` 1024, `max_seq_len` 512. Training is pooled shuffled stages 0–3, with no growth and no graduation. `params_before` and `params_after` are both 18406322. `final_stage` stays 0. `n_graduation` is 0 and `n_grow_applied` is 0. Each seed has 100 evals and the last eval step is 100000.

Probe columns here are the original held-out stage probes (`probe_pseudo_stages` false). The result text says scratch_final trains on the pooled shuffled mix and scores those original probes. The same text repeats the random_order warning: those probes are shuffled pseudo-stages and must not be read as true stage ladders.

Seed 0 and seed 1 ran uninterrupted. Their rows are the verbatim summary lines in `logs/e2_scratch_final_012.log`.

Seed 2 stopped in a CompShare downtime near step 34000 and resumed from `curriculum.pt` (`resumed_from_step` 34000, `remaining_steps` 66000, checkpoint `reports/e2_scratch_final_012/events/scratch_final/seed2/curriculum.pt` inside the run tree). Its row is `reports/e2_scratch_final_s2.json` `results[0]`. Summary `elapsed_s` 10251.205 is the resume segment only. The seed-2 report meta records a separate `elapsed_s` of 10320.737. The summary table uses 10251.205.

The merged JSON is reconstructed, on the box, 2026-11-07. The GPU never wrote `e2_scratch_final_012.json`: seed 2 had crashed near 34k and was resumed on its own. Seed 0 and seed 1 were copied unmodified from the log summary lines. Seed 2 was copied unmodified from its report. The committed file sets `"reconstructed": true`.

The seed-2 report meta calls the resume an E2 pilot under an explicit `--steps` override, and says it is not a Chinchilla-complete stage-3 run. Stage 2/3 reads on that report are a prefix cap (`max_train` 200000, `max_probe` 4096, `full_scan` false). Handwritten banks on that report are extra probes (`extra_handwrittenN_acc`). With `graduation.handwritten_gate` defaulting false, they do not advance a stage.

Graduation and growth event logs for seeds 0, 1, and 2 hash to the empty-file sha256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.

| seed | last_step | n_evals | final_stage | params_after | n_graduation | n_grow_applied | elapsed_s | probe0_acc | probe1_acc | probe2_acc | probe3_acc | extra_handwritten0_acc | extra_handwritten2_acc | extra_handwritten3_acc |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 100000 | 100 | 0 | 18406322 | 0 | 0 | 14040.214 | 0.6167452830188679 | 0.43842168194499803 | 0.5810643977795271 | 0.43579951708723247 | 0.48044692737430167 | 0.32391590013140603 | 0.3111814345991561 |
| 1 | 100000 | 100 | 0 | 18406322 | 0 | 0 | 14053.835 | 0.626406023222061 | 0.4348345954563571 | 0.5811941480683184 | 0.43487113874575306 | 0.5 | 0.3455978975032852 | 0.310126582278481 |
| 2 | 100000 | 100 | 0 | 18406322 | 0 | 0 | 10251.205 | 0.611121190130624 | 0.42088481466719807 | 0.5825352238144067 | 0.43437964432967574 | 0.4692737430167598 | 0.32588699080157685 | 0.31680731364275666 |

Sources, copied from the summary table: seed 0 and seed 1 are `logs/e2_scratch_final_012.log (verbatim summary line)`. Seed 2 is `reports/e2_scratch_final_s2.json results[0] (resumed from step 34000)`. Probe losses and the rest of each result object are in `reports/growbrain_e2_scratch_final_012.json`.
