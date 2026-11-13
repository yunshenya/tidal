# Growbrain E2 results

Two finished curriculum runs. Budget 100000 steps, eval every 1000 steps, CUDA on a CompShare RTX 4090. The scratch_final report records `gpu_name` `NVIDIA GeForce RTX 4090`, torch `2.11.0+cu128`, CUDA `12.8`. This page keeps that training record. The true-stage re-evaluation re-scores the same checkpoints eval-only. The shared-span section scores every arm on the same 128-token prefix, CJK tokens only. Task probes are a separate bank, in 「任务题正式对照（task probes）」. The later `staged_final` control is in 「staged_final 对照臂」. It is not part of the two curriculum archives or the 13-checkpoint rescore. An offline estimate of stage-0 mastery graduation, from existing logs only, is in 「若改用 stage-0 mastery 毕业（离线估计）」. Logged graduations used the loss plateau. An offline check of whether the existing data and evaluation can distinguish capability, from those logs and the task banks, is in 「现有数据与评测能否区分能力（离线核对）」. A CPU eval-only rescore of the existing end `curriculum.pt` files, with the by-skill, headline, and first-CJK scorer from growbrain PR [#33](https://github.com/yunshenya/growbrain/pull/33), is in 「按技能、headline 与 first-CJK 的 CPU 只评测重打分」. It is not a new training run. A later CPU eval-only rescore of the same 15 end checkpoints on a local stage-0 train-format diagnostic bank (not committed to growbrain, not a replacement of the official 「问：/答：」 transfer bank) is in 「本地 stage-0 训练格式诊断题库的 CPU 只评测重打分」. It is not a new training run. Stage0 memory scores stay in `reports/growbrain_stage0_fit.json` and `reports/growbrain_stage0_held.json`.

## Files

Committed training summaries, copied from the run artifacts:

- `reports/growbrain_e2_multiseed_12.csv`
- `reports/growbrain_e2_multiseed_12.json`
- `reports/growbrain_e2_scratch_final_012.csv`
- `reports/growbrain_e2_scratch_final_012.json` (`"reconstructed": true`)
- `reports/growbrain_e2_scratch_final_012.sha256` (box-archive checksums)

True-stage re-evaluation, eval-only:

- `reports/growbrain_e2_true_stage_reeval.csv`
- `reports/growbrain_e2_true_stage_reeval.json` (`probe_pseudo_stages` false on every row)

Shared-span CJK rescore at L=128, eval-only:

- `reports/growbrain_e2_cjk_l128.csv`
- `reports/growbrain_e2_cjk_l128.json`

Task probes, eval-only, for the 13 checkpoints rescored on 2026-11-10:

- `reports/growbrain_e2_taskprobe_arms.csv` (the same file also holds the later `staged_final` seed means)
- `reports/growbrain_e2_taskprobe_runs.csv` (those 13 checkpoints only)

`staged_final` training-log stage-0 task probes, one row per eval:

- `reports/growbrain_e2_staged_final_trajectory.csv`

Offline crossing table for stage-0 graduation bars, read off existing logs on 2026-11-11 (CST). Not a new training run:

- `reports/growbrain_e2_mastery_graduation_crossings.csv`

Offline check of whether existing data and evaluation can distinguish capability, read off existing logs and task banks on 2026-11-12 (Asia/Shanghai). Not a new training run. Stage-0 per-skill exact counts for the 13 end-run rescores:

- `reports/growbrain_e2_taskprobe_stage0_by_skill.csv`

CPU eval-only rescore of the 15 end checkpoints on 2026-11-13 (Asia/Shanghai), after growbrain PR [#33](https://github.com/yunshenya/growbrain/pull/33). Not a new training run. Full-bank exact and CJK match the original end-run scores. Headline, first-CJK, and by-skill splits are new:

- `reports/growbrain_e2_rescore_stage_summary.csv`
- `reports/growbrain_e2_rescore_stage0_by_skill.csv`
- `reports/growbrain_e2_rescore_by_skill.csv` (adds `staged_final` stage 2 / 3 skill rows to the stage-0 table)

Local stage-0 train-format diagnostic bank, CPU eval-only, same 15 end checkpoints, 2026-11-13 (Asia/Shanghai). Not a new training run. The bank itself is not in git. Score tables only:

- `reports/growbrain_e2_diag_bank_v1_stage_summary.csv`
- `reports/growbrain_e2_diag_bank_v1_by_skill.csv`
- `reports/growbrain_e2_diag_bank_v2_stage_summary.csv`
- `reports/growbrain_e2_diag_bank_v2_by_skill.csv`

External archives, on the bot box, not in git:

- `/workspace/scratch/e2_multiseed_12/`
- `/workspace/scratch/e2_scratch_final_012/`

`curriculum.pt` checkpoints stay in those archives.

## How to compare arms

Cross-arm comparison uses the held-out stage probes one column at a time: `probe0_acc` through `probe3_acc` on a row whose `probe_pseudo_stages` is false. The equal-weight mean of those four columns is not a capability measure. Handwritten scores stay the banks the runs already log (`hw0_acc`, `hw2_acc`, `hw3_acc` on the multi-seed summary; `extra_handwritten0_acc`, `extra_handwritten2_acc`, `extra_handwritten3_acc` on scratch_final). No stage-1 handwritten column was logged.

`tidal/growbrain_e2_probes.py` applies that split to a summary row. `python scripts/growbrain_e2_compare.py` prints it for the training files and then prints the true-stage re-evaluation. A null probe cell is unscored. It is not a zero.

In this training archive:

- `coupled`, `decoupled`, and `graduation_only` log probe0–3 as the stage ladder (the flag is absent; the reader treats them as stage probes).
- `scratch_final` sets `probe_pseudo_stages` false. Those probe0–3 are the original held-out stage probes.
- `growth_only` logs probe0 and probe1. `probe2_acc` and `probe3_acc` are null because `final_stage` stayed 0. Stages 2 and 3 are unscored in this archive.
- `random_order` probe0–3 are shuffled pseudo-stages. The flag is absent on these rows; the reader still treats that arm as pseudo-stages. They are in the pseudo-stage table below, and they are not comparable to the other arms' probe0–3. This training archive has no second, true stage-probe columns for `random_order`. Those columns are the true-stage re-evaluation below.

Dual reporting, when a run logs both: the comparable columns are the object with `probe_pseudo_stages` false (`probe0_acc`–`probe3_acc`, plus the handwritten fields already on that object). The shuffled pseudo-stage scores are a separate object with `probe_pseudo_stages` true. That pseudo-stage object is not the stage ladder, including when it is the only probe0–3 set on the arm. The true-stage re-evaluation is that later pass. Every row there sets `probe_pseudo_stages` false, including `random_order`, and that is the stage ladder. `growth_only` probe2 and probe3, null in this archive, are scored there. `scratch_final` probe0–3 in that pass match the training-time scores in the scratch_final table.

The trainer that writes the training rows is growbrain `scripts/e2_run.py` (`configs/e2.toml`). Those files are the record of the runs above. They do not rescore checkpoints. The re-evaluation file is a separate eval-only pass, described below.

Task probes are a third reading, in 「任务题正式对照（task probes）」. The bank is growbrain `data/probes/taskmatch/` at `e112519`. Primary numbers are stage-0 exact-match and CJK token accuracy on stages 0, 2, and 3. Stage-2 and stage-3 exact-match on that bank are a difficulty floor and stay out of the arm comparison. Those scores are not `probe0_acc`–`probe3_acc` and not the shared-span CJK table.

## Multi-seed, seeds 1 and 2

Arms `coupled`, `decoupled`, `random_order`, `growth_only`, `graduation_only`, each at seeds 1 and 2. The matrix ran serially after a parallel launch ran out of memory.

Growth moves 2819058 parameters to 18406322. `graduation_only` never grows, so both counts stay 2819058. `growth_only` applies growth and stays `final_stage` 0. The other four arms end at `final_stage` 3.

`graduation_only` seed 1 and seed 2 are verified reconstructions. The combined `e2_multiseed_12.json` was never written, so those two rows were rebuilt from `graduation_only/seed1/evals.jsonl` and `graduation_only/seed2/evals.jsonl`. The summary JSON says so in `source`. `elapsed_s` is null on both rows.

`random_order` probe0–probe3 are shuffled pseudo-stages. They are not in the stage-probe table. The run note already says they must not be read as a true stage ladder. The true-stage re-evaluation scores the same checkpoints on the stage ladder. Those shuffled numbers must not be compared with that table.

`growth_only` leaves `probe2_acc` and `probe3_acc` null (unscored). The re-evaluation scores probe2 and probe3 for that arm.

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

### Stage probes and handwritten accuracy

Stage ladder for the arms whose probe0–3 are held-out stage probes. `growth_only` probe2 and probe3 are unscored (null). `random_order` is not in this table.

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

### Pseudo-stage probes

`random_order` only. These four columns are shuffled pseudo-stages, not a stage ladder, and not the comparison with the table above. Handwritten `hw0_acc` / `hw2_acc` / `hw3_acc` on these rows are the same extra banks as the other arms.

| arm | seed | probe0_acc | probe1_acc | probe2_acc | probe3_acc | hw0_acc | hw2_acc | hw3_acc | source |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| random_order | 1 | 0.5460860315567231 | 0.5521018940143866 | 0.5404972941026684 | 0.5446413076314927 | 0.4553072625698324 | 0.3068331143232589 | 0.30520393811533053 | e2_ms_random_order_s1.json |
| random_order | 2 | 0.536184091973907 | 0.5411234889784309 | 0.5408373774798961 | 0.5420818889963135 | 0.4748603351955307 | 0.3114323258869908 | 0.30133614627285515 | e2_ms_random_order_s2.json |

## scratch_final, seeds 0, 1, and 2

Full width from scratch, 18406322 parameters, about 18.4M. The seed-2 report config and every result `cfg_after` are `d_model` 384, `n_layers` 10, `n_heads` 6, `d_ff` 1024, `max_seq_len` 512. Training is pooled shuffled stages 0–3, with no growth and no graduation. `params_before` and `params_after` are both 18406322. `final_stage` stays 0. `n_graduation` is 0 and `n_grow_applied` is 0. Each seed has 100 evals and the last eval step is 100000.

Probe columns here are the original held-out stage probes (`probe_pseudo_stages` false). The result text says scratch_final trains on the pooled shuffled mix and scores those original probes. The same text repeats the random_order warning: those probes are shuffled pseudo-stages and are the separate table above, not this one.

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

## True-stage re-evaluation

All 13 checkpoints were re-scored eval-only. growbrain `main` at `48ad257`, `scripts/e2_eval_heldout.py`. No training and no optimizer steps. `probe_pseudo_stages` false. CPU. The data seed is the checkpoint seed. Arms `coupled`, `decoupled`, `random_order`, `growth_only`, and `graduation_only` are seeds 1 and 2. `scratch_final` is seeds 0, 1, and 2.

The training archive above is unchanged. `random_order` probe0–3 there stay shuffled pseudo-stages. `growth_only` probe2 and probe3 there stay null. This pass is the stage ladder for every arm, including those two. The shuffled `random_order` columns must not be compared with the columns in this section.

`reports/growbrain_e2_true_stage_reeval.json` sets `probe_pseudo_stages` false on every row so the reader does not apply the training-archive `random_order` default. `python scripts/growbrain_e2_compare.py` prints this table after the training archive.

### Per seed

| arm | seed | probe0_acc | probe1_acc | probe2_acc | probe3_acc | data_seed | device |
|---|---:|---:|---:|---:|---:|---:|---|
| coupled | 1 | 0.633141328011611 | 0.4703068951773615 | 0.5216368289416734 | 0.4355225638845222 | 1 | cpu |
| coupled | 2 | 0.6398993105950653 | 0.479473893981666 | 0.5207305134916094 | 0.4332172210281595 | 2 | cpu |
| decoupled | 1 | 0.6309869375907112 | 0.47668393782383417 | 0.5218101520886408 | 0.4336541049535616 | 1 | cpu |
| decoupled | 2 | 0.6242969883889695 | 0.47150259067357514 | 0.5278183650882641 | 0.4327959401000932 | 2 | cpu |
| graduation_only | 1 | 0.5837633981738786 | 0.4575528098844161 | 0.553586626929058 | 0.4227885812230906 | 1 | cpu |
| graduation_only | 2 | 0.5623594018790525 | 0.46552411319250697 | 0.5521181049430623 | 0.42314771359193587 | 2 | cpu |
| growth_only | 1 | 0.6803791727140783 | 0.4830609804703069 | 0.5497171153218631 | 0.41637378540417613 | 1 | cpu |
| growth_only | 2 | 0.6754354136429608 | 0.4906337186129932 | 0.5512024656428013 | 0.4186830290098728 | 2 | cpu |
| random_order | 1 | 0.6183554063860668 | 0.41689916301315266 | 0.568346932906513 | 0.42822036113137335 | 1 | cpu |
| random_order | 2 | 0.6163370827285921 | 0.42885611797528894 | 0.56643747343266 | 0.42791610268332547 | 2 | cpu |
| scratch_final | 0 | 0.6167452830188679 | 0.43842168194499803 | 0.5810643977795271 | 0.43579951708723247 | 0 | cpu |
| scratch_final | 1 | 0.626406023222061 | 0.4348345954563571 | 0.5811941480683184 | 0.43487113874575306 | 1 | cpu |
| scratch_final | 2 | 0.611121190130624 | 0.42088481466719807 | 0.5825352238144067 | 0.43437964432967574 | 2 | cpu |

### Per probe

Seed means of each probe. Rows follow the per-seed arm order. They are not a ranking. `equal_weight_mean` is the unweighted mean of probe0–3. It is not a capability measure.

| arm | n | probe0 | probe1 | probe2 | probe3 | equal_weight_mean |
|---|---:|---:|---:|---:|---:|---:|
| coupled | 2 | 0.6365203193033382 | 0.47489039457951376 | 0.5211836712166413 | 0.43436989245634083 | 0.5167410693889585 |
| decoupled | 2 | 0.6276419629898404 | 0.4740932642487047 | 0.5248142585884524 | 0.4332250225268274 | 0.5149436270884562 |
| graduation_only | 2 | 0.5730614000264656 | 0.46153846153846156 | 0.5528523659360602 | 0.4229681474075132 | 0.5026050937271251 |
| growth_only | 2 | 0.6779072931785195 | 0.48684734954165004 | 0.5504597904823322 | 0.41752840720702444 | 0.5331857101023816 |
| random_order | 2 | 0.6173462445573294 | 0.42287764049422083 | 0.5673922031695865 | 0.42806823190734944 | 0.5089210800321216 |
| scratch_final | 3 | 0.618090832123851 | 0.43138036402285107 | 0.5815979232207508 | 0.43501676672088707 | 0.5165214715220849 |

Rounded to three decimals, probe0/1/2/3: `coupled` .637/.475/.521/.434 (n=2); `decoupled` .628/.474/.525/.433 (n=2); `graduation_only` .573/.462/.553/.423 (n=2); `growth_only` .678/.487/.550/.418 (n=2); `random_order` .617/.423/.567/.428 (n=2); `scratch_final` .618/.431/.582/.435 (n=3). `graduation_only` is context 128. The other arms are context 512. Its probe2 cell is not comparable to theirs.

`scratch_final` probe0–3 in this pass match `reports/growbrain_e2_scratch_final_012.json` on seeds 0, 1, and 2. On the multi-seed training summary, the logged stage-probe cells match this pass except `decoupled` seed 1 `probe2_acc`: the training row is 0.521811120374378 and this pass is 0.5218101520886408. `growth_only` probe2 and probe3 were null there. `random_order` there is the pseudo-stage table, not this one.

### Shared-span CJK at L=128

Eval-only rescore of the same 13 checkpoints. Every sample is truncated to 129 ids (`max_seq_len` 128), including models whose native context is 512. CJK-only accuracy counts a gold token in U+4E00–U+9FFF or U+3400–U+4DBF. Markup, punctuation, whitespace, and ASCII letters and digits are excluded. The item-bootstrap SE is 2000 resamples of items, token-weighted. Full probe2 is 4096 items. The same-stage trigram row is the audit's L=128 trigram. It was not recomputed in this pass. Tokens 129–512 were not scored separately.

Rows follow the per-seed arm order. They are not a ranking.

| arm | n | probe0 | probe1 | probe2 | probe3 | hw0 | hw2 | hw3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| coupled | 2 | 0.3834266007287871 | 0.36933614330874603 | 0.39773418905344626 | 0.27343459138149717 | 0.37884615384615383 | 0.23541009463722395 | 0.22616357885223679 |
| decoupled | 2 | 0.37529281624154087 | 0.3717070600632244 | 0.40214061981978416 | 0.2684905315100995 | 0.3942307692307692 | 0.2338328075709779 | 0.22706732941708088 |
| graduation_only | 2 | 0.33995965642894327 | 0.3601159114857745 | 0.34717033415639487 | 0.2571100439822557 | 0.3730769230769231 | 0.19834384858044163 | 0.2080885675553547 |
| growth_only | 2 | 0.45506897449245187 | 0.3838250790305585 | 0.426001157894216 | 0.2507714247852947 | 0.4115384615384615 | 0.2385646687697161 | 0.22300045187528242 |
| random_order | 2 | 0.35378709005726183 | 0.3240252897787145 | 0.44501239540602994 | 0.2637136925533056 | 0.39615384615384613 | 0.2602523659305994 | 0.2471757794848622 |
| scratch_final | 3 | 0.37539042165538783 | 0.3364945556726379 | 0.458347971953229 | 0.27313590748371436 | 0.42948717948717946 | 0.2807570977917981 | 0.2571170356981473 |
| same-stage trigram |  | 0.336 | 0.322 | 0.350 | 0.244 | 0.227 | 0.176 | 0.139 |

Item-bootstrap SE of that CJK accuracy. probe2 and probe3 are about 0.002. probe0 and probe1 are about 0.008–0.013. handwritten0, 24 items, is about 0.03–0.04.

| arm | probe0 | probe1 | probe2 | probe3 | hw0 | hw2 | hw3 |
|---|---:|---:|---:|---:|---:|---:|---:|
| coupled | 0.008048350730401131 | 0.012110630419864427 | 0.0020874909411957047 | 0.0012785331764531603 | 0.0389148077626322 | 0.012898194625154145 | 0.011518567576897336 |
| decoupled | 0.008397330716418485 | 0.012052767673558592 | 0.0020894790468516753 | 0.0012568270921467341 | 0.035005879931695114 | 0.01223498783095854 | 0.011277893348947857 |
| graduation_only | 0.007188066408231791 | 0.012579416179003121 | 0.0019723599444345795 | 0.0012542366383985117 | 0.036965210992174966 | 0.012937742045649746 | 0.011795265803321311 |
| growth_only | 0.010138681401330233 | 0.012393255634229041 | 0.0023775143933881474 | 0.0012485590673253074 | 0.03452536525332455 | 0.015309664674087214 | 0.010199660931753986 |
| random_order | 0.008008892917588155 | 0.00920468165387842 | 0.002399636111797082 | 0.0012744655776448371 | 0.039415047435721745 | 0.012897189909820993 | 0.011621921862024709 |
| scratch_final | 0.007619584313645635 | 0.009088681758996096 | 0.0024003965033931114 | 0.0012680492274362214 | 0.042692016771938124 | 0.014486248714431304 | 0.011646000722695516 |

### Reading

`graduation_only`'s official probe2 advantage reverses on the shared tokens. CJK probe2 is 0.34717033415639487, at the same-stage trigram floor 0.350. `coupled` is 0.39773418905344626 and `scratch_final` is 0.458347971953229. The item-bootstrap intervals do not overlap: graduation about [0.343, 0.351], coupled about [0.392, 0.403], scratch about [0.452, 0.465].

`scratch_final` beats `coupled` by 0.06061378289978275 CJK accuracy on probe2, about 6 points. Curriculum plus growth is worse than direct training on stage-2 text. It is not a tie.

`coupled` and `decoupled` are about the same. probe2 CJK is 0.39773418905344626 and 0.40214061981978416.

`growth_only` against `scratch_final` flips sign by stage. probe0 CJK is 0.45506897449245187 against 0.37539042165538783, about 8 points higher for `growth_only`. probe2 CJK is 0.426001157894216 against 0.458347971953229, about 3.2 points higher for `scratch_final`. That matches staying on stage 0. On both seeds the training summary records `n_grow_applied` 3, `n_graduation` 0, `params_before` 2819058, `params_after` 18406322 (about 18.4M), and `final_stage` 0. It grew 3 times, to about 18.4M, and those 100000 steps stayed on stage 0 data. It is not evidence that growth helps.

probe3 CJK and handwritten0 are too tight or too small to call. The 18.4M probe3 CJK scores sit between 0.2507714247852947 and 0.27343459138149717. handwritten0 is 24 items, and its item-bootstrap SE is about 0.03–0.04.

`scratch_final` true-stage scores are identical to its training-time scores, so the earlier scratch comparison stands on those official columns. `random_order`'s earlier shuffled pseudo-stage scores must not be compared to these columns. Task-probe scores of these checkpoints are a separate reading. See 「任务题正式对照（task probes）」.

Only 2 or 3 seeds. The runs share a step budget, not a compute budget: equal steps, not equal compute. Tokens 129–512 were not scored separately. The equal-weight mean of probe0–3 is still not a capability measure.

### Evaluation limits

Probe audit, 2026-11-10, eval-only, growbrain `main` at `48ad257`. The equal-weight mean of probe0–3 cannot distinguish capability at the 1–2 point gaps in this table.

Probe accuracy is character-level next-token top-1, including markup. probe0 is 42% markup tokens (41.7% special markup in the audit's token-class mix).

A same-stage character trigram gets close to the model scores. probe0 trigram .557 against 18M-model probe0 about .62–.68. probe3 trigram .382 against model probe3 about .42–.44.

`graduation_only` is scored at context 128 (2819058 parameters). The other arms are scored at context 512 (18406322 parameters). Short prefixes are easier: a trigram on probe2 is .493 at L=128 and .420 at L=512. `graduation_only` probe2 is not comparable to the 512-context arms.

probe0 has 87.8% character 8-gram overlap with stage-0 train.

handwritten2 and handwritten3, 24 items each, do separate the 2.8M model from the 18.4M models. Those banks are tiny, and they are not the headline.

The shared-span L=128 CJK rescore is the table above. An expanded handwritten exact-match probe set is still in progress. Tokens 129–512 were not scored separately. Task probes are the following section. They use a different bank. Their arm comparison is stage-0 exact-match and CJK token accuracy.

## 任务题正式对照（task probes）

上文的 probe0–3 与 L=128 共享窗 CJK 是 stage probe。这一节是另一套题，对同一批 13 个 checkpoint 做只评测重打分。不训练，不写 `curriculum.pt`。题库随 growbrain PR [#30](https://github.com/yunshenya/growbrain/pull/30) 合并为 `e112519`，路径 `data/probes/taskmatch/stage0.jsonl`、`stage2.jsonl`、`stage3.jsonl`，360 / 584 / 660 题。没有 stage1 题库。打分日是 2026-11-10，CPU。13 个 checkpoint 都是第 100000 步。`growth_only` 的训练日志里 probe2 / probe3 为空，这里仍对 stage2 和 stage3 打了分。`random_order` 的训练日志 probe0–3 是打乱的 pseudo-stage，这里六条臂用同一题库。

臂均值在 `reports/growbrain_e2_taskprobe_arms.csv`。逐个 checkpoint 在 `reports/growbrain_e2_taskprobe_runs.csv`。`staged_final` 的三行种子均值在同一份 arms CSV 的末尾。runs CSV 仍是这 13 个 checkpoint。

### 题型与计分

每条题是 prompt，接 `问：`，再接 `答：`。金答案是整句。计分是 teacher-forced argmax：答案的每个位置用金标上文取概率最大的 token。只统计 CJK 字，范围 U+4E00–U+9FFF。exact-match 要求这些字全部命中。CJK token acc 是命中字数除以 CJK 字数。stage0 / stage2 / stage3 的 CJK 分母在每个 checkpoint 上都是 2362 / 8013 / 10567。

共享预测窗是 `min(model.cfg.max_seq_len, 128)`。13 个 run 的截断条数都是 0。上文官方 probe2 里，`graduation_only` 的上下文是 128，其余臂是 512。这一节的窗对六条臂都是 128。

`答：` 在 E2 训练集里出现过。含 `答：` 或 `答:` 的行数：stage0 18/22409，stage1 217/556，stage2 3812/200000，stage3 7/200000。每条题的 prompt 都以 `答：` 结尾。stage2 exact-match 在 13 个 run 上都是 0。stage3 exact-match 落在 0–0.8%，单次最高是 coupled seed1 的 0.0075757576。这是整句金答案的难度地板。比臂不使用这两列。

teacher-forced 的 CJK acc 计入答案内部的字词搭配。这一节没有自由生成的分数。stage0 上 exact-match 更严。stage2 和 stage3 上能分开臂的是 CJK token acc。

官方 `--task-probes-only` 在 graduation_only seed1 上，与同一次前向的 exact-match、exact-match 区间、CJK token acc 一致。官方 JSON 只给 exact-match 的 bootstrap 区间。runs CSV 的 `cjk_lo` / `cjk_hi` 是同一次前向按条目重采样 `cjk_hits/cjk_total` 得到的。

stage0 的 exact 命中集中在从 prompt 抄短答案的 `where` 和 `color`。`handoff` 在每个 run 的 stage0 exact-match 上都是 0。stage3 上少见的 exact 命中几乎都落在 `conditional`。

### 训练切分过滤

用真实 E2 train jsonl 重跑 `scripts/build_task_probes.py --seed 0`，`--max-train` 200000。8-gram 近重丢弃 0 条。train-trigram 整句命中丢弃 0 条。三份 jsonl 与 `e112519` 的题库逐字节相同。

| stage | train 文档 | 8-gram 丢弃 | train-trigram 整句丢弃 | 保留 | 保留题最高 8-gram | train-trigram exact | train-trigram CJK acc |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 22409 | 0 | 0 | 360 | 0 | 0 | 0.2218458933 |
| 2 | 200000 | 0 | 0 | 584 | 0.0227272727 | 0 | 0.1114439037 |
| 3 | 200000 | 0 | 0 | 660 | 0.1692307692 | 0 | 0.114602063 |

stage0 的 train 文档是 22409，少于 `--max-train` 200000。stage3 保留题的最高 8-gram 是 0.1692307692，低于 0.5 的丢弃线。

题库上自拟合的 trigram 写在已提交的 manifest 里。它拟合的是题库本身。exact-match 是 0.0027777778（1/360）、0、0。CJK token acc 是 0.5774767146、0.6641707226、0.6064162014。比臂用的地板是表里的 train-trigram。

### 主报指标

比臂用四项：stage0 exact-match，以及 stage0、stage2、stage3 的 CJK token acc。

stage2 exact-match 在 13 个 run 上都是 0。stage3 exact-match 是 0–0.8%。这两列是难度地板，不进入臂间比较。

stage0 exact-match 高于 train-trigram 的 exact 地板 0。三阶段 CJK token acc 高于 train-trigram 地板 0.2218458933、0.1114439037、0.114602063。

### 臂表

百分数。种子均值 ± 样本标准差（除以 n−1）。n=2 的臂是 seed 1 和 seed 2。`scratch_final` 是 seed 0、1、2，n=3。行序与上文 per-probe 表一致。下表是 2026-11-10 的六条臂。`staged_final` 的三行在同一份 arms CSV 的末尾，种子是 1 和 2，读法在「staged_final 对照臂」。未舍入的比例在 `reports/growbrain_e2_taskprobe_arms.csv`。表内百分数由该文件的比例乘 100，保留两位。s2 exact 与 s3 exact 不是主指标。train-trigram 那一行不是模型臂，不在 arms CSV 里。它的 CJK 百分数保留一位，未舍入值是过滤表里的 0.2218458933、0.1114439037、0.114602063。

| 臂 | n | s0 exact | s0 CJK | s2 exact | s2 CJK | s3 exact | s3 CJK |
|---|---:|---:|---:|---:|---:|---:|---:|
| coupled | 2 | 0.97 ± 0.20 | 48.18 ± 4.19 | 0.00 ± 0.00 | 49.51 ± 1.46 | 0.38 ± 0.54 | 45.47 ± 4.22 |
| decoupled | 2 | 5.42 ± 0.20 | 57.49 ± 1.56 | 0.00 ± 0.00 | 49.31 ± 3.56 | 0.08 ± 0.11 | 42.09 ± 0.76 |
| graduation_only | 2 | 5.00 ± 1.18 | 53.90 ± 4.13 | 0.00 ± 0.00 | 34.70 ± 2.48 | 0.00 ± 0.00 | 31.59 ± 0.37 |
| growth_only | 2 | 6.39 ± 4.71 | 58.64 ± 5.33 | 0.00 ± 0.00 | 48.27 ± 1.08 | 0.23 ± 0.32 | 40.10 ± 4.08 |
| random_order | 2 | 19.86 ± 0.98 | 66.81 ± 0.66 | 0.00 ± 0.00 | 52.08 ± 5.71 | 0.08 ± 0.11 | 43.62 ± 0.63 |
| scratch_final | 3 | 20.65 ± 11.09 | 68.18 ± 1.83 | 0.00 ± 0.00 | 58.03 ± 1.66 | 0.05 ± 0.09 | 47.60 ± 2.64 |
| train-trigram |  | 0 | 22.2 | 0 | 11.1 | 0 | 11.5 |

每题都打完了。stage0 / stage2 / stage3 的 `n_scored` 是 360 / 584 / 660。

### 逐个 checkpoint

39 行在 `reports/growbrain_e2_taskprobe_runs.csv`。13 个 checkpoint，每个 stage 一行。`n_truncated` 全是 0。`exact_lo` / `exact_hi` 与 `cjk_lo` / `cjk_hi` 是条目 bootstrap 95% 区间。

`scratch_final` 的 stage0 exact-match：seed0 0.1583333333 [0.1222222222, 0.2000000000]，seed1 0.3333333333 [0.2833333333, 0.3833333333]，seed2 0.1277777778 [0.0944444444, 0.1611111111]。seed1 的区间与另外两粒不相交。

`growth_only` 的 stage0 exact-match：seed1 0.0972222222 [0.0666666667, 0.1305555556]，seed2 0.0305555556 [0.0138888889, 0.0527777778]。两粒区间不相交。

### 结论

四项主指标的种子均值里，`scratch_final` 都高于其余五条臂。未舍入值：stage0 exact-match 0.2064814814814815 ± 0.11091417732873596，CJK token acc 0.6817668642393452 ± 0.01827043194865611、0.5803070011231748 ± 0.016563743939135966、0.4760417652439986 ± 0.02637857612828585。百分数是 20.65 ± 11.09，以及 68.18 ± 1.83、58.03 ± 1.66、47.60 ± 2.64。

`random_order` 在 stage0 上接近。exact-match 0.19861111111111113 ± 0.009820927516479812（19.86 ± 0.98），CJK 0.6680779000846739 ± 0.006586091950086429（66.81 ± 0.66）。stage2 / stage3 的 CJK 是 0.5208411331586172 ± 0.05709448239457085 与 0.43616920601873754 ± 0.006290152118059556（52.08 ± 5.71、43.62 ± 0.63），低于 `scratch_final`。stage3 CJK 上 `coupled` 的均值 0.4547175167975773 高于 `random_order`。

`coupled` 的 stage0 exact-match 最低，0.009722222222222222 ± 0.001964185503295966，约 1%（0.97%）。两粒种子是 0.0111111111 与 0.0083333333。

`graduation_only` 停在 2819058 参数（约 2.8M）。stage2 CJK 0.34699862723075003 ± 0.024796830839064005，stage3 CJK 0.3159364057916154 ± 0.0036804081541837886（34.70 ± 2.48、31.59 ± 0.37）。其余五条臂的 `params_after` 是 18406322（约 18.4M）。stage2 上这五条臂的最低均值是 `growth_only` 的 0.4827155871708474。stage3 上是 `growth_only` 的 0.400965269234409。`graduation_only` 每个种子的 stage2 CJK 区间上沿最高 0.3729128015，18.4M 臂的区间下沿最低 0.4569725568（decoupled seed2）。stage3 上 `graduation_only` 的区间上沿最高 0.3255375330，18.4M 臂的区间下沿最低 0.3619994373（growth_only seed2）。2.8M 在 stage2 / stage3 的 CJK 上低于全部 18.4M 臂。

`coupled` 与 `decoupled` 在四项主指标上的均值都低于 `scratch_final`。课程加扩容没有超过从全宽直接训练的 `scratch_final`。`coupled` 的 stage3 CJK 均值 0.4547175167975773 ± 0.04215740249337799，与 `scratch_final` 的 0.4760417652439986 ± 0.02637857612828585 之差小于两边的样本标准差。

种子方差大。`scratch_final` 的 stage0 exact-match 三粒是 0.1277777778、0.1583333333、0.3333333333，约 13–33%。n 是 2 或 3。六条臂的步数都是 100000。参数量见 Run counts 的 `params_after`，墙钟见该表 `elapsed_s`。步数对齐，计算量没有对齐。等宽的 `staged_final` 不在这六条臂里。见「staged_final 对照臂」。

### 后续

评分路径只做评测。下面几项不改这一节的点估计。

1. 官方 JSON 没有 CJK token acc 的区间。stage2 / stage3 上用来比臂的指标，官方输出只有点估计。runs CSV 里的 CJK 区间是事后按条目重采样补上的。
2. 模型分数里没有 `first_cjk_acc`。manifest 里已有 trigram 的 first-CJK 界。stage2 / stage3 的 exact-match 贴在 0 时，首字准确率是更细的整句类读数。
3. `bootstrap_ci` 的上界下标是 `int(0.975 * n_boot) - 1`。1000 次重采样时取下标 974，常用取法是下标 975。上界偏紧一格。点估计不走这个下标。
4. `is_cjk_char` 只含 U+4E00–U+9FFF，不含 CJK Ext-A（U+3400–U+4DBF）。上文 L=128 共享窗把 Ext-A 算进 CJK。这三份题库没有 Ext-A。1604 条被接受的答案都在 U+4E00–U+9FFF。本题库的比率不受影响。
5. 已合并的 `manifest.json` 仍是 `train_filter.checked=false`。jsonl 不用改。待更新的是 filter 块。写入时不要带绝对训练路径。

## staged_final 对照臂

`staged_final` 来自 growbrain PR [#31](https://github.com/yunshenya/growbrain/pull/31)（`0d8d8f1`）。它走分阶段课程，毕业开着，阈值与 `graduation_only` 相同，从最终宽度起步：`d_model` 384，`n_layers` 10，`n_heads` 6，`d_ff` 1024，`max_seq_len` 512。`cfg_before` 与 `cfg_after` 相同。`params_before` 与 `params_after` 都是 18406322（约 18.4M）。`n_growth`、`n_grow_attempted`、`n_grow_applied`、`n_grow_skipped` 都是 0。`growth_sizes_remaining` 是 3。`growth_events.jsonl` 是空文件。结果对象的 `growth_note` 写明：从 `growth.sizes[-1]` 起步，按分阶段课程训练，毕业开着。

和 `scratch_final` 比，宽度相同，步数都是 100000，`eval_every` 是 1000，`max_seq_len` 都是 512。数据顺序不同。`scratch_final` 把 stage 0–3 混在一起打乱，没有毕业。`staged_final` 按阶段走，毕业三次，`final_stage` 是 3。本页没有另记 FLOPs。墙钟也不能当计算量：这次是 `NVIDIA RTX 6000D`，上文 `scratch_final` 是 RTX 4090。对齐的是配置里的宽度和 100000 步。

训练日是 2026-11-11。两条种子在 `2026-11-11T03:41:42Z` 同时启动，命令是 `scripts/e2_run.py --device cuda --arms staged_final`，配置 `configs/e2.toml`。torch `2.14.1+cu132`，CUDA `13.2`。种子是 1 和 2。`max_train` 200000，`max_probe` 4096，`full_scan` false。`n_evals` 100，最后一步 100000。结果对象的 `elapsed_s` 是 seed1 17989.782、seed2 17977.665。报告顶层的 `elapsed_s` 是 18046.884 与 18034.843。墙钟用结果对象里的数。

毕业记在 `graduation_events.jsonl`。每个种子 3 条，`reason` 都是 `plateau`。训练日志在这三步把 `n_graduation` 记成 1、2、3。`n_growth` 全程是 0。

| seed | step | from_stage | to_stage |
|---:|---:|---:|---:|
| 1 | 8000 | 0 | 1 |
| 1 | 15000 | 1 | 2 |
| 1 | 39000 | 2 | 3 |
| 2 | 9000 | 0 | 1 |
| 2 | 16000 | 1 | 2 |
| 2 | 37000 | 2 | 3 |

训练日志的 probe0–3 是 held-out stage probe（`probe_pseudo_stages` false）。这一节的排序用任务题。probe 准确率留作这次运行的记录。seed1 是 0.6307828374455733、0.4854523714627342、0.527763172801241、0.434800925257742。seed2 是 0.6327331277213353、0.4802710243124751、0.5367779130150191、0.4372154890954552。

任务题是训练时写入 `task_probes.jsonl` 的分数，与 2026-11-10 那次只评测重打分分开。每行有 `step`、`arm`、`seed`、`stage`、`when`、`n_scored`、`n_truncated`、`exact_match`、`cjk_token_acc`。文件里没有逐题命中，也没有 `exact_lo`、`exact_hi`、`cjk_lo`、`cjk_hi`、`cjk_tokens`。条目 bootstrap 95% 区间从这批文件算不出来。`scratch_final`、`graduation_only`、`coupled` 的区间用 `reports/growbrain_e2_taskprobe_runs.csv` 里已有的数。

训练中的 stage0 快照，字段是 `when`=`eval`：每个种子 100 行，stage 都是 0，步数 1000 到 100000，步长 1000。`when`=`end` 是跑完时的 stage 0、2、3，步数 100000。两粒种子上，step 100000 的 `eval` 行与 `end` 的 stage0 行，exact-match 和 CJK token acc 相同。`n_truncated` 全是 0。`n_scored` 是 stage0 360、stage2 584、stage3 660。

arms CSV 追加了 `staged_final` 的三行。均值和样本标准差（除以 n−1）与上一节同一算法。runs CSV 仍是 39 行：新日志填不上那份表头里的区间和 `cjk_tokens`。

### 种子对齐

对照用 seed 1 和 seed 2。`scratch_final` 的 seed 0 不进下表。上一节臂表里 `scratch_final` 的 n=3 含 seed 0，那个均值不是这里的种子对齐。

终局点估计：stage0 exact-match 是 seed1 46/360 = 0.12777777777777777（12.78%），seed2 45/360 = 0.125（12.50%）。CJK token acc，stage0 是 0.6430990685859441 / 0.6257408975444538（64.3% / 62.6%），stage2 是 0.5855484837139648 / 0.5015599650567827（58.6% / 50.2%），stage3 是 0.44847165704551906 / 0.4244345604239614（44.8% / 42.4%）。

下表的对照臂取 runs CSV 的 10 位小数。方括号是该文件里的条目 bootstrap 95% 区间。`staged_final` 一格只有点估计。s2 exact 与 s3 exact 仍不进排序，所以表里没有这两列。

| 臂 | seed | s0 exact | s0 CJK | s2 CJK | s3 CJK |
|---|---:|---|---|---|---|
| scratch_final | 1 | 0.3333333333 [0.2833333333, 0.3833333333] | 0.7027942422 [0.6738511590, 0.7350794332] | 0.5823037564 [0.5681451112, 0.5953669898] | 0.4456326299 [0.4328743546, 0.4577949637] |
| scratch_final | 2 | 0.1277777778 [0.0944444444, 0.1611111111] | 0.6697713802 [0.6468431772, 0.6939045936] | 0.5628353925 [0.5499752598, 0.5741815936] | 0.4927604807 [0.4826095131, 0.5023041475] |
| staged_final | 1 | 0.12777777777777777 | 0.6430990685859441 | 0.5855484837139648 | 0.44847165704551906 |
| staged_final | 2 | 0.125 | 0.6257408975444538 | 0.5015599650567827 | 0.4244345604239614 |
| graduation_only | 1 | 0.0583333333 [0.0333333333, 0.0833333333] | 0.5681625741 [0.5446985447, 0.5944420321] | 0.3645326345 [0.3558938231, 0.3729128015] | 0.3185388474 [0.3109704243, 0.3255375330] |
| graduation_only | 2 | 0.0416666667 [0.0222222222, 0.0638888889] | 0.5097375106 [0.4857261067, 0.5322019147] | 0.3294646200 [0.3201411646, 0.3383200490] | 0.3133339642 [0.3060434372, 0.3206635071] |
| coupled | 1 | 0.0111111111 [0.0027777778, 0.0222222222] | 0.5114309907 [0.4937395659, 0.5294612795] | 0.5054286784 [0.4933937335, 0.5162526264] | 0.4845273020 [0.4741908087, 0.4948453608] |
| coupled | 2 | 0.0083333333 [0.0000000000, 0.0194444444] | 0.4521591871 [0.4364459070, 0.4680207433] | 0.4847123424 [0.4706174680, 0.4975542456] | 0.4249077316 [0.4130129943, 0.4348941020] |

`staged_final` 的种子均值 ± 样本标准差（n=2，除以 n−1）在 arms CSV。未舍入：stage0 exact-match 0.12638888888888888 ± 0.0019641855032959585，CJK 0.634419983065199 ± 0.01227408045243374；stage2 CJK 0.5435542243853737 ± 0.05938885108430637；stage3 CJK 0.4364531087347402 ± 0.016996794021139683。百分数由比例乘 100，保留两位：12.64 ± 0.20，63.44 ± 1.23，54.36 ± 5.94，43.65 ± 1.70。

stage2 / stage3 exact-match 是地板。seed1 stage2 是 1/584 = 0.0017123287671232876，seed2 stage3 是 1/660 = 0.0015151515151515152，另外两格是 0。上一节「13 个 run 的 stage2 exact-match 都是 0」只覆盖那 13 个 checkpoint。

### stage0 训练中轨迹

100 个 `when=eval` 的 stage0 点在 `reports/growbrain_e2_staged_final_trajectory.csv`。列是 `seed,step,when,stage,n_scored,n_truncated,exact_match,cjk_token_acc`。step 100000 那一行就是终局 stage0。

exact-match 在第一次毕业之前是 0。seed1 到 step 8000，seed2 到 step 9000。CJK 在第一次毕业时高于 step 1000。seed1 从 step 1000 的 0.1968670618120237 到 step 8000 的 0.34462320067739205。seed2 从 step 1000 的 0.28662150719729046 到 step 6000 的 0.3835732430143946；step 9000 是 0.32726502963590176。

毕业当步与下一步：

| seed | step | exact 当步 | exact 下一步 | CJK 当步 | CJK 下一步 |
|---:|---:|---:|---:|---:|---:|
| 1 | 8000 | 0.0 | 0.005555555555555556 | 0.34462320067739205 | 0.4123624047417443 |
| 1 | 15000 | 0.002777777777777778 | 0.027777777777777776 | 0.4500423370025402 | 0.5474174428450466 |
| 1 | 39000 | 0.08055555555555556 | 0.06111111111111111 | 0.6147332768839966 | 0.5770533446232007 |
| 2 | 9000 | 0.0 | 0.002777777777777778 | 0.32726502963590176 | 0.47332768839966133 |
| 2 | 16000 | 0.008333333333333333 | 0.05 | 0.4928027095681626 | 0.6011854360711262 |
| 2 | 37000 | 0.1 | 0.1 | 0.5702794242167655 | 0.6350550381033023 |

100 个点对步数的相关系数：seed1 exact-match 0.4987772179，CJK 0.6802397576；seed2 exact-match 0.4249060486，CJK 0.4772223007。相邻步里，exact-match 的上升 / 下降 / 持平是 seed1 45 / 42 / 12，seed2 46 / 42 / 11。CJK 是 seed1 54 / 44 / 1，seed2 51 / 47 / 1。exact-match 和 CJK 的连续下降都是 seed1 最长 4 个 eval、seed2 最长 3 个 eval。

峰在终局之前，峰和终局之间来回跳。seed1 exact-match 的峰是 step 74000 的 0.1638888888888889，下一步 75000 是 0.05，终局是 0.12777777777777777。同一种子的 CJK 峰也在 step 74000，0.6875529212531752；75000 是 0.5563082133784928；77000 回到 0.6841659610499576；终局是 0.6430990685859441。seed2 exact-match 的峰是 step 87000 的 0.2111111111111111，上一步 86000 是 0.03333333333333333，终局是 0.125。CJK 的峰是 step 28000 的 0.6900931414055885；step 96000 是 0.5152413209144793，97000 是 0.6062658763759525，终局是 0.6257408975444538。终局低于峰。路径是波动，离开 stage0 之后没有一条单调下滑。

### 读法

四条臂里，终局 stage0 CJK 的位置是 `scratch_final`，然后 `staged_final`，然后 `graduation_only`，然后 `coupled`。两粒种子都是这个次序。`staged_final` 更靠近 `scratch_final`。

stage0 CJK 上，`scratch_final` 减 `graduation_only` 是 seed1 0.1346316681、seed2 0.1600338696。`staged_final` 减 `graduation_only` 是 0.0749364944859441、0.1160033869444538。这两段相除是 0.0749364944859441 / 0.1346316681 = 0.5566037734，以及 0.1160033869444538 / 0.1600338696 = 0.7248677248。两粒平均 0.6407357491。这一段换的是宽度和训练上下文：`graduation_only` 是 2819058 参数、上下文 128，`staged_final` 是 18406322 参数、`max_seq_len` 512，课程顺序相同，步数都是 100000。它占 stage0 CJK 缺口的大约三分之二。两粒并不一样：seed1 是 0.5566037734，接近一半；seed2 是 0.7248677248，接近四分之三。

剩下的是数据顺序。`scratch_final` 减 `staged_final` 的 stage0 CJK 是 0.0596951736140559 与 0.0440304826555462，约 6.0 与 4.4 个点，两粒都在 4–6 个点。`staged_final` 的点估计落在 `scratch_final` 该种子区间的下沿之下。`staged_final` 自己的区间没有记下来，这里只报告点的位置。

stage2 / stage3 的 CJK，seed1 上 `staged_final` 减 `scratch_final` 是 +0.0032447273139648 与 +0.00283902714551906。两个点都落在 `scratch_final` seed1 的区间内：stage2 [0.5681451112, 0.5953669898]，stage3 [0.4328743546, 0.4577949637]。seed2 是 -0.0612754274432173 与 -0.0683259202760386，约 6.1 与 6.8 个点，落在 `scratch_final` seed2 区间下沿之下。

stage0 exact-match 的种子噪声盖过臂间差距。`scratch_final` 自己是 seed1 0.3333333333 [0.2833333333, 0.3833333333]，seed2 0.1277777778 [0.0944444444, 0.1611111111]，两格区间不相交。`staged_final` 两粒是 0.12777777777777777 与 0.125，贴着 `scratch_final` seed2 的点，离 seed1 的区间下沿 0.2833333333 很远。主证据用 stage0 CJK，以及 stage2 / stage3 的 CJK。

`coupled` 的 stage0 exact-match 是 0.0111111111 与 0.0083333333。`staged_final` 在同样的最终宽度上是 0.12777777777777777 与 0.125。只把数据改成分阶段顺序，到不了 `coupled` 的约 1%。`coupled` 还扩了三次容。这一臂的 `n_growth` 是 0，扩容不在这次对照里。

在相同宽度、相同 100000 步下，分阶段顺序的终局低于直接在最终宽度上训练的 `scratch_final`。stage0 CJK 两粒都低 4–6 个点。stage2 / stage3 在 seed1 上相差约 0.3 个点，在 seed2 上低 6–7 个点。

限制：只有 2 个种子。stage2 / stage3 的 exact-match 是地板，不用于排序。`n_growth` 是 0，这臂不检验扩容。部署上 stage2 / stage3 的训练文件被截到前 210000 行。结果 JSON 里的 `max_train` 与 provenance `used_train` 都是 200000，`capped_prefix` 为 true，loader 读的是前 200000 行。210000 大于 200000，读到的行与未截断文件的前 200000 行相同。

stage-0 mastery 毕业的离线估计（只用已有日志）见「若改用 stage-0 mastery 毕业（离线估计）」。

## 若改用 stage-0 mastery 毕业（离线估计）

2026-11-11（CST），用已有训练日志估计 E2 的毕业若改走 stage-0 mastery，日程和 stage 0 任务分会落在哪里。没有新的训练，也没有新的 GPU 步。代码是 growbrain `main` `0d8d8f1` 的 `growbrain/curriculum.py`，配置是 `configs/e2.toml`，与「staged_final 对照臂」同一份。跨越表是 `reports/growbrain_e2_mastery_graduation_crossings.csv`。`random_order` 和 `scratch_final` 故意不在这张表里。

### 毕业规则

E2 的毕业在代码里有两条路径。eval 每 1000 步一次。

掌握路径：`probe{k}_acc >= mastery`，连续 `sustain` 次 eval。`probe0_acc` 是 held-out stage-0 文档上的 next-token top-1。配置是 `mastery = 0.85`，`sustain = 2`。

plateau 路径：连续 `window=3` 次 eval 的 ALP < `plateau_eps=0.01`，并且下一阶段的 ALP 更大。这些 run 上实际触发的是这一条。掌握路径没有触发过。

`handwritten_gate` 在全部 E2 run 里关着。若打开，离开 stage 0 还要 `extra_handwritten0_acc >= 0.39`，离开 stage 1 用的是 probe1，并且旧阶段下降不超过 5 个百分点。0.39 来自 `scripts/calibrate_e2_thresholds.py`，是 seed-0 阶梯臂上 extra-bank 决策簇（最高 0.2957）与 stage-1 probe 簇（最低 0.4771）的中点。它把两簇切开，不是掌握程度。配置注释已经写明：这扇门开着时，每条阶梯臂的第一次 plateau 都会被拒绝。

现有规则不用任务题 exact-match，也不用任务题 CJK。下面这两项只是在日志上试过的候选线。

### staged_final 的毕业前后

`staged_final` 的六次毕业全是 plateau。seed1 是 8000 / 15000 / 39000，seed2 是 9000 / 16000 / 37000，与上一节同一份 `graduation_events.jsonl`。第一次毕业之前，step 1000 到 8000（seed1）或 9000（seed2），`stage_weights` 是 [1, 0, 0, 0]。这段是纯 stage 0。

下表是第一次毕业前、毕业当步、以及毕业后第一次 eval，三位小数。任务题的未舍入值在上一节「stage0 训练中轨迹」。`probe0_acc` 与手写库的未舍入跨越在 CSV。

| seed | step | 位置 | probe0_acc | extra_handwritten0_acc | task EM s0 | task CJK s0 |
|---:|---:|---|---:|---:|---:|---:|
| 1 | 1000 | 纯 stage 0 | 0.611 | 0.179 | 0.000 | 0.197 |
| 1 | 4000 | 纯 stage 0 | 0.683 | 0.210 | 0.000 | 0.290 |
| 1 | 8000 | 毕业，plateau | 0.708 | 0.196 | 0.000 | 0.345 |
| 1 | 9000 | 毕业后 1 次 eval，stage 1 权重 75% | 0.640 | 0.422 | 0.006 | 0.412 |
| 2 | 1000 | 纯 stage 0 | 0.612 | 0.193 | 0.000 | 0.287 |
| 2 | 6000 | 纯 stage 0 | 0.705 | 0.215 | 0.000 | 0.384 |
| 2 | 9000 | 毕业，plateau | 0.717 | 0.224 | 0.000 | 0.327 |
| 2 | 10000 | 毕业后 1 次 eval | 0.654 | 0.427 | 0.003 | 0.473 |

终局 step 100000，seed1 / seed2：`probe0_acc` 0.631 / 0.633，手写 0.469 / 0.464，任务 exact-match 0.128 / 0.125，任务 CJK 0.643 / 0.626。probe 的未舍入终局是上一节的 0.6307828374455733 / 0.6327331277213353。CSV 的 `end_value` 是 0.6308 / 0.6327，是同一格。

纯 stage 0 上，`probe0_acc` 在 0.71 附近走平。最后 3000 步，每 1000 步增加 0.002 到 0.004。100000 步上的最高就是毕业当步：seed1 0.708，seed2 0.717（CSV `max_any` 0.7075 / 0.7171）。配置的 `probe0_acc >= 0.85`、连续 2 次 eval，两粒都没有跨过。0.80 和 0.75 也没有。

### 首次跨越

下表只列 `staged_final` 的 sustain 2。步数是已经跑完的日志里，该指标第一次连续两次 eval 达到这条线的步。晚于 8000 / 9000 的跨越已经含有后续阶段的数据，不能当成「若一直停在 stage 0，会在这一步毕业」。最后一列只回答：纯 stage 0、第一次毕业之前，这条线有没有跨过。sustain 1、只碰到一次、以及其余臂，以 CSV 为准。正文的 0.618、0.670、0.682、0.128、0.207 是三位读法，CSV 的线是 0.6181、0.6698、0.6818、0.1278、0.2065。

| 阈值 | seed1 sustain 2 | seed2 sustain 2 | 第一次毕业前的纯 stage 0 |
|---|---|---|---|
| probe0 ≥ 0.85（配置 mastery） | 从未 | 从未 | 否。100000 步最高 0.708 / 0.717 |
| probe0 ≥ 0.80 | 从未 | 从未 | 否 |
| probe0 ≥ 0.75 | 从未 | 从未 | 否 |
| probe0 ≥ 0.70 | 7000 | 7000 | 是。比 plateau 的 8000 / 9000 早 1000 到 2000 步 |
| probe0 ≥ 0.618（scratch_final 终局均值） | 3000 | 3000 | 是。早于 plateau |
| hw0 ≥ 0.39（校准门） | 10000 | 11000 | 否。毕业前最高 0.232 / 0.240。日志上的首次跨越在 stage 1 的数据已经混入之后 |
| hw0 ≥ 0.45 | 19000 | 21000 | 否 |
| hw0 ≥ 0.50 | 从未（77000 碰到一次） | 从未（32000 碰到一次） | 否 |
| 任务 CJK s0 ≥ 0.60 | 37000 | 18000 | 否 |
| 任务 CJK s0 ≥ 0.670（scratch_final 终局最小，seed2） | 从未（74000 碰到一次） | 29000 | 否 |
| 任务 CJK s0 ≥ 0.682（scratch_final 终局均值） | 从未（74000 碰到一次） | 从未（28000 碰到一次） | 否 |
| 任务 EM s0 ≥ 0.10 | 60000 | 24000 | 否。毕业前每次 eval 的 exact-match 都是 0 |
| 任务 EM s0 ≥ 0.128（scratch_final seed2 终局 12.78%） | 从未（59000 碰到一次） | 28000 | 否 |
| 任务 EM s0 ≥ 0.207（scratch_final 终局均值） | 从未 | 从未（87000 碰到一次） | 否 |
| 任务 EM s0 ≥ 0.333（scratch_final seed1） | 从未 | 从未 | 否 |

CSV 每一行是一条臂、一粒种子、一个指标、一条线。列是 `arm,seed,metric,bar,bar_source,first_grad_step,value_at_first_grad,max_pre_grad,max_any,end_value,first_cross_sustain1,first_cross_sustain2,cross_before_first_grad_sustain2`。步数是训练步。`first_cross_sustain1` 是第一次达到，`first_cross_sustain2` 是第一次连续两次 eval 达到。空单元格表示没有这次跨越。`cross_before_first_grad_sustain2` 在 sustain 2 的跨越不晚于第一次毕业时填写。`growth_only` 没有毕业，`first_grad_step` 为空；它若以 sustain 2 跨过某条线，最后一列记下的就是这一步。`tp_em` 与 `tp_cjk` 只出现在 `staged_final`。其余臂的中途 eval 没有任务题。

### 其余臂

多种子的 seed 1、2，加上 pilot 的 seed 0。中途 eval 只有 `probe0_acc` 和手写库。

`graduation_only` / `coupled` / `decoupled` 的第一次毕业是 seed0 的 8000，或 seed 1、2 的 9000。全是 plateau。当时参数量 2819058（约 2.8M），上下文 128。毕业当步的 `probe0_acc` 是 0.627 / 0.636 / 0.632（CSV 0.6269 / 0.636 / 0.6324），三臂相同。手写库在 0.221–0.246（CSV 从 0.2207 到 0.2458）。100000 步上 `probe0_acc` 的 `max_any`，seed 1 和 seed 2 在 0.636–0.645，最高是 `decoupled` seed1 的 0.6447。`graduation_only` seed0 的峰值是 0.6269；`coupled` / `decoupled` 的 seed0 是 0.6385 / 0.6374。0.85、0.80、0.75、0.70 在这三条臂的每一粒上都没有跨过。手写 ≥ 0.39 的 sustain 2 首次在 10000–12000，都在 stage 1 已经混入之后。

`growth_only` 的 `stage` 字段停在 0，`stage_weights` 是 [0.25, 0.25, 0.25, 0.25]，练的是四段均匀混合。它不是「一直停在纯 stage 0」的代理。扩容到 18.4M 之后，`probe0_acc` 最高仍是 seed1 的 0.6804，seed0 / seed2 是 0.6751 / 0.6754，停在 0.680 这一档，低于 0.70。这臂没有毕业。

`scratch_final`（seed 0 / 1 / 2）终局 `probe0_acc` 是 0.617 / 0.626 / 0.611，全程最高 0.630。这低于 `staged_final` 第一次毕业时的 0.708 / 0.717。stage 0 的任务分是各臂里最高的。`probe0_acc` 对不齐主任务指标。这臂没有毕业，也不在跨越 CSV 里。`random_order` 也不在表里。训练档案里它的 probe0–3 是打乱的伪阶段。步数占比见「stage 0 所占步数」。

### stage 0 所占步数

按日志里的 `stage_weights` 做步数加权，分母是 100000 步。

`staged_final` 大约有 17.5k 与 18.5k 步在 stage 0，占 17.5–18.5%。`graduation_only` / `coupled` / `decoupled` / `random_order` 是 17–19%。`scratch_final` 是 5.3%，大约 5.3k 步。stage0 exact-match 15.8 / 33.3 / 12.8%，CJK 67.3 / 70.3 / 67.0%（seed 0 / 1 / 2）。这是 stage 0 任务分最好的一臂。

### 读法

不值得为 mastery 毕业再开一次 GPU。相对 `scratch_final`，这些日志指向的是更差。日志里没有打平它的路径。早先「最多与 scratch 打平」这个读法过宽。

配置的掌握线 `probe0_acc >= 0.85`、连续 2 次 eval，不会触发。`staged_final` 在 100000 步上的最高是 0.708 / 0.717。每一次实际毕业都是 plateau：3 次 eval 的 ALP < 0.01，且下一阶段的 ALP 更大。0.85 在 100000 步内都达不到，run 会留在 stage 0 直到结束，stage 2 / 3 几乎得不到训练。相对 `scratch_final`，这是更差的结果。

把线降到 `probe0_acc >= 0.70`，两粒的 sustain 2 都在 7000，比 plateau 早 1000 到 2000 步。日程几乎不变。上一节的 `staged_final` 已经低于 `scratch_final`。日志里没有另一套分数，能让这一两千步的提前变成打平。

手写库、任务题 CJK、任务题 exact-match 上试过的线，在第一次毕业前的纯 stage 0 上都没有达到。毕业前：任务 exact-match 每次 eval 都是 0；CJK 停在 0.29–0.38；手写库大约 0.20–0.24，最高 0.232 / 0.240。这些规则会把 run 留在 stage 0，留很久，或者留满 100000 步。stage 2 和 stage 3 就几乎得不到训练。

日志里更晚的跨越，包括手写 ≥ 0.39 的 10000 / 11000，已经混入了后续阶段。那些步是实际 run 的记录，不是停在 stage 0 时的毕业步。stage 0 的任务能力是在后续阶段混进来之后才上去的。stage 1 混入后的第一次 eval，手写库大约 +0.2：seed1 从 0.196 到 0.422（+0.226），seed2 从 0.224 到 0.427（+0.203）。CJK 从 0.345 到 0.412（+0.067），以及从 0.327 到 0.473（+0.146），大约 +0.07 到 +0.15。同一次 eval 上 `probe0_acc` 从 0.708 掉到 0.640，从 0.717 掉到 0.654。

`scratch_final` 只有大约 5.3% 的步在 stage 0，stage 0 的 exact-match 和 CJK 最好。分阶段的臂用了 17–19%。

能对上纯 stage 0 的，只有前 8000 步（seed1）和前 9000 步（seed2）。之后的分数含有后续阶段的数据，不是「一直停在 stage 0」的反事实。没有任何一条 run 在 9000 步之后仍只练 stage 0。这里不写 mastery 规则下的终局分数。

试过的每条线落在两组里的一组。

（a）在 plateau 当时或之前，纯 stage 0 上就已经触发，run 不会被留在 stage 0。`probe0_acc` 不超过 0.70 的线，以及 `scratch_final` 的 probe 锚 0.6181，属于这里。0.70 只把第一次毕业从 8000 / 9000 提前到 7000，日程几乎不变。0.6181 在 3000 就跨过，离开 stage 0 更早；那一次提前毕业的任务终局，日志里没有对应的 run，这里不补。

（b）在 8000–9000 的纯 stage 0 上没有达到。run 被留在 stage 0，stage 2 / 3 的训练被拿掉。手写库、任务 CJK、任务 exact-match 属于这里。probe0 的 0.75、0.80 和配置的 0.85 也属于这里：纯 stage 0 上没有达到，100000 步内也没有达到。

两组都给不出能超过 `scratch_final` 的机制。

9000 步纯 stage 0 之后，stage 0 技能仍可能再跳一次。日志里没有支持这次晚跳的点。

日志上还剩的、可证伪的 GPU 对照，是最终宽度的 `stage0_only`，100000 步，seed 1 和 seed 2。它检验的是纯 stage 0 数据。它不检验 mastery 毕业。stage 2 / 3 的 CJK 按数据分配，几乎可以确定会明显低于 `scratch_final` seed1 / seed2 的 58.2 / 56.3 与 44.6 / 49.3。这是从数据分配推的。日志里没有这条对照的测量。不建议跑这一对照。

限制：每臂用来估计的种子是两粒。`scratch_final` 另有 seed 0，其中 seed1 的 exact-match 33% 是离群点。阶梯臂的 pilot seed 0 写在 CSV 里，也写在第一次毕业的 8000 上，不另作一条终局结论。旧臂的任务题是只评测重打分，共享窗口 128，截断数是 0。`staged_final` 的任务题写在训练过程里，截断数也是 0。题目这一层可以对。两条流水线各写各的，逐位相同不成立。`graduation_only` / `coupled` / `decoupled` / `growth_only` 的中途 eval 没有任务题。

原始日志在分析机器上，不在 git 里：`/workspace/scratch/e2_staged_final/s{1,2}/events/staged_final/seed{1,2}/{evals,task_probes,graduation_events}.jsonl`；`/workspace/scratch/e2_multiseed_12/ckpts/{arm}/seed{1,2}/{evals,graduation_events,growth_events}.jsonl`；`/workspace/scratch/e2_pilot_full/events/{arm}/seed0/`；`/workspace/scratch/e2_scratch_final_012/events/scratch_final/seed{0,1,2}/`；`/workspace/scratch/e2_task_probes/per_run_table.csv`。代码和配置在同一台机器的 growbrain `0d8d8f1`。

现有数据与评测能否把能力分开的离线核对（只用已有日志和题库）见「现有数据与评测能否区分能力（离线核对）」。

## 现有数据与评测能否区分能力（离线核对）

2026-11-12（Asia/Shanghai，UTC+8）。用已有日志和已有题库核对：这套数据和评测能不能把更强的模型和更弱的模型分开。没有新的训练，没有 GPU，没有 `--resume`。这一节不改上文已发表的臂分，也不改 `reports/growbrain_e2_mastery_graduation_crossings.csv`。离线备忘于当天从已有日志算出，在分析机器上，不在 git 里：`/workspace/scratch/e2_eval_distinguish.md`。关键表在本节。13 个终局重打分的 stage-0 逐技能命中在 `reports/growbrain_e2_taskprobe_stage0_by_skill.csv`，列是 `arm,seed,stage,skill,n,n_exact,exact,cjk`。`exact` 与 `cjk` 是 item 文件里的原值。13 个 run，6 个技能。这一节沿用 2026-11-11 的 mastery 反事实，不重做那一节。

### 直接回答

只能分开一部分。官方把 probe0–3 等权平均，分不开能力。被当成 stage 0 头条的 exact-match，也分不开。评测和数据对齐之前，不要为另一条同类的 growbrain 训练臂再租 GPU。

能分开的，有这四项。

stage2 / stage3 的任务 CJK 能把 2.8M 和 18.4M 分开。`graduation_only` 的种子均值是 0.347 / 0.316。每条 18.4M 臂都明显高于它，种子区间与小模型不相交。未舍入均值和区间上、下沿已经写在「任务题正式对照」：stage2 均值 0.34699862723075003，区间上沿最高 0.3729128015（18.4M 的区间下沿最低 0.4569725568，decoupled seed2）；stage3 均值 0.3159364057916154，区间上沿最高 0.3255375330（18.4M 的区间下沿最低 0.3619994373，growth_only seed2）。

共享窗 L=128 的 corpus probe2 是另一格，不是上面的任务 CJK。去掉 128 对 512 的混淆之后，它讲的是同一件大小模型的事。`graduation_only` 停在同阶段 trigram 地板上（0.347 对 0.350），比 `scratch_final` 低 11 个点。未舍入值在「Shared-span CJK at L=128」。这一节没有重算。

`coupled` 的 stage0 exact-match 能和同宽、不扩容的 `staged_final` 分开。seed1 / seed2：`coupled` 是 4/360 与 3/360（1.11% / 0.83%），`staged_final` 是 12.78% / 12.50%。这个差距不是格式地板，也不是种子噪声。上文「种子对齐」里的未舍入格子是 0.0111111111 与 0.0083333333，以及 46/360 = 0.12777777777777777 与 45/360 = 0.125。

stage0 CJK 能做粗排序。`scratch_final` 是 67.0–70.3%，`random_order` 是 66.3–67.3%，`staged_final` 是 64.3% / 62.6%（seed1 / seed2），然后是分阶段的 2.8M 和混合臂，`coupled` 在最后（51.1% / 45.2%，seed1 / seed2）。单次 run 的条目 bootstrap 区间大约 ±2–3 个点。scratch 与 `staged_final` 的差距是 4–6 个点，现有种子上方向一致。中间臂的未舍入 CJK 在「任务题正式对照」和「种子对齐」。

`where` 这一片的 stage0 exact-match 比全技能平均稳。`scratch_final` 三粒是 43.8–46.9%。CSV 里 seed 0 与 seed 2 是 30/64，seed 1 是 28/64。`random_order` 更高，64.1–68.8%（seed1 / seed2 是 41/64 与 44/64）。`coupled` 是 0–4.7%（seed1 / seed2 是 3/64 与 0/64）。

分不开的，有这四项。

stage2 / stage3 的 exact-match 是难度地板。13 个旧臂的 stage2 全是 0。stage3 是 0–0.76%。`staged_final` 终局同样是地板，备忘的读法是 end EM 0 / 0.15%。上一节已发表的格子是 seed1 stage2 为 1/584、seed2 stage3 为 1/660，另外两格是 0。答案偏长：stage2 中位 13 字，stage3 中位 16 字，在 10–27 个 CJK 字上全有或全无。不用这两列排臂。

官方 probe0–3 的等权平均不是能力。2026-11-10 的 probe 审计和共享窗重打分已经写过：这些 probe 主要跟着该臂见过多少该阶段的数据走。`graduation_only` 官方 probe2 的「赢」是 128 对 512 的产物。同阶段字符 trigram 在好几条 probe 上离 18.4M 模型只有几个点。数字在「Shared-span CJK at L=128」。这一节没有重算。

stage0 exact-match 不能当头条，尤其不能用来宣称 `scratch_final` 在这项上赢了 `staged_final` 或 `random_order`。`scratch_final` seed1 是 33.3% [28.3, 38.3]，seed2 是 12.8% [9.4, 16.1]。两个区间不相交。未舍入区间在「种子对齐」：[0.2833333333, 0.3833333333] 与 [0.0944444444, 0.1611111111]。缺口的绝大部分是 `who`。seed 1 / 0 / 2 是 90.6% / 17.2% / 9.4%，即 58 / 11 / 6 道，分母 64。去掉 `who`，剩下 296 道上三粒是 20.9% / 15.5% / 13.5%，种子顺序仍是 1 / 0 / 2。`staged_final` 的 12.8% / 12.5% 贴着 scratch 的 seed2，不贴着 33% 那一粒离群点。两粒或三粒种子撑不起「scratch 在 stage 0 的 exact-match 上赢了」。

`probe0_acc` 不能当能力，也不能当掌握。它和任务分反向。见下。

所以：这套题能在后期 CJK 上把小模型和大模型分开，也能把塌掉的 `coupled` 和还在工作的同宽臂分开。它分不清后期课程上几个点的变动是不是真的。它也分不清 stage-0 的任务技能是不是 stage-0 的数据教出来的。后一件事挡住新的训练臂。

### 死技能和种子分歧

13 个终局 item 文件，stage 0，360 题。没有任何一道题在 13 次里都 exact。161/360 一次都没有。

| 技能 | n | 13 次里从未 exact | 命中 run 的平均比例 |
|---|---:|---:|---:|
| handoff | 72 | 72 | 0 |
| count | 64 | 56 | 0.96% |
| attribute | 48 | 25 | 5.4% |
| who | 64 | 3 | 11.7% |
| color | 48 | 3 | 17.8% |
| where | 64 | 2 | 29.3% |

`handoff` 是死题类。72/360 = 20% 的题库，对档案里每一次 exact-match 的贡献都是 0。把它平均进去，只是把每条臂往 0 压。`count` 接近死题：8 道只在恰好一次 run 里命中，56 道从未命中。`who` 的平均命中比例是 11.7%。scratch seed1 的离群点几乎全落在这个技能上。逐 run 的 `n` 与 `n_exact` 在 CSV。上表的平均比例是备忘的读法。

种子分歧不是抽题噪声。n=360 上，二项 95% 半宽在 p=0.20 是 4.1 个点，在 p=0.33 是 4.9 个点。scratch seed1 对 seed2 差 20.6 个点，是这个半宽的好几倍，条目 bootstrap 区间也不相交。同一臂里，exact 命中的集合几乎对不上：

| 臂 | exact 题数 | 交 / 并（Jaccard） | 两粒不一致的题数 |
|---|---|---:|---:|
| coupled | 4 vs 3 | 0 / 7（0） | 7 |
| graduation_only | 21 vs 15 | 3 / 33（0.09） | 30 |
| decoupled | 19 vs 20 | 6 / 33（0.18） | 27 |
| growth_only | 35 vs 11 | 7 / 39（0.18） | 32 |
| random_order | 69 vs 74 | 33 / 110（0.30） | 77 |
| scratch 0 vs 1 | 57 vs 120 | 35 / 142（0.25） | 107 |
| scratch 1 vs 2 | 120 vs 46 | 25 / 141（0.18） | 116 |

模型并没有稳定地解同一小组题。5–20% 的 exact-match 均值、n=2，排不了名。Jaccard 常见落在 0.09–0.30。`coupled` 是 0。

没命中的题上，CJK 仍有粗的大对小、scratch 对 coupled 信号：scratch 的未中题 CJK 是 0.57–0.63，random 是 0.60，coupled 是 0.45–0.51。它紧不到能支持 1–2 个点的课程结论。scratch seed1 的未中题 CJK 更低，是 0.568，因为那一粒已经把容易的题挑走了。

### 中途 exact-match 不宜当门

`staged_final` 是唯一有中途任务题的臂：每 1000 步，只评 stage 0，截断数 0，n=360。第一次毕业之后，exact-match 相邻步绝对变化的均值是 0.031 / 0.032（seed1 / seed2），最大是 0.114 和 0.178。终局率大约 0.13 时，二项标准误是 0.018。轨迹每次 eval 在动好几个标准误。seed2 的峰是 step 87000 的 21.1%，终局是 12.5%。未舍入的峰在「stage0 训练中轨迹」：0.2111111111111111，终局 0.125。没有平滑，不能把掌握线画在原始的中途 exact-match 上。平滑之后，这个数仍然跳。

CJK 稍稳一些。相邻步绝对变化的均值大约 0.03，抬起来之后相对带宽更窄。`hw0` 跳一次，然后停住。

### `probe0_acc` 与任务分反向

在 `staged_final` 的轨迹上，`probe0_acc` 对 stage0 任务 CJK 的 Pearson 相关是 −0.56 / −0.68（seed1 / seed2），对 exact-match 是 −0.32 / −0.47。`hw0` 的方向是对的：对任务 CJK 是 +0.91 / +0.89，对 exact-match 是 +0.51 / +0.59。机器可读摘要里，对 CJK 的未舍入相关是 `probe0` 的 −0.5592604893047963 / −0.6799796048096802，`hw0` 的 0.9131369339531065 / 0.8867982987415428。对 exact-match 的相关，备忘只给了上面的两位小数。

机制在第一次毕业时就能看见。权重在毕业当步仍是 [1, 0, 0, 0]，下一步 stage 1 已经混入。本表用备忘的读法。上一节同一批日志用三位小数，格子没有改。seed2 在 9000 的手写库，跨越 CSV 的 `value_at_first_grad` 是 0.2235，上一节写成 0.224，备忘写成 0.223。seed1 在 9000 的任务 EM，轨迹是 0.005555555555555556，上一节写成 0.006，备忘写成 0.0056。seed2 在 10000 的任务 EM，轨迹是 0.002777777777777778，上一节写成 0.003，备忘写成 0.0028。

| seed | 位置 | probe0 | hw0 | 任务 EM | 任务 CJK |
|---|---|---:|---:|---:|---:|
| 1 | 8k，仍是 [1, 0, 0, 0] | 0.708 | 0.196 | 0 | 0.345 |
| 1 | 9k，[0.25, 0.75, 0, 0] | 0.640 | 0.422 | 0.0056 | 0.412 |
| 2 | 9k，仍是 [1, 0, 0, 0] | 0.717 | 0.223 | 0 | 0.327 |
| 2 | 10k，stage 1 已混入 | 0.654 | 0.427 | 0.0028 | 0.473 |
| 1 / 2 | 100k 终局 | 0.631 / 0.633 | 0.469 / 0.464 | 0.128 / 0.125 | 0.643 / 0.626 |

`probe0` 的最高点在纯 stage 0 的末尾，然后后续数据一进来就掉，任务分同时往上走。`scratch_final` 终局的 probe0 更低：0.617 / 0.626 / 0.611，全程最高 0.630，低于 `staged_final` 第一次毕业时的 0.708 / 0.717，而 stage0 任务分是各臂里最好的。配置的 `probe0 >= 0.85` 不会触发，上一节已经写了。把线降到 0.70，会在任何任务技能出现之前毕业。负相关是这一节新的，把那条结论收紧了。

`hw0` 是更好的中途代理。它是 24 段 next-token 段落，条目 bootstrap 标准误大约 0.03–0.04。它撑不起 1–2 个点的结论。

teacher-forced 的 CJK 不是 exact-match。CJK 明显高于 stage0 的 train-trigram 地板 0.222，模型落在 0.45–0.70，因为金标前缀之后，答案内部的搭配也能得分。exact-match 更严，也更吵、更会按技能塌掉、中途更跳。已有的 greedy / teacher-forced 输出里，first-CJK 落在两者中间，而且只有 7 个旧臂 run，路径是 `/workspace/scratch/e2_coupled_s0_diag/preds/`。`coupled` seed1 的 first-CJK 是 37.8%，exact-match 是 1.1%。scratch seed1 是 59.7% 对 33.3%。它不能直接拿来排臂：`decoupled` seed1 的 first-CJK 是 29.2%，低于 `coupled` seed1，exact-match 反而更高（5.3% 对 1.1%）。`coupled` 在 72 道 `handoff` 里有 50 道 first-CJK 命中，exact 是 0。首字是一个便宜的先验。first-CJK 值得加，要带技能分解，不能换成新的单一头条。`handoff` 上不要用 first-CJK 替换 exact-match。

官方 JSON 仍没有 CJK 区间，也没有 `first_cjk_acc`。`bootstrap_ci` 的上界下标是 `int(0.975 * n_boot) - 1`，上界偏紧一格。`is_cjk_char` 只含 U+4E00–U+9FFF。这三份题库没有 Ext-A，比率不受影响。已提交的 `manifest.json` 仍是 `train_filter.checked=false`。2026-11-10 的同阶段检查是 0 条 8-gram 丢弃、0 条 train-trigram 整句命中，填的只是 manifest。`staged_final` 的中途题和 13 个 checkpoint 的重打分，在题目这一层可以对，两条流水线不是逐位相同。这些限制上文已经写过，这里仍然有效。

### stage-0 技能来自后续数据：格式对不上

现象比「stage 1 之后分数跳了」更清楚。这是数据与评测的格式错位，不是题目泄漏，也不是故事理解的神秘迁移。官方过滤器没有丢掉题，整句也不在训练集里。

纯 stage 0（step 1k 到第一次毕业，权重 [1, 0, 0, 0]）：每次 eval 的任务 exact-match 都是 0。seed1 是 8 次 eval，seed2 是 9 次。任务 CJK 停在 0.197–0.345（seed1）和 0.287–0.384（seed2）。`hw0` 平在 0.18–0.24。`probe0` 已经升到 0.71 并走平。

stage 1 混入后的第一次 eval（75% stage 1，25% 回放）：两粒的 `hw0` 大约都是 +0.22。任务 CJK 是 +0.068 / +0.146。上一节按三位小数把同一次跳跃写成 +0.067 / +0.146（0.345 到 0.412，0.327 到 0.473），那一格没有改。任务 exact-match 变成 0.56% / 0.28%，即 2 道和 1 道。它没有变成可用的 exact-match 信号。

stage 1 窗口的其余部分（seed1：9k–15k；seed2：10k–16k）：exact-match 停在 0–1.67%，最多 6/360。CJK 和 `hw0` 停在新的、更高的水平上。

stage 2 混入后的第一次 eval：exact-match 从 0.28% 到 2.78%（seed1，16k），从 0.83% 到 5.00%（seed2，17k）。CJK 再跳 +0.097 / +0.108。stage0 的 exact-match 是从这里才出现的。未舍入的当步与下一步在「stage0 训练中轨迹」。

所以 stage 1 主要教的是格式，以及 `hw0` 量到的那部分。stage0 题库上可用的 exact-match 随着 stage 2 出现。`scratch_final` 只有大约 5.3% 的步在 stage 0，stage0 CJK 仍然最好。阶梯臂用了 17–19%，在后续阶段混入之前 exact-match 是 0。更多的 stage-0 数据并不是 stage-0 任务分的来源。

官方过滤器对全部 12 对都做了，不只是同阶段：3 份题库（stage 0、2、3）对 4 份训练切分（0、1、2、3）。规则是 OverlapIndex 的精确命中，加上 8-gram 出现比例 ≥ 0.5，以及 train-trigram 整句 exact。12 对的丢弃都是 0。train-trigram exact 在 12 对里都是 0。stage0 题对 stage1 训练的最差 8-gram 出现比例是 0.429，低于 0.5。≥ 0.25 的有 12 道，≥ 0.10 的有 15 道。对 stage2 的最差是 0.056。同阶段的 train-trigram CJK 与 2026-11-10 的过滤一致，备忘读成 0.222 / 0.111 / 0.115。未舍入值在「训练切分过滤」：0.2218458933、0.1114439037、0.114602063。

没有任何一道 stage0 题的完整 `prompt+answer` 是任一训练文档的子串，任一 stage 都没有。完整 prompt 也不是子串。

更弱的重叠是模板复用，测试题并不在训练集里：

14/360 道 stage0 题的问句一行，原样出现在 stage1 训练里。这 14 道只有 5 个不同的问句：`问：书包在哪里？` 占 8 道，另外四句 `谁在…` 占 6 道。stage1 里确有 `问：书包在哪里？\n答：书包在学校。它就在那里。`。问句相同，上下文和答案都不同。题库的上下文是一行事实，例如「书包在河里。」，答案是「书包在河里」。

有一道颜色题的上下文或答案片段「帽子是白色」出现在 stage1 的 `qa_what` 答案里（「阿姨面前放着帽子。那个帽子是白色的。」）。

少量短的泛答案（「有一个苹果」「小鱼在河里」「皮球是黄色」等）是无关的 stage0 或 stage2 文档的子串。stage2 有 17 道这样的答案子串，匹配的问句行是 0，匹配的 prompt 是 0。这是大语料里的短语碰撞。

stage2 和 stage3 题库对任何训练切分，prompt、答案、问句行、整段子串的命中都是 0。

已提交的 manifest 只对过 `stage1_train_mix.txt`（38 篇手写）加上同阶段过滤，仓库里 `checked=false`。缺的那次检查是 stage0 题对完整的 stage1 训练（556 篇）。它不改题库。它也解释不了中途的跳跃：14 道问句相同、答案不同，不能在 360 道上造出 7–15 个点的 CJK 跳跃，而 exact-match 在整个 stage1 窗口里都贴近 0。2026-11-10 的任务题报告已经查过手写的 `stage1_train_mix.txt`：exact 命中是 0。这一节没有再查验证切分、corpus probe，或用手写库对任务题另做一遍。

错位在题库测的格式上。stage0 任务题是固定框的短阅读：一句事实，然后 `问：…？\n答：`，答案通常几乎是事实的复写。例子是「小猫在床上。」/ `问：小猫在哪里？` / 「小猫在床上」。360 道题，不同的问句行只有 99 句。

stage0 训练是 22409 篇，故事和对话。probe 标点下同时含「问」和「答」的文档是 1/22409（0.0045%）。以「问：」开头的行是 0。「答：」出现在 18 篇里。「问：」或「问:」出现在 13 篇里，几乎都是叙述（「莉莉问她妈妈」），不是题库的框。技能词在故事里有：630 篇提到「在哪里」（同一份 `train.jsonl` 上本地重计；这一节原先写的是 631），476 篇提到「什么颜色」，但都不是「问：」问句。上文「题型与计分」里 stage0 的 18/22409 是含 `答：` 或 `答:` 的行数。备忘的 18 是文档数。上文的行数留在原处。

stage1 训练是 556 篇（38 篇手写 + 518 篇模板）。其中 39.0% 是「问：」/「答：」框，217 篇，另有 32 个对话轮。含「在哪里」的问句 24 句，「去了哪里」13 句，含「谁」的问句 66 句，其中字面含「谁在」的是 32 句。这一节原先把 66 写成了「谁在」。「什么颜色」「几个」「怎么样」「在谁手里」都是 0。交接只出现在叙述从句里（「阿姨把笔递给妈妈」，22 篇），不是问答。模板（`qa_where`、`qa_where_thing`、`qa_who` 等）和题库是同一族，但常常是闭卷：事实在答案里，不在上下文那一句里。所以 stage 1 抬得动 `hw0` 和 CJK，而「从 prompt 里抄答案」的 exact-match 要等到 stage 2 更杂的问答进来才离开 0。stage2 里同时有「问：」和「答：」的文档是 1961 篇，「问：」行 401 行，其中有少量「什么颜色」和「几个」。

stage3 和 stage0 一样，问答框极少：200000 篇里同时有「问：」和「答：」的是 1 篇，问句行是 0。

所以 stage0 题库测的是一种问答抽取格式：在 stage 0 里几乎没有，在 stage 1 里密，在 stage 2 里以低比例存在。它测的不是「stage0 数据教了什么」。中途的跳跃是先接触格式，再接触更宽的问答。`growth_only` 已经给出较弱的同一现象：它见到的问答文档比 `scratch_final` 多（混合里大约 10% 对大约 0.5%），exact-match 仍然只有 9.7% / 3.1%，因为均匀的 25% 混合不等于「stage 0 教会了这套题」。未舍入的两粒是 0.0972222222 与 0.0305555556，在「逐个 checkpoint」。

这也改写「分阶段课程在 stage 0 上输给了 scratch」。它输在一套只练 stage 0 过不了、后续阶段的数据开始过得了的测试上。这是评测和数据是否对齐的证据。对阶梯更干净的不利证据，仍然是 stage2 / stage3 的 CJK，以及共享窗的 corpus CJK。那里 `scratch_final` 领先或打平，而 `coupled` 在这些后期阶段上并没有像 stage0 exact-match 那样单独塌掉。

### 下一步

在改模型设计之前，先改评测和数据。另一条同类训练臂会继承同一套评测。不要为它租 GPU。

可以离线做、不用新训练的：

1. stage0 exact-match 按技能报。排序里拿掉 `handoff`，也拿掉 `count` 的大部分。真正会动的是 `{where, color}`。头条不再把死技能平均进去。13 个终局重打分的逐技能命中数已经在 `reports/growbrain_e2_taskprobe_stage0_by_skill.csv`。`staged_final` 没有终局的逐题输出，只有合计的 exact-match 和 CJK。把它的两份 `curriculum.pt` 做一次只评测的 CPU 重打分，可以补上这个洞。checkpoint 在 `/workspace/scratch/e2_staged_final/s{1,2}/events/staged_final/seed{1,2}/curriculum.pt`。这一节没有打。2026-11-13 的 CPU 只评测重打分见「按技能、headline 与 first-CJK 的 CPU 只评测重打分」。
2. 官方任务题报告加上 `first_cjk_acc` 和 CJK 区间，并按技能拆开。打分器里已经有零件，官方 JSON 没写出来。不要在 `handoff` 上用 first-CJK 替换 exact-match。stage2 / stage3 要用 first-CJK 或前缀匹配，才能从这些题库得到有梯度的数，而不必改写答案。那也是一次 checkpoint 评测，不是新训练。2026-11-13 的重打分已经把 first-CJK、CJK 区间和按技能拆分写进官方 JSON，并记在后面那一节。
3. 把已提交 manifest 里的 `train_filter.checked` 填上。同阶段的 0 丢弃是 2026-11-10 量的。跨阶段的 0 丢弃是这一节：12 对都是 0。写进仓库时用相对路径。
4. `probe0_acc` 退出毕业、退出排臂、退出「掌握」。以后若还要中途的门，跟任务走的是 `hw0` 或平滑后的 stage0 任务 CJK。`probe0` 不跟。在本节的格式错位修好之前，门仍然不要打开：任务 exact-match 和 CJK 的任何一条线都会把 run 留在 stage 0，而纯 stage 0 上这些线不动。这已经是 mastery 离线估计的结果。
5. 臂间比较若取决于 `who`，或取决于两粒种子的均值，就不要把 stage0 exact-match 当头条。现有题库上若只留一个 stage0 数，用 CJK，或用 `where` 的 exact-match，并把种子区间留在表里。

仍要新的评测或新的数据，但仍然不是一条新的训练臂：

1. 把 stage0 的数据和 stage0 的测试对齐，两条路取一条。或者在 stage0 训练里放入足够密的「问：」/「答：」、从 prompt 抄答案的框，让只练 stage 0 的 run 至少有可能被这套题打分。或者停止用这套题给只练 stage 0 的 checkpoint 打分，另做一套测 stage0 故事里真正有的东西的题。在其中一件发生之前，「课程有没有教会 stage 0」不是一个提得清楚的实验。
2. stage2 / stage3 的 exact-match 要换的是目标：更短的可接受答案，或官方的前缀、first-CJK、span 分数。这些题作为难度是成立的。作为整句 exact-match 的臂间比较，它们没有用。
3. exact-match 若仍要当头条，就增加种子，或事先登记一个技能切片。n=2 吸收不了 `who` 上 90.6% 对 9.4% 的摆动（seed1 对 seed2）。

这些都不需要新的 coupled、staged 或 mastery 训练臂。日志上还剩的、可证伪的训练实验，是最终宽度、只练 stage 0、100000 步，并且打分用的题必须是 stage-0 数据真正教得会的。这是数据与评测的实验。等这套题存在再做。上一节写下的 `stage0_only` 对照，按数据分配会在 stage2 / stage3 的 CJK 上输掉。那一条仍然不建议当成课程实验去跑。没有对齐过的题，那次训练回答不了它想回答的问题。

### 读法

现有数据和评测只能部分区分能力。能站住的是：stage2 / stage3 任务 CJK 上 2.8M 对 18.4M，区间不相交；同宽下 `coupled` 的 stage0 exact-match 1.11% / 0.83% 对 `staged_final` 的 12.78% / 12.50%；stage0 CJK 的粗排序；`where` 的 exact-match 比全技能平均稳。

站不住的是：stage2 / stage3 的 exact-match；probe0–3 的等权平均；stage0 exact-match 头条；`probe0_acc` 的毕业、排序和掌握。0.70 的 `probe0` 线会在任务技能出现之前毕业。

「分阶段在 stage 0 上输给 scratch」记的是评测和数据没有对齐。纯 stage 0 上任务 exact-match 每次都是 0。可用的 exact-match 出现在 stage 2 混入之后。对阶梯不利、而且题和数据还对得上的证据，仍是 stage2 / stage3 的 CJK。

在格式对齐之前，不为另一条同类 growbrain 训练臂租 GPU。

### 限制

`staged_final` 的中途和终局任务题只有合计：exact-match、CJK、n=360、截断数 0。没有逐题或逐技能的输出，所以看不出 stage 1 和 stage 2 分别跳的是哪个技能。两份终局 `curriculum.pt` 在分析机器上。这一节没有打分。终局的按技能、headline 与 first-CJK 已在后面「按技能、headline 与 first-CJK 的 CPU 只评测重打分」补上。中途仍然没有按技能的数。

first-CJK 和 greedy 输出只覆盖 13 个旧臂 run 里的 7 个，没有 `staged_final`，也没有 stage 2 / 3。后面那一节已经用同一批终局检查点，把 first-CJK 补到 15 个终局检查点的 stage 0 / 2 / 3。greedy 自由生成仍然没有。

没有重打 token 129–512，也没有重跑共享窗或 n-gram 的 corpus-probe 审计。

训练重叠用的是官方 200000 篇前缀、官方的 0.5 与 trigram 整句规则，加上子串搜索。没有查验证切分，没有把 corpus probe 或手写库再对一遍任务题，也没有在官方字符 trigram 之外另训一个语言模型。

种子仍是每臂 2 粒，`scratch_final` 是 3 粒。步数对齐，计算量没有对齐，上文已经写过。

`growth_only` 的 `final_stage` 仍是 0，权重是 [0.25, 0.25, 0.25, 0.25]。它不是只练 stage 0 的对照。没有任何一条 run 在 8k–9k 之后仍只练 stage 0。

teacher-forced 的 CJK 抬高，以及偏紧一格的 bootstrap 上界下标，都没有重推。

### 来源

备忘于 2026-11-12 从已有日志离线算出。路径是 `/workspace/scratch/e2_eval_distinguish.md`。工作目录是 `/workspace/scratch/e2_eval_distinguish_work/`。代码和配置是同一台机器上的 growbrain `0d8d8f1`。训练切分、题库和 item 分数不在 git 里。关键表在本节，逐技能计数在 `reports/growbrain_e2_taskprobe_stage0_by_skill.csv`。

## 按技能、headline 与 first-CJK 的 CPU 只评测重打分

2026-11-13（Asia/Shanghai，UTC+8）。在 growbrain `main` `c66007cff1d16f2875406b4de898fa4fb14a1733`（PR [#33](https://github.com/yunshenya/growbrain/pull/33)）上，对已有终局 `curriculum.pt` 做 CPU 只评测重打分。不是新的训练，没有 `--resume`，没有租卡，没有改写原检查点。打分器是 PR #33 一并带上的按技能、headline 与 first-CJK。整库 `exact_match` 与 `cjk_token_acc` 与原先终局点估计逐格相同；headline、first-CJK 和按技能拆分是新数。

命令：

```
python scripts/e2_eval_heldout.py --task-probes-only --eval <arm> <seed> <curriculum.pt> --device cpu --threads 2 --out <report.json>
```

Python 是箱上的 `/workspace/scratch/gbvenv/bin/python`（3.13.5，torch 2.14.1+cpu）。题库是仓库里的 `data/probes/taskmatch/{stage0,stage2,stage3}.jsonl`，360 / 584 / 660 题。仍然没有 stage-1 任务题库。计分是 teacher-forced 答案段 argmax，只计 CJK 字，共享预测窗 128。CJK 与 first-CJK 的 95% 区间是按条目重采样，Hyndman–Fan type 7，1000 次，种子 0。stage 0 的 headline 去掉 `count` 和 `handoff`（224 题）。stage 2 和 stage 3 没有排除，headline 等于整库。官方 JSON 有整库 exact / CJK / first-CJK 的区间，没有 headline 区间；下表的 headline 只报点估计。

`staged_final` 两份终局检查点：seed1 / seed2，参数 18406322（384/10/6，`d_ff` 1024，`max_seq_len` 512），`n_growth` 为 0，`n_graduation` 为 3（全是 `plateau`），终局 `grad_stage` 为 3，步数 100000。毕业步与「staged_final 对照臂」相同：seed1 是 8000 / 15000 / 39000，seed2 是 9000 / 16000 / 37000。另外 13 份旧臂终局检查点与 2026-11-10 的任务题重打分是同一批；`curriculum.pt` 的 sha256 与当时本地清单一致（`scratch_final` 的三粒原先不在那份清单里，这次另记了哈希）。每份官方报告都有 `trained: false`，`resumed_training: false`，`optimizer_steps: 0`，`wrote_training_checkpoint: false`。

墙钟：`staged_final` seed1 从 2026-11-13 02:43:11 CST 起 29.913 秒，seed2 从 02:43:44 CST 起 28.891 秒；13 个旧臂从 02:45:04 到 02:50:27 CST，各次耗时合计 323.163 秒，按模型串行。以上时刻都是 Asia/Shanghai（UTC+8）。

未舍入比例在：

- `reports/growbrain_e2_rescore_stage_summary.csv`
- `reports/growbrain_e2_rescore_stage0_by_skill.csv`

`staged_final` 的 stage 2 / 3 按技能行在 `reports/growbrain_e2_rescore_by_skill.csv`。原始官方 JSON 和备忘仍在分析机器上，不在 git 里：`/workspace/scratch/e2_staged_final_rescore/`，`/workspace/scratch/e2_old_arms_first_cjk_rescore/`。

`probe0_acc` 仍然只作诊断，不进这组排序。它与 stage-0 任务 CJK 反向，见「现有数据与评测能否区分能力（离线核对）」和「`probe0_acc` 与任务分反向」。毕业规则没有改：默认仍是 `mastery_metric=stage_probe`，阈值 0.85。

2026-11-12 那一节写过：`staged_final` 当时没有终局的按技能拆分，也没有 first-CJK，看不出终局各技能落在哪里；first-CJK 当时只覆盖 13 个旧臂里的 7 个，没有 `staged_final`，也没有 stage 2 / 3。这两处终局的洞，这次补上了。中途仍然没有按技能的数：纯 stage 0 窗口的 `curriculum.pt` 没有留在盘上，训练日志的 `task_probes.jsonl` 也只有合计。这次重打分看的是终局技能剖面，不是第一次和第二次毕业时各自抬起来的是哪一个技能。

整库点估计与原先终局完全一致：`staged_final` 对训练日志里的终局 `task_probes`，13 个旧臂对 2026-11-10 的 PR #30 报告，15 次 × 3 个阶段，exact 与 CJK 都没有错位。旧臂部分 95% 区间在小数第四位可能不同，因为这次打分器按 Hyndman–Fan type 7 取百分位；比较用点估计。例如 coupled seed1 stage 0 的 exact 区间，旧文件是 `[0.002777…, 0.022222…]`，这次是 `[0.002777…, 0.022291…]`。

### staged_final 分阶段

百分数，方括号是整库条目 bootstrap 95% 区间。headline 没有官方区间。stage 2 / 3 的整句 exact-match 仍贴近 0（seed1 stage 2 为 1/584，seed2 stage 3 为 1/660，另外两格是 0）。这两阶段比臂用 first-CJK 或 CJK，不用整句 exact-match。

| seed | stage | n | exact [CI] | CJK [CI] | first-CJK [CI] | headline exact | headline CJK |
|---:|---:|---:|---|---|---|---|---|
| 1 | 0 | 360 | 12.78% [9.44, 16.40] | 64.31% [61.92, 66.91] | 44.44% [39.17, 49.72] | 20.54% | 77.08% |
| 1 | 2 | 584 | 0.17% [0.00, 0.51] | 58.55% [57.53, 59.49] | 59.42% [55.31, 63.53] | 0.17% | 58.55% |
| 1 | 3 | 660 | 0.00% [0.00, 0.00] | 44.85% [43.78, 45.86] | 49.24% [45.45, 52.88] | 0.00% | 44.85% |
| 2 | 0 | 360 | 12.50% [9.16, 16.11] | 62.57% [60.41, 64.98] | 56.39% [51.67, 61.67] | 20.09% | 75.26% |
| 2 | 2 | 584 | 0.00% [0.00, 0.00] | 50.16% [49.23, 51.10] | 56.34% [52.05, 60.45] | 0.00% | 50.16% |
| 2 | 3 | 660 | 0.15% [0.00, 0.45] | 42.44% [41.35, 43.43] | 33.18% [29.39, 36.52] | 0.15% | 42.44% |

stage 0 去掉 `count` 和 `handoff` 之后，136 题里 exact 命中是 0。全部 exact 都在剩下的 224 题里，所以 headline exact 是 46/224 = 20.54%（seed1）和 45/224 = 20.09%（seed2），高于整库的 12.78% / 12.50%。CJK 从 64.31% / 62.57% 升到 77.08% / 75.26%。

### staged_final 的 stage 0 按技能

| seed | skill | n | n_exact | exact | CJK | first-CJK |
|---:|---|---:|---:|---:|---:|---:|
| 1 | attribute | 48 | 8 | 16.67% | 73.89% | 35.42% |
| 1 | color | 48 | 20 | 41.67% | 88.33% | 100.00% |
| 1 | count | 64 | 0 | 0.00% | 72.19% | 0.00% |
| 1 | handoff | 72 | 0 | 0.00% | 37.36% | 19.44% |
| 1 | where | 64 | 18 | 28.12% | 81.56% | 100.00% |
| 1 | who | 64 | 0 | 0.00% | 70.09% | 26.56% |
| 2 | attribute | 48 | 0 | 0.00% | 65.29% | 68.75% |
| 2 | color | 48 | 24 | 50.00% | 90.00% | 100.00% |
| 2 | count | 64 | 0 | 0.00% | 63.12% | 0.00% |
| 2 | handoff | 72 | 0 | 0.00% | 39.03% | 47.22% |
| 2 | where | 64 | 21 | 32.81% | 82.50% | 96.88% |
| 2 | who | 64 | 0 | 0.00% | 69.20% | 40.62% |

exact 命中几乎全在 `color` 和 `where`。seed1 的 `attribute` 是 8/48，seed2 是 0/48。`count`、`handoff`、`who` 两粒的 exact 都是 0。`color` 的 first-CJK 两粒都是 100%；`where` 是 100% / 96.88%。`count` 的 CJK 是 72.19% / 63.12%，first-CJK 却是 0：前缀或格式没有对上，不是字重叠为 0。`handoff` 的 first-CJK 高于 exact，与上一节的警告一致：这个技能不要用 first-CJK 替换 exact-match。

### 各臂 stage 0 的 headline 与 first-CJK

headline 去掉 `count` 和 `handoff`。first-CJK 仍是 360 题整库，方括号是区间。`staged_final` 两行来自同一次 SHA 与同一条命令，放在这里只为对照。

| 臂 | seed | 整库 exact [CI] | 整库 CJK [CI] | first-CJK [CI] | headline exact | headline CJK |
|---|---:|---|---|---|---|---|
| graduation_only | 1 | 5.83% [3.33, 8.33] | 56.82% [54.47, 59.44] | 32.78% [28.06, 37.78] | 9.38% | 70.35% |
| graduation_only | 2 | 4.17% [2.22, 6.39] | 50.97% [48.57, 53.22] | 25.28% [20.83, 29.44] | 5.80% | 58.09% |
| growth_only | 1 | 9.72% [6.67, 13.06] | 62.40% [60.50, 64.51] | 55.56% [50.83, 61.11] | 15.62% | 72.69% |
| growth_only | 2 | 3.06% [1.39, 5.28] | 54.87% [52.92, 57.05] | 28.33% [23.61, 33.06] | 4.91% | 64.07% |
| coupled | 1 | 1.11% [0.28, 2.23] | 51.14% [49.37, 52.95] | 37.78% [33.06, 42.78] | 1.34% | 56.51% |
| coupled | 2 | 0.83% [0.00, 1.94] | 45.22% [43.64, 46.80] | 30.56% [26.11, 35.56] | 1.34% | 47.28% |
| decoupled | 1 | 5.28% [3.33, 7.78] | 56.39% [54.09, 58.85] | 29.17% [24.72, 34.17] | 8.48% | 67.78% |
| decoupled | 2 | 5.56% [3.33, 8.06] | 58.59% [56.65, 60.47] | 43.89% [38.89, 48.89] | 8.48% | 63.16% |
| random_order | 1 | 19.17% [15.28, 23.61] | 66.34% [63.80, 69.13] | 46.11% [41.11, 51.67] | 30.80% | 83.28% |
| random_order | 2 | 20.56% [16.67, 24.72] | 67.27% [64.62, 70.13] | 42.78% [38.06, 48.33] | 33.04% | 83.59% |
| scratch_final | 0 | 15.83% [12.22, 20.00] | 67.27% [65.08, 69.77] | 49.72% [45.00, 55.00] | 25.45% | 81.47% |
| scratch_final | 1 | 33.33% [28.33, 38.33] | 70.28% [67.38, 73.51] | 59.72% [54.72, 65.00] | 53.57% | 90.24% |
| scratch_final | 2 | 12.78% [9.44, 16.12] | 66.98% [64.68, 69.39] | 45.00% [39.72, 50.00] | 18.75% | 79.95% |
| staged_final | 1 | 12.78% [9.44, 16.40] | 64.31% [61.92, 66.91] | 44.44% [39.17, 49.72] | 20.54% | 77.08% |
| staged_final | 2 | 12.50% [9.16, 16.11] | 62.57% [60.41, 64.98] | 56.39% [51.67, 61.67] | 20.09% | 75.26% |

`coupled` 的 headline 仍然差：exact 两粒都是 3/224 = 1.34%，CJK 是 56.51% / 47.28%。去掉 `count` 和 `handoff` 几乎没有把它抬起来。seed1 的整库 exact 是 4/360：3 题在 `where`，1 题在 `count`；headline 去掉那 1 题，剩下 3/224。seed2 的 3/360 全在 `color`，headline 仍是 3/224。它的 exact 几乎本来就在 headline 集合里。

`random_order` 的 stage 0 headline 在分阶段和混合臂里最高，exact 是 30.80% / 33.04%，CJK 是 83.28% / 83.59%，高于 `staged_final`，也高于 2.8M 的 `graduation_only`。这与上一节的读法一致：后期问答格式把 stage 0 的分抬起来，不是这条课程本身更强。`random_order` 不是小模型臂，也不是终局仍停在 2.8M 的臂。这次多种子跑里，它在 3 次已应用的 `capacity_saturation` 扩容之后，终局参数是 18406322（约 18.4M）：2819058 → 5768530 → 10829106 → 18406322（约 2.8M → 5.77M → 10.8M → 18.4M）。只有 `graduation_only` 停在 2819058，`n_grow_applied` 为 0。`coupled`、`decoupled`、`growth_only` 和 `random_order` 的终局都是 18406322。`scratch_final` seed 1 的 headline 可以更高，来自 `who` 的种子摆动。

`scratch_final` 的 headline exact 仍在摆：25.45% / 53.57% / 18.75%（seed 0 / 1 / 2）。去掉 `count` 和 `handoff` 之后，历史里的 `who` 摆动还在 headline 里。`who` 的 exact 是 11/64、58/64、6/64（17.19% / 90.62% / 9.38%）。headline 不再把死技能平均进去，但还不是一条干净的头条。`attribute` 也不稳：scratch 是 2/48、12/48、0/48；`staged_final` 是 8/48 与 0/48。

### 各臂 stage 2 / 3 的 first-CJK 与 CJK

headline 等于整库。不用整句 exact-match 排序。stage 2 的 exact 在 13 个旧臂上都是 0；`staged_final` seed1 是 1/584，seed2 是 0。stage 3 的 exact 仍是 0–0.76%。

| 臂 | seed | s2 CJK [CI] | s2 first-CJK [CI] | s3 CJK [CI] | s3 first-CJK [CI] |
|---|---:|---|---|---|---|
| graduation_only | 1 | 36.45% [35.59, 37.29] | 15.75% [12.84, 18.66] | 31.85% [31.10, 32.55] | 20.15% [17.27, 22.88] |
| graduation_only | 2 | 32.95% [32.01, 33.83] | 4.11% [2.57, 5.65] | 31.33% [30.60, 32.07] | 22.27% [19.09, 25.45] |
| growth_only | 1 | 49.03% [47.97, 50.09] | 44.35% [39.90, 48.46] | 42.98% [42.04, 43.97] | 56.67% [52.73, 60.16] |
| growth_only | 2 | 47.51% [46.47, 48.55] | 27.74% [23.97, 31.34] | 37.21% [36.20, 38.23] | 8.33% [6.36, 10.61] |
| coupled | 1 | 50.54% [49.34, 51.63] | 42.64% [38.52, 46.58] | 48.45% [47.42, 49.48] | 54.55% [50.76, 58.18] |
| coupled | 2 | 48.47% [47.06, 49.76] | 28.94% [25.00, 32.53] | 42.49% [41.30, 43.49] | 27.27% [23.79, 30.31] |
| decoupled | 1 | 51.83% [50.85, 52.76] | 30.31% [26.37, 33.91] | 41.55% [40.43, 42.60] | 37.58% [33.79, 41.21] |
| decoupled | 2 | 46.80% [45.70, 47.80] | 43.49% [39.21, 47.43] | 42.63% [41.64, 43.59] | 44.24% [40.60, 47.73] |
| random_order | 1 | 56.12% [54.98, 57.36] | 69.86% [66.10, 73.46] | 44.06% [43.02, 45.01] | 55.00% [51.21, 58.49] |
| random_order | 2 | 48.05% [47.04, 49.08] | 19.86% [16.61, 23.29] | 43.17% [42.15, 44.10] | 42.73% [38.94, 46.21] |
| scratch_final | 0 | 59.58% [58.49, 60.75] | 54.28% [50.34, 58.22] | 48.97% [47.92, 49.98] | 51.67% [48.03, 55.30] |
| scratch_final | 1 | 58.23% [56.81, 59.54] | 55.99% [51.54, 59.94] | 44.56% [43.29, 45.78] | 42.27% [38.48, 45.91] |
| scratch_final | 2 | 56.28% [55.00, 57.42] | 41.61% [37.16, 45.72] | 49.28% [48.26, 50.23] | 56.97% [53.18, 60.61] |
| staged_final | 1 | 58.55% [57.53, 59.49] | 59.42% [55.31, 63.53] | 44.85% [43.78, 45.86] | 49.24% [45.45, 52.88] |
| staged_final | 2 | 50.16% [49.23, 51.10] | 56.34% [52.05, 60.45] | 42.44% [41.35, 43.43] | 33.18% [29.39, 36.52] |

`staged_final` 的 stage 2 first-CJK 是 59.42% / 56.34%，与 `scratch_final` 的 54.28% / 55.99% / 41.61% 大致相当，两粒都高于 `graduation_only` 的 15.75% / 4.11%。stage 3 的 first-CJK，seed1 是 49.24%，靠近 `scratch_final` 的区间；seed2 是 33.18%，偏弱。stage 2 / 3 的 first-CJK 仍然高度集中在技能上，常见接近 0 或接近 100。CJK 上，2.8M 的 `graduation_only` 与 18.4M 各臂仍然分开，区间不相交，与上一节用整库 CJK 写下的结论相同。

### 读法

宽度仍然解释了 `staged_final` 对 `graduation_only` 的大部分缺口。课程顺序相同、步数都是 100000 时，18.4M 在 stage 2 / 3 的 CJK 和 first-CJK 上都明显高于 2.8M。stage 0 的 headline CJK 也是 `staged_final` 的 77.08% / 75.26% 高于 `graduation_only` 的 70.35% / 58.09%。这与「staged_final 对照臂」里用整库 stage 0 CJK 算过的「缺口大约三分之二来自宽度和上下文」并不冲突；这次没有重算那一组差的比。

在相同的最终宽度和 100000 步下，分阶段的 `staged_final` 终局仍然没有整体超过 `scratch_final`。stage 0 headline CJK 是 77.08% / 75.26%，低于 scratch 的 81.47% / 90.24% / 79.95%。headline exact 是 20.54% / 20.09%，贴着 scratch 较低的两粒（25.45% / 18.75%），离 seed1 的 53.57% 很远。stage 0 first-CJK 是 44.44% / 56.39%，与 scratch 的 49.72% / 59.72% / 45.00% 有重叠，不是分阶段获胜。stage 2 first-CJK 与 scratch 相当。stage 3 的 seed2 first-CJK 偏弱。

stage 0 的整句 exact-match，即使去掉 `handoff` 和 `count`，仍不是一条干净的头条。会动的主要是 `color` 和 `where`。`who` 和 `attribute` 还有种子残差。`scratch_final` 的 headline exact 摆动仍然主要来自 `who`。

`random_order` 较高的 stage 0 headline，不应读成课程更强。上一节已经写过：纯 stage 0 上这套题的 exact-match 是 0，可用的 exact 出现在后续阶段混入之后。`random_order` 更早、更均匀地见到后期问答格式。

`coupled` 在 stage 0 任务题上仍然弱。headline exact 1.34% / 1.34%，headline CJK 56.51% / 47.28%，仍低于同宽、不扩容的 `staged_final`。同宽下把 `coupled` 和还在工作的臂分开，这一条仍然成立。

评测和数据对齐之前，不为另一条同类的 growbrain 训练臂租 GPU。这一节没有改题库，也没有改训练集，也没有改毕业门。

同一批终局检查点上，另有一份本地的 stage-0 训练格式诊断库重打分，见「本地 stage-0 训练格式诊断题库的 CPU 只评测重打分」。那一份没有写入 growbrain，也不替换这一节的官方「问：/答：」题库。

### 限制

没有重打 token 129–512，也没有重跑共享窗或 n-gram 的 corpus-probe 审计。corpus `probe0–3` 这次没有重打（`--task-probes-only`）。

没有中途检查点，补不上「stage 1 和 stage 2 分别抬起哪个技能」。终局按技能表不能代替那条轨迹。

官方报告没有 headline 的 bootstrap 区间。`bootstrap` 仍是条目重采样；teacher-forced CJK 会计入答案内部搭配，这一节仍然没有自由生成分数。

种子仍是每臂 2 粒，`scratch_final` 是 3 粒。步数对齐，计算量没有对齐，上文已经写过。`growth_only` 的 `final_stage` 仍是 0，权重是 [0.25, 0.25, 0.25, 0.25]，不是只练 stage 0 的对照。盘上仍然没有一份在 8k–9k 之后还只练 stage 0 的检查点。

仍然没有 stage-1 任务题库。

### 来源

官方 JSON：`/workspace/scratch/e2_staged_final_rescore/staged_final_seed{1,2}.json`，以及 `/workspace/scratch/e2_old_arms_first_cjk_rescore/out/<arm>_seed<seed>.json`（13 份）。备忘是这两个目录里的 `MEMO.md`。代码是箱上的 growbrain `c66007cff1d16f2875406b4de898fa4fb14a1733`，工作树未改。对照用的旧整库分数：`staged_final` 来自训练日志，13 个旧臂来自 `/workspace/scratch/e2_task_probes/out/`。未舍入表在 `reports/growbrain_e2_rescore_stage_summary.csv`、`reports/growbrain_e2_rescore_stage0_by_skill.csv` 和 `reports/growbrain_e2_rescore_by_skill.csv`。

## 本地 stage-0 训练格式诊断题库的 CPU 只评测重打分

2026-11-13（Asia/Shanghai，UTC+8）。对同一批 15 个终局 `curriculum.pt`，用与上一节相同的 PR [#33](https://github.com/yunshenya/growbrain/pull/33) 打分器（teacher-forced 答案段 argmax，exact / CJK / first-CJK，只计 CJK 字，共享预测窗 128；区间是条目重采样，Hyndman–Fan type 7，1000 次，种子 0），在 CPU 上做只评测重打分。不是新的训练，没有 `--resume`，没有租卡，没有改 growbrain 仓库或原检查点。growbrain 仍是 `main` `c66007cff1d16f2875406b4de898fa4fb14a1733`，工作树未改。

这一节用的是本地诊断题库，没有写入 growbrain，也不是官方「问：/答：」转移题库的替代品。官方 `data/probes/taskmatch/` 和毕业门都没有改。题库本身仍留在分析机器上，不进 git；这里只提交分数表。格式贴近 stage 0 训练里的 46 篇 `template_qa_dialogue`：`<user>` / `<bot>` / `<eot>`，可选前文。没有强制「问：/答：」，没有 handoff 改写金答案，也没有「在谁手里」。槽位是新的，不是那 46 条原句对，也不是官方 360 题的原句。

未舍入比例在：

- `reports/growbrain_e2_diag_bank_v1_stage_summary.csv`
- `reports/growbrain_e2_diag_bank_v1_by_skill.csv`
- `reports/growbrain_e2_diag_bank_v2_stage_summary.csv`
- `reports/growbrain_e2_diag_bank_v2_by_skill.csv`

### v1

32 题：颜色、计数各 8 题闭卷问答；处所、人物各 8 题开卷对话阅读。头条不排除任何技能。墙钟 2026-11-13 03:11:45–03:12:09 CST，约 24 秒，15/15 成功，无截断。

整库 exact 是 0–2/32。15 个检查点上一共 9 次命中，全部在开卷处所或人物；闭卷颜色+计数 exact 是 0/240。CJK 约 54.17%–66.15%，臂间重叠大。`random_order` 是 0/32 与 1/32，并不领先；它在上一节官方 stage 0 headline 上是 30.80% / 33.04%。官方题库上的优势没有转到这份训练格式库。

前缀仍会塌。计数金答案是训练形态「有N个X。」，部分臂的 first-CJK 是 7–8/8，`scratch_final` 与 `random_order` 是 0/8。颜色常是全 0 或全 8：有的臂以「这」起头，greedy 却是默认色或复述问句（`这是红色的。`、`这是什么颜色？`）；有的丢掉「这是」（`红色的。`、`绿色的。`）；有的是对话敷衍（`你猜。`、`我也不知道。`）。处所金答案以实体起头，不少输出以「在」起头，其中一部分是套话 `在这里。`。人物金答案以人名起头，模型常从「你／我」起头（`你猜。`、`我在河边。`）。v1 人物问句是「谁在…？」，那 46 条训练问答里没有这个框；训练里有的是「X在做什么？」。

闭卷颜色和计数用的是新槽，上下文里没有事实。stage 0 预训练不可能知道这些具体颜色或个数。闭卷 exact 贴在地板上是预期，不是金答案少了句号。不把 canonical 金答案放宽成各臂实际发出的第一个字：那样会抬一部分臂的 first-CJK，同时把走训练框「这是…／有…」的臂判错。first-CJK 只看 canonical。处所也不把金答案改成「在Y。」，因为第一字在实体照抄和「在」之间是分裂的，而且不少「在」并不是对的处所。

### v2

最小修订，仍是同一套新槽。头条的颜色和计数改为开卷：前文的 `<bot>` 里先出现训练形态的事实（「这是C的N。」、「有N个X。」），再问同一技能，金答案就是那句，包括句末「。」。人物问句改为训练框「X在做什么？」；金答案和前文事实不变。处所与 v1 相同。另留 4+4 道与 v1 前四题同题同金的闭卷对照，技能名是 `color_closed` 和 `count_closed`，排除在头条之外。合计 40 题，头条 32 题。没有把金答案改成「红色的。」或「一个X。」来刷分，也没有把题做成那 46 条训练原句。

墙钟 2026-11-13 03:20:50–03:21:00 CST，约 10 秒（单进程一次跑完），15/15 成功，无截断。处所 8 题与 v1 逐条一致。闭卷对照与 v1 对应的 0001–0004 逐条一致。

闭卷对照 exact 是 0/120，与「没有上下文就不可答」一致。开卷 exact 合计 34/480。头条 exact 是 0–8/32，高于 v1 的 0–2/32，仍然低。最高是 `decoupled` seed2 的 8/32（颜色 5、计数 2、处所 1）。`staged_final` seed1 与 `scratch_final` seed0 是 5/32。五个检查点仍是 0/32，包括 `random_order` 的两个种子。全体 40 题 CJK 约 55.74%–72.34%（`random_order` seed1 至 `staged_final` seed1），头条 32 题 CJK 约 58.85%–79.17%，臂间仍重叠。

下表的 headline 是 32 道开卷题。全体 40 题的 CJK / first-CJK，以及闭卷对照，在 `reports/growbrain_e2_diag_bank_v2_stage_summary.csv`。按技能未舍入在 `reports/growbrain_e2_diag_bank_v2_by_skill.csv`。

| 臂 | seed | headline exact | headline first-CJK | headline CJK |
|---|---:|---:|---:|---|
| graduation_only | 1 | 0/32 | 5/32 | 58.85% |
| graduation_only | 2 | 3/32 | 16/32 | 65.62% |
| growth_only | 1 | 3/32 | 22/32 | 72.40% |
| growth_only | 2 | 0/32 | 11/32 | 69.79% |
| coupled | 1 | 1/32 | 14/32 | 70.31% |
| coupled | 2 | 3/32 | 16/32 | 68.23% |
| decoupled | 1 | 0/32 | 7/32 | 59.90% |
| decoupled | 2 | 8/32 | 21/32 | 76.56% |
| random_order | 1 | 0/32 | 19/32 | 59.90% |
| random_order | 2 | 0/32 | 3/32 | 69.27% |
| scratch_final | 0 | 5/32 | 29/32 | 79.17% |
| scratch_final | 1 | 3/32 | 12/32 | 73.44% |
| scratch_final | 2 | 2/32 | 11/32 | 71.35% |
| staged_final | 1 | 5/32 | 18/32 | 74.48% |
| staged_final | 2 | 1/32 | 5/32 | 64.58% |

前缀坍缩有所缓解，但没有消失，也不是单向变好。计数的 first-CJK，v1 里 `scratch_final` 与 `random_order` 是 0/8；开卷后 `scratch_final` seed0 / seed2 到 8/8，`random_order` 到 5/8 与 2/8。人物改用「X在做什么？」之后，`staged_final` seed1 与 `random_order` seed1 的 first-CJK 从 0/8 到 7/8，不再一律从「你／我」起头。颜色也有升有降：`scratch_final` seed0 的 first-CJK 从 0/8 到 7/8，`decoupled` seed2 从 0/8 到 8/8，但 `decoupled` seed1 从 8/8 降到 3/8，`growth_only` seed2 的计数从 4/8 降到 0/8。first-CJK 变高常常只是开头对齐了训练框，后面的数字、颜色或名词仍错。处所的「在」与实体分裂还在，与 v1 相同。

`random_order` 仍然不占优：两个种子的头条 exact 都是 0/32。臂之间比 v1 更容易分开：可以把 `decoupled` seed2 的 8/32、`staged_final` seed1 / `scratch_final` seed0 的 5/32，与五个 0/32 分开。多数臂仍挤在 0–3/32。种子差很大：`decoupled` 是 0/32 与 8/32，`staged_final` 是 5/32 与 1/32。这些终局分不能当成臂的稳定排序，也还不够用来解释课程。

### 限制

题量小。闭卷对照只有 8 题，每个开卷技能 8 题。报的是原始命中数，不要把 8/32 读成稳定的能力分。

终局检查点都见过后续阶段。盘上没有纯 stage 0 窗口的权重。这份库上的终局分不能证明「只练 stage 0 就会了」。

计分仍是 teacher-forced，不是自由生成的整句匹配。first-CJK 只看 canonical 金答案。

没有继续改前缀或金答案去贴各臂的习惯说法。再改下去容易变成给终局检查点的癖好拟合，而不是诊断题库。

仍然没有 stage-1 任务题库。官方转移题库和毕业门未改。

### 读法

与「现有数据与评测能否区分能力（离线核对）」里的对齐判断相容。官方「问：/答：」题库仍应读成后期格式和迁移测验，不是 stage-0 能力的干净头条。`random_order` 在官方 stage 0 headline 上领先、在这份训练格式库上并不领先，是那一读法的直接证据。

把颜色和计数改成开卷阅读之后，训练格式库上出现了一些非地板的 exact，说明「前文里已经有训练形态的事实，再问同一技能」是测得动的。这仍是诊断信号，不是结业考，也还不够解释课程。

不要在此基础上租 GPU，也不要另开一条同类的训练臂。

### 来源

题库、备忘和原始 JSON 在分析机器上，不在 git 里：`/workspace/scratch/e2_stage0_diag_bank_draft/v1/`，`/workspace/scratch/e2_stage0_diag_bank_draft/v2/`。根目录的 `stage0.jsonl` 和 `out/` 是冻结的 v1 副本。包装器是同目录的 `run_diag_eval.py`，只打这份 stage 0 库。检查点 sha256 与上一节官方题库重打分一致。代码是箱上的 growbrain `c66007cff1d16f2875406b4de898fa4fb14a1733`，工作树未改。未舍入表在 `reports/growbrain_e2_diag_bank_v1_stage_summary.csv`、`reports/growbrain_e2_diag_bank_v1_by_skill.csv`、`reports/growbrain_e2_diag_bank_v2_stage_summary.csv` 和 `reports/growbrain_e2_diag_bank_v2_by_skill.csv`。
