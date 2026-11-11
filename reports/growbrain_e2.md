# Growbrain E2 results

Two finished curriculum runs. Budget 100000 steps, eval every 1000 steps, CUDA on a CompShare RTX 4090. The scratch_final report records `gpu_name` `NVIDIA GeForce RTX 4090`, torch `2.11.0+cu128`, CUDA `12.8`. This page keeps that training record. The true-stage re-evaluation re-scores the same checkpoints eval-only. The shared-span section scores every arm on the same 128-token prefix, CJK tokens only. Task probes are a separate bank, in 「任务题正式对照（task probes）」. The later `staged_final` control is in 「staged_final 对照臂」. It is not part of the two curriculum archives or the 13-checkpoint rescore. Stage0 memory scores stay in `reports/growbrain_stage0_fit.json` and `reports/growbrain_stage0_held.json`.

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
