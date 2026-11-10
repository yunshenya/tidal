# Growbrain E2 results

Two finished curriculum runs. Budget 100000 steps, eval every 1000 steps, CUDA on a CompShare RTX 4090. The scratch_final report records `gpu_name` `NVIDIA GeForce RTX 4090`, torch `2.11.0+cu128`, CUDA `12.8`. This page keeps that training record. The true-stage re-evaluation re-scores the same checkpoints eval-only. The shared-span section scores every arm on the same 128-token prefix, CJK tokens only. Stage0 memory scores stay in `reports/growbrain_stage0_fit.json` and `reports/growbrain_stage0_held.json`.

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

`scratch_final` true-stage scores are identical to its training-time scores, so the earlier scratch comparison stands on those official columns. `random_order`'s earlier shuffled pseudo-stage scores must not be compared to these columns.

Only 2 or 3 seeds. The runs share a step budget, not a compute budget: equal steps, not equal compute. Tokens 129–512 were not scored separately. The equal-weight mean of probe0–3 is still not a capability measure.

### Evaluation limits

Probe audit, 2026-11-10, eval-only, growbrain `main` at `48ad257`. The equal-weight mean of probe0–3 cannot distinguish capability at the 1–2 point gaps in this table.

Probe accuracy is character-level next-token top-1, including markup. probe0 is 42% markup tokens (41.7% special markup in the audit's token-class mix).

A same-stage character trigram gets close to the model scores. probe0 trigram .557 against 18M-model probe0 about .62–.68. probe3 trigram .382 against model probe3 about .42–.44.

`graduation_only` is scored at context 128 (2819058 parameters). The other arms are scored at context 512 (18406322 parameters). Short prefixes are easier: a trigram on probe2 is .493 at L=128 and .420 at L=512. `graduation_only` probe2 is not comparable to the 512-context arms.

probe0 has 87.8% character 8-gram overlap with stage-0 train.

handwritten2 and handwritten3, 24 items each, do separate the 2.8M model from the 18.4M models. Those banks are tiny, and they are not the headline.

The shared-span L=128 CJK rescore is the table above. An expanded handwritten exact-match probe set is still in progress. Tokens 129–512 were not scored separately.
