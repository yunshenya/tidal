#!/usr/bin/env bash
# Full phase-1 pipeline after synthetic generation finished. CPU-friendly: <=4 torch threads total.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
export TIDAL_THREADS=2 OMP_NUM_THREADS=2
$PY -m tidal.synth_clean
$PY -m tidal.dataset > logs/dataset.log
$PY -m tidal.baselines T > logs/baselines_T.log
$PY -m tidal.baselines TS > logs/baselines_TS.log
train() { $PY -m tidal.train "$@" > "logs/train_$5.log" 2>&1; }
for R in T TS; do
  # main ablation: GRU real-only vs real+synth, 3 seeds each (2 jobs in parallel x 2 threads)
  for S in 0 1 2; do
    train $R real gru 0.5 ${R}_real_gru_s$S $S &
    train $R real+synth gru 0.5 ${R}_rs_gru_s$S $S &
    wait
  done
  # synth weight sweep + transformer variant (1 seed)
  train $R real+synth gru 0.2 ${R}_rs02_gru_s0 0 &
  train $R real+synth gru 1.0 ${R}_rs10_gru_s0 0 &
  wait
  train $R real tf 0.5 ${R}_real_tf_s0 0 &
  train $R real+synth tf 0.5 ${R}_rs_tf_s0 0 &
  wait
done
echo TRAIN_DONE
