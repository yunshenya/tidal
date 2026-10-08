#!/bin/bash
# Phase-4A integration A/B: identical to phase-3 P3ft_s{0,1,2} (init P3pub_s0, real+llm, same seeds) + 11 emotion columns.
cd "$(dirname "$0")/.."; mkdir -p logs/p4
for s in 0 1 2; do tag=P4emo_s$s; [ -f models/$tag.pt ] && continue
  TIDAL_THREADS=2 OMP_NUM_THREADS=2 nice -n 10 .venv/bin/python -u -W ignore -m tidal.vap $tag --dataset p3 --data real,llm --init P3pub_s0 --seed $s --extra emo > logs/p4/$tag.log 2>&1
  echo "$(date +%T) done $tag $(grep best_val logs/p4/$tag.log)"; done
