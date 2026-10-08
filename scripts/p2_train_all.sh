#!/bin/bash
# Phase-2 training queue: single thread, lowest CPU/IO priority so the shadow cron always wins.
cd "$(dirname "$0")/.."; mkdir -p logs/p2
run(){ tag=$1; shift; [ -f models/$tag.pt ] && return; echo "$(date +%T) start $tag"; TIDAL_THREADS=1 OMP_NUM_THREADS=1 nice -n 19 ionice -c3 .venv/bin/python -u -W ignore -m tidal.vap $tag "$@" > logs/p2/$tag.log 2>&1; echo "$(date +%T) done $tag $(grep best_val logs/p2/$tag.log)"; }
for s in 0 1 2; do
  run P2a_s$s --data real,llm --seed $s
  run P2n_s$s --data real,llm --seed $s --no-pretrain
  run P2b_s$s --data real,llm,live,1on1 --seed $s
done
run P2L_nolive_s0 --data real,llm,1on1 --seed 0
run P2L_no1on1_s0 --data real,llm,live --seed 0
run P2L_synonly_s0 --data llm,live,1on1 --seed 0 --val syn
run P2L_nogroup_s0 --data live,1on1 --seed 0 --val syn
echo ALLDONE
