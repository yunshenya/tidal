#!/bin/bash
# control for P4emo: same 11 extra columns but only the has-text flag (emotion values zeroed)
cd "$(dirname "$0")/.."
for s in 0 1 2; do tag=P4emoflag_s$s; [ -f models/$tag.pt ] && continue
  TIDAL_THREADS=2 OMP_NUM_THREADS=2 nice -n 10 .venv/bin/python -u -W ignore -m tidal.vap $tag --dataset p3 --data real,llm --init P3pub_s0 --seed $s --extra emoflag > logs/p4/$tag.log 2>&1
  echo "$(date +%T) done $tag $(grep best_val logs/p4/$tag.log)"; done
