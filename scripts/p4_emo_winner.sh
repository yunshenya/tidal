#!/bin/bash
# rule A on the 4B winner: <kind> + text-emotion features (same protocol as P4emo, init from P4pub_<kind>)
cd "$(dirname "$0")/.."; k=$1
for s in 0 1 2; do tag=P4emo_${k}_s$s; [ -f models/$tag.pt ] && continue
  TIDAL_THREADS=2 OMP_NUM_THREADS=2 nice -n 10 .venv/bin/python -u -W ignore -m tidal.vap $tag --dataset p3 --kind $k --data real,llm --init P4pub_$k --seed $s --extra emo > logs/p4/$tag.log 2>&1
  echo "$(date +%T) done $tag $(grep best_val logs/p4/$tag.log)"; done
