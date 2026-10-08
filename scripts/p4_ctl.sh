#!/bin/bash
# re-run of the phase-3 P3ft protocol with the current code (control for P4emo; checks reproducibility)
cd "$(dirname "$0")/.."
while pgrep -f "scripts/p4_emo.sh" >/dev/null; do sleep 20; done
for s in 0 1 2; do tag=P4ctl_s$s; [ -f models/$tag.pt ] && continue
  TIDAL_THREADS=2 OMP_NUM_THREADS=2 nice -n 10 .venv/bin/python -u -W ignore -m tidal.vap $tag --dataset p3 --data real,llm --init P3pub_s0 --seed $s > logs/p4/$tag.log 2>&1
  echo "$(date +%T) done $tag $(grep best_val logs/p4/$tag.log)"; done
