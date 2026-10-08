#!/bin/bash
# Phase-4B backbone queue: identical protocol to phase-3 P3pub_s0 -> P3ft_s{0,1,2} (GRU reused from phase 3).
# Restartable: a run is skipped if its checkpoint exists.
cd "$(dirname "$0")/.."; mkdir -p logs/p4
run(){ th=$1; tag=$2; shift 2; [ -f models/$tag.pt ] && return; echo "$(date +%T) start $tag"; TIDAL_THREADS=$th OMP_NUM_THREADS=$th nice -n 19 ionice -c3 .venv/bin/python -u -W ignore -m tidal.vap $tag --dataset p3 "$@" > logs/p4/$tag.log 2>&1; echo "$(date +%T) done $tag $(grep best_val logs/p4/$tag.log)"; }
PUB=tg,irc,twitch,candor,aishell4
bb(){ th=$1; k=$2; run $th P4pub_$k --kind $k --data $PUB --val pub --pre-epochs 10 --epochs 20 --patience 4
      for s in 0 1 2; do run $th P4ft_${k}_s$s --kind $k --data real,llm --init P4pub_$k --seed $s; done; }
( bb 3 mamba3 ) & ( bb 2 tx_kv; bb 2 m3_ablate_m2 ) & ( bb 2 mamba3_siso ) & wait
echo ALLDONE
