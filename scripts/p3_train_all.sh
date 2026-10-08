#!/bin/bash
# Phase-3 training queue (CPU, nice 19). Three lanes run in parallel with 2 threads each; every run is skipped if its
# checkpoint exists, so the queue can be restarted after an interruption.
cd "$(dirname "$0")/.."; mkdir -p logs/p3
run(){ tag=$1; shift; [ -f models/$tag.pt ] && return; echo "$(date +%T) start $tag"; TIDAL_THREADS=2 OMP_NUM_THREADS=2 nice -n 19 ionice -c3 .venv/bin/python -u -W ignore -m tidal.vap $tag --dataset p3 "$@" > logs/p3/$tag.log 2>&1; echo "$(date +%T) done $tag $(grep best_val logs/p3/$tag.log)"; }
PUB=tg,irc,twitch,candor,aishell4
lane1(){ run P3pub_s0 --data $PUB --val pub --pre-epochs 10 --epochs 20 --patience 4
         for s in 0 1 2; do run P3ft_s$s --data real,llm --init P3pub_s0 --seed $s; done; }
lane2(){ run P3mix_s0 --data real,llm,$PUB --seed 0 --pre-epochs 10 --epochs 30 --patience 5
         run P3mix_s1 --data real,llm,$PUB --seed 1 --pre-epochs 10 --epochs 30 --patience 5; }
lane3(){ run P3L_nolive_s0 --data real,llm,tg,irc,candor,aishell4 --seed 0 --pre-epochs 10 --epochs 30 --patience 5
         run P3mix_s2 --data real,llm,$PUB --seed 2 --pre-epochs 10 --epochs 30 --patience 5; }
lane1 & lane2 & lane3 & wait
echo ALLDONE
