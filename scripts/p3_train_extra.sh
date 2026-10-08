#!/bin/bash
# Phase-3 extra runs (added after the gated datasets arrived): danmaku in-domain reference + AI-streamer fine-tune.
# Waits for a free lane (<3 tidal.vap jobs), skips runs whose checkpoint exists.
cd "$(dirname "$0")/.."
run(){ tag=$1; shift; [ -f models/$tag.pt ] && return; echo "$(date +%T) start $tag"; TIDAL_THREADS=2 OMP_NUM_THREADS=2 nice -n 19 ionice -c3 .venv/bin/python -u -W ignore -m tidal.vap $tag --dataset p3 "$@" > logs/p3/$tag.log 2>&1; echo "$(date +%T) done $tag $(grep best_val logs/p3/$tag.log)"; }
wait_lane(){ while [ "$(pgrep -fc 'm tidal.vap')" -ge 3 ]; do sleep 30; done; }
PUB=tg,irc,twitch,candor,aishell4
wait_lane; run P3dm_s0 --data real,llm,$PUB,danmaku --seed 0 --pre-epochs 10 --epochs 30 --patience 5
while [ ! -f models/P3mix_s0.pt ]; do sleep 30; done
wait_lane; run P3ai_s0 --data ai,real,llm --init P3mix_s0 --seed 0 --epochs 10 --patience 3
echo EXTRADONE
