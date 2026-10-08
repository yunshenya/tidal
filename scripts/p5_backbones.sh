#!/bin/bash
# Phase-5 backbone bake-off. Fresh tags. Skip only if THIS run's checkpoint exists.
cd /workspace/tidal
mkdir -p logs/p5
run(){
  th=$1; tag=$2; shift 2
  if [ -f models/$tag.pt ]; then echo "$(date +%T) skip $tag"; return; fi
  echo "$(date +%T) start $tag"
  TIDAL_THREADS=$th OMP_NUM_THREADS=$th nice -n 19 ionice -c3 .venv/bin/python -u -W ignore -m tidal.vap "$tag" --dataset p3 "$@" > logs/p5/$tag.log 2>&1
  echo "$(date +%T) done $tag $(grep best_val logs/p5/$tag.log | tail -1)"
}
PUB=tg,irc,twitch,candor,aishell4
bb(){
  th=$1; k=$2
  run $th P5bb_pub_$k --kind $k --data $PUB --val pub --pre-epochs 10 --epochs 20 --patience 4
  for s in 0 1 2; do
    run $th P5bb_ft_${k}_s$s --kind $k --data real,llm --init P5bb_pub_$k --seed $s
  done
}
bb 3 mamba3_siso &
bb 3 m3_ablate_m2 &
wait
echo ALLDONE
