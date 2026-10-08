#!/bin/bash
# phase 3 audio chain: oto prep -> oto pretrain -> (wait for MagicData CV) -> oto eval (MagicData ft + Krisp)
cd "$(dirname "$0")/.."; export TIDAL_THREADS=2 OMP_NUM_THREADS=2; L=logs/p3/audio_chain.log
log(){ echo "$(date +%H:%M:%S) $*" >> $L; }
while pgrep -f "audio_oto prep" >/dev/null; do sleep 20; done
[ -f models/audio_fe_oto.pt ] || { log start oto pretrain; nice -n 10 .venv/bin/python -u -m tidal.audio_oto pretrain >> logs/p3/audio_oto.out 2>&1; log done pretrain; }
while pgrep -f "audio_train cv" >/dev/null; do sleep 30; done
log start eval; nice -n 10 .venv/bin/python -u -m tidal.audio_oto eval >> logs/p3/audio_oto.out 2>&1; log done eval rc=$?
