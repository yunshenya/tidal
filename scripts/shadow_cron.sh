#!/usr/bin/env bash
# One shadow-mode tick, as run by cron. Light: 1 thread, nice 10, idle I/O class, hard 10-min timeout, single instance.
set -uo pipefail
cd "$(dirname "$0")/.."
umask 077
mkdir -p shadow/logs shadow/state && chmod 700 shadow/state
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 TIDAL_THREADS=1 PYTHONWARNINGS=ignore
LOG=shadow/logs/shadow.log
# keep the log small: rotate at 5 MB, keep one old copy
if [ -f "$LOG" ] && [ "$(stat -c %s "$LOG")" -gt 5242880 ]; then mv -f "$LOG" "$LOG.1"; fi
exec 9>shadow/state/cron.lock
flock -n 9 || { echo "$(date '+%F %T') previous tick still running; skip" >> "$LOG"; exit 0; }
nice -n 10 ionice -c3 timeout 600 .venv/bin/python -m tidal.shadow.run "$@" >> "$LOG" 2>&1
