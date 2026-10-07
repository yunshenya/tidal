#!/usr/bin/env bash
# Daily shadow report (markdown + json) into shadow/reports/ (gitignored).
set -uo pipefail
cd "$(dirname "$0")/.."
umask 077
mkdir -p shadow/reports
export OMP_NUM_THREADS=1 PYTHONWARNINGS=ignore
D=$(date +%F)
nice -n 10 timeout 900 .venv/bin/python -m tidal.shadow.report --out "shadow/reports/report-$D.md" --json "shadow/reports/report-$D.json" > /dev/null 2>> shadow/logs/report.log
cp -f "shadow/reports/report-$D.md" shadow/reports/latest.md 2>/dev/null || true
