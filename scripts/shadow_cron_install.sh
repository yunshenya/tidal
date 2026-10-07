#!/usr/bin/env bash
# Install / remove the shadow-mode cron entries (user crontab). usage: shadow_cron_install.sh [install|remove|status]
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MARK="# tidal-shadow"
cur="$(crontab -l 2>/dev/null | grep -v "$MARK" || true)"
case "${1:-install}" in
  install)
    { [ -n "$cur" ] && echo "$cur"
      echo "3-59/10 * * * * $ROOT/scripts/shadow_cron.sh $MARK"
      echo "17 9 * * * $ROOT/scripts/shadow_report_cron.sh $MARK"; } | crontab -
    pgrep -x cron >/dev/null || echo "WARNING: cron daemon not running (start it with: sudo cron)";;
  remove) if [ -n "$cur" ]; then echo "$cur" | crontab -; else crontab -r 2>/dev/null || true; fi;;
  status) crontab -l 2>/dev/null | grep "$MARK" || echo "not installed"; pgrep -ax cron || echo "cron daemon not running";;
esac
