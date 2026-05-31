#!/usr/bin/env bash
#
# weekly_source_rescore.sh — Monday-morning re-score of every source on
# every sport. Idempotent. Cron-safe.
#
# What it does:
#   1. cd to the repo root (resolved relative to this script's location)
#   2. ensure data/source_history.db exists (run backfills if missing)
#   3. run `python -m flashcat source-accountability` which writes both
#      a dated archive (`source-accountability-YYYY-MM-DD.md`) and the
#      `source-accountability-latest.md` mirror under paw-reports/sportsbetting/
#   4. exit 0 on success; non-zero on failure (cron-mailable)
#
# Cron suggestion (every Monday at 09:00 UTC):
#   0 9 * * 1 /path/to/SportsBetting/scripts/weekly_source_rescore.sh \
#     >> /var/log/flashcat-weekly.log 2>&1
#
# Environment:
#   - PYTHON          override python interpreter (default: python3)
#   - FLASHCAT_SKIP_TENNIS_PERLIVE=1
#                     skip the tennis-data.co.uk re-pull (offline mode)
#   - FLASHCAT_TENNIS_START / FLASHCAT_TENNIS_END
#                     override the year window for the tennis per-event ledger
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python3}"
export PYTHONPATH="src${PYTHONPATH:+:${PYTHONPATH}}"

log() {
    printf '[weekly_source_rescore %s] %s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "$*"
}

log "starting from ${REPO_ROOT}"

# 1) Ensure the source history DB exists. If it doesn't, run the backfill
#    scripts. We intentionally don't re-backfill if the DB exists — the
#    backfills are idempotent but slow (NFL pulls ~150k PBP rows). Phil's
#    operational expectation is that backfills run once per quarter, and the
#    weekly rescore just reads what's already in the DB plus the latest
#    tennis-data.co.uk pull.
if [ ! -f data/source_history.db ]; then
    log "data/source_history.db missing — running backfills first"
    "${PYTHON}" scripts/backfill_nfl_historical.py     || log "  nfl backfill failed (continuing)"
    "${PYTHON}" scripts/backfill_tennis_historical.py  || log "  tennis backfill failed (continuing)"
    "${PYTHON}" scripts/backfill_nba_historical.py     || log "  nba backfill failed (continuing)"
else
    log "data/source_history.db present — skipping backfills"
fi

# 2) Build CLI flags from env.
FLAGS=()
if [ "${FLASHCAT_SKIP_TENNIS_PERLIVE:-0}" = "1" ]; then
    FLAGS+=(--skip-tennis-per-event)
fi
if [ -n "${FLASHCAT_TENNIS_START:-}" ]; then
    FLAGS+=(--tennis-start "${FLASHCAT_TENNIS_START}")
fi
if [ -n "${FLASHCAT_TENNIS_END:-}" ]; then
    FLAGS+=(--tennis-end "${FLASHCAT_TENNIS_END}")
fi

# 3) Run the scorer. ``flashcat source-accountability`` writes both the
#    dated archive and the latest mirror.
log "running flashcat source-accountability ${FLAGS[*]:-}"
"${PYTHON}" -m flashcat source-accountability \
    --out-dir paw-reports/sportsbetting \
    "${FLAGS[@]}"

# 4) Show the latest verdict roll-up to stdout so cron mails it if there's
#    something to flag.
LATEST=paw-reports/sportsbetting/source-accountability-latest.md
if [ -f "${LATEST}" ]; then
    log "verdict roll-up from ${LATEST}:"
    # Pull just the TL;DR + Verdict roll-up sections, stop at the next H2.
    awk '
        /^## TL;DR/                {p=1}
        p && /^## Per-source notes/ {exit}
        p
    ' "${LATEST}"
fi

log "done."
