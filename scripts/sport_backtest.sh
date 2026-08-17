#!/usr/bin/env bash
# Rotating weekly sport backtest runner.
#
# Usage:
#   bash scripts/sport_backtest.sh <sport> [start] [end]
#
# Examples:
#   bash scripts/sport_backtest.sh mlb
#   bash scripts/sport_backtest.sh mlb 2022-01-01 2023-12-31
#   bash scripts/sport_backtest.sh nba
#
# Idempotent — re-running on the same window overwrites
# data/<sport>_walk_forward_backtest.json and INSERT OR REPLACE'es
# predictions in source_history.db.
#
# Cron-friendly: emits a non-zero exit code on failure, never prompts.
#
# This is the weekly feature-expansion harness. See
# docs/FEATURE_EXPANSION_PLAYBOOK.md for the methodology and rotation
# order. The production gate is NEVER touched by this script.

set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <sport> [start_date YYYY-MM-DD] [end_date YYYY-MM-DD]" >&2
    echo "       supported sports (week-1 rotation): mlb (nba, nfl, cfb, atp, wta, pga = stubs)" >&2
    exit 64
fi

SPORT="${1,,}"  # lowercase
DEFAULT_END=$(date -u +%Y-%m-%d)
DEFAULT_START=$(date -u -d "2 years ago" +%Y-%m-%d 2>/dev/null || \
                python3 -c "import datetime; print((datetime.date.today() - datetime.timedelta(days=730)).isoformat())")

START="${2:-$DEFAULT_START}"
END="${3:-$DEFAULT_END}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:-src}"
# Honor an existing PYTHONPATH if the caller set one to include src.
case "$PYTHONPATH" in
    *src*) ;;
    *) PYTHONPATH="src:$PYTHONPATH" ;;
esac
export PYTHONPATH

DATA_OUT="data/${SPORT}_walk_forward_backtest.json"

echo ">>> sport=$SPORT window=$START..$END output=$DATA_OUT"

case "$SPORT" in
    mlb)
        python3 scripts/mlb_walk_forward_backtest.py \
            --start "$START" \
            --end "$END" \
            --output "$DATA_OUT" \
            "${@:4}"
        ;;
    nba)
        python3 scripts/nba_walk_forward_backtest.py \
            --start "$START" \
            --end "$END" \
            --output "$DATA_OUT" \
            "${@:4}"
        ;;
    nfl|cfb|atp|wta|pga)
        cat <<EOF
NOT YET IMPLEMENTED — $SPORT is in the rotation queue (see
docs/FEATURE_EXPANSION_PLAYBOOK.md::Rotation order). The walk-forward
harness in src/flashcat/mlb_features/ is sport-agnostic; the next
weekly PR will add a $SPORT feature builder that plugs into it.

To stub a $SPORT run you'd need:
  1. src/flashcat/<sport>_features/feature_builder.py
       — load historical game rows for the sport
       — build the feature catalog (see playbook step 2)
  2. scripts/${SPORT}_walk_forward_backtest.py
       — driver mirroring scripts/mlb_walk_forward_backtest.py
  3. tests/test_${SPORT}_walk_forward.py
       — leakage + smoke tests
EOF
        exit 65
        ;;
    *)
        echo "unknown sport: $SPORT" >&2
        echo "supported: mlb (live), nba/nfl/cfb/atp/wta/pga (stubs)" >&2
        exit 64
        ;;
esac

if [[ -f "$DATA_OUT" ]]; then
    echo ">>> wrote $DATA_OUT"
    python3 - <<PYEOF
import json, sys
with open("$DATA_OUT") as f:
    d = json.load(f)
o = d.get("overall") or {}
roi = o.get("roi")
n = o.get("n_bets") or 0
clv = o.get("clv_proxy_pp")
roi_s = f"{roi*100:+.2f}%" if roi is not None else "n/a"
clv_s = f"{clv*100:+.2f}pp" if clv is not None else "n/a"
print(f">>> headline: n={n} ROI={roi_s} CLV_proxy={clv_s}")
PYEOF
fi
