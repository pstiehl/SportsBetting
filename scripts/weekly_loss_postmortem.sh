#!/usr/bin/env bash
#
# weekly_loss_postmortem.sh — aggregate losing backtest bets by probable
# cause across all sports that have a walk-forward backtest JSON in
# ``data/<sport>_walk_forward_backtest.json``.
#
# What it does:
#   1. Find every data/<sport>_walk_forward_backtest.json file.
#   2. For each, read ``loss_buckets`` (count of LOSING bets per bucket).
#   3. Roll up per-sport AND a cross-sport bucket-share table.
#   4. Identify the dominant bucket per sport + the overall top driver.
#   5. Write paw-reports/sportsbetting/loss-postmortem-YYYY-MM-DD.md
#      and update loss-postmortem-latest.md mirror.
#
# Cron-safe; idempotent. Exit 0 on success even if 0 sports backtested
# yet (we just produce an empty report explaining why).
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON="${PYTHON:-python3}"

DATE="$(date -u +%Y-%m-%d)"
OUT_DIR="paw-reports/sportsbetting"
OUT_PATH="${OUT_DIR}/loss-postmortem-${DATE}.md"
LATEST_PATH="${OUT_DIR}/loss-postmortem-latest.md"
mkdir -p "${OUT_DIR}"

"${PYTHON}" - "${OUT_PATH}" "${LATEST_PATH}" <<'PY'
import json, sys, glob, os
from collections import defaultdict, Counter

out_path, latest_path = sys.argv[1], sys.argv[2]

paths = sorted(glob.glob("data/*_walk_forward_backtest.json"))

per_sport = {}  # sport -> {bucket: count, ...}
per_sport_meta = {}  # sport -> dict with n_bets, roi, clv, dominant_bucket
for p in paths:
    sport = os.path.basename(p).split("_")[0]
    try:
        data = json.loads(open(p).read())
    except Exception as e:
        print(f"WARN: failed to read {p}: {e}", file=sys.stderr)
        continue
    buckets = data.get("loss_buckets", {}) or {}
    overall = data.get("overall", {}) or {}
    per_sport[sport] = buckets
    total_losses = sum(buckets.values())
    dominant = max(buckets.items(), key=lambda kv: kv[1])[0] if buckets else None
    per_sport_meta[sport] = {
        "n_bets": data.get("n_bets") or overall.get("n_bets") or 0,
        "roi": overall.get("roi"),
        "clv_proxy_pp": overall.get("clv_proxy_pp"),
        "total_losses": total_losses,
        "dominant_bucket": dominant,
        "window": data.get("window"),
    }

# Cross-sport bucket totals (share-weighted across sports = simple sum)
cross = Counter()
for sport, buckets in per_sport.items():
    for k, v in buckets.items():
        cross[k] += v

# Cross-sport dominant
total_cross = sum(cross.values())
overall_top = cross.most_common(1)[0] if cross else (None, 0)

lines = []
lines.append("# Weekly loss post-mortem — cross-sport aggregate")
lines.append("")
lines.append(f"Generated: {__import__('datetime').datetime.utcnow().isoformat()}Z")
lines.append("")
if not per_sport:
    lines.append("No walk-forward backtest JSON files found under `data/`.")
    lines.append("")
    lines.append("This script aggregates `data/<sport>_walk_forward_backtest.json`")
    lines.append("files produced by `scripts/<sport>_walk_forward_backtest.py`. Run")
    lines.append("the per-sport drivers first, then re-run this script.")
else:
    # Headline: sports backtested + total bets + top driver
    sports_n = len(per_sport)
    total_bets_all = sum(m.get("n_bets") or 0 for m in per_sport_meta.values())
    total_losses_all = sum(m["total_losses"] for m in per_sport_meta.values())
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- **{sports_n} sport(s) backtested** "
                 + ", ".join(sorted(per_sport.keys())))
    lines.append(f"- **{total_bets_all:,} bets**, **{total_losses_all:,} losing bets** classified")
    if overall_top[0]:
        share = overall_top[1] / total_cross * 100 if total_cross else 0
        lines.append(
            f"- **Cross-sport top driver**: `{overall_top[0]}` "
            f"({overall_top[1]:,} losing bets — {share:.1f}% of all classified losses)"
        )
    lines.append("")

    # Per-sport table
    lines.append("## Per-sport headline")
    lines.append("")
    lines.append("| Sport | n_bets | ROI | CLV proxy | Losses | Dominant bucket | Window |")
    lines.append("|---|---:|---:|---:|---:|---|---|")
    for sport in sorted(per_sport.keys()):
        m = per_sport_meta[sport]
        roi = m["roi"]
        roi_s = f"{roi*100:+.2f}%" if roi is not None else "—"
        clv = m["clv_proxy_pp"]
        clv_s = f"{clv*100:+.2f}pp" if clv is not None else "—"
        win = m.get("window") or {}
        win_s = f"{win.get('start','?')} → {win.get('end','?')}"
        lines.append(
            f"| {sport} | {m['n_bets']:,} | {roi_s} | {clv_s} | "
            f"{m['total_losses']:,} | `{m['dominant_bucket'] or '—'}` | {win_s} |"
        )
    lines.append("")

    # Per-sport bucket breakdowns
    lines.append("## Per-sport loss buckets")
    lines.append("")
    for sport in sorted(per_sport.keys()):
        buckets = per_sport[sport]
        if not buckets:
            continue
        total = sum(buckets.values())
        lines.append(f"### {sport}")
        lines.append("")
        lines.append("| Bucket | Count | % of losses |")
        lines.append("|---|---:|---:|")
        for bucket, n in sorted(buckets.items(), key=lambda kv: -kv[1]):
            pct = n / total * 100 if total else 0
            lines.append(f"| `{bucket}` | {n:,} | {pct:.1f}% |")
        lines.append("")

    # Cross-sport bucket roll-up
    lines.append("## Cross-sport bucket roll-up")
    lines.append("")
    lines.append("Bucket names with the same string across sports get summed. NBA-specific buckets (`pace_signal_wrong`, `rest_disadvantage`) and MLB-specific buckets (`pitcher_signal_wrong`) are reported alongside the shared ones (`pure_variance`, `line_moved_against`, `rolling_signal_wrong`, `generic`).")
    lines.append("")
    lines.append("| Bucket | Cross-sport count | % of all classified losses |")
    lines.append("|---|---:|---:|")
    for bucket, n in cross.most_common():
        pct = n / total_cross * 100 if total_cross else 0
        lines.append(f"| `{bucket}` | {n:,} | {pct:.1f}% |")
    lines.append("")

    # Interpretation hints
    lines.append("## Reading the buckets")
    lines.append("")
    lines.append("- **`pure_variance`** — pick prob in [0.45, 0.55]. Acceptable. Coin-flippy picks lose half the time by definition.")
    lines.append("- **`line_moved_against`** / **`market_proxy_dominant`** — model edge vs market proxy < 1pp. We're losing to the vig. Edge gate fixes this in production; feature work doesn't.")
    lines.append("- **`pitcher_signal_wrong`** (MLB) / **`pace_signal_wrong`** (NBA) / **`rolling_signal_wrong`** (any) — model overweighted a specific feature class. This IS a feature-quality signal. If one bucket > 20% of losses for a sport, it's a candidate for Phase-2 work.")
    lines.append("- **`rest_disadvantage`** (NBA) — we backed the team with worse rest. If this is significant, the rest features need bigger coefficients (or sign flipping).")
    lines.append("- **`generic`** — fallback. Should be small. A large generic bucket means our taxonomy is under-fitting losses.")

text = "\n".join(lines) + "\n"
open(out_path, "w").write(text)
open(latest_path, "w").write(text)
print(f"wrote {out_path}")
print(f"wrote {latest_path}")
PY

echo "[weekly_loss_postmortem $(date -u +%Y-%m-%dT%H:%M:%SZ)] done."
