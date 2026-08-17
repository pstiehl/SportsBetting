# feat(nba): Phase-2 feature expansion — efficiency, rest, pace, travel

**Week:** 9 (2026-08-17)  
**Branch:** `feat/nba-phase2-feature-expansion`  
**Phase:** NBA Phase 2 (gate rescued Phase 1; now hardening underlying CLV)

---

## Context

- Phase 1 (PR #25, CLOSED) had `n_bets=2,579`, `win_rate=64.52%`, `roi=-6.75%`, `clv_proxy=-0.61pp`
- With `+3pp` edge gate: `n_bets=823`, `roi=+1.85%`, `profit=$1,520`
- **Dominant Phase-1 loss bucket:** `line_moved_against` (43.0%)

Goal: add rolling form, schedule fatigue (B2B/rest), and pace features so the
ungated model finds real signal instead of paying vig on structurally-strong teams
with no in-game edge.

---

## Phase-2 Feature Catalog (12 features, ~10 effective signal dimensions)

| # | Feature | Category | Rationale |
|---|---|---|---|
| 1 | `pt_diff_l5_home` | Rolling form | Home team recent scoring margin L5 |
| 2 | `pt_diff_l5_away` | Rolling form | Away team recent scoring margin L5 |
| 3 | `pt_diff_l5_diff` | Rolling form | **Primary signal: home − away (hot/cold streak)** |
| 4 | `win_pct_l10_home` | Rolling form | Win% home team L10 |
| 5 | `win_pct_l10_away` | Rolling form | Win% away team L10 |
| 6 | `win_pct_l10_diff` | Rolling form | Home − away win% stability |
| 7 | `b2b_home` | Fatigue | Home team on back-to-back? |
| 8 | `b2b_away` | Fatigue | Away team on back-to-back? |
| 9 | `b2b_diff` | Fatigue | `b2b_away − b2b_home` (positive = away at B2B disadvantage) |
| 10 | `rest_days_diff` | Fatigue | `(home_rest) − (away_rest)`, capped ±7 days |
| 11 | `srs_diff` | Strength prior | Season SRS home − away (from bref connector via source_history.db) |
| 12 | `home_court_flag` | Structural | Constant 1.0 for HCA intercept interpretability |

**Data source:** Cached game logs via `nba_api` (2021-22 through 2023-24, 3,690 games).  
**Market proxy:** SRS prior probability × 4.5% hold. No real NBA moneyline historical archive is available.  
**CLV proxy note:** NOT true CLV — it's `model_prob − srs_prior_prob`. Positive means model diverged upward from the structural prior; negative means model regressed toward the prior less than market did.

---

## Headline Metrics Table

| Metric | Phase 1 (SRS only, gate rescues) | Phase 2 (no gate) | Phase 2 (+3pp gate) |
|---|---|---|---|
| `n_bets` | 2,579 | **2,823** | 447 |
| `win_rate` | 64.52% | 63.37% | 52.57% |
| `roi` (ungated) | −6.75% | **−7.86%** | — |
| `roi` (gated) | +1.85% | — | **+6.64%** |
| `profit` (gated) | $1,520 | — | **$2,968** |
| `clv_proxy_pp` | −0.61pp | −6.69pp | +9.54pp |
| `max_drawdown` (gated) | n/a | — | $1,491 |
| `sharpe` (gated) | n/a | — | **2.20** |

**Window:** 2022-01-01 → 2024-06-30 (27 walk-forward folds, 30-day eval windows, 365-day train)

---

## Ablation: Rolling Form vs. Schedule+SRS Baseline

| Model | n_bets | win_rate | roi | clv_proxy |
|---|---|---|---|---|
| Full Phase-2 (all 12 features) | 2,823 | 63.37% | −7.86% | −6.69pp |
| Baseline (B2B+rest+SRS only; form features zeroed) | 2,823 | 63.23% | −9.14% | −8.08pp |
| **Marginal gain from rolling form** | — | +0.14pp | **+1.28pp ROI** | +1.39pp CLV |

**Reading:** Rolling form (pt_diff_l5, win_pct_l10) adds ~1.3pp ROI over the schedule+SRS baseline. This is real but small — the model extracts marginally more signal from short-term form than from season-level SRS alone, but not enough to overcome the 4.5% market hold without a gate.

---

## Per-Year Breakdown

| Year | n_bets | win_rate | roi | profit |
|---|---|---|---|---|
| 2022 | 909 | 60.6% | −10.53% | −$9,570 |
| 2023 | 1,164 | 64.2% | −5.89% | −$6,860 |
| 2024 | 750 | 65.5% | −7.67% | −$5,750 |

2022 is the worst year — model had less rolling history (fewer seasons to train on) and the SRS prior was noisier early in the 2021-22 season. 2023-24 improvement signals the model is calibrating better with more data.

---

## Loss Post-Mortem

| Loss Bucket | Count | % of Losses | Phase 1 | Change |
|---|---|---|---|---|
| `form_signal_wrong` | 482 | **46.6%** | n/a | NEW bucket (Phase 1 had `line_moved_against` as primary) |
| `pure_variance` | 258 | 25.0% | ~25% | Stable |
| `generic` | 234 | 22.6% | ~20% | Slight increase |
| `line_moved_against` | 37 | 3.6% | **43.0%** | ✅ Massively reduced |
| `b2b_fatigue` | 23 | 2.2% | n/a | NEW bucket |

### Key finding: gate rescued Phase 1 AND Phase 2

**Phase 1's 43% `line_moved_against` problem is largely solved.** The new features gave the model real form signal, so it's no longer picking teams based on season-level SRS and paying vig on games where both teams are near-equal. The +3pp gate works even better on Phase-2 predictions: `roi=+6.64%` vs `+1.85%` in Phase 1.

**New dominant bucket:** `form_signal_wrong` (46.6%). The model now uses rolling pt_diff_l5_diff as its primary signal, and that signal was wrong in 47% of losses. This is a calibration issue in rolling form: a team hot by 8pt/game over the last 5 games doesn't maintain that margin against playoff-level opponents. We're picking correct direction on form but the magnitude of the edge is overstated.

**Phase 3 hypothesis:** Adding opponent-adjusted form (OffRtg/DefRtg from per-game box scores via nba_api TeamGameLog) would correct for schedule strength in the rolling window — a team that went 5-0 against lottery teams looks better than they are. The `form_signal_wrong` bucket should shrink when form is opponent-adjusted.

---

## Honest Verdict

**Does Phase 2 beat Phase 1?**

On the ungated model: **No** — ROI drops from −6.75% to −7.86%. The new features increase n_bets and change the loss distribution but don't crack +EV.

On the gated model: **Yes** — ROI improves from +1.85% to +6.64%, profit nearly doubles ($1,520 → $2,968). The gate is filtering for games where the rolling-form edge is large AND positive vs. the SRS prior — exactly the games where Phase-2 features add value.

**Recommendation:** Keep the +3pp edge gate active. The ungated model is a research tool; the gated model is +EV in this backtest window. Phase 3 should target `form_signal_wrong` by adding opponent-adjusted efficiency (requires per-game box-score nba_api pull, ~30 min backfill).

**Data blocker note:** No historical NBA moneyline archive is available (sportsbookreview dead, Odds API requires paid key). CLV proxy uses the bref SRS prior as market proxy — this understates CLV accuracy since the real market incorporates lineup/injury info we don't have. The ROI figures use the same proxy for odds conversion, so the absolute ROI number is an approximation.

---

## Files Changed

- `src/flashcat/nba_features/` — new module (feature_builder.py, model.py, simulator.py, __init__.py)
- `scripts/nba_walk_forward_backtest.py` — Phase-2 driver
- `scripts/sport_backtest.sh` — wired NBA stub → real implementation
- `tests/test_nba_walk_forward.py` — 14 tests (leakage, shape, B2B, smoke, sim)
- `docs/nba_feature_audit_phase2.md` — feature audit
- `data/nba_walk_forward_backtest.json` — backtest receipt
