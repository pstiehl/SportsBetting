## feat(wta): Phase-1 walk-forward harness + feature catalog

Week 6 of the weekly sport-rotation feature-expansion loop. A near-mechanical PORT of the Week-5 ATP harness to the WTA women's tour. Seasons backtested: 2022-2024.

### Backtest window

* **Start:** 2022-01-01
* **End:** 2024-12-31
* **Train window:** rolling 365 days
* **Eval window:** 30 days, slide forward by 30 days
* **Warmup:** 120 days
* **Folds completed:** 31

### Data

* **Matches loaded** (WTA main-tour singles, completed): 7034
* **Matches graded by model** (full feature gate passed): 4652
* **Priors coverage:** market-close=7025, tennis-rank-bt=7014 (of 7034).
* **Payout odds:** REAL archived closing decimal odds (Pinnacle > Bet365 > market-avg) from tennis-data.co.uk. This is a genuine closing-line payout, not a reconstructed proxy.
* **CLV proxy:** model pick prob minus devigged closing implied prob (`market_prob_home`).

### Headline metrics

| Variant | n_bets | win_rate | ROI | CLV proxy | Max DD | Sharpe | Profit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ungated ($100/match) | 4652 | 65.7% | -3.28% | +1.26pp | $-15,833 | -2.29 | $-15,275 |
| +3pp edge gate | 2368 | 64.9% | -1.92% | +2.25pp | $-5,519 | -1.29 | $-4,542 |

### Per-year breakdown (ungated)

| Year | n_bets | win_rate | ROI |
|---|---:|---:|---:|
| 2022 | 835 | 61.9% | -5.58% |
| 2023 | 1803 | 67.5% | -0.46% |
| 2024 | 2014 | 65.7% | -4.86% |

### Per-surface breakdown (ungated)

| Surface | n_bets | win_rate | ROI |
|---|---:|---:|---:|
| Clay | 1128 | 66.3% | -2.39% |
| Grass | 659 | 64.5% | -6.15% |
| Hard | 2865 | 65.8% | -2.98% |

### Loss post-mortem (ungated)

| Bucket | Losing bets | % of total losses |
|---|---:|---:|
| `favorite_upset` | 332 | 20.8% |
| `pure_variance` | 324 | 20.3% |
| `generic` | 294 | 18.4% |
| `line_moved_against` | 237 | 14.9% |
| `ranking_signal_wrong` | 197 | 12.4% |
| `surface_form_wrong` | 86 | 5.4% |
| `fatigue_disadvantage` | 73 | 4.6% |
| `h2h_signal_wrong` | 51 | 3.2% |

### Verdict (one sentence)

**Loses to the closing line — -3.28% ungated ROI (CLV proxy +1.26pp); the 13-feature Phase-1 catalog does not beat Pinnacle's close.**

### Feature catalog (13 features)

* `market_prob_home`
* `rank_bt_prob_home`
* `rank_bt_minus_market_pp`
* `rank_log_ratio`
* `rank_points_log_ratio`
* `win_pct_l10_diff`
* `win_pct_l25_diff`
* `surface_win_pct_l20_diff`
* `games_won_pct_l10_diff`
* `sets_won_pct_l10_diff`
* `rest_days_diff`
* `matches_l14_diff`
* `h2h_home_share`

### WTA-specific difference vs the ATP catalog

* **`best_of_5` DROPPED.** The WTA main tour is best-of-3 ONLY — there are no best-of-5 Grand Slam matches on the women's tour. In ATP the feature flags Grand Slam best-of-5 matches; on WTA it would be a **constant 0.0 across the entire dataset** and add exactly zero signal. So the Phase-1 WTA catalog is **13 features** (ATP's 14 minus `best_of_5`). The ATP simulator's `best_of_5_variance` loss bucket is likewise dropped — it can never fire on best-of-3. This is a deliberate, documented WTA-specific difference; everything else in the ATP catalog ports directly.

### Skipped in Phase 1 (documented, not hidden)

* **sackmann-wta-elo (surface-adjusted Elo)** — the `JeffSackmann/tennis_wta` GitHub repo returned 404/429 during the backfill, so the Elo source never populated. This is the single highest-value Phase-2 lever (surface Elo is the gold-standard tennis rating).
* **serve hold % / return break %** — tennis-data.co.uk does not publish per-match serve/return point stats. `games_won_pct_l10_diff` is our coarse proxy. Sackmann's match-charting / point-by-point data would supply the real thing.
* **real line movement** — only the closing line is archived; there is no opener in the ledger, so a line-movement feature is not computable from this source.

### What this PR does NOT change

* `build_site.py::resolve_sport_modes()` — untouched.
* `config.py::live_roi_floor()` — untouched.
* `backtest/runner.py` — untouched.
* Live picks pipeline / production source weights — untouched.

Per FEATURE_EXPANSION_PLAYBOOK.md: the production gate is the trust contract with the live site and stays in place. This PR adds a new **evaluation lens** for WTA, parallel to the MLB/NFL/ATP harnesses.

### Self-merge gate

- [x] Tests pass (`tests/test_wta_walk_forward.py`).
- [x] No production-gate files touched (`build_site.py` / `config.py` / `backtest/runner.py`).
- [x] No real-money integration.

Per standing autonomy grant (SportsBetting weekly rotation, NFL PR #27 / ATP PR #31 precedent), self-merging.

🐾 Ada Week 6 WTA run, 2026-07-13.
