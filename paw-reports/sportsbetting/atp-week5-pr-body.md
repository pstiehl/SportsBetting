## feat(atp): Phase-1 walk-forward harness + feature catalog

Week 5 of the weekly sport-rotation feature-expansion loop. Seasons backtested: 2022-2024.

### Backtest window

* **Start:** 2022-01-01
* **End:** 2024-12-31
* **Train window:** rolling 365 days
* **Eval window:** 30 days, slide forward by 30 days
* **Warmup:** 120 days
* **Folds completed:** 31

### Data

* **Matches loaded** (ATP main-tour singles, completed): 7769
* **Matches graded by model** (full feature gate passed): 5356
* **Priors coverage:** market-close=7759, tennis-rank-bt=7756 (of 7769).
* **Payout odds:** REAL archived closing decimal odds (Pinnacle > Bet365 > market-avg) from tennis-data.co.uk. This is a genuine closing-line payout, not a reconstructed proxy.
* **CLV proxy:** model pick prob minus devigged closing implied prob (`market_prob_home`).

### Headline metrics

| Variant | n_bets | win_rate | ROI | CLV proxy | Max DD | Sharpe | Profit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ungated ($100/match) | 5356 | 66.4% | -3.94% | +1.77pp | $-22,688 | -2.80 | $-21,096 |
| +3pp edge gate | 2572 | 66.2% | -3.85% | +3.24pp | $-12,432 | -2.71 | $-9,904 |

### Per-year breakdown (ungated)

| Year | n_bets | win_rate | ROI |
|---|---:|---:|---:|
| 2022 | 1091 | 66.0% | -4.60% |
| 2023 | 2050 | 65.0% | -6.06% |
| 2024 | 2215 | 67.8% | -1.65% |

### Per-surface breakdown (ungated)

| Surface | n_bets | win_rate | ROI |
|---|---:|---:|---:|
| Clay | 1569 | 65.6% | -4.34% |
| Grass | 727 | 67.5% | -3.38% |
| Hard | 3060 | 66.5% | -3.87% |

### Loss post-mortem (ungated)

| Bucket | Losing bets | % of total losses |
|---|---:|---:|
| `favorite_upset` | 464 | 25.7% |
| `pure_variance` | 315 | 17.5% |
| `line_moved_against` | 274 | 15.2% |
| `generic` | 245 | 13.6% |
| `ranking_signal_wrong` | 219 | 12.2% |
| `fatigue_disadvantage` | 101 | 5.6% |
| `surface_form_wrong` | 101 | 5.6% |
| `h2h_signal_wrong` | 50 | 2.8% |
| `best_of_5_variance` | 33 | 1.8% |

### Verdict (one sentence)

**Loses to the closing line — -3.94% ungated ROI (CLV proxy +1.77pp); the 14-feature Phase-1 catalog does not beat Pinnacle's close.**

### Feature catalog (14 features)

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
* `best_of_5`

### Skipped in Phase 1 (documented, not hidden)

* **sackmann-atp-elo (surface-adjusted Elo)** — the `JeffSackmann/tennis_atp` GitHub repo returned 404/429 during the Week-5 backfill, so the Elo source never populated. This is the single highest-value Phase-2 lever (surface Elo is the gold-standard tennis rating).
* **serve hold % / return break %** — tennis-data.co.uk does not publish per-match serve/return point stats. `games_won_pct_l10_diff` is our coarse proxy. Sackmann's match-charting / point-by-point data would supply the real thing.
* **real line movement** — only the closing line is archived; there is no opener in the ledger, so a line-movement feature is not computable from this source.

### What this PR does NOT change

* `build_site.py::resolve_sport_modes()` — untouched.
* `config.py::live_roi_floor()` — untouched.
* `backtest/runner.py` — untouched.
* Live picks pipeline / production source weights — untouched.

Per FEATURE_EXPANSION_PLAYBOOK.md: the production gate is the trust contract with the live site and stays in place. This PR adds a new **evaluation lens** for ATP, parallel to the MLB/NFL harnesses.

### Self-merge gate

- [x] Tests pass (`tests/test_atp_walk_forward.py`).
- [x] No production-gate files touched (`build_site.py` / `config.py` / `backtest/runner.py`).
- [x] No real-money integration.

Per standing autonomy grant (SportsBetting weekly rotation, NFL PR #27 precedent), self-merging.

🐾 Ada Week 5 ATP run, 2026-07-06.
