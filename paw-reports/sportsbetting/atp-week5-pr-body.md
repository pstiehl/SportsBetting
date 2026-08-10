## feat(atp): Phase-1 walk-forward harness + feature catalog

Week 5 of the weekly sport-rotation feature-expansion loop. Seasons backtested: 2022-2024.

### Backtest window

* **Start:** 2022-01-01
* **End:** 2024-12-31
* **Train window:** rolling 365 days
* **Eval window:** 98 days, slide forward by 98 days
* **Warmup:** 120 days
* **Folds completed:** 10

### Data

* **Matches loaded** (ATP main-tour singles, completed): 7769
* **Matches graded by model** (full feature gate passed): 5356
* **Priors coverage:** market-close=7759, tennis-rank-bt=7756 (of 7769).
* **Payout odds:** REAL archived closing decimal odds (Pinnacle > Bet365 > market-avg) from tennis-data.co.uk. This is a genuine closing-line payout, not a reconstructed proxy.
* **CLV proxy:** model pick prob minus devigged closing implied prob (`market_prob_home`).

### Headline metrics

| Variant | n_bets | win_rate | ROI | CLV proxy | Max DD | Sharpe | Profit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ungated ($100/match) | 5356 | 66.4% | -3.79% | +1.98pp | $-21,522 | -2.70 | $-20,290 |
| +3pp edge gate | 2689 | 67.0% | -2.32% | +3.59pp | $-8,936 | -1.64 | $-6,232 |

### Per-year breakdown (ungated)

| Year | n_bets | win_rate | ROI |
|---|---:|---:|---:|
| 2022 | 1091 | 66.2% | -4.32% |
| 2023 | 2050 | 65.3% | -5.51% |
| 2024 | 2215 | 67.6% | -1.93% |

### Per-surface breakdown (ungated)

| Surface | n_bets | win_rate | ROI |
|---|---:|---:|---:|
| Clay | 1569 | 65.8% | -4.04% |
| Grass | 727 | 67.3% | -3.90% |
| Hard | 3060 | 66.6% | -3.63% |

### Loss post-mortem (ungated)

| Bucket | Losing bets | % of total losses |
|---|---:|---:|
| `favorite_upset` | 474 | 26.4% |
| `pure_variance` | 305 | 17.0% |
| `line_moved_against` | 255 | 14.2% |
| `generic` | 241 | 13.4% |
| `ranking_signal_wrong` | 223 | 12.4% |
| `surface_form_wrong` | 112 | 6.2% |
| `fatigue_disadvantage` | 96 | 5.3% |
| `h2h_signal_wrong` | 56 | 3.1% |
| `best_of_5_variance` | 36 | 2.0% |

### Verdict (one sentence)

**Loses to the closing line — -3.79% ungated ROI (CLV proxy +1.98pp); the 14-feature Phase-1 catalog does not beat Pinnacle's close.**

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
