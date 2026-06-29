## feat(nfl): Phase-1 walk-forward harness + feature catalog

Week 4 of the weekly sport-rotation feature-expansion loop. Seasons backtested: 2022-2024.

### Backtest window

* **Start:** 2022-09-01
* **End:** 2024-12-31
* **Train window:** rolling 365 days
* **Eval window:** 30 days, slide forward by 30 days
* **Warmup:** 120 days
* **Folds completed:** 25

### Data

* **Games loaded** (regular season, 3 seasons): 799
* **Games graded by model** (full feature gate passed): 416
* **Source streams attached as priors:** 538 NFL Elo (web.archive 2023 snapshot, 2022-only coverage), nflfastR EPA (model fit walk-forward by week), market-close moneylines (devigged for the market prob feature).

### Headline metrics

| Variant | n_bets | win_rate | ROI | CLV proxy | Max DD | Sharpe | Profit |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ungated ($100/game) | 416 | 68.0% | +1.45% | +7.51pp | $-1,215 | 0.32 | $601 |
| +3pp edge gate | 345 | 67.2% | +1.11% | +9.12pp | $-894 | 0.24 | $384 |

### Top loss buckets (ungated)

| Bucket | Losing bets | % of total losses |
|---|---:|---:|
| `divisional_misjudged` | 42 | 31.6% |
| `generic` | 25 | 18.8% |
| `pure_variance` | 24 | 18.0% |
| `rolling_signal_wrong` | 16 | 12.0% |
| `bye_off_overrated` | 9 | 6.8% |
| `rest_disadvantage` | 6 | 4.5% |
| `prior_disagreement_wrong` | 6 | 4.5% |
| `line_moved_against` | 5 | 3.8% |

### Verdict (one sentence)

**BEATS vig — +1.45% ungated ROI on 416 bets.**

### Feature catalog (17 features)

* `elo_prob_home`
* `epa_prob_home`
* `market_prob_home`
* `elo_minus_market_pp`
* `epa_minus_market_pp`
* `priors_avg`
* `off_epa_l4_diff`
* `def_epa_l4_diff`
* `success_rate_l4_diff`
* `off_epa_l8_diff`
* `pass_epa_l4_diff`
* `rush_epa_l4_diff`
* `rest_diff`
* `home_off_bye`
* `away_off_bye`
* `divisional`
* `week_number`

### What this PR does NOT change

* `build_site.py::resolve_sport_modes()` — untouched.
* `config.py::live_roi_floor()` — untouched.
* Live picks pipeline — untouched.
* Any production source weights — untouched.

Per the FEATURE_EXPANSION_PLAYBOOK.md anti-pattern list: the production gate stays in place. This PR adds a new **evaluation lens** for NFL, parallel to the MLB/CFB harnesses.

### Self-merge gate

- [x] No CI failures (tests pass; see `tests/test_nfl_walk_forward.py`).
- [x] No production-gate weakening (build_site.py untouched).
- [x] No real-money integration.

Per standing autonomy grant (MEMORY.md, *Standing autonomy grant — Phil's SportsBetting model*), self-merging.

🐾 ada-cloud Week 4 NFL run, 2026-06-29.
