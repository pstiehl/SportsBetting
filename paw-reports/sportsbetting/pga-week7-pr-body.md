## feat(pga): Phase-1 walk-forward harness + feature catalog (Week 7)

Week 7 of the weekly sport-rotation feature-expansion loop. A near-mechanical
PORT of the Week-6 WTA harness to PGA Tour **head-to-head matchups** (same
event shape as tennis singles: two players, pick who finishes ahead).

### Data status: HARNESS ONLY — backtest BLOCKED on historical data access

Per FEATURE_EXPANSION_PLAYBOOK.md §6 ("honest evaluation over green
dashboards"), this PR ships the harness code + tests **without a fabricated
backtest**. The real walk-forward run is blocked:

> No PGA rows in source_history.db, and the only known source of genuine historical closing MATCHUP odds (DataGolf's matchup-odds archive, https://datagolf.com/matchup-odds-archive) is a PAID-tier endpoint that is OFF LIMITS per Phil's standing constraint and requires an unset DATAGOLF_API_KEY. Free golf-results datasets (ESPN leaderboards, opendatabay/Kaggle PGA CSVs) supply finishing positions but NOT closing matchup probabilities, so there is no honest CLV proxy to grade against. UNBLOCK: set DATAGOLF_API_KEY with paid-tier matchup archive access (needs Phil sign-off), OR wire a free source that pairs player finishes with closing H2H matchup odds, then run this driver — the harness will produce real metrics with no code changes.

No outcomes were synthesized and no ROI number was manufactured. The harness
is leakage-gated, unit-tested against a synthetic fixture, and will produce
real metrics the moment a data source or key is provided — with zero code
changes (just run `scripts/pga_walk_forward_backtest.py`).

### What landed

* `src/flashcat/pga_features/` — feature_builder / model / simulator, ported
  from `wta_features/` with golf-specific adaptations.
* `scripts/pga_walk_forward_backtest.py` — this driver (real backtest when
  data exists; honest DATA-BLOCKED receipt otherwise).
* `tests/test_pga_walk_forward.py` — leakage + gate + sign-invariant tests
  (all passing).
* `data/pga_walk_forward_backtest.json` — the honest receipt (data_status =
  HARNESS_ONLY_DATA_BLOCKED, n_bets = 0).

### Metrics

| Metric | Value |
|---|---|
| data_status | `HARNESS_ONLY_DATA_BLOCKED` |
| n_bets | 0 (no graded PGA matchups available) |
| win_rate / roi / clv_proxy_pp | — (blocked) |

(No loss post-mortem table — there are no graded bets to bucket. That is the
honest outcome, not a gap to paper over.)

### Feature catalog (12 features)

* `market_prob_home`
* `skill_bt_prob_home`
* `skill_bt_minus_market_pp`
* `win_pct_log_ratio`
* `skill_diff`
* `h2h_form_l10_diff`
* `finish_quality_l10_diff`
* `made_cut_pct_l10_diff`
* `course_tier_quality_l10_diff`
* `rest_days_diff`
* `starts_l28_diff`
* `h2h_home_share`

### Sport-mapping vs the WTA catalog

* tennis surface -> PGA **course-difficulty tier** (easy/standard/hard/major)
* tennis ranking points -> DataGolf **pre-tournament win% / skill estimate**
* tennis rank-BT prior -> **datagolf-sg** Bradley-Terry matchup prior
* tennis games/sets-won share -> **made-cut rate + exponential finish-quality**
* **DROPPED:** `sets_won_pct_l10_diff` / `games_won_pct_l10_diff` — no golf
  analog. **ADDED:** `made_cut_pct_l10_diff`, `skill_diff`. Net = 12 features.

### What this PR does NOT change

* `build_site.py::resolve_sport_modes()` — untouched.
* `config.py::live_roi_floor()` — untouched.
* `backtest/runner.py` — untouched.
* Live picks pipeline / production source weights — untouched.

### Self-merge gate

- [x] Tests pass (`tests/test_pga_walk_forward.py` + full suite).
- [x] No production-gate files touched.
- [x] No real-money integration. No fabricated backtest outcomes.

### Verdict (one sentence)

**Phase-1 PGA harness landed and unit-tested; the backtest is BLOCKED on
historical data (DataGolf matchup archive is paid/off-limits, no free
H2H-with-closing-odds source found) — cannot yet say whether the feature set
beats vig, and we will NOT pretend otherwise.**

Per standing autonomy grant (SportsBetting weekly rotation, WTA PR #33
precedent), self-merging once CI is green — the production gate is untouched.

🐾 Ada Week 7 PGA run, 2026-07-20.
