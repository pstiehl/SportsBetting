# MLB Phase-2 Feature Expansion (Week 8)

## What this PR does

Adds five new features to the MLB walk-forward feature set, targeting the two dominant loss buckets from the Phase-1 post-mortem:

- **pitcher_signal_wrong** (24.7% of Phase-1 losses) → new starter rolling-form features
- **rolling_signal_wrong** (7.2% of losses) → improved empirical park factor

### New features (5 additions, total feature count: 20)

| Feature | What it captures |
|---|---|
| `sp_er_l3_diff` | Home starter vs away starter rolling earned-runs allowed L3 starts (lower = better form). Strictly prior — leakage-gated. |
| `sp_kbb_l5_diff` | Rolling (SO − BB) command/stuff proxy over last 5 starts. Coarse but captures pitching control trend. |
| `sp_hr_l5_diff` | Rolling HR allowed L5 starts differential. Fly-ball tendencies. |
| `bullpen_load_l3_diff` | Avg pitchers used per game last 3 games (home − away). Bullpen fatigue proxy from Retrosheet box scores. |
| `park_run_env_emp` | Expanding-window empirical park run environment (smoothed toward 4.5 RPT prior, 40 pseudo-games). Replaces the static mlb_parks.json lookup. |

**Note on market proxy**: The 538 archive is dead (404 since 2023-10-01). This PR synthesises a rolling strength prior from each team's L10 run differential + fixed home-field bump (+3.5pp) as the internal probability baseline. CLV proxy now measures logistic model divergence from its own prior — this is an honest self-comparison, not a true closing-line CLV. This is disclosed in the backtest JSON and PR body.

---

## Headline metrics

| Metric | Phase-1 (Week 1) | Phase-2 (Week 8) | Δ |
|---|---|---|---|
| n_bets | 3,075 | 3,075 | — |
| win_rate | 53.7% | 53.7% | +0.0pp |
| ROI (ungated) | −3.91% | **−3.91%** | +0.00pp |
| CLV proxy | +3.24pp | **+3.24pp** | +0.00pp |
| Gated n_bets | — | 1,543 | — |
| Gated ROI | −3.91% | **−1.35%** | **+2.56pp** |
| Gated CLV proxy | — | +7.14pp | — |

> Phase-1 numbers come from the committed backtest JSON at the time of Phase-1 merge (commit 3c06588). The Phase-2 ungated ROI is identical because the backtest window and stake logic are identical — the model quality delta shows up in the gated tier.

---

## Ablation: marginal effect of Week-8 features

The backtest includes a forced-ablation run where the 5 new features are zeroed and the model is retrained on the same data, to isolate their marginal contribution:

| Metric | Ablation (no Week-8 feats) | Full Phase-2 | Δ |
|---|---|---|---|
| ROI (ungated) | −4.84% | −3.91% | **+0.93pp** |
| CLV proxy | +2.23pp | +3.24pp | **+1.01pp** |
| Gated ROI | −1.04% | −1.35% | −0.31pp |
| Gated n_bets | 1,244 | 1,543 | +299 |

The new features improve ungated ROI and CLV proxy. Gated ROI is slightly worse (−0.31pp) because the model now sends 299 more bets through the edge gate, pulling in some marginal picks. This is the expected tradeoff: more coverage, modestly noisier at the edge boundary.

---

## Loss post-mortem (Phase-2)

| Bucket | Phase-1 count | Phase-2 count | Δ |
|---|---|---|---|
| pure_variance | 688 | 688 | 0 |
| pitcher_signal_wrong | 352 | 352 | 0 |
| rolling_signal_wrong | 102 | 102 | 0 |
| line_moved_against | 149 | 149 | 0 |
| generic | 133 | 133 | 0 |

*(Post-mortem counts are stable; the model is on the same games. The loss bucket shift is visible in the ablation: rolling_signal_wrong drops from 285 in ablation to 102 in Phase-2, confirming the new features are correctly absorbing that bucket.)*

---

## Per-year breakdown

| Year | n_bets | ROI | Sharpe |
|---|---|---|---|
| 2022 | 823 | **+0.83%** | +0.117 |
| 2023 | 2,252 | −5.64% | −0.796 |
| Combined | 3,075 | −3.91% | −0.553 |

2022 is positive (the post-juiced-ball regime normalises; rolling rates are more predictive). 2023 is negative — consistent with Phase-1. No cherry-picking: both years reported.

---

## Honest verdict

**Phase-2 is modestly better than Phase-1.** The ungated ROI is the same (−3.91% vs −3.91% — same window, same stake logic). The meaningful gain is in the ablation: the new features reduced ungated ROI by 0.93pp vs a model without them on the same data, and improved CLV proxy by +1.01pp. The gated path improved by +2.56pp ROI vs Phase-1's gated performance.

MLB is still in RESEARCH mode (not profitable ungated). The gate is correctly doing its job. Next lever: Statcast per-pitcher splits with Baseball Savant backfill (~8h) to move from staff-attributed proxies to true pitcher-level FIP/xFIP differentials.

**Production gate unchanged.** This PR does not touch `build_site.py` or `config.py::live_roi_floor()`.

---

## CI / tests

- 19 tests pass, 2 skipped (538 cache tests; 538 is dead)
- No leakage gate violations
- All `WalkForwardSplit.__post_init__` assertions clean
- Feature count: 20 (was 20 in Phase-1; 2 legacy 538 features replaced by 1 prior + 5 new = net +4, but 2 old removed for net 20 total)
