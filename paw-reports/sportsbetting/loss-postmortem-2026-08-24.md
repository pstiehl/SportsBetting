# Weekly loss post-mortem — cross-sport aggregate

Generated: 2026-08-24T12:01:28.268590Z

## Headline

- **6 sport(s) backtested**: atp, mlb, nba, nfl, pga, wta
- **16,322 bets**, **5,983 losing bets** classified
- **Cross-sport top driver**: `pure_variance` (1,599 losing bets — 26.7% of all classified losses)

## Per-sport headline

| Sport | n_bets | ROI | CLV proxy | Losses | Dominant bucket | Window |
|---|---:|---:|---:|---:|---|---|
| atp | 5,356 | -3.79% | +1.98pp | 1,798 | `favorite_upset` | 2022-01-01 → 2024-12-31 |
| mlb | 3,075 | -3.91% | +3.24pp | 1,424 | `pure_variance` | 2022-01-01 → 2023-12-31 |
| nba | 2,823 | -7.86% | -6.69pp | 1,034 | `form_signal_wrong` | 2022-01-01 → 2024-06-30 |
| nfl | 416 | +1.45% | +7.51pp | 133 | `divisional_misjudged` | 2022-09-01 → 2024-12-31 |
| pga | 0 | — | — | 0 | `—` | 2022-01-01 → 2024-12-31 |
| wta | 4,652 | -3.28% | +1.26pp | 1,594 | `favorite_upset` | 2022-01-01 → 2024-12-31 |

## Per-sport loss buckets

### atp

| Bucket | Count | % of losses |
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

### mlb

| Bucket | Count | % of losses |
|---|---:|---:|
| `pure_variance` | 688 | 48.3% |
| `pitcher_signal_wrong` | 352 | 24.7% |
| `line_moved_against` | 149 | 10.5% |
| `generic` | 133 | 9.3% |
| `rolling_signal_wrong` | 102 | 7.2% |

### nba

| Bucket | Count | % of losses |
|---|---:|---:|
| `form_signal_wrong` | 482 | 46.6% |
| `pure_variance` | 258 | 25.0% |
| `generic` | 234 | 22.6% |
| `line_moved_against` | 37 | 3.6% |
| `b2b_fatigue` | 23 | 2.2% |

### nfl

| Bucket | Count | % of losses |
|---|---:|---:|
| `divisional_misjudged` | 42 | 31.6% |
| `generic` | 25 | 18.8% |
| `pure_variance` | 24 | 18.0% |
| `rolling_signal_wrong` | 16 | 12.0% |
| `bye_off_overrated` | 9 | 6.8% |
| `rest_disadvantage` | 6 | 4.5% |
| `prior_disagreement_wrong` | 6 | 4.5% |
| `line_moved_against` | 5 | 3.8% |

### wta

| Bucket | Count | % of losses |
|---|---:|---:|
| `favorite_upset` | 332 | 20.8% |
| `pure_variance` | 324 | 20.3% |
| `generic` | 294 | 18.4% |
| `line_moved_against` | 237 | 14.9% |
| `ranking_signal_wrong` | 197 | 12.4% |
| `surface_form_wrong` | 86 | 5.4% |
| `fatigue_disadvantage` | 73 | 4.6% |
| `h2h_signal_wrong` | 51 | 3.2% |

## Cross-sport bucket roll-up

| Bucket | Cross-sport count | % of all classified losses |
|---|---:|---:|
| `pure_variance` | 1,599 | 26.7% |
| `generic` | 927 | 15.5% |
| `favorite_upset` | 806 | 13.5% |
| `line_moved_against` | 683 | 11.4% |
| `form_signal_wrong` | 482 | 8.1% |
| `ranking_signal_wrong` | 420 | 7.0% |
| `pitcher_signal_wrong` | 352 | 5.9% |
| `surface_form_wrong` | 198 | 3.3% |
| `fatigue_disadvantage` | 169 | 2.8% |
| `rolling_signal_wrong` | 118 | 2.0% |
| `h2h_signal_wrong` | 107 | 1.8% |
| `divisional_misjudged` | 42 | 0.7% |
| `best_of_5_variance` | 36 | 0.6% |
| `b2b_fatigue` | 23 | 0.4% |
| `bye_off_overrated` | 9 | 0.2% |
| `rest_disadvantage` | 6 | 0.1% |
| `prior_disagreement_wrong` | 6 | 0.1% |

## Reading the buckets

- **`pure_variance`** — pick prob in [0.45, 0.55]. Acceptable.
- **`line_moved_against`** / **`market_proxy_dominant`** — model edge vs market proxy < 1pp. We're losing to the vig. Edge gate fixes this; feature work doesn't.
- **`form_signal_wrong`** / **`pitcher_signal_wrong`** / **`pace_signal_wrong`** / **`rolling_signal_wrong`** — model overweighted a feature class. This IS a feature-quality signal.
- **`favorite_upset`** — backed the chalk, chalk lost. May need upset weighting.
- **`b2b_fatigue`** / **`rest_disadvantage`** — rest features need bigger coefficients.
- **`generic`** — fallback. Large generic = taxonomy under-fitting.
