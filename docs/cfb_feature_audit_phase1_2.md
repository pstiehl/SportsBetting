# CFB Feature Audit — Phase 1+2 (Combined)

**Date:** 2026-08-24  
**Status:** Phase-1 was lost in workspace wipe; this doc covers the combined re-run.

---

## 1. Existing Connectors

### Production connectors (`src/flashcat/sources/`)

| Connector | Type | Features consumed |
|---|---|---|
| `cfb_cfbfastr_epa.py` (`cfb-cfbfastr-epa`) | Historical + live | Season-level off/def PPA per team, HFA dummy, conference tier dummy. Falls back to ESPN score-derived synthetic PPA when CFBD API gated. |
| `cfb_espn_fpi.py` (`espn-fpi-cfb`) | Live only (no historical archive) | ESPN FPI gameProjection (home-win %). Not usable for backtesting. |
| `cfb_market_consensus.py` | Market proxy | Consensus market probabilities where available |

### Data availability

| Source | Historical data available | Date range | Notes |
|---|---|---|---|
| CFBD API (`collegefootballdata.com`) | ❌ Gated (401 without paid API key) | — | PPA/EPA per play unavailable. Returns 401 on /games and /ppa/teams. |
| ESPN public scoreboard | ✅ Free | 2021–2024 | Final scores + team names + conference IDs. No PPA, no play-by-play. |
| Market closing lines | ❌ Not available | — | No free CFB historical odds archive. SBR is dead; Odds API requires paid key. |

### What's in `source_history.db` for CFB

No rows in `predictions` table for CFB sources (no prior backfill). This is the first CFB walk-forward run.

---

## 2. Feature Wish List vs. Implementation

| Category | Desired | Implemented (Phase 1+2) | Blocked |
|---|---|---|---|
| Rolling team efficiency | Off/Def efficiency L3/L5/L10 | ✅ L5 off/def (pts scored/allowed) | L10 omitted (CFB teams play 12-15 games/season; L10 overlaps near full season) |
| QB metrics | Completion %, YPA, TD/INT L5 | ❌ | Requires play-by-play (CFBD gated) |
| Home field advantage | Home/away split, neutral site flag | ✅ `home_field_flag` constant | Neutral site flag requires game location metadata (ESPN scoreboard doesn't return it reliably) |
| Conference strength | SOS proxy | ✅ `conf_tier_diff` (P5 vs G5) | Full SOS metric requires opponent schedule lookup |
| Rest days / bye week | Days since last game, bye flag | ✅ `rest_days_diff`, `bye_home`, `bye_away` | — |
| Turnover differential | L5 turnovers | ⚠️ Proxied via `margin_volatility_l5` | Real turnovers need play-by-play |
| Recruiting composite | Year-over-year roster quality delta | ❌ | Requires paid Rivals/247Sports data |
| Market context | Line movement, opening vs closing spread delta | ❌ | No free CFB historical odds |

---

## 3. Data Blockers (Honest Assessment)

1. **CFBD API gated (401)**: The primary CFB analytics source requires a signup-only API key. The ESPN fallback gives us game scores but no play-by-play or PPA metrics. All CFB EPA/PPA features would require either (a) a CFBD subscription or (b) scraping ESPN play-by-play (significant engineering + ToS risk).

2. **No closing line data**: CFB historical moneylines are not freely available. The CLV proxy uses the EPA connector's predicted probability as the market stand-in. Since the market proxy is derived from the same data as the model features (ESPN scores), the -2.38pp CLV proxy is not a reliable calibration signal — the model and the proxy share information.

3. **ESPN scoreboard limited to 400 events per query**: Regular-season FBS has ~60-80 games per Saturday. The `limit=400` parameter per date-range query is sufficient for monthly ranges. Verified operational for 2021-2024.

4. **Conference metadata sparse**: ESPN's `conferenceId` numeric codes don't consistently populate for all teams (G5 schools particularly). The `_ESPN_CONFERENCE_IDS` map covers the major conferences. ~15-20% of games have `None` conference for one team, defaulting to `conf_tier_diff=0`.

---

## 4. Backtest Results Summary

See `data/cfb_walk_forward_backtest.json` for full detail.

| Metric | Value |
|---|---|
| n_bets | 1,823 |
| win_rate | 70.9% |
| ROI | +3.60% |
| CLV proxy (pp) | -2.38pp ⚠️ (see data blocker #2) |
| max_drawdown | $6,936 |
| Sharpe | 0.531 |
| Dominant loss bucket | upset_heavy_favorite (48%) |

**Season breakdown:**
| Season | n | ROI |
|---|---|---|
| 2022 | 24 | -28.0% (warmup period, tiny n) |
| 2023 | 902 | +11.0% |
| 2024 | 897 | -3.0% |

**Verdict:** PROFITABLE aggregate (+3.60%) but **split across regimes**. 2023 was strongly positive (+11%), 2024 was slightly negative (-3%). The +3.60% combined ROI barely clears the vig threshold, and the negative CLV proxy (-2.38pp) indicates the model is not finding true market edge — both model and proxy use the same data source (ESPN scores). Real closing line data would be needed to confirm +EV. **TREAT AS RESEARCH MODE ONLY.**

---

## 5. Next Iteration (Phase 3 Candidates)

Priority order if data access improves:

1. **CFBD subscription** (~$5-10/mo): unlocks PPA, recruiting data, play-by-play. Expected to add QB efficiency, turnover count, and real recruiting composite delta.
2. **The Odds API paid tier**: Real CFB historical closing lines. Would fix the CLV proxy problem immediately.
3. **Neutral site detection**: CFB has ~20 neutral-site bowl/playoff games per season where HFA = 0. Worth detecting from ESPN venue metadata.
4. **L10 rolling windows**: With 3+ seasons of data, L10 is feasible for early-season games in season 3+.
