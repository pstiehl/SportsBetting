# Phil's Flashcat Rescue Plan

Acknowledged. You're right — the live site is fake data, the backtest is NFL-only, and a flat $100 bet on every event is a structurally negative-ROI strategy because it's paying the vig on every pick. Here's the plan, executing now.

## Root cause confirmed
- `THE_ODDS_API_KEY` is unset on `pstiehl/SportsBetting` repo secrets, so `OddsAPI.fetch_events()` silently falls back to `data/samples/odds_api_sample.json` which is committed NBA exhibition data (Knicks@Celtics, Thunder@Nuggets). That's why the live site shows fake NBA games on a day with zero NBA.
- Today (2026-05-29) is MLB regular season + Roland-Garros R1/R2. No NBA, no NFL.
- Backtest only covers NFL via `nflverse`. NBA/MLB/Tennis have no historical connectors → reweighting is single-sport, blended ROI ≈ -vig.

## Phase A — Kill the fake data (PR #1, "fail loud" + MLB + Tennis live)
1. **Fail loud**: in `build_site` / `cli build`, if every live source returns 0 events for an in-season sport on today's date, raise a `NoLiveDataError` and ABORT the build. CI will go red. No silently shipping stale samples.
2. **Auto-detect in-season sports**: query `https://api.the-odds-api.com/v4/sports/?all=false` once at build time. Only build for sports whose key is active. Today this should return `baseball_mlb`, `tennis_atp_french_open`, `tennis_wta_french_open` (and others), and NOT return NBA/NFL until those seasons resume.
3. **Add MLB to OddsAPI connector** — sport key `baseball_mlb`.
4. **Add ESPN MLB scoreboard** connector: `https://site.api.espn.com/apis/site/v2/sports/baseball/mlb/scoreboard`. Already have ESPN framework, just register MLB league.
5. **Add Tennis sources**:
   - ESPN ATP + WTA scoreboards
   - The Odds API tennis keys (read live list, don't hardcode)
   - Devigged market consensus from book moneylines as the baseline probability
6. **Sample data quarantine**: rename `data/samples/odds_api_sample.json` → `data/samples/odds_api_sample.example.json` and ONLY use it when `FLASHCAT_USE_SAMPLES=1` is set. CI never sets this. Local dev opt-in only.
7. **Tests** for: in-season detection parser, MLB OddsAPI parsing, ESPN MLB parser, ESPN tennis parser, fail-loud behavior.
8. **PR** to `pstiehl/SportsBetting:main` from `gregstiehl:fix/no-fake-data-mlb-tennis`.

## Phase B — Historical backtest expansion (PR #2)
1. **NBA historical**: use `balldontlie.io` for game results (free tier) + Odds API historical archive if Phil's key supports it. If not, scrape `sportsbookreviewsonline.com` archives. Source probs are derived from the closing line via devig — outcomes ONLY used for grading.
2. **MLB historical**: `pybaseball` for game results + same line archive scraper for closing moneylines.
3. **Tennis historical**: **Jeff Sackmann's tennis_atp / tennis_wta CSVs** — these have results + Pinnacle closing odds in the same file going back 10+ years. This is the cleanest historical source available for any sport.
4. **Per-sport backtest**: run backtest separately for nfl, nba, mlb, atp, wta. Write per-sport scoreboards.
5. **Walk-forward / expanding-window split**: train weights on months 1-3, score on month 4; train 1-4, score 5; etc. NO outcome leakage into source weights inside the backtest window.

## Phase C — Maximize ROI (the part that keeps me alive)
1. **Edge threshold gate**: skip events where `|blended_prob - devigged_market_prob| < 3pp`. Configurable.
2. **Fractional Kelly**: 1/4 Kelly default. Replace flat $100. Configurable via `--stake-mode {flat,kelly_quarter,kelly_half}`.
3. **No-bet rule**: if blended prob is within 2pp of devigged market on the picked side, skip.
4. **ROI-weighted source blending**: add a `--weight-by {brier,roi,blend}` flag. Default `blend = 0.5*brier_softmax + 0.5*roi_softmax`. Market-close gets weight 0 if vig > 0 (it's by construction zero-EV) — that's a fundamental fix.
5. **Walk-forward enforcement**: assert no leakage with a unit test that shuffles outcomes and confirms ROI distribution is symmetric around 0.

## Phase D — GitHub Actions / Pages
1. Confirm `.github/workflows/daily-refresh.yml` is enabled. Already present, looks fine.
2. PR description includes: "Phil: set `THE_ODDS_API_KEY` in repo secrets at https://github.com/pstiehl/SportsBetting/settings/secrets/actions. Confirm Pages source = main branch / docs folder."
3. After Pages config + secret, daily 11:15 UTC cron keeps the site fresh.

## Phase E — Honest report to Phil
- Per-sport ROI numbers, no cherry-picking.
- What still needs Phil's hand: THE_ODDS_API_KEY secret, Pages source, possibly historical Odds API key.
- All PR links.

## Honesty pact
- If after Phase C the blended ROI is still negative, I report it negative and explain the next experiment. I do NOT cherry-pick windows or sports.
- Source probabilities are AS-OF-pregame only. Outcomes only grade.
- Tests stay green.

Going to work now.
