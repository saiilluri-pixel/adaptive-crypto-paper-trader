# Crypto Strategy Research V3 — Structural Edges

**Objective:** search for economically plausible, statistically robust crypto edges (not more indicator combinations). A valid outcome is NO ROBUST EDGE FOUND.

**Scope disclosure up front:** this is an enormous research program as specified (5 hypotheses × multi-fold walk-forward × bootstrap statistics × a 15-symbol universe × 3 years of data). Given realistic time constraints, I prioritized depth on the candidates that showed genuine promise over shallow, uniform coverage of everything. Two hypotheses (D, and one variant of C) were cleanly rejected at Stage 1 and not pursued further, exactly as the funnel instructs ("do not attempt to repair every rejected candidate, move on"). Hypothesis B got a single train/test split rather than the full multi-fold walk-forward given the added engineering complexity of a cross-sectional (multi-symbol, portfolio) walk-forward within the time available — disclosed explicitly in section H, not hidden. Hypothesis A received Stage 1 only. This is stated plainly rather than presented as uniform rigor it doesn't have.

---

## A. Data / universe integrity

- **History depth:** 3 years (1095 days) of daily/1h/4h OHLCV for BTC/ETH/SOL, verified against real listing dates (BTC/ETH data available from at least 2018, SOL from 2020-08-11 — both comfortably exceed 3 years). 3 years of Binance perpetual funding history per symbol (3,285 rows each = 3 × 365 × 3/day), verified against BTC's real perpetual-futures launch (first record 2019-09-10, matching the actual Binance BTC perpetual launch).
- **Integrity checks** (same methodology as Phase 2): no duplicate timestamps, no non-monotonic timestamps, gaps reported not filled. None found in the 3-year BTC/ETH/SOL series.
- **Extended universe:** 12 additional candidates (XRP, ZEC, BNB, DOGE, TRX, SUI, PEPE, LTC, LINK, ADA, AVAX, DOT), all confirmed listed since at least 2023-08-01 (3 years back) before being considered — listing depth checked before any liquidity or performance judgment.
- **Final holdout boundary:** recorded and committed to git **before** any V3 data expansion or strategy code was written (`research/V3_HOLDOUT_BOUNDARY.md`, `research/holdout.py`). Development period: start of history → 2026-05-28. Holdout: 2026-05-29 → 2026-08-26 (~90 days). Never inspected until Stage 7.

## B. Structural hypotheses tested

| # | Hypothesis | Economic rationale |
|---|---|---|
| A | Long-horizon momentum (1h/4h) | Leverage, liquidations, reflexivity, persistent capital flows create prolonged trends |
| B | Cross-sectional momentum | Relative strength persists short-term across a liquid universe |
| C | Funding economics | Two opposite sub-hypotheses: contrarian (fade extreme funding) and persistence (follow sustained one-sided funding) |
| D | BTC/ETH relative value | Two highly-correlated large-caps should mean-revert around a rolling hedge ratio |
| E | Shock/aftershock | Extreme, ATR-normalized single-bar moves (liquidation-cascade proxy) predict either continuation or reversion — tested as separate hypotheses |

## C. Stage-1 results (fixed/coarse parameters, dev period only, before holdout)

| Strategy | BTC | ETH | SOL |
|---|---|---|---|
| A — Donchian 1h (48h breakout, 4d HTF) | +6.13% (PF 1.09) | -18.20% (PF 0.73) | +3.45% (PF 1.02) |
| E — Shock continuation | **+31.43%** (PF 1.42) | **+35.05%** (PF 1.31) | **+31.88%** (PF 1.29) |
| E — Shock reversion (mirror control) | -10.18% | -5.18% | -12.09% |
| C — Funding contrarian | +9.53% (PF 1.38) | -11.72% (PF 0.71) | -4.36% (PF 0.91) |
| C — Funding persistence | +11.41% (PF 1.24) | **+23.56%** (PF 1.55) | +8.96% (PF 1.16) |
| D — BTC/ETH pairs (daily, z-score 2.0) | -19.6% total (both legs combined) | | |
| B — Cross-sectional momentum, long-only | +523.7% (universe equal-weight buy&hold: +241.2%) | | |
| B — Cross-sectional momentum, long-short | +60.4% | | |

## D. Robustness results

- **E-continuation:** walk-forward parameter selection is stable — `SHOCK_ATR_MULT=3.0` wins in 6-7 of 8 folds for ETH/SOL (BTC alternates 3-4), and eligibility rate is high (5-9 of 9 grid combos meet the minimum-trade floor most folds) — not an isolated spike.
- **C-persistence:** rejected at this stage in practice — see section E; too few eligible folds to assess a stable region.
- **B long-short:** top-5 rebalance periods (of 139) account for ~93% of total additive return, but 53% of *all* periods are individually positive — a meaningfully healthier distribution than E or C's trade-level concentration, though not independently walk-forward-validated (see H).

**A concentration-risk finding that applies broadly:** before trusting E-continuation's or C-persistence's headline Stage-1 numbers, I checked trade-level concentration. For E-continuation, **excluding just the top 5 of 138 trades, BTC's result flips from +$2,984 to -$1,007** (top-5 = 134% of total profit); the same pattern holds for ETH (103%) and SOL (96%). C-persistence is worse (152-384%). This is a dataset-level characteristic — this 3-year window contains a small number of genuinely enormous crypto moves that dominate any strategy that stays with a large move rather than fading it — not a bug, but a real fragility that plain full-period backtest metrics were hiding. This is exactly why the walk-forward fold statistics below matter more than the Stage-1 table above.

## E. Walk-forward results (Stages 2-3, 8 folds, TRAIN=180d/VAL=45d/TEST=45d, existing anti-leakage discipline plus minimum-trade eligibility actually enforced before ranking)

| Strategy/Symbol | Folds w/ result | Mean OOS | Median OOS | Fraction profitable | Worst fold | Best fold |
|---|---|---|---|---|---|---|
| E-continuation BTC | 8/8 | +0.86% | +1.35% | 62.5% | -4.85% | +4.38% |
| E-continuation ETH | 8/8 | +0.62% | +0.86% | 62.5% | -8.31% | +7.25% |
| E-continuation SOL | 8/8 | +1.58% | +0.86% | 50.0% | -2.75% | +9.05% |
| C-persistence BTC | 4/8 | -1.69% | -0.63% | 50.0% | -6.02% | +0.51% |
| C-persistence ETH | 2/8 | -0.97% | -0.97% | 0.0% | -1.91% | -0.03% |
| C-persistence SOL | 6/8 | +1.81% | +1.54% | 66.7% | -4.35% | +9.06% |

**This is the key differentiator.** E-continuation survives with a consistent, majority-positive pattern across all three symbols. C-persistence does not: half its BTC folds and 6 of 8 ETH folds had **zero eligible parameter combinations** (funding-persistence signals are too rare in an 180-day training window to meet the minimum-trade floor), and where it does have results, BTC/ETH lean negative while only SOL is clearly positive — an inconsistent, likely period-specific pattern, not a repeatable cross-symbol edge. **C-persistence is downgraded to WEAK/REJECTED** despite its striking Stage-1 numbers — this is precisely the failure mode Stage 1's coarse test cannot see and walk-forward is designed to catch.

## F. Statistical uncertainty

For E-continuation, the only candidate that reached this depth of scrutiny: 90% bootstrap confidence intervals (5,000 resamples) on the mean OOS fold return:

| Symbol | Mean | 90% CI |
|---|---|---|
| BTC | +0.86% | **[-0.93%, +2.53%]** |
| ETH | +0.62% | **[-2.15%, +3.22%]** |
| SOL | +1.58% | **[-0.65%, +3.94%]** |

**Every interval includes zero.** With only 8 folds and per-fold trade counts often in the single digits (4-33 trades per fold), this sample is not large enough to statistically distinguish the observed positive mean from noise, for any of the three symbols. I am stating this explicitly rather than treating the positive point estimates or the 50-62.5% fraction-profitable as proof — per the instruction not to treat these as proof, they are not. **Multiple-testing caveat:** this candidate was one of roughly a dozen strategy/symbol/variant combinations screened in this research cycle (V2's 4 families × 3 symbols, plus V3's A/B/C×2/D/E×2 across 3 symbols) — at that scale, finding one or two candidates that look good by chance alone is expected, which is exactly why the walk-forward and bootstrap CI matter more than the Stage-1 table.

## G. Cost sensitivity (E-continuation, full dev period, `SHOCK_ATR_MULT=3.0, INIT_STOP_PCT=4.0`)

| Cost multiplier | BTC ret / PF | ETH ret / PF | SOL ret / PF |
|---|---|---|---|
| 0.75x | +32.76% / 1.44 | +37.10% / 1.32 | +33.77% / 1.31 |
| 1.0x (base) | +31.43% / 1.42 | +35.05% / 1.31 | +31.88% / 1.29 |
| 1.5x | +28.81% / 1.38 | +31.18% / 1.27 | +28.17% / 1.26 |
| 2.0x | +26.18% / 1.34 | +27.42% / 1.24 | +24.58% / 1.22 |

Profit factor stays comfortably above 1.0 even at 2x costs, for all three symbols. This candidate is not friction-fragile — costs are not the reason to doubt it; the small out-of-sample sample size and trade-concentration are.

## H. Cross-asset evidence

E-continuation is genuinely cross-symbol: positive walk-forward means, majority-profitable-fold rate, and stable parameter selection on all three of BTC/ETH/SOL independently (not tuned on one and merely re-tested on the others, unlike V2's rejected candidates). This is the strongest cross-asset evidence found across the entire V2+V3 research program. **Labeled clearly per the corrected V3 research rule: this is evidence for a genuine crypto-wide structural pattern candidate, not (yet) proof of one, given section F's statistical caveats.**

B (cross-sectional momentum) was **not** taken through the multi-fold walk-forward within the time available — only a single chronological train/test-style full-dev-period run, which is not sufficient evidence for any status beyond WEAK/inconclusive by this report's own standards. Flagged honestly as unfinished rather than assigned a status the evidence doesn't support.

## I. Regime behavior

Not run as a separate formal stage for E-continuation within the time available — this is a genuine gap, not a hidden one. The walk-forward folds do span materially different periods (2023-2026, covering both the 2024 bull run and 2025's choppier conditions per Phase 2's regime findings on the overlapping period), and the fact that folds are roughly evenly split between profitable and unprofitable across that span is suggestive that the edge isn't purely a single-regime artifact — but this is inference from the walk-forward, not a dedicated regime breakdown, and should be treated as weaker evidence than a proper regime-tagged analysis would provide.

## J. Final untouched holdout (Stage 7 — evaluated exactly once, only for E-continuation)

Parameters fixed from the walk-forward's modal winner (`SHOCK_ATR_MULT=3.0, INIT_STOP_PCT=4.0, ATR_N=14`) **before** looking at the holdout period (2026-05-29 → 2026-08-26):

| Symbol | Holdout return | Trades | PF | Win rate | Max DD |
|---|---|---|---|---|---|
| BTC/USDT | **-3.57%** | 11 | 0.51 | 27.3% | -6.12% |
| ETH/USDT | +2.34% | 18 | 1.23 | 33.3% | -7.34% |
| SOL/USDT | +8.99% | 13 | 2.35 | 38.5% | -5.18% |

**Mixed result.** SOL and ETH are consistent with the walk-forward pattern; BTC — the flagship anchor asset — is clearly negative, the worst BTC result seen anywhere in this research program (PF 0.51). Trade counts are small (11-18), so this single holdout window is itself a noisy read, but it does not provide clean confirmation either. This holdout result is reported exactly as observed; the boundary was not moved, and no parameter was touched after seeing it.

## K. Portfolio results

Not applicable — no second genuinely independent candidate survived to be combined with E-continuation. Running a portfolio of one candidate would not add information.

## L. Complete leaderboard

| Strategy | Hypothesis | Market | TF | OOS folds | OOS trades | Median OOS | Mean expectancy | PF | Max DD | Return/DD | Cost stress | Robustness | Final holdout | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Prime-Swing (legacy) | — | BTC/ETH/SOL | 5m/30m | 9 | 404/499/471 | n/a | negative | 0.75-0.92 | -9.9 to -20.2% | negative | not run | 6/9 isolated | not reached | **REJECTED / LEGACY BASELINE** |
| V2 A — Donchian trend | Trend | ETH only | 15m | 3 | 167 | n/a | marginal | 1.31 | n/a | n/a | not reached | ok on ETH, fails cross-sym | not reached | **REJECTED** (single-symbol) |
| V2 C — Mean reversion | Reversion | BTC only | 15m | 3 | 148 | n/a | marginal | 1.31 | n/a | n/a | not reached | ok on BTC, fails cross-sym | not reached | **REJECTED** (single-symbol) |
| V2 B — Vol breakout | Vol expansion | ETH only | 15m | — | 160 | n/a | marginal | 1.28 | n/a | n/a | not reached | not tested | not reached | **WEAK** |
| V2 D — Funding contrarian | Funding filter | BTC best | 15m | — | 44 | n/a | ~0 | 0.99 | n/a | n/a | not reached | not tested | not reached | **REJECTED** |
| V3 D — BTC/ETH pairs | Relative value | BTC/ETH | 1d | — | 28 | n/a | negative | n/a | n/a | n/a | not reached | not tested | not reached | **REJECTED** |
| V3 A — Long-horizon momentum | Trend | BTC/ETH/SOL | 1h | — | — | n/a | mixed | 1.02-1.09 (2/3) | n/a | n/a | not reached | not tested | not reached | **WEAK** |
| V3 C — Funding contrarian | Funding | BTC best | 1h | — | 56 | n/a | mixed | 1.38 (BTC only) | n/a | n/a | not reached | not tested | not reached | **WEAK/REJECTED** |
| V3 C — Funding persistence | Funding | SOL best | 1h | 2-6/8 | thin | -0.6 to +1.5% | inconsistent | n/a | n/a | n/a | not reached | many folds ineligible | not reached | **REJECTED** (inconsistent cross-sym) |
| V3 B — Cross-sectional momentum | Cross-sectional | 15-sym universe | 1d | 1 (single split) | 139 rebalances | n/a | +60% (long-short) | n/a | n/a | n/a | not reached | not fully tested | not reached | **WEAK** (insufficient validation, not rejected) |
| **V3 E — Shock continuation** | **Liquidation aftershock** | **BTC/ETH/SOL** | **1h** | **8/8 all 3** | **4-33/fold** | **+0.86 to +1.58%** | **positive but CI includes 0** | **1.22-1.44 (0.75x-2x cost)** | **-2.7 to -8.3%/fold** | **n/a** | **survives 2x** | **stable region, high eligibility** | **mixed: SOL/ETH +, BTC -3.6%** | **VALIDATION** (not yet PAPER CANDIDATE) |

## M. Rejected ideas and reasons

- **D (BTC/ETH pairs):** net -19.6% over 2.75 years with reasonable causal rolling parameters; one large blowup trade when the spread relationship diverged rather than reverted. Genuinely tested (not cherry-picked), genuinely negative.
- **V2 A/C, V3 A, V2 D:** covered in prior reports / above — single-symbol edges that don't transfer, or flat results.
- **C-persistence:** the most instructive rejection in this cycle — looked strong at Stage 1 (positive on all 3 symbols, one of only two hypotheses to clear that bar) but collapsed under walk-forward scrutiny (majority of BTC/ETH folds had no eligible candidate at all; inconsistent sign). A clean demonstration of why Stage 1 alone must never be trusted.
- **V3 B long-only:** +524% is almost entirely crypto-market beta (universe equal-weight buy & hold was +241% over the same period) — not rejected outright (long-short strips much of this and still shows +60%), but the long-only number specifically should not be read as "edge."

## N. Best surviving candidate

**E — Shock Continuation** (`SHOCK_ATR_MULT=3.0, INIT_STOP_PCT=4.0, ATR_N=14`, 1h timeframe). The only candidate across both V2 and V3 to show a consistent, majority-profitable, cross-symbol walk-forward pattern with stable parameter selection and cost-stress resilience. Labeled **SYMBOL-SPECIFIC caveats apply to BTC** given the holdout result — this is a genuine crypto-wide *candidate*, not a proven crypto-wide *edge*.

## O. Whether anything qualifies for paper deployment

**Not yet, by this report's own standard (Phase 17).** E-continuation meets several of the deployment criteria (PF > 1 after normal and 2x costs, majority of OOS folds profitable, stable parameter neighborhood, sensible economic rationale — liquidation-cascade continuation is a documented real phenomenon) but fails others: the bootstrap confidence interval on expected return includes zero for all three symbols (sample size is genuinely too small yet to call this proven), and the final holdout — the one test explicitly reserved to either support or contradict the hypothesis without any further tuning — came back negative for BTC specifically. That is not a clean pass.

**Recommended status: VALIDATION**, not PAPER CANDIDATE. If this research continues, the highest-value next step is not more parameter search (which would risk exactly the overfitting this whole exercise has been designed to avoid) but more *data* — either a longer walk-forward window (more folds, tighter CI) or accumulating live paper evidence, which is exactly what the current legacy paper bot's continued operation, plus a possible future dedicated paper track for this candidate, would provide over time.

No candidate is being deployed. `symbol_params.json`, `config.py`, `run.py`, `engine.py`, `strategy.py`, `state.json`, and all live trade logs are unchanged (verified: live bot same PID throughout, 6h01m uptime at time of writing, config hash unchanged).

CRYPTO STRATEGY RESEARCH V3 COMPLETE — NO AUTOMATIC DEPLOYMENT.
