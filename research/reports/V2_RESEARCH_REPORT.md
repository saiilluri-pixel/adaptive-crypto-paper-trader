# Crypto Strategy Research V2

**Objective:** discover and validate a genuinely robust crypto strategy for BTC/ETH/SOL, using the corrected `PaperEngine`, prioritizing positive expectancy and cross-time/cross-asset robustness over maximum historical return.

**Result up front: no candidate cleared the funnel. This report does not manufacture a winner.**

---

## 1. Strategy families tested

Four independent hypotheses, each a single transparent mechanism (not stacked indicators), each exposing 2-3 explicit parameters, each driven through a new [`research/strategies/base.py`](../strategies/base.py) interface (`on_bar()` → `"long"`/`"short"`/`None`, strictly causal — no future-bar access is even possible through the interface) and executed by [`research/generic_runner.py`](../generic_runner.py), which drives the **exact same production `PaperEngine`** (fees, slippage, risk-based sizing, risk guards, corrected trailing-stop math, corrected OHLC-aware stop-fill logic) as the Prime-Swing baseline — no new fills, sizing, or fee logic was written for this research. Exit is uniformly the shared ATR-scaled trailing stop for every family (Supertrend-flip exit disabled), so every candidate is judged on identical execution mechanics.

| Family | Hypothesis | Mechanism |
|---|---|---|
| A — Trend/Momentum | Crypto has persistent directional moves after meaningful breakouts | Donchian breakout of the prior 20-bar range, gated by a 24h higher-timeframe trend filter |
| B — Volatility Breakout | Large moves follow volatility expansion | True-range spike vs. rolling ATR + level breakout (distinct from A: requires expansion, not just a level cross) |
| C — Mean Reversion | Short-horizon overextensions revert in non-trending regimes | Rolling z-score of price vs. its own mean/std, gated to only trade *against* HTF trend absence (not fighting a strong trend) |
| D — Funding-Aware Filter | Extreme perpetual funding predicts reversal | Contrarian: fade extreme funding (short when crowded-long funding, long when crowded-short funding). Real Binance public funding-rate history (no auth needed), 1095 rows/symbol over the year — not fabricated. Tested as an independent hypothesis, not combined with A/B/C. |

Timeframe: 15m (Phase 4's middle candidate), HTF context from a 96-bar/24h lookback. Data: same integrity-checked full-year cache used for Prime-Swing validation, plus newly cached 15m/1h OHLCV and Binance funding-rate history.

## 2. Hypotheses — see table above.

## 3. Coarse results (Stage 1, fixed simple parameters, one run per family × symbol)

| Family | BTC/USDT | ETH/USDT | SOL/USDT |
|---|---|---|---|
| A Donchian trend | -10.92% (PF 0.81) | **+26.30%** (PF 1.27, Sharpe 2.27) | +1.89% (PF 1.02) |
| B Vol breakout | -10.05% (PF 0.83) | **+25.14%** (PF 1.28, Sharpe 2.22) | -14.63% (PF 0.87) |
| C Mean reversion | **+22.81%** (PF 1.24, Sharpe 2.00) | -8.12% (PF 0.89) | -15.14% (PF 0.73) |
| D Funding contrarian | +0.47% (PF 0.99, ~breakeven) | -8.34% (PF 0.77) | -6.27% (PF 0.87) |

## 4. Rejected candidates and why

- **D (funding contrarian) — rejected at Stage 1.** Best case (BTC) is statistically indistinguishable from breakeven (PF 0.99); ETH and SOL both negative. No symbol shows genuine promise. Not "obviously negative" in the catastrophic sense, but not promising enough to spend a grid search on.
- **B (volatility breakout) — deprioritized at Stage 1.** Same single-symbol pattern as A (ETH positive, BTC/SOL negative), and mechanistically similar enough to A (both are breakout-family) that it's likely capturing an overlapping signal rather than an independent one. Given limited research budget, A was carried forward as the family's representative; B's full Stage 1 numbers are preserved in `research/reports/stage1_coarse.json` for the record, not hidden.
- **A (Donchian trend) — rejected at Stage 4 (cross-symbol).** Passed Stage 2 (best combo not an isolated spike — 4 neighbors, mean neighbor return +11%, not the single-spike pattern that doomed several Prime-Swing walk-forward folds) and reached Stage 3. But applying the winning combo (tuned on ETH) to BTC and SOL: **BTC -9.37%, SOL -12.57%** — loses on both. Classic single-symbol-only edge with no economic reason ETH should behave differently from BTC/SOL here.
- **C (mean reversion) — rejected at Stage 4 (cross-symbol).** Same story: passed Stage 2, reached Stage 3, but the BTC-tuned combo applied to ETH and SOL: **ETH -3.54%, SOL -8.11%**. Loses on both.

## 5. Surviving candidates

**None.** All four families are rejected — two at Stage 1 (D outright, B deprioritized as redundant with A), two at Stage 4 (A and C, after passing Stages 2-3, on cross-symbol generalization).

## 6. Walk-forward results (Stage 3, for A and C before their Stage-4 rejection)

Same train(100d)→validation(25d)→test(25d) discipline as the Prime-Swing walk-forward, same anti-leakage guarantee (test never influences selection), same MIN_TRADES eligibility floor (now actually enforced — see Phase 0). Small grids (9 combos for A, 18 for C — each family only has 2-3 tunable knobs).

| | Fold 0 OOS | Fold 1 OOS | Fold 2 OOS | Mean OOS |
|---|---|---|---|---|
| A (Donchian, ETH) | +2.12% | -3.85% | +3.32% | **+0.53%** |
| C (Z-score, BTC) | +1.02% | -1.75% | +3.26% | **+0.84%** |

Both means are near breakeven with real fold-to-fold variance (swings of 5-7 points between folds on a 3-fold sample). More tellingly: **the selection score (train+val) was negative in every single fold for both families** — meaning even the *best available* parameter combination in-sample was usually unprofitable, and the mildly-positive OOS means above are close to what you'd expect from noise around a near-zero or negative true expectancy, not a discoverable edge. This OOS weakness, on top of the Stage 4 cross-symbol failure, is two independent reasons neither family should be trusted.

## 7. Cost sensitivity

**Not reached.** Per the funnel and "do not attempt to repair every rejected candidate, move on" — A and C were already rejected at Stage 4 before cost-stress (Stage 5) would run, and D/B were rejected/deprioritized before Stage 2. Running 1.5x/2x cost stress on candidates that already show negative-to-flat cross-symbol profit factor (0.77-0.97) would not change the conclusion — added friction can only make a losing or breakeven result worse.

## 8. Parameter robustness

Stage 2 neighbor analysis (full detail in `research/reports/stage2_robustness.json`):
- **A (ETH):** best combo `DONCHIAN_N=10, INIT_STOP_PCT=3.0`, 4 grid neighbors, mean neighbor return **+11.0%**, 2/4 neighbors positive — not an isolated spike, a genuinely supportive region *for ETH specifically*.
- **C (BTC):** best combo `ZSCORE_N=10, Z_THRESH=2.5, INIT_STOP_PCT=2.0`, 5 neighbors, mean neighbor return **+7.3%**, 2/5 positive — also not isolated.

Both pass the robustness bar the Prime-Swing walk-forward largely failed (6/9 isolated spikes there). The parameter *region* is real for the symbol it was found on — the problem surfaced later, at Stage 4, is that the region doesn't transfer to the other two symbols at all. Robust-but-narrow is still not the target outcome.

## 9. Cross-symbol evidence

The decisive result of this research cycle. Neither surviving-to-Stage-4 family showed positive performance on more than one symbol, in either direction (native-symbol-tuned parameters applied elsewhere, or vice versa — Stage 1's own table already shows the reverse-direction evidence: A's ETH-good combo family shows BTC/SOL both negative even with their *own* independently-fit Stage-1 defaults, not just the ETH-tuned ones). This is exactly the Phase 9 rejection criterion "single-symbol-only edge with no economic explanation," applied twice.

## 10. Leaderboard

| Strategy | Version | Symbols | TF | Trades (Stage1) | OOS Return | Max DD | Profit Factor | Expectancy | Return/DD | Cost stress | Robustness | Status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Prime-Swing (BOS/CHoCH+Fib+NW/Supertrend) | legacy | BTC/ETH/SOL | 5m/30m | 404/499/471 | -0.66% pooled mean (9 folds) | -9.9% to -20.2% | 0.75-0.92 | negative all 3 | negative | not run | 6/9 folds isolated spike | **REJECTED / LEGACY BASELINE** |
| A — Donchian trend | v1 | ETH (native) | 15m | 167 | +0.53% mean (3 folds) | n/a | 1.31 (Stage1 ETH) | marginal | n/a | not reached | not isolated on ETH, but fails cross-symbol | **REJECTED** (Stage 4) |
| B — Vol breakout | v1 | ETH (native) | 15m | 160 | not walk-forward tested | n/a | 1.28 (Stage1 ETH) | marginal | n/a | not reached | not tested | **WEAK** (deprioritized, redundant with A) |
| C — Mean reversion | v1 | BTC (native) | 15m | 148 | +0.84% mean (3 folds) | n/a | 1.31 (Stage1 BTC) | marginal | n/a | not reached | not isolated on BTC, but fails cross-symbol | **REJECTED** (Stage 4) |
| D — Funding contrarian | v1 | BTC (best case) | 15m | 44 | not applicable | n/a | 0.99 (~breakeven) | ~0 | n/a | not reached | not tested | **REJECTED** (Stage 1) |

Full raw data behind every row: `research/reports/stage1_coarse.json`, `stage2_robustness.json`, `stage3_walkforward.json`, `stage4_cross_symbol.json` (all research-only, none touch live files).

## 11. Best candidate, if any

**None qualifies for paper-candidate status.** A and C are the closest — both passed parameter-robustness (Stage 2) on their native symbol and showed near-breakeven (not catastrophic) walk-forward results (Stage 3) — but both failed the cross-asset requirement (Stage 4) decisively, and both had negative in-sample selection scores throughout their walk-forward folds, meaning the walk-forward's mild positive mean is not well-supported. Neither should be considered "PROMISING" let alone a paper candidate under the stated bar (robustness across BTC/ETH/SOL was an explicit priority, not optional).

## 12. Is any candidate strong enough for live-market paper trading?

**No.** This research cycle's honest outcome is: **NO ROBUST EDGE FOUND**, across four genuinely distinct, simply-specified hypotheses, tested with anti-leakage walk-forward validation and an actually-cross-symbol-checked funnel. This sits alongside Phase 2's finding that the existing Prime-Swing configuration also shows no out-of-sample edge — two independent research efforts, on the same corrected engine, neither producing a strategy that clears a reasonably strict bar.

What this does *not* mean: that no strategy exists in this space, or that these four families are permanently disqualified in every possible parameterization/timeframe. It means the specific, simple, transparent versions tested here — with real costs, real risk guards, real corrected execution, and genuine unseen-data validation — did not find one. The next research generation (if pursued) should treat this as informative: family A/C's *mechanism* had a real, non-spike parameter region on one symbol each, so the failure mode is specifically "doesn't transfer across assets," not "the whole idea is noise" — a narrower, more actionable finding than a blanket rejection would be. Combining a trend filter with mean-reversion entries (currently untested — Phase 3 explicitly said not to combine initially) is one plausible next step, but is out of scope for this generation per your instructions and would need to be declared as a new research generation, not retrofitted onto these results.

Nothing here changes `symbol_params.json`, `config.py`, or the running `com.btcpaper.bot` process (verified: same PID throughout, 5h14m uptime, config hash unchanged). No new strategy code is imported by `run.py` or any live path.
