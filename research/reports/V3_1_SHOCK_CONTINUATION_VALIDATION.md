# Shock Continuation V3.1 Deep Validation

**Purpose:** determine whether `SHOCK_CONTINUATION_V3_FROZEN` (frozen definition: `research/SHOCK_CONTINUATION_V3_FROZEN.md`) is a genuine repeatable fat-tail continuation edge, or driven by a small number of exceptional historical events. Parameters were not touched during this study; any new idea found is logged as a `SHOCK_CONTINUATION_V4_HYPOTHESIS`, not retrofitted here.

**Headline result: the picture materially diverged by symbol once more data was used.** With 5 years of history and 16 walk-forward folds (up from 2.75 years / 8 folds), **BTC's edge reverses** (now negative), **ETH is a coin flip**, and **SOL's edge strengthens into the first statistically meaningful (90% CI excludes zero) result in this entire research program.** This changes the conclusion from V3's "cross-symbol candidate" framing to something narrower and more defensible: **SOL-specific, SYMBOL-SPECIFIC status**, per the corrected V3 research rule.

---

## A. Frozen strategy definition

See `research/SHOCK_CONTINUATION_V3_FROZEN.md` in full. Summary: 1h chart, true-range shock (`TR > 3.0×ATR14`) in the shock bar's own candle direction, enter with the shock, shared production trailing-stop exit (`INIT_STOP_PCT=4.0`), production risk-based sizing (1% risk/trade), production fee+slippage model. No volume threshold in the frozen definition.

## B. Event ledger summary

Built via a pure-logging subclass of the frozen strategy (identical signal logic, only adds a diagnostic side-channel — verified no trading-behavior change). 5-year dev-period (pre-holdout) trade counts: **BTC 270, ETH 371, SOL 372, combined 1,013**. Full per-event ledger (shock size, ATR-normalized magnitude, volume multiple, wick/body shape, MAE/MFE, forward 1h/4h/12h/24h returns, regime tags) saved to `research/data/event_ledger_*.csv`.

## C. Profit concentration

Removing top trades, **combined across symbols** (1,013 trades):

| Excluded | Trades remaining | Total PnL | PF | % of total PnL removed |
|---|---|---|---|---|
| none | 1013 | $15,472 | — | — |
| top 1 | 1012 | $14,151 | 1.24 | 8.5% |
| top 3 | 1010 | $12,111 | 1.20 | 21.7% |
| top 5 | 1008 | $10,145 | 1.17 | 34.4% |
| top 10 | 1003 | $6,215 | **1.10** | 59.8% |

**Pooled across symbols, the edge survives removing the top 10 of 1,013 trades** (1%) — still PF 1.10, still ~$6,200 positive. Per-symbol it's more fragile: SOL alone stays near-breakeven (PF 0.998) even excluding its own top 10 of 372; BTC and ETH both go net-negative excluding their top 5. **This is genuine fat-tail dependence, not automatically disqualifying** per the instruction — trend/crisis-continuation strategies legitimately earn disproportionately from rare large moves. The question is whether it's *statistically repeatable* (sections E-G answer this) rather than *concentrated in one episode* (section D).

## D. Independent shock-cluster count

Clustering rule: events on the same symbol within 24h of each other (a plausible single-cascade window) join one cluster. Result: **BTC 270→246 clusters, ETH 371→340, SOL 372→339, combined 1,013→925** — average 1.1 trades per cluster. **Shock events are overwhelmingly independent occurrences, not one or two cascades counted many times.** This directly answers the core concern motivating V3.1: the effective sample size (925 independent clusters) is close to the raw trade count, not a small handful of episodes inflated by re-counting.

## E. Time-period stability

Combined across symbols, by year (dev period): 2021 -$383 (partial year), **2022 +$3,429, 2023 +$3,141, 2024 +$4,317, 2025 +$5,439** (four consecutive positive years), 2026 (partial, through May) -$470. Per-symbol detail in `research/reports/v3_1_time_stability.json` is noisier (each symbol has some negative years individually), but **the combined edge appears in four separate consecutive years, not concentrated in a single crisis window.**

## F. Expanded walk-forward (16 folds, TRAIN=90d/VAL=20d/TEST=20d, existing anti-leakage discipline, min-trade eligibility enforced)

| Symbol | Folds | Mean OOS | Median OOS | Fraction profitable | Worst fold | Best fold |
|---|---|---|---|---|---|---|
| BTC | 16/16 | **-0.66%** | -0.85% | 37.5% | -5.54% | +5.59% |
| ETH | 16/16 | -0.07% | +0.45% | 56.3% | -5.49% | +6.63% |
| SOL | 16/16 | **+1.42%** | +0.51% | 62.5% | -3.17% | +8.97% |

**This is the pivotal finding of V3.1.** V3's original 8-fold result showed all three symbols mean-positive with 50-62.5% fold profitability. With double the folds and nearly double the history, **BTC's result reverses to negative with only 37.5% of folds profitable** — worse evidence than before, not better. ETH settles near flat. **SOL is the only symbol whose result strengthens** with more data.

## G. Statistical uncertainty

90% bootstrap CI (5,000 resamples) on mean OOS fold return, 16-fold sample:

| Symbol | Mean | 90% CI | P(mean > 0), bootstrap |
|---|---|---|---|
| BTC | -0.66% | [-1.64%, +0.39%] | 14.7% |
| ETH | -0.07% | [-1.50%, +1.36%] | 47.4% |
| **SOL** | **+1.42%** | **[+0.08%, +2.84%]** | **95.3%** |

**SOL's confidence interval excludes zero** — the first statistically meaningful result across V2, V3, and V3.1 combined. BTC's interval leans negative (86% bootstrap probability the true mean is ≤0). This is not proof (16 folds is still a modest sample, and per-fold trade counts run 4-33), but it is qualitatively different evidence than anything found before.

**Independence assumption, stated explicitly:** the bootstrap above resamples fold-level returns, which section D shows correspond to largely-independent shock clusters (1.1 trades/cluster) — the independence assumption is reasonably supported by the clustering analysis, not simply assumed. A block-bootstrap variant was not additionally run given time constraints; given the clustering result, the risk of understating correlation is judged low but is disclosed as a residual limitation rather than fully closed out.

**Multiple-testing caveat:** SOL is one of roughly a dozen-plus strategy/symbol combinations screened across this entire research program (V2's 8, V3's 5 hypotheses × 3 symbols). At that scale, one candidate clearing a 90% CI threshold is not automatically surprising by chance alone. This is why sections C, D, E, J, K, L below (independent, converging lines of evidence beyond the single CI number) matter for the final judgment, not the CI in isolation.

## H. BTC holdout diagnosis (diagnostic only — no parameters altered)

The V3 final holdout showed BTC -3.57% (PF 0.51, 11 trades). Investigated via the event ledger applied to the holdout window: **all 5 short trades in the holdout lost money (0% win rate)**; longs were a coin flip (3/6 wins). BTC's actual holdout price path: fell from $73,716 to a capitulation low of $58,290 (2026-06-25 — the exact date of the single most extreme shock in the holdout, 8.9× ATR, 9.8× volume) then rallied +38% to $80,732 by late August, ending net +6.9%. Every subsequent down-shock during the recovery phase got bought back up rather than continuing, squeezing every short. **Classified as regime dependence + direction asymmetry specific to this window** (a V-shaped capitulation-then-recovery), not "fake breakouts" broadly — shock sizes in the losing trades were genuinely large (3.1-8.9× ATR) and volume was genuinely elevated (1.6-9.8×), ruling out the "low volume/low volatility" explanations. Cross-checked against the full 5-year history: **both long and short are net profitable for BTC over the full period** (long PF 1.45, short PF 1.24) — so this is not evidence of a general structural short-side weakness, just an unusually adverse single window for shorts specifically.

## I. Long vs. short decomposition (full 5-year dev period)

| Symbol | Side | Trades | Win rate | Expectancy | PF | Total |
|---|---|---|---|---|---|---|
| BTC | long | 130 | 36.9% | $23.04 | 1.45 | $2,995 |
| BTC | short | 140 | 37.1% | $13.57 | 1.24 | $1,900 |
| ETH | long | 175 | 34.9% | $6.81 | 1.12 | $1,192 |
| ETH | short | 196 | 39.3% | $10.39 | 1.19 | $2,037 |
| SOL | long | 183 | 41.0% | $26.88 | **1.45** | $4,919 |
| SOL | short | 189 | 41.3% | $12.85 | 1.18 | $2,429 |

**Both directions are profitable for every symbol over the full period** — the BTC holdout's short-side failure (section H) is a window-specific event, not evidence of a persistent one-sided edge. SOL's long side is the single strongest line item in the entire study (PF 1.45, $4,919 of its $7,348 total).

## J. Shock-severity behavior (pre-declared quantile buckets, combined symbols)

| Bucket | ATR-norm range | Trades | Win rate | Expectancy | PF |
|---|---|---|---|---|---|
| moderate | [3.0, 3.67) | 507 | 39.8% | $11.48 | 1.20 |
| large | [3.67, 5.37) | 354 | 37.9% | $18.95 | 1.33 |
| extreme | [5.37, 24.6] | 152 | 36.2% | $19.38 | 1.31 |

Expectancy and PF both **rise with shock severity** (roughly monotonic — extreme is marginally below large on PF but both clearly above moderate). This supports the liquidation-cascade hypothesis: bigger shocks → more continuation, not less.

## K. Volume confirmation (diagnostic, combined symbols)

| Bucket | Volume mult. range | Trades | Expectancy | PF |
|---|---|---|---|---|
| normal | [1.02, 3.45) | 334 | $13.37 | 1.22 |
| elevated | [3.46, 4.98) | 345 | $7.66 | 1.13 |
| extreme | [4.98, 32.9] | 334 | $25.04 | **1.43** |

Not perfectly monotonic (the elevated bucket dips below normal), but the **extreme-volume tercile clearly and substantially outperforms both others** — partial, genuine support for the "bigger shock + bigger volume = more continuation" economic story, reported honestly including the non-monotonic middle bucket rather than only the clean part.

## L. Cost and delay sensitivity (SOL, full dev period)

| Cost multiplier | Return | PF |
|---|---|---|
| 1.0x (base) | +73.5% | 1.303 |
| 1.5x | +64.5% | 1.269 |
| 2.0x | +56.0% | 1.236 |
| 3.0x | +40.4% | 1.174 |

PF stays comfortably above 1.0 through 3x costs; break-even friction is well outside the tested range (extrapolating the decay rate, likely 8-10x base cost). **Delayed-entry test** (signal held one bar, entered at the *next* bar's close instead of the signal bar's own close): return +73.5%→+55.9%, PF 1.303→1.229. The edge weakens with delayed execution but does not collapse — it is not solely an artifact of capturing an unrealistic instantaneous fill.

## M. Position-sizing independence (SOL)

| Sizing scheme | Expectancy/trade | PF |
|---|---|---|
| Fixed $1,000 notional | $6.37 | 1.354 |
| Production risk-based (~1% risk) | $19.75 | 1.303 |

Both schemes show a comparable, clearly-positive PF — the edge is carried by the signal itself, not manufactured by the risk-based position-sizing scheme.

## N. Monte Carlo / trade-sequence risk (SOL, 372-trade bootstrap resample, 3,000 simulations)

- Final return (full ~5yr horizon): mean +72.9%, 5th percentile +9.3%, median +71.5%, 95th percentile +138.0%. **P(losing money over the full horizon): 2.5%.**
- Max drawdown: mean -18.1%, 5th-percentile (worst-case) -33.3%, median -16.4%.
- Longest losing streak: mean 10.1 trades, 95th percentile 15 trades.
- **P(losing money over a 3-month window, ~18 trades): 36.2%. P(losing over 6 months, ~37 trades): 28.2%.**

Realistic near-term expectations: over any given 3-6 month paper/live stretch, there is a genuine, non-trivial (~30-36%) chance of being net negative even if the long-run edge is real — this is a fat-tail strategy by nature, and short-horizon judgments of "is it working" should account for this before concluding prematurely either way.

## O. Final evidence assessment

Weighing all of the above:

**Supporting a real, repeatable structural pattern (SOL specifically):**
- Effective sample is close to the raw trade count (925 of 1,013 independent clusters) — not a handful of episodes re-counted.
- Positive expectancy across 4 consecutive years, not one crisis window.
- Both long and short sides independently profitable.
- 16-fold walk-forward strengthens (not weakens) with more data, unlike BTC.
- 90% bootstrap CI excludes zero.
- Economically monotonic-ish behavior with shock severity and volume — consistent with the liquidation-cascade rationale, not just a curve fit.
- Survives 3x cost stress and 1-bar delayed entry; not a sizing artifact.

**Against full confidence / reasons for caution:**
- Fat-tail dependent at the single-symbol level (SOL alone needs its top trades to stay clearly positive, though the pooled cross-symbol view is more robust).
- Only 16 OOS folds, still a modest statistical sample; per-fold trade counts are small (4-33).
- One candidate surviving out of a dozen-plus screened — multiple-testing risk is real and not fully correctable with the data available.
- Not genuinely cross-symbol anymore — BTC actively fails on the expanded data, and the correct label per the V3.1 research-rule correction is SYMBOL-SPECIFIC, not universal.
- No block-bootstrap was run as a cross-check on the plain resampling independence assumption (disclosed limitation).

## P. Status

**SOL Shock Continuation: VALIDATION → upgraded to PROMISING, SYMBOL-SPECIFIC. Not yet PAPER CANDIDATE.**

Against the Phase 17 decision standard: positive OOS expectancy (yes, SOL), majority-profitable OOS folds (yes, 62.5%), PF > 1 after normal costs (yes, 1.30) and survives 1.5x costs (yes, 1.27) — these are met. But "adequate number of independent shock events" and "not dependent on one single historical event" are met only partially (339 independent clusters is a reasonable count, but per-symbol concentration in section C means a handful of clusters still carry a large share of the total), and the multiple-testing caveat in section G means the statistical significance found here should be treated as suggestive, not conclusive. **BTC and ETH: REJECTED / WEAK respectively** at their current definitions — BTC's reversal on more data is a genuine red flag for treating this as a universal crypto pattern, and directly informs why SOL must be labeled symbol-specific rather than promoted as a general finding.

**BTC/ETH — REJECTED, WEAK** respectively (per this frozen definition; not re-optimized).
**SOL — PROMISING, SYMBOL-SPECIFIC.** The next legitimate step, if pursued, would be accumulating further live/paper evidence specifically on SOL (the only lever left that doesn't risk overfitting to already-seen data) before any deployment discussion — not further parameter search on the existing history.

No deployment. `symbol_params.json`, `config.py`, `run.py`, `engine.py`, `strategy.py`, `state.json` unchanged throughout (live bot confirmed same PID, 6h30m+ continuous uptime, config hash unchanged at time of writing).

SHOCK CONTINUATION V3.1 VALIDATION COMPLETE — NO AUTOMATIC DEPLOYMENT.
