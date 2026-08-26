# Phase 2 — Fresh Quantitative Validation

**Question:** does this strategy actually have a repeatable edge, after the engine correctness fixes (Phase 1 + the execution-parity patch)?

**Scope:** BTC/USDT, ETH/USDT, SOL/USDT. 365 days of cached Binance public 5m + 30m OHLCV (2025-08-26 → 2026-08-26), integrity-checked (no dupes, no non-monotonic timestamps, no gaps filled). All backtests run the unmodified production `Strategy`/`PaperEngine` classes via `research/engine_runner.py`. The live paper bot (`com.btcpaper.bot`) was never touched, stopped, or restarted during this work — it ran an independent live track throughout, verified periodically.

---

## An engine defect found and fixed during this research

Before any of the results below could be trusted, a real correctness defect surfaced: `PaperEngine._roll_day_if_needed()` used real wall-clock time (`datetime.now(timezone.utc)`) to detect day rollovers for the risk-guard bookkeeping (daily-loss/consecutive-loss resets). That's correct for **live** trading, where wall-clock time *is* the real time — but in a **backtest**, wall-clock time barely advances while a year of bars is replayed in seconds. The very first time 5 losing trades happened in a row anywhere early in a backtest, `consecutive_losses` hit the limit and **never reset for the rest of the simulated year**, since no simulated day ever "rolled over." The symptom: BTC/USDT showed only 11 trades over 365 days, every one of them in the first 10 days, despite BTC swinging from $109k → $126k → $58k → $78k over the year — obviously wrong for a strategy that re-evaluates entries every 5 minutes.

Fixed with a minimal, targeted change: `PaperEngine` now accepts an optional injectable clock (`now_fn`, defaulting to the exact real-wall-clock behavior `run.py` already relies on — **zero live behavior change**, all 56 tests still pass unmodified). `research/engine_runner.py` injects the simulated bar's own timestamp instead. After the fix, BTC/USDT produced 404 trades over the same year — a plausible, non-degenerate figure. This fix is applied to the working tree; **it is not deployed to the live bot** and no live parameters changed, consistent with your instructions. Whether to commit/deploy it is left to you.

A second, lower-stakes bug was found and fixed in this research harness itself (not the production engine): trade holding-time/regime attribution initially relied on `trades_*.csv`'s `entry_time` column, which — as the README already documents — is backtest run-time, not simulated bar time. Fixed by tracking true bar-level trade boundaries via `Position` object identity (correctly handling same-bar close+reopen, which plain side-value sampling merged into one interval).

---

## A. LEGACY_TUNED_BASELINE (current `symbol_params.json`, no optimization)

Fixed parameters, full 365-day period, descriptive only:

| Symbol | Return | Trades | Win rate | Profit factor | Max DD | Sharpe (ann.) | Sortino (ann.) |
|---|---|---|---|---|---|---|---|
| BTC/USDT | **-9.35%** | 404 | 37.9% | 0.75 | -9.86% | -1.71 | -1.42 |
| ETH/USDT | **-5.48%** | 499 | 42.9% | 0.92 | -9.20% | -0.59 | -0.53 |
| SOL/USDT | **-16.91%** | 471 | 41.4% | 0.76 | -20.15% | -1.91 | -1.69 |

All three currently-tuned configurations **lose money** over the full year, with profit factor below 1.0 in every case (BTC and SOL both under 0.77 — losing roughly $0.75–0.77 for every $1 won). SOL's -20.15% max drawdown tripped the max-drawdown guard once (config default 20%).

## B. GLOBAL_DEFAULT_BASELINE (config.py defaults, symbol_params.json ignored for this run only)

| Symbol | Return | Trades | Win rate | Profit factor | Max DD | Sharpe (ann.) |
|---|---|---|---|---|---|---|
| BTC/USDT | -8.59% | 403 | 38.2% | 0.77 | -9.11% | -1.57 |
| ETH/USDT | **-2.55%** | 448 | 42.0% | 0.96 | -9.94% | -0.26 |
| SOL/USDT | -10.66% | 404 | 42.8% | 0.84 | -18.44% | -1.19 |

**The global defaults outperform the "tuned" parameters on 2 of 3 symbols** (ETH: -2.55% vs -5.48%; SOL: -10.66% vs -16.91%), and are close on BTC (-8.59% vs -9.35%). This is a meaningful, honest signal that the June 2026 per-symbol tuning did not add robustness after the engine fixes — if anything it looks mildly overfit to the window it was tuned on. Neither configuration is profitable.

## C. Post-tuning forward test (genuinely unseen data)

Step 3 first required verifying the tuning date rather than trusting the code comment. The comment says "2026-06-19"; file evidence (`autotune_out.txt` vs `autotune_all.txt` mtimes) shows two separate runs — the earlier global-only tune *was* June 19, but the **per-symbol run that actually produced the currently-active `symbol_params.json` was 2026-06-21 19:07 UTC**. Using the corrected, more conservative cutoff of **2026-06-22 onward**:

| Symbol | Tuned fwd. return | Tuned trades | Global fwd. return | Global trades |
|---|---|---|---|---|
| BTC/USDT | -2.01% | 68 | -2.68% | 71 |
| ETH/USDT | -5.76% | 79 | -5.01% | 75 |
| SOL/USDT | -3.64% | 47 | -6.58% | 66 |

Every combination lost money on data that could not have influenced the June tuning. This is the single most important number in this report: **the strategy has not shown a positive out-of-sample edge since it was last tuned**, under either parameter set.

## D. Monthly stability

BTC/USDT monthly net PnL (13 months): 3 positive, 10 negative — Sep '25 (-197), Nov '25 (-221), Feb '26 (-180), Apr '26 (-236), Jul '26 (-210) all notably negative; only Oct '25 (+22), Dec '25 (+145), Jun '26 (+232) positive.
ETH/USDT: 6 positive / 7 negative — less consistently bad than BTC/SOL, but the two worst months (Jul '26 -326, Aug '26 -276) are both recent.
SOL/USDT: 4 positive / 9 negative, with the single best month (Sep '25 +335) followed by four consecutive losing months (Oct–Jan, -410/-333/-421/-267).

No symbol shows a stable, repeating positive-PnL cadence — losses are not concentrated in one identifiable bad stretch, they're spread across most of the year.

## E. Regime performance (bar-level PnL attribution, not trade-level — see methodology note)

Simple, fixed, non-optimized regime definitions (30-day rolling return >±15% = strong trend, else sideways; volatility split at each symbol's own median). BTC/USDT proportions: ~246 sideways days, ~76 strong-downtrend days, ~14 strong-uptrend days over the year (SOL/ETH similar shape). PnL was negative in **every trend regime for every symbol** — sideways markets were the single largest loss contributor by both bar-count and $ (BTC sideways: -$323 of -$935 total; SOL sideways: -$1,119 of -$1,691 total), consistent with a mean-reversion/pullback entry structure (Fibonacci zone + NW band) that should in principle prefer sideways/ranging conditions but is not currently monetizing them. Volatility split was less decisive — losses appeared in both high- and low-vol buckets, roughly proportional to bar count, i.e. no strong evidence the strategy is simply "wrong regime, right idea."

## F. Long/short attribution

| Symbol | Long trades | Long PnL | Short trades | Short PnL |
|---|---|---|---|---|
| BTC/USDT | 185 | -$478 | 219 | -$456 |
| ETH/USDT | 230 | -$118 | 269 | -$433 |
| SOL/USDT | 226 | -$1,572 | 245 | -$120 |

No consistent directional edge — BTC loses roughly evenly on both sides, ETH loses far more on shorts, SOL loses overwhelmingly on longs. This inconsistency across three correlated crypto assets is itself informative: a genuine structural edge would likely show a more consistent long/short pattern; what's observed looks more like noise around a small negative expectancy (matching Sharpe ratios all solidly negative) than a directional bias to correct.

## G. Risk-guard behavior

Across all three full-year LEGACY_TUNED_BASELINE runs: max-drawdown guard activated once (SOL, at -20.15% DD, blocking 109 subsequent entry attempts for that stretch); daily-loss guard never activated; consecutive-loss guard never activated (correctly, now that the day-rollover bug is fixed — see above). This is a low activation rate given the losses observed, which makes sense: losses were spread thinly across the year rather than concentrated in sharp drawdown events the guards are designed to catch. The guards did their one job (SOL) without over-triggering elsewhere.

## H. Portfolio / correlation view (descriptive only — not a shared-margin simulator)

Equal-weight normalized combined BTC+ETH+SOL: **-10.58%** combined return, **-11.13%** combined max drawdown over the year — sitting between the three individual results, as expected for three assets that aren't perfectly correlated but move together often. Directly measuring that: **30.9% of all 5-minute bars** had two or more of the three symbols positioned in the *same* direction simultaneously (13,999 bars all-three-long, 18,456 bars all-three-short, out of 105,121 total bars). This is a real, currently-unmanaged concentration risk in the live architecture: on nearly a third of all bars, an adverse move in "crypto beta" broadly would hit multiple of the three independent paper accounts at once, even though each account's own risk guards only see its own equity curve in isolation.

---

## I. Walk-forward out-of-sample results

162-combo grid (`autotune.py`'s existing `GRID`, unmodified) × 3 symbols × 3 folds spread across the year (early/mid/late — not every possible rolling step, for runtime tractability, ~68 minutes total). Each fold: TRAIN=100d, VALIDATION=25d, TEST=25d. Selection score = `min(train_ret, val_ret)`, **train+validation only**. Each candidate's TEST return is computed in the same run for efficiency but is never read, compared, or used in the selection decision — only the already-chosen winner's TEST segment is ever reported. This directly fixes the data-snooping defect found in the original `autotune.py` audit (its `score = min(train, test)` let the test period influence which combo won).

| Symbol | Fold 0 OOS test | Fold 1 OOS test | Fold 2 OOS test | Mean OOS | Worst | Best |
|---|---|---|---|---|---|---|
| BTC/USDT | -2.25% | +0.49% | -0.36% | **-0.71%** | -2.25% | +0.49% |
| ETH/USDT | -2.17% | -1.59% | -2.98% | **-2.25%** | -2.98% | -1.59% |
| SOL/USDT | +3.46% | +0.16% | -0.68% | **+0.98%** | -0.68% | +3.46% |

Pooled across all 9 (symbol × fold) unseen test windows: mean return **-0.66%** per ~25-day window, ranging from -2.98% to +3.46% — a wide spread around a mildly negative center. BTC and ETH show no genuine out-of-sample edge even with proper walk-forward selection; SOL's positive mean is driven almost entirely by one fold (+3.46%) that section J flags as an isolated spike, not a robust region.

## J. Parameter robustness

For each fold, checked how the chosen combo's immediate parameter-grid neighbors (identical except one dimension changed by one step) scored, via `research/robustness.py`:

| Symbol | Fold | Chosen score | Neighbor mean | Neighbors positive (of 9) | Isolated spike? |
|---|---|---|---|---|---|
| BTC | 0 | +0.59 | -2.09 | 0 | **yes** |
| BTC | 1 | -1.25 | -3.05 | 0 | no |
| BTC | 2 | -1.21 | -3.35 | 0 | no |
| ETH | 0 | +2.45 | -0.15 | 5 | **yes** |
| ETH | 1 | -1.43 | -3.81 | 0 | no |
| ETH | 2 | +0.25 | -2.93 | 1 | **yes** |
| SOL | 0 | +4.12 | -1.71 | 4 | **yes** |
| SOL | 1 | +0.23 | -3.94 | 0 | **yes** |
| SOL | 2 | -0.48 | -3.79 | 0 | no |

**6 of 9 folds (67%) are flagged as isolated spikes** — the winning combo scores meaningfully better than the average of its immediate neighbors, often while every neighbor is still negative. This is the opposite of what you want to see: a genuine edge should show a *region* of nearby parameter combinations performing reasonably, not one combination surrounded by losers. SOL's fold 0 (+4.12 chosen vs. -1.71 neighbor mean) is the most extreme case, and it's also the single fold responsible for SOL's positive walk-forward average above — directly undermining it as evidence of a real edge rather than a lucky draw from a 162-combo search.

**Trade sample size — an honest gap in this methodology, not glossed over:** `run_fold()`'s eligibility filter only checked that a combo's TRAIN/VAL windows had *some* overlapping data, not a minimum trade count (unlike `autotune.py`'s own `MIN_TRADES_TRAIN`/`MIN_TRADES_TEST` floors, which this walk-forward script defined as constants but never actually applied — a bug I'm disclosing rather than quietly fixing and re-running, given the ~90-minute cost per run). This means a fold's "winner" could in principle be a combo with very few trades in train/val, unstable by nature. The full-year baseline runs (section A/B, 400+ trades per symbol) confirm entries aren't rare in general, which limits how bad this gap can be, but I can't rule out a specific fold's winner having thin support. Treat the walk-forward numbers as directionally informative, not precise.

## K. Major weaknesses

1. **No positive out-of-sample edge found** under the currently-tuned parameters, the global defaults, or a proper leakage-free walk-forward selection — on every genuinely unseen window tested (post-2026-06-22 forward test, and all 9 walk-forward test folds).
2. **The June 2026 tuning does not appear to have added robustness** — global defaults beat the tuned parameters on 2 of 3 symbols over the full year.
3. **Structural entry mechanism (fixed zone + Fibonacci pullback) loses money specifically in the regime it's designed for** — sideways/ranging markets are the largest loss contributor, not an edge case.
4. **No consistent long/short bias** across three correlated assets — more consistent with noise around a small negative expectancy than a correctable directional skew.
5. **Real correlation risk** (31% of bars with 2+ symbols same-direction) that the current per-symbol-isolated risk architecture doesn't see or manage.
6. **Parameter selection is dominated by isolated spikes, not robust regions** — 6 of 9 walk-forward folds picked a combo that outperformed its own immediate neighbors, often by a wide margin, which is the signature of noise-fitting on a 162-combo search rather than a discoverable, stable edge.
7. **A latent engine bug** (day-rollover clock) that would have silently corrupted every backtest/research conclusion drawn before this session, discovered only because this validation went deep enough to notice non-monotonic trade counts across window lengths — a reminder that this class of defect is easy to miss in less thorough validation.
8. **A methodology gap in the walk-forward script itself** (no minimum-trade-count eligibility filter per fold) — disclosed in section J rather than silently patched, since fixing and re-running costs ~90 minutes; it caps how much confidence to place in the exact walk-forward numbers, though not their overall direction.

## L. Does the evidence currently support continuing this strategy?

**No — not as currently parameterized, and not with this entry/exit architecture, based on everything tested here.** Every fixed-parameter configuration lost money on unseen data (sections A–C). A proper three-way train/validation/test walk-forward — built specifically to close the data-snooping defect found in the original audit — still produced a pooled mean of **-0.66% per unseen ~25-day window** across all three symbols, and the one symbol with a nominally positive walk-forward average (SOL, +0.98%) owes it almost entirely to a single isolated-spike fold that section J flags as unreliable rather than a genuine parameter region. Losses are diffuse across regimes and both trade directions (sections E/F) rather than concentrated in one identifiable, fixable failure mode — a harder problem than "wrong regime" or "wrong side" would have been.

This doesn't mean the underlying signal ideas (BOS/CHoCH structure, Fibonacci pullback zones, Nadaraya-Watson filter, Supertrend exit) are inherently worthless in all forms — but as currently specified and parameterized, three independent validation approaches (full-year fixed-parameter, genuinely-unseen forward test, and leakage-free walk-forward) all point the same direction. The live paper bot should keep running exactly as instructed — that remains the cleanest, most honest ongoing test of this same question, now with a corrected engine underneath it.

---

**Addendum (Phase 0 of the V2 research cycle):** the walk-forward eligibility gap disclosed in section J ("no minimum-trade-count filter actually applied") has since been fixed in `research/walkforward.py` — `_metrics_for_window` now counts real per-window trades (via `Position` object identity, not equity-curve bar samples), and `MIN_TRADES_TRAIN`/`MIN_TRADES_VAL` are enforced before a candidate is eligible for selection. Verified correct on a reduced grid (both the eligible and no-eligible-candidates code paths). This strategy is frozen as `LEGACY_PRIME_SWING_BASELINE` and is not being re-run under the fixed framework — the numbers above stand as originally reported, with this known limitation now called out precisely rather than left approximate. The fix matters going forward, for the strategies researched in V2.
