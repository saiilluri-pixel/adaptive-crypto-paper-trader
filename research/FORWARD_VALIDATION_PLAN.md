# FORWARD_VALIDATION_PLAN — SOL/USDT Shock Continuation V3.1

Committed BEFORE the first forward-shadow trade. This document is the
pre-registration: it fixes the strategy, the historical prior, the minimum
observation period, the evaluation metrics, and the decision rules in
advance, specifically so that none of them can be quietly redefined after
seeing forward results. If this file is ever edited after the first forward
trade, that edit must itself be a new commit with its own message explaining
why, on top of (not replacing) this original text.

> **ADDENDUM (2026-08-28, dated clarification, added after the epoch started, not a rewrite of the above):** a read-only post-deployment verification found that this document did not pin an explicit, machine-usable forward-epoch market-time boundary, and that `state.json`'s `epoch_start_iso` field (process-construction wall-clock time) could be mistaken for one. This is now fixed:
>
> - **Forward-epoch boundary is pinned in `shadow/shock_continuation_sol_v3_1/FORWARD_EPOCH_BOUNDARY.json`**: `forward_epoch_start_ts_ms = 1787878800000` (`forward_epoch_start_iso = 2026-08-28T01:00:00Z`) — the first 1h SOL/USDT bar processed after the initial deployment's indicator-warmup replay completed.
> - **Warmup shocks do NOT count.** The 15 shocks detected while replaying 501 bars of pre-existing history during the initial bootstrap (`phase=warmup` in `events.csv`) are diagnostic context only and are permanently excluded from every forward counter (independent-event tally, the 30-event stopping rule, and all evaluation metrics).
> - **Catchup shocks DO count** if `market_ts_ms >= forward_epoch_start_ts_ms`. A restart after this point in time replays any bars missed while the process was down and tags them `phase=catchup` — these are genuine post-epoch-start market events processed late, not pre-epoch history, and must not be excluded by a naive `phase=='live'`-only filter.
> - **`state.json`'s `epoch_start_iso` is NOT the evaluation boundary.** It is the wall-clock moment the runner process was constructed (`2026-08-28T00:47:13Z` at initial deployment), not a market-bar timestamp, and is never read by the evaluation helper below.
> - **Trade forward-membership is keyed on `signal_ts_ms`, never on exit phase** — a trade opened during genuine live trading may legitimately exit during a later `phase=catchup` replay after a restart, and must still count as a forward trade.
>
> A read-only evaluation helper implementing this precisely, `shadow/shock_continuation_sol_v3_1/forward_filter.py` (`is_forward_event`, `is_forward_trade`, `cluster_forward_events`, `compute_counts`), plus 14 deterministic tests covering all of the above (`test_forward_epoch_boundary.py`), were added in the same commit as this addendum. Neither `forward_filter.py` nor its tests write to `state.json`, `events.csv`, or `trades.csv` — evaluation reads the ledgers, it never mutates them.

## STATUS

**VALIDATION — INCONCLUSIVE.**

Not PROMISING. Not PAPER CANDIDATE. Not VALIDATED. Not a PROFITABLE EDGE.

## PURPOSE

Determine whether the weakly-positive but statistically inconclusive
historical SOL Shock Continuation result persists on genuinely unseen future
Binance data. This is not a deployment. This is not a re-run of the existing
history. It exists specifically to collect out-of-sample evidence that could
not have been curve-fit, cherry-picked, or influenced by any prior analysis
in this research program.

## HISTORICAL PRIOR

**The historical evidence does NOT establish a positive edge.**

Corrected 16-fold walk-forward (`research/reports/v3_1_expanded_walkforward_CORRECTED.json`,
folds evenly spread across the full pre-holdout data range, `fold_windows()`
fixed in commit `d91c362` after a methodology bug was found during the
pre-deployment leakage check — see `research/reports/V3_1_SHOCK_CONTINUATION_VALIDATION.md`
correction box):

| Metric | SOL (corrected) |
|---|---|
| Mean OOS return per fold | ≈ +0.25% |
| Profitable-fold fraction | ≈ 53.3% |
| 90% bootstrap CI on mean OOS return | ≈ [-1.32%, +1.92%] — **includes zero** |
| P(mean > 0) | ≈ 59.7% |

BTC and ETH do **not** support a universal cross-symbol edge under this same
frozen definition (BTC mean ≈ -0.76%, ETH mean ≈ -0.54%, both CIs include
zero) — this candidate survives, if at all, only as a SOL-specific pattern,
per the V3 research-rule correction that permits symbol-specific candidates.

**The historical confidence interval is conditional on selection.** SOL
Shock Continuation was the one candidate that reached this stage out of a
multi-strategy (V2: 4 families; V3: 5 hypotheses), multi-symbol (up to 15
liquid symbols in the V3 universe) search. The reported CI is not equivalent
to a clean single-hypothesis significance test — some further discount for
search/multiple-testing risk applies on top of whatever the CI shows, and
that discount is not quantified here. This forward test is the closest thing
available to a hypothesis test that wasn't touched by that search.

**Multiple-testing/search-selection risk remains material** and is not
resolved by this forward test alone — a positive forward result raises this
candidate's status but does not eliminate the selection-bias caveat, since
the decision to run a forward test on SOL specifically (rather than BTC,
ETH, or a different candidate) was itself made after seeing the historical
data.

## FROZEN STRATEGY

`research/SHOCK_CONTINUATION_V3_FROZEN.md`, implemented in
`research/strategies/shock_aftershock.py::ShockContinuationStrategy`. Not
altered for this forward test. Any new idea discovered during forward
observation is logged as a `SHOCK_CONTINUATION_V4_HYPOTHESIS`, evaluated
later on separate unseen data — never retrofitted into this frozen
definition or this forward epoch.

**Frozen parameters:** `ATR_N=14`, `SHOCK_ATR_MULT=3.0`, `INIT_STOP_PCT=4.0`.
**Symbol:** SOL/USDT only. **Timeframe:** 1h. **Exit:** shared PaperEngine
trailing stop (ratchet-only, `exit_on_flip=False`), closed-bar OHLC stop
checks only (`check_stop_bar`, no intrabar tick checks) — matching
`research/generic_runner.py`, the exact methodology that produced the
walk-forward numbers above, so forward results remain comparable to them.
**Costs/sizing:** `config.TAKER_FEE`, `config.SLIPPAGE`,
`config.RISK_PCT_PER_TRADE`, `config.POSITION_PCT`, and the shared risk
guards (`MAX_DRAWDOWN_PCT`, `DAILY_LOSS_LIMIT_PCT`, `CONSECUTIVE_LOSS_LIMIT`)
— unmodified from the shared `config` module, identical to the legacy bot
and to the validated backtest.

## PRE-LAUNCH HASHES (computed from `shadow/shock_continuation_sol_v3_1/frozen_spec.py`)

```
frozen_spec_md_sha256:         f89f05057f52fcb4112b3b70ceea8234b60c7b96e22fa30e15d739107cfd864e
strategy_impl_sha256:          d1b63495e44c9ac91c0f22f4f2c3178aa2e63079cdde1698d059541e0c9477c2
frozen_params_sha256:          a69c5b4aab3657ab134e2651befc7dae55820fca532a73d25ff91d4d6002418d
execution_assumptions_sha256:  8bd9172d81134a194ddfa76a58f71eebca5018bf94e3a498ace247d74b5f4128
combined_hash:                 2fc1bc8708e190a095a730f17aa9abdf29d4bccfc7fdaff9f1040da8e3bb3918
git_commit (this plan):        d91c36258cf53e806aee06899cb4877f60c9c655
```

If any of these hashes change while the forward epoch is running, the epoch
is considered ended per the "no performance-based modification" rule below
— the runner checks this on every restart (`run_shadow.py::_check_config_drift`)
and refuses to silently continue past a mismatch.

## MINIMUM OBSERVATION PERIOD

**At least 90 calendar days AND at least 30 independent shock events/trades.**

If fewer than 30 independent events have occurred after 90 days, the
experiment continues past 90 days until 30 is reached. Independence is
measured via the same 24h chain-rule clustering used in V3.1
(`research/outlier_and_clustering.py::cluster_events`, `CLUSTER_WINDOW_HOURS=24`)
applied to the forward trade log.

**Prefer at least 50 independent events before any strong promotion
decision.** 30 is the floor for a REJECTED/continue call, not for a
PAPER CANDIDATE call.

**Do not terminate early because results look excellent.** A short hot
streak is not evidence of a real edge; it is exactly the kind of thing this
program has already shown can look significant and not be (see the
now-corrected 16-fold result above).

**Do not terminate early merely because initial trades lose.** A short cold
streak is equally uninformative on its own.

## EVALUATION METRICS (recorded at the end of the minimum observation period)

- Independent shock-cluster count (24h chain rule)
- Raw trade count
- Net expectancy per trade
- Profit factor
- Total net return
- Max drawdown
- Win rate
- Payoff ratio (avg win / avg loss)
- Long expectancy vs. short expectancy (separately)
- Longest losing streak
- Cost actually experienced (fees + slippage, realized vs. modeled)
- Shock-severity distribution (ATR-normalized magnitude)
- Volume-multiple distribution
- Forward event clustering (cluster sizes, concentration)
- Comparison against the historical corrected distribution (does the forward
  severity/volume/clustering profile resemble the historical sample, or does
  it look like a different regime?)

Total return alone is never a sufficient success criterion — see decision
rules below.

## DECISION RULES

**REJECTED** if forward evidence materially contradicts the hypothesis: persistent
negative expectancy or profit factor below 1 with an adequate sample (≥30
independent events).

**VALIDATION** (i.e., stay in the current inconclusive state, keep
observing) if evidence remains mixed, or sample size remains insufficient
(<30 independent events and <90 days, or the metrics are genuinely
ambiguous even with an adequate sample).

**PAPER CANDIDATE** only if the forward evidence materially strengthens the
case, requiring ALL of:
- Positive net expectancy
- Profit factor > 1 after modeled costs
- Acceptable drawdown (no formal numeric threshold pre-registered here;
  judged against the historical worst-fold drawdown as context, not a
  specific number, per the user's explicit instruction not to require an
  arbitrary return target)
- Adequate independent event count (≥30 minimum, ≥50 preferred)
- Results not dependent on one exceptional trade (outlier-dependence check,
  same method as V3.1 section C, applied to the forward sample)
- Shock behavior broadly consistent with the historical mechanism (severity/
  volume/direction distributions resemble the historical sample, not a
  different regime)
- No material implementation or data-quality problem discovered during the
  observation window

**Do not require an arbitrary return target. Do not promote solely because
total PnL is positive.**

## NO PERFORMANCE-BASED MODIFICATIONS DURING VALIDATION

If a genuine software defect is found during the observation window: stop
the shadow process, document the defect and its discovery date/commit,
preserve the entire epoch's data (do not delete or overwrite), fix the
defect with its own tests, and begin a NEW forward epoch (new hash, new
`FORWARD_VALIDATION_PLAN.md` addendum, new `shadow/` sub-run or clearly
marked epoch boundary in the same files). Pre-fix and post-fix results are
never silently mixed into one evaluation.

If no defect is found, the strategy, parameters, and execution assumptions
are not altered for any reason — including if forward results look
unexpectedly good or unexpectedly bad — until the minimum observation period
completes and a decision is made per the rules above.

## SAFETY

Binance PUBLIC market data only (ccxt `fetch_ohlcv`, `fetch_ticker`), no API
keys, no authenticated endpoints, no `create_order`/`cancel_order`/
`withdraw`/`transfer` anywhere in the shadow code path. Fully isolated from
the legacy `com.btcpaper.bot` (separate runtime directory, separate state,
separate launchd service `com.btcpaper.shocksol`). No real orders can be
sent — see the final deployment report for the explicit verification.
