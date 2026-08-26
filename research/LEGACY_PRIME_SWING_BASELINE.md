# LEGACY_PRIME_SWING_BASELINE

**Status: REJECTED / LEGACY BASELINE** (per Phase 2 fresh-validation evidence, see [reports/PHASE2_VALIDATION_REPORT.md](reports/PHASE2_VALIDATION_REPORT.md)).

This is the frozen label for the Prime-Swing (BOS/CHoCH + Fibonacci zone + Nadaraya-Watson filter) entry, Supertrend-ATR trailing-stop exit strategy that has been the subject of all correctness work in this repository to date. It remains available as a benchmark for future strategy research (Crypto Strategy Research V2) — any new candidate should be judged, in part, against how it compares to this baseline.

**Preserved, unmodified, not autotuned further:**
- `strategy.py` — entry/exit signal logic
- `symbol_params.json` — per-symbol tuned parameters (BTC/ETH/SOL), as of 2026-06-21
- `research/legacy_tuned_baseline.json` — full frozen config snapshot (symbols, params, risk settings, fees/slippage, timeframes)
- `research/reports/PHASE2_VALIDATION_REPORT.md` — the complete validation report
- `research/reports/fixed_baseline.json`, `regime.json`, `portfolio.json`, `walkforward.json`, `robustness.json` — the raw research outputs behind that report (regenerable via `research/run_fixed_baseline.py`, `research/run_regime_portfolio.py`, `research/walkforward.py`, `research/robustness.py` against the cached data in `research/data/`)

**Evidence summary:** no positive out-of-sample expectancy under the tuned parameters, global defaults, or a leakage-free walk-forward, across all three symbols. See the report for full detail.

**What "frozen" means here:** no further parameter search, no symbol_params.json changes, no strategy.py edits, driven by this strategy's own results. It may still receive genuine correctness fixes if one is found (as happened with the day-rollover clock defect, which benefits every strategy using `PaperEngine`, this one included) — that is a shared-infrastructure fix, not tuning.

**The live paper bot (`com.btcpaper.bot`) continues running this exact configuration** as an independent, ongoing real-world data point — per instruction, it is not being stopped, and nothing here changes what it's doing.
