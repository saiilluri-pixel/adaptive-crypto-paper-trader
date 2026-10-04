# Adaptive Crypto Paper Trader

A research-grade **paper-trading** platform for systematic crypto strategies on Binance,
built to *honestly* search for a real, out-of-sample trading edge — and to document what
does and (mostly) does **not** work.

> ⚠️ **100% PAPER TRADING. NOT FINANCIAL ADVICE.**
> This software places **no real orders** and uses **no API keys** — it reads only
> public Binance market data and simulates fills. Nothing here is a recommendation to
> buy or sell anything. Crypto trading carries substantial risk of loss. If you adapt
> this for live trading you do so entirely at your own risk. **No warranty** (see LICENSE).

---

## What this is

- A **single shared-cash paper portfolio**: a Spot-legal long book plus a *simulated*
  short/margin book, drawing from one capital pool (first-come-first-serve).
- A **champion/challenger adaptation loop** that walk-forward-tests parameter changes
  before promoting them, with automatic rollback.
- **Gap-integrity-safe** market data (a cursor that refuses to skip missing candles).
- A deterministic **pytest suite** (~260 tests) covering portfolio invariants, risk
  rules, strategy causality, and safety (asserts *no* auth/order code paths exist).
- A read-only **dashboard** (`adaptive/dashboard.py`).

Core package: [`adaptive/`](adaptive/) — `runner.py` (loop), `strategies.py`,
`risk_engine.py`, `portfolio.py` / `short_portfolio.py`, `meta_controller.py`
(scoring/ranking), `market_data.py`, `research_worker.py` (backtest engine),
`adaptation.py` + `champion_store.py`, `regime.py`, `trailing.py`, `cursor.py`.

## Honest research findings

This repo's main value is **negative results, rigorously obtained.** On real multi-year
Binance data with a 0.1% fee + 0.02% slippage model and **causal in-sample / out-of-sample
splits**, we tested a lot and found no strong alpha. Summary:

| Idea | Verdict (out-of-sample) |
|---|---|
| TA signals (trend/breakout/mean-reversion/ATR) at 1h/15m | ❌ Overfit — great in-sample (PF 2–3.5), collapsed to PF 0.5–1.0 OOS |
| Aggressive parameter tuning | ❌ Deepened the overfit; live paper went net-negative |
| Cross-exchange spread (Binance vs Coinbase) | ❌ Real but ~0.007% — ~100× below the ~0.7% cost to trade it |
| Funding-rate extreme → reversion (BTC/ETH, ~7yr) | ❌ In-sample only; OOS PF 0.70–0.88, negative expectancy |
| Cross-sectional / weekly momentum | ⚠️ Unstable; OOS results flipped sign across universes/windows |
| **Trend-filtered equal-weight basket (SMA200 / 4h / daily)** | ✅ **Kept — as risk management, not alpha** |

**The one survivor** — hold each of 5 coins (BTC/ETH/BNB/XRP/SOL) only while its 4h close
is above its SMA200, equal-weight, else cash — is a **drawdown brake, not an edge**:
out-of-sample it captured most of the basket's upside while **roughly halving max drawdown**
(≈34% vs ≈63%) and improving Sharpe, and it turned the 2022 bear from −68% into −28%.
It will **lag buy-and-hold in strong bulls** by design. See `trend_basket/`.

**Honest takeaway:** price/momentum signals on public candles with retail fees do not
appear to carry durable edge after costs — the visible-to-everyone signals are competed
away. Real edges tend to need infrastructure this design can't reach (low-latency
market-making, derivatives/funding capture at scale, private data). PRs that find
otherwise — *with out-of-sample proof* — are very welcome.

## Run it

```bash
pip install -r requirements.txt          # pandas, numpy, scipy, ccxt, matplotlib
python3 -m pytest test_adaptive_*.py -q  # ~260 tests
python3 adaptive/runner.py               # adaptive paper bot (writes to adaptive_runtime/)
python3 adaptive/dashboard.py --port 8788
python3 trend_basket/trend_basket_paper.py 10000   # trend-filtered basket, one daily decision
```

Runtime output (state, logs, ledgers, the cached OHLCV/funding history) is written under
`adaptive_runtime/` and `trend_basket/` and is **git-ignored** — it regenerates from the
public Binance API.

## Contributing / help find an edge

The backtest harness (`adaptive/research_worker.py`) and the deep-history fetchers make it
easy to test a hypothesis causally. If you propose a strategy, please include an
**out-of-sample** result (fit on one window, report on another) and costs — in-sample-only
numbers are not evidence. Bug fixes and code-quality PRs equally welcome.

## License

MIT — see [LICENSE](LICENSE). Set the copyright holder before you push.
