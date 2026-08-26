# BTC Paper Trading — Prime Swing ⊕ Supertrend ATR (fused)

Simulated futures paper-trading of BTC/USDT against **Binance live public prices**.
No API keys are used — only public market-data endpoints (`ccxt`). Nothing is ever
sent to an exchange; all fills, fees and P&L are simulated.

## The fused strategy
- **Entry — Prime Strategy Swing** (Pine v6 port)
  - Structure (swing pivots → BOS/CHoCH → trend) on **30m**
  - Fixed zone with Fibonacci 15/30/50/70/85 levels
  - Nadaraya-Watson band filter on **5m**
  - `BUY` / `BUY PRIME` → open **long**; `SELL` / `SELL PRIME` → open **short**
  - Session / weekday / max-1-per-day gates from the source are **dropped** (24/7 crypto)
- **Exit — Supertrend ATR w/ Trailing Stop Loss** (Pine v4 port)
  - Per-position ratcheting trailing stop (initial 3%, tightens after +1% profit)
  - Close on **trailing-stop hit** OR **Supertrend `direction` flips against the position**
  - ATR period 1, multiplier 3.0 (source defaults)
  - Flip-exit arms only *after* the trend confirms the position (`ARM_FLIP_AFTER_ENTRY`)

One position at a time, one equity curve. Defaults: 10,000 USDT, 100% equity/trade,
0.04% taker fee + 0.02% slippage per fill.

## Files
| file | role |
|---|---|
| `config.py` | all parameters |
| `feed.py` | Binance public OHLCV + last price (ccxt, no auth) |
| `strategy.py` | Prime signal engine + Supertrend/TSL, stateful & non-repainting |
| `engine.py` | simulated futures portfolio, fills, fees, trade log |
| `run.py` | live runner (warmup → poll live prices → trade; publishes `state.json`) |
| `dashboard.py` | live web dashboard (stdlib http server) at http://localhost:8787 |
| `backtest.py` | replay recent history through the same engine (validation) |

## Run
```bash
python3 run.py              # live paper trade (runs until Ctrl-C)
python3 dashboard.py        # web dashboard at http://localhost:8787 (run alongside run.py)
python3 run.py --warmup     # warm up + one snapshot, then exit
python3 backtest.py --days 30   # offline validation over recent history
```

## Monitor / stop
```bash
open http://localhost:8787  # live web dashboard (auto-refreshes every 3s)
tail -f paper.log           # live heartbeats + trades
cat trades.csv              # closed-trade log
pkill -f run.py             # stop the live trader
pkill -f dashboard.py       # stop the dashboard
```

## Validation (30d to 2026-06-19)
+3.23% vs BTC buy-&-hold −17.83%; 33 trades, 39% win, −5.94% max DD.
Backtest stops use bar adverse-extreme; the live runner uses real tick prices.
Backtest `trades.csv` timestamps are run-time (not bar-time); live timestamps are real.
