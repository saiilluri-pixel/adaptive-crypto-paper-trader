"""
Offline validation: replay recent Binance history through the SAME strategy +
engine used live, with trading enabled. Confirms the fused pipeline produces
sane trades and P&L before a long live run.

Stops are checked against each bar's adverse extreme (low for longs, high for
shorts) as a bar-resolution approximation of intrabar fills; the live runner
checks the real tick price.

    python3 backtest.py [--days N]
"""
import argparse
import time

import ccxt
import pandas as pd

import config
from strategy import Strategy
from engine import PaperEngine

TF_MS = {"5m": 300_000, "15m": 900_000, "30m": 1_800_000}


def fetch_history(ex, symbol, tf, days):
    ms = TF_MS[tf]
    need = int(days * 86_400_000 / ms) + 5
    since = ex.milliseconds() - need * ms
    rows = []
    while True:
        batch = ex.fetch_ohlcv(symbol, tf, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + ms
        if len(batch) < 1000:
            break
        time.sleep(ex.rateLimit / 1000)
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close", "volume"])
    df = df.drop_duplicates("ts").reset_index(drop=True)
    df["dt"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    return df.iloc[:-1]  # drop in-progress bar


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()

    ex = getattr(ccxt, config.EXCHANGE)({"enableRateLimit": True})
    print(f"Fetching ~{args.days}d of {config.SYMBOL} history…")
    struct = fetch_history(ex, config.SYMBOL, config.STRUCTURE_TF, args.days)
    chart = fetch_history(ex, config.SYMBOL, config.CHART_TF, args.days)
    print(f"  {len(struct)}x{config.STRUCTURE_TF}, {len(chart)}x{config.CHART_TF} bars\n")

    strat = Strategy()
    eng = PaperEngine(log_fn=lambda m: None)  # silent; we summarize at end

    events = []
    for _, r in struct.iterrows():
        events.append((int(r.ts) + TF_MS[config.STRUCTURE_TF], 0, "S", r))
    for _, r in chart.iterrows():
        events.append((int(r.ts) + TF_MS[config.CHART_TF], 1, "C", r))
    events.sort(key=lambda e: (e[0], e[1]))

    equity_curve = []
    for _, _, kind, r in events:
        if kind == "S":
            strat.feed_struct_bar(r.open, r.high, r.low, r.close)
            continue
        # OHLC-aware stop check FIRST, using the stop as it stood before this
        # bar (before any ratchet) -- catches an intrabar/gap stop touch that
        # a close-only check would miss, without ever crediting a fill better
        # than the market actually offered. Same order as live's bar-close
        # handling (run.py VariantTrader.on_chart_bar).
        if eng.pos is not None:
            eng.check_stop_bar(r.open, r.high, r.low)
        snap = strat.feed_chart_bar(r.open, r.high, r.low, r.close)
        if eng.pos is not None:
            eng.manage_on_bar(snap)    # ratchet TSL + Supertrend flip exit
        eng.on_signals(snap)           # entries when flat
        equity_curve.append(eng.mark(r.close))

    # ── results ──
    e = eng
    ret = (e.equity / config.START_CAPITAL - 1) * 100
    wr = (e.wins / e.n_trades * 100) if e.n_trades else 0
    peak, mdd = -1e18, 0.0
    for v in equity_curve:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)
    bh = (chart.iloc[-1].close / chart.iloc[0].close - 1) * 100

    print("──────────── FUSED STRATEGY — BACKTEST ────────────")
    print(f"  Period           : {chart['dt'].iloc[0]:%Y-%m-%d} → {chart['dt'].iloc[-1]:%Y-%m-%d}")
    print(f"  Start capital    : {config.START_CAPITAL:,.2f} USDT")
    print(f"  End equity       : {e.equity:,.2f} USDT")
    print(f"  Total return     : {ret:+.2f}%")
    print(f"  Buy & hold BTC   : {bh:+.2f}%")
    print(f"  Trades           : {e.n_trades}  (win {wr:.0f}%)")
    print(f"  Max drawdown     : {mdd*100:.2f}%")
    print(f"  Realized P&L     : {e.realized:+,.2f} USDT")
    print(f"  Trade log        : {e.csv_path}")
    print("───────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
