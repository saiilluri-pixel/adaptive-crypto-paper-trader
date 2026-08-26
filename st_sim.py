"""
Quick evidence backtest: the SUPERTREND-flip strategy (long+short, always in
market, flip on direction change) on 15m — comparing the TradingView default
ATR period 1 vs a slower ATR 10. Shows why ATR 1 whipsaws.
"""
import ccxt
import config
from backtest import fetch_history
from strategy import Strategy

FEE = 0.0006  # taker + slippage per side

def sim(symbol, atr_period, days=60):
    ex = getattr(ccxt, config.EXCHANGE)({"enableRateLimit": True})
    chart = fetch_history(ex, symbol, "15m", days)
    p = {"SWING_LEN": 5, "NW_BANDWIDTH": 4.0, "NW_MULT": 3.0, "NW_WINDOW": 500,
         "ST_ATR_PERIOD": atr_period, "ST_ATR_MULT": 3.0, "ST_INIT_STOP": 3.0}
    strat = Strategy(p)
    eq = 10000.0
    pos, entry = 0, None
    trades, wins, prev = 0, 0, None
    curve = []
    for _, r in chart.iterrows():
        strat.feed_chart_bar(r.open, r.high, r.low, r.close)
        d = strat.st_dir
        if prev is not None and d != prev:           # flip
            if pos != 0:
                ret = (r.close / entry - 1) * pos
                eq *= (1 + ret)
                eq *= (1 - FEE)                       # exit fee
                trades += 1
                wins += 1 if ret > 0 else 0
            pos = d
            entry = r.close
            eq *= (1 - FEE)                           # entry fee
        prev = d
        m = eq * (1 + (r.close / entry - 1) * pos) if pos and entry else eq
        curve.append(m)
    peak, mdd = -1e18, 0.0
    for v in curve:
        peak = max(peak, v); mdd = min(mdd, v / peak - 1)
    ret = (curve[-1] / 10000 - 1) * 100
    bh = (chart.iloc[-1].close / chart.iloc[0].close - 1) * 100
    wr = wins / trades * 100 if trades else 0
    return ret, trades, wr, mdd * 100, bh


print(f"{'symbol':9s} {'ATR':>4s} | {'return':>8s} {'trades':>7s} {'win%':>5s} {'maxDD':>7s} | b&h")
for sym in config.SYMBOLS:
    for atr in (1, 10):
        ret, tr, wr, mdd, bh = sim(sym, atr)
        print(f"{sym:9s} {atr:>4d} | {ret:+7.2f}% {tr:>7d} {wr:>4.0f}% {mdd:>6.1f}% | {bh:+.1f}%")
