"""
Per-symbol walk-forward parameter optimizer.

For each symbol and each parameter combo it runs ONE replay over ~60 days of
Binance history, scoring on a TRAIN window and validating on an unseen TEST
window (last 30%). A combo is eligible only if it ACTUALLY TRADES enough in
both windows (so we don't ship something that sits flat). Eligible combos are
ranked by CONSISTENCY = min(train_return, test_return) — rewarding configs that
profit in both the seen and unseen periods, not curve-fits.

Writes the winners to symbol_params.json, which the live runner reads per symbol.
Offline only; never touches the live trader (engine runs log_trades=False).

    python3 autotune.py --all                # tune every config.SYMBOLS
    python3 autotune.py --symbol ETH/USDT    # tune one
"""
import argparse
import itertools
import json
import os
import time

import ccxt

import config
from backtest import fetch_history, TF_MS
from strategy import Strategy
from engine import PaperEngine

HERE = os.path.dirname(os.path.abspath(__file__))

GRID = {
    "SWING_LEN":     [3, 5, 8],
    "NW_MULT":       [2.0, 3.0],
    "ST_ATR_PERIOD": [1, 10, 14],
    "ST_ATR_MULT":   [2.0, 3.0, 4.0],
    "ST_INIT_STOP":  [2.0, 3.0, 5.0],
}
MIN_TRADES_TRAIN = 12          # must actually trade in the training window
MIN_TRADES_TEST = 3            # …and in the unseen window
DEFAULT = {"SWING_LEN": 5, "NW_MULT": 2.0, "ST_ATR_PERIOD": 14,
           "ST_ATR_MULT": 4.0, "ST_INIT_STOP": 5.0}   # current BTC-tuned global


def _max_dd(curve):
    peak, mdd = -1e18, 0.0
    for v in curve:
        peak = max(peak, v)
        if peak > 0:
            mdd = min(mdd, v / peak - 1)
    return mdd * 100


def run_replay(struct, chart):
    strat, eng = Strategy(), PaperEngine(log_fn=lambda m: None, log_trades=False)
    events = []
    for _, r in struct.iterrows():
        events.append((int(r.ts) + TF_MS[config.STRUCTURE_TF], 0, "S", r))
    for _, r in chart.iterrows():
        events.append((int(r.ts) + TF_MS[config.CHART_TF], 1, "C", r))
    events.sort(key=lambda e: (e[0], e[1]))
    times, eq, ntr, wins = [], [], [], []
    for ct, _, kind, r in events:
        if kind == "S":
            strat.feed_struct_bar(r.open, r.high, r.low, r.close)
            continue
        # same order as backtest.py / live: OHLC stop check (pre-ratchet
        # stop) before manage_on_bar's ratchet -- see backtest.py
        if eng.pos is not None:
            eng.check_stop_bar(r.open, r.high, r.low)
        snap = strat.feed_chart_bar(r.open, r.high, r.low, r.close)
        if eng.pos is not None:
            eng.manage_on_bar(snap)
        eng.on_signals(snap)
        times.append(ct); eq.append(eng.mark(r.close))
        ntr.append(eng.n_trades); wins.append(eng.wins)
    return times, eq, ntr, wins


def windowed(times, eq, ntr, wins, split_ts, sc):
    si = 0
    while si < len(times) and times[si] <= split_ts:
        si += 1
    si = max(1, min(si, len(times) - 1))
    eqs = eq[si - 1]
    train = {"ret": (eqs / sc - 1) * 100, "trades": ntr[si - 1],
             "wins": wins[si - 1], "mdd": _max_dd(eq[:si])}
    test = {"ret": (eq[-1] / eqs - 1) * 100, "trades": ntr[-1] - ntr[si - 1],
            "wins": wins[-1] - wins[si - 1], "mdd": _max_dd(eq[si:])}
    for d in (train, test):
        d["win"] = (d["wins"] / d["trades"] * 100) if d["trades"] else 0.0
    return train, test


def tune_symbol(ex, symbol, days):
    print(f"\n############ {symbol} ############", flush=True)
    struct = fetch_history(ex, symbol, config.STRUCTURE_TF, days)
    chart = fetch_history(ex, symbol, config.CHART_TF, days)
    split_ts = int(chart.iloc[int(len(chart) * 0.70)].ts) + TF_MS[config.CHART_TF]
    sc = config.START_CAPITAL
    keys = list(GRID)
    combos = list(itertools.product(*GRID.values()))
    print(f"  {len(chart)} bars; {len(combos)} combos…", flush=True)

    results = []
    t0 = time.time()
    for i, vals in enumerate(combos, 1):
        params = dict(zip(keys, vals))
        for k, v in params.items():
            setattr(config, k, v)
        times, eq, ntr, wins = run_replay(struct, chart)
        train, test = windowed(times, eq, ntr, wins, split_ts, sc)
        score = min(train["ret"], test["ret"])
        results.append({"params": params, "train": train, "test": test, "score": score})
        if i % 40 == 0 or i == len(combos):
            print(f"    {i}/{len(combos)} ({time.time()-t0:.0f}s)", flush=True)

    eligible = [r for r in results
                if r["train"]["trades"] >= MIN_TRADES_TRAIN
                and r["test"]["trades"] >= MIN_TRADES_TEST]
    eligible.sort(key=lambda r: r["score"], reverse=True)

    def row(r, tag=""):
        p = r["params"]; tr = r["train"]; te = r["test"]
        ps = (f"swing={p['SWING_LEN']} nwM={p['NW_MULT']} atrP={p['ST_ATR_PERIOD']} "
              f"atrM={p['ST_ATR_MULT']} stop={p['ST_INIT_STOP']}%")
        return (f"  {tag:3s}{ps:50s}| TRAIN {tr['ret']:+7.2f}% {tr['trades']:3d}t {tr['win']:3.0f}%w "
                f"| TEST {te['ret']:+7.2f}% {te['trades']:3d}t {te['win']:3.0f}%w | score {r['score']:+.1f}")

    print(f"  eligible combos (≥{MIN_TRADES_TRAIN} train & ≥{MIN_TRADES_TEST} test trades): "
          f"{len(eligible)}/{len(results)}")
    print("  ── top 6 by consistency (min of train/test return) ──")
    for r in eligible[:6]:
        print(row(r))
    base = next((r for r in results if r["params"] == DEFAULT), None)
    if base:
        print("  ── current global params ──"); print(row(base, "now"))

    chosen = eligible[0] if eligible else None
    if chosen is None:
        # fallback: most trades among any profitable-on-test combo, else just most trades
        prof = [r for r in results if r["test"]["ret"] > 0 and r["train"]["trades"] >= 5]
        chosen = (max(prof, key=lambda r: r["test"]["trades"]) if prof
                  else max(results, key=lambda r: r["train"]["trades"]))
        print("  !! no combo met both-window trade+profit floors — "
              "falling back to most-active reasonable combo (flagged).")
    print("  ── CHOSEN ──"); print(row(chosen, "★"))
    return chosen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--symbol", default=config.SYMBOL)
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    ex = getattr(ccxt, config.EXCHANGE)({"enableRateLimit": True})
    symbols = config.SYMBOLS if args.all else [args.symbol]

    path = os.path.join(HERE, config.SYMBOL_PARAMS_FILE)
    try:
        with open(path) as f:
            out = json.load(f)
    except Exception:
        out = {}

    summary = []
    for sym in symbols:
        chosen = tune_symbol(ex, sym, args.days)
        out[sym] = chosen["params"]
        summary.append((sym, chosen))
        with open(path, "w") as f:           # write incrementally
            json.dump(out, f, indent=2)

    print("\n══════════════ SUMMARY — written to symbol_params.json ══════════════")
    for sym, r in summary:
        p = r["params"]
        flag = "" if (r["train"]["ret"] > 0 and r["test"]["ret"] > 0) else "  (NOT robustly profitable)"
        print(f"  {sym:9s} train {r['train']['ret']:+6.2f}% / test {r['test']['ret']:+6.2f}% "
              f"({r['test']['trades']}t) -> swing={p['SWING_LEN']} nwM={p['NW_MULT']} "
              f"atrP={p['ST_ATR_PERIOD']} atrM={p['ST_ATR_MULT']} stop={p['ST_INIT_STOP']}%{flag}")
    print("\nRestart the bot to apply:  launchctl bootout/bootstrap com.btcpaper.bot")


if __name__ == "__main__":
    main()
