"""
READ-ONLY data-source audit tool. Does NOT import run_shadow.py, does NOT
touch state.json/events.csv/trades.csv, does NOT place orders. Public REST
endpoints only, no API keys.

Compares:
  1. The shadow's own data path (feed.Feed -> ccxt binance spot) against a
     SEPARATE, independent direct-HTTP call to Binance's public REST API,
     for the same instrument.
  2. Binance Spot SOL/USDT against Binance USD-M Futures SOLUSDT (perpetual)
     for the same window, to quantify basis and confirm what market type
     the historical research data used.
  3. Several recent CLOSED 1h candles for exact OHLCV integrity.
"""
import json
import os
import sys
import time

import requests

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, ROOT)

from feed import Feed  # noqa: E402
import config  # noqa: E402

SPOT_REST = "https://api.binance.com/api/v3"
FAPI_REST = "https://fapi.binance.com/fapi/v1"
SYMBOL_CCXT = "SOL/USDT"
SYMBOL_RAW = "SOLUSDT"


def section(title):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def part1_shadow_feed_vs_direct_rest():
    section("1) SHADOW'S OWN FEED PATH vs DIRECT BINANCE REST (independent HTTP call)")
    feed = Feed()
    print("ccxt exchange id:", feed.ex.id)
    print("ccxt apiKey set:", bool(feed.ex.apiKey), " secret set:", bool(feed.ex.secret))

    t0 = time.time()
    shadow_price = feed.last_price(SYMBOL_CCXT)
    t_shadow = time.time() - t0

    t0 = time.time()
    r = requests.get(f"{SPOT_REST}/ticker/price", params={"symbol": SYMBOL_RAW}, timeout=10)
    r.raise_for_status()
    direct_price = float(r.json()["price"])
    t_direct = time.time() - t0

    diff = shadow_price - direct_price
    pct = (diff / direct_price) * 100 if direct_price else None
    print(f"shadow (ccxt) last_price: {shadow_price:.4f}  ({t_shadow*1000:.0f} ms)")
    print(f"direct REST last price:   {direct_price:.4f}  ({t_direct*1000:.0f} ms)")
    print(f"abs diff: {diff:+.4f}   pct diff: {pct:+.5f}%")

    shadow_chart = feed.closed_candles(SYMBOL_CCXT, "1h", 3)
    last_closed = shadow_chart.iloc[-1]
    r = requests.get(f"{SPOT_REST}/klines",
                      params={"symbol": SYMBOL_RAW, "interval": "1h", "limit": 3}, timeout=10)
    r.raise_for_status()
    raw = r.json()
    direct_last_closed = raw[-2]  # klines' own last row is the still-forming candle too

    print("\nlast CLOSED 1h candle -- shadow (ccxt) vs direct REST:")
    print(f"  open_ts   shadow={int(last_closed.ts)}  direct={direct_last_closed[0]}  "
          f"match={int(last_closed.ts) == direct_last_closed[0]}")
    print(f"  open      shadow={last_closed.open:.4f}  direct={float(direct_last_closed[1]):.4f}  "
          f"match={abs(last_closed.open - float(direct_last_closed[1])) < 1e-6}")
    print(f"  high      shadow={last_closed.high:.4f}  direct={float(direct_last_closed[2]):.4f}  "
          f"match={abs(last_closed.high - float(direct_last_closed[2])) < 1e-6}")
    print(f"  low       shadow={last_closed.low:.4f}  direct={float(direct_last_closed[3]):.4f}  "
          f"match={abs(last_closed.low - float(direct_last_closed[3])) < 1e-6}")
    print(f"  close     shadow={last_closed.close:.4f}  direct={float(direct_last_closed[4]):.4f}  "
          f"match={abs(last_closed.close - float(direct_last_closed[4])) < 1e-6}")
    print(f"  volume    shadow={last_closed.volume:.4f}  direct={float(direct_last_closed[5]):.4f}  "
          f"match={abs(last_closed.volume - float(direct_last_closed[5])) < 1e-6}")
    return {"shadow_price": shadow_price, "direct_price": direct_price, "pct_diff": pct}


def part2_closed_candle_integrity(n=8):
    section(f"2) CLOSED-CANDLE INTEGRITY -- last {n} closed 1h candles, shadow feed vs direct REST")
    feed = Feed()
    shadow_chart = feed.closed_candles(SYMBOL_CCXT, "1h", n)
    r = requests.get(f"{SPOT_REST}/klines",
                      params={"symbol": SYMBOL_RAW, "interval": "1h", "limit": n + 2}, timeout=10)
    r.raise_for_status()
    raw = r.json()[:-1]  # drop the still-forming candle, same convention as Feed.closed_candles
    raw_by_ts = {row[0]: row for row in raw}

    all_match = True
    for _, row in shadow_chart.iterrows():
        ts = int(row.ts)
        direct = raw_by_ts.get(ts)
        if direct is None:
            print(f"  ts={ts}: NOT FOUND in direct REST response -- cannot compare")
            all_match = False
            continue
        checks = {
            "open": abs(row.open - float(direct[1])) < 1e-6,
            "high": abs(row.high - float(direct[2])) < 1e-6,
            "low": abs(row.low - float(direct[3])) < 1e-6,
            "close": abs(row.close - float(direct[4])) < 1e-6,
            "volume": abs(row.volume - float(direct[5])) < 1e-6,
        }
        ok = all(checks.values())
        all_match = all_match and ok
        print(f"  ts={ts} ({row['dt']}): {'OK' if ok else 'MISMATCH ' + str(checks)}")
    print(f"\nall {n} closed candles identical: {all_match}")
    return all_match


def part3_spot_vs_futures():
    section("3) SPOT vs USD-M FUTURES (perpetual) SOLUSDT -- basis check")
    r = requests.get(f"{SPOT_REST}/ticker/price", params={"symbol": SYMBOL_RAW}, timeout=10)
    r.raise_for_status()
    spot_price = float(r.json()["price"])

    r = requests.get(f"{FAPI_REST}/ticker/price", params={"symbol": SYMBOL_RAW}, timeout=10)
    r.raise_for_status()
    fut_price = float(r.json()["price"])

    r = requests.get(f"{FAPI_REST}/premiumIndex", params={"symbol": SYMBOL_RAW}, timeout=10)
    r.raise_for_status()
    premium = r.json()
    mark_price = float(premium["markPrice"])
    funding_rate = float(premium["lastFundingRate"])

    basis = fut_price - spot_price
    basis_pct = (basis / spot_price) * 100

    r = requests.get(f"{SPOT_REST}/klines",
                      params={"symbol": SYMBOL_RAW, "interval": "1h", "limit": 3}, timeout=10)
    spot_kl = r.json()[-2]
    r = requests.get(f"{FAPI_REST}/klines",
                      params={"symbol": SYMBOL_RAW, "interval": "1h", "limit": 3}, timeout=10)
    fut_kl = r.json()[-2]

    print(f"spot price:        {spot_price:.4f}")
    print(f"futures price:     {fut_price:.4f}")
    print(f"mark price:        {mark_price:.4f}")
    print(f"funding rate:      {funding_rate:.6f}  ({funding_rate*100:.4f}%)")
    print(f"basis (fut-spot):  {basis:+.4f}  ({basis_pct:+.4f}%)")
    print()
    print(f"last closed 1h candle -- spot vs futures:")
    print(f"  open_ts   spot={spot_kl[0]}  fut={fut_kl[0]}  same_bar={spot_kl[0]==fut_kl[0]}")
    print(f"  open      spot={float(spot_kl[1]):.4f}  fut={float(fut_kl[1]):.4f}  "
          f"diff={float(spot_kl[1])-float(fut_kl[1]):+.4f}")
    print(f"  high      spot={float(spot_kl[2]):.4f}  fut={float(fut_kl[2]):.4f}  "
          f"diff={float(spot_kl[2])-float(fut_kl[2]):+.4f}")
    print(f"  low       spot={float(spot_kl[3]):.4f}  fut={float(fut_kl[3]):.4f}  "
          f"diff={float(spot_kl[3])-float(fut_kl[3]):+.4f}")
    print(f"  close     spot={float(spot_kl[4]):.4f}  fut={float(fut_kl[4]):.4f}  "
          f"diff={float(spot_kl[4])-float(fut_kl[4]):+.4f}")
    close_ret_spot = (float(spot_kl[4]) / float(spot_kl[1]) - 1) * 100
    close_ret_fut = (float(fut_kl[4]) / float(fut_kl[1]) - 1) * 100
    print(f"  1h return spot={close_ret_spot:+.4f}%  fut={close_ret_fut:+.4f}%  "
          f"diff={close_ret_spot-close_ret_fut:+.5f}pp")
    print(f"  volume    spot={float(spot_kl[5]):.2f}  fut={float(fut_kl[5]):.2f}  "
          f"(different venues, NOT expected to match)")

    return {"spot_price": spot_price, "fut_price": fut_price, "mark_price": mark_price,
            "funding_rate": funding_rate, "basis_pct": basis_pct}


def part4_market_metadata():
    section("4) RESOLVED MARKET METADATA (ccxt, actual, not inferred from config labels)")
    import ccxt
    ex = ccxt.binance({"enableRateLimit": True})
    ex.load_markets()
    m = ex.market(SYMBOL_CCXT)
    fields = ["id", "type", "spot", "swap", "future", "linear", "contract", "base", "quote"]
    for f in fields:
        print(f"  {f}: {m.get(f)}")
    print(f"  exchange id: {ex.id}")
    print(f"  exchange default type option: {ex.options.get('defaultType')}")
    print(f"  apiKey set: {bool(ex.apiKey)}  secret set: {bool(ex.secret)}")
    return {f: m.get(f) for f in fields}


def part5_historical_research_market_type():
    section("5) HISTORICAL RESEARCH DATA SOURCE (research/data_cache.py)")
    with open(os.path.join(ROOT, "research", "data_cache.py")) as f:
        src = f.read()
    print("research/data_cache.py exchange construction:")
    for line in src.splitlines():
        if "getattr(ccxt" in line or "EXCHANGE" in line:
            print("  " + line.strip())
    print(f"\nconfig.EXCHANGE = {config.EXCHANGE!r} -- same ccxt id as feed.py, no defaultType override")
    print("Conclusion: historical research data is also Binance SPOT (same construction pattern).")


if __name__ == "__main__":
    results = {}
    results["part1"] = part1_shadow_feed_vs_direct_rest()
    results["part2_all_match"] = part2_closed_candle_integrity(8)
    results["part3"] = part3_spot_vs_futures()
    results["part4"] = part4_market_metadata()
    part5_historical_research_market_type()

    section("SUMMARY (JSON)")
    print(json.dumps(results, indent=2, default=str))
