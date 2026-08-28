"""
READ-ONLY, standalone diagnostic. Completely separate process from
run_shadow.py -- imports nothing from it, feeds nothing into any trading
decision, writes only to this diagnostics/ directory. Public Binance
WebSocket streams only: no API key, no authentication, no order placement.

Subscribes to SOLUSDT kline_1h, aggTrade, and bookTicker on Binance's public
combined stream. Records exchange_event_ts (Binance's own event timestamp)
and local_receive_ts (this machine's wall clock on receipt) for every
message, plus price/bid/ask/spread and closed-candle OHLCV when a kline
closes.

    python3 ws_capture.py --seconds 300
"""
import argparse
import asyncio
import json
import os
import time

import websockets

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(HERE, "ws_capture_output.jsonl")

STREAMS = ["solusdt@kline_1h", "solusdt@aggTrade", "solusdt@bookTicker", "solusdt@depth10@100ms"]
URL = "wss://stream.binance.com:9443/stream?streams=" + "/".join(STREAMS)


def now_ms():
    return time.time() * 1000.0


async def capture(seconds, out_path):
    n = 0
    with open(out_path, "w") as f:
        async with websockets.connect(URL, open_timeout=10) as ws:
            deadline = time.time() + seconds
            while time.time() < deadline:
                try:
                    raw = await asyncio.wait_for(ws.recv(), timeout=5)
                except asyncio.TimeoutError:
                    continue
                local_receive_ts = now_ms()
                msg = json.loads(raw)
                stream = msg.get("stream", "")
                data = msg.get("data", {})

                row = {"local_receive_ts": local_receive_ts, "stream": stream}
                if stream.endswith("@bookTicker"):
                    row.update({
                        "type": "bookTicker",
                        "exchange_event_ts": None,  # bookTicker carries no event-time field
                        "best_bid": float(data.get("b")), "best_ask": float(data.get("a")),
                        "bid_qty": float(data.get("B")), "ask_qty": float(data.get("A")),
                    })
                    row["spread"] = row["best_ask"] - row["best_bid"]
                elif stream.endswith("@aggTrade"):
                    row.update({
                        "type": "aggTrade",
                        "exchange_event_ts": data.get("E"),
                        "trade_price": float(data.get("p")), "trade_qty": float(data.get("q")),
                        "trade_time_T": data.get("T"),
                    })
                elif "@kline" in stream:
                    k = data.get("k", {})
                    row.update({
                        "type": "kline", "exchange_event_ts": data.get("E"),
                        "kline_open_ts": k.get("t"), "kline_close_ts": k.get("T"),
                        "open": float(k.get("o")), "high": float(k.get("h")),
                        "low": float(k.get("l")), "close": float(k.get("c")),
                        "volume": float(k.get("v")), "is_closed": k.get("x"),
                    })
                elif "@depth" in stream:
                    bids = data.get("bids", [])
                    asks = data.get("asks", [])
                    row.update({
                        "type": "depth", "exchange_event_ts": None,
                        "best_bid": float(bids[0][0]) if bids else None,
                        "best_ask": float(asks[0][0]) if asks else None,
                        "bid_depth5_qty": sum(float(b[1]) for b in bids[:5]),
                        "ask_depth5_qty": sum(float(a[1]) for a in asks[:5]),
                    })
                else:
                    row["type"] = "unknown"

                f.write(json.dumps(row) + "\n")
                f.flush()
                n += 1
    print(f"captured {n} messages -> {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=int, default=300)
    ap.add_argument("--out", default=OUT_PATH)
    args = ap.parse_args()
    asyncio.run(capture(args.seconds, args.out))
