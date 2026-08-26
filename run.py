"""
Live multi-symbol, multi-variant paper-trading runner.

Runs every config.VARIANTS (e.g. A = profit-tuned/slow exits, B = more-trades/
tight) across every config.SYMBOLS, all on the SAME live Binance prices, so the
variants can be compared head-to-head. Each (variant, symbol) is its own paper
account (START_CAPITAL each).

Publishes one combined state.json:
    {ts, variants:{V:{label, symbols:{SYM:{...}}, totals}}, grand_total}
Per (variant,symbol): trades_<V>_<SYM>.csv and equity_<V>_<SYM>.csv.

    python3 run.py            # run live
    python3 run.py --warmup   # warm up, print snapshots, exit
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

import config
from feed import Feed
from strategy import Strategy
from engine import PaperEngine, Position

TF_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "30m": 1_800_000,
         "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000}
HERE = os.path.dirname(os.path.abspath(__file__))


def log_line(msg):
    line = f"{datetime.now(timezone.utc):%H:%M:%S}  {msg}"
    print(line, flush=True)
    with open(os.path.join(HERE, config.LOG_FILE), "a") as f:
        f.write(line + "\n")


class VariantTrader:
    """One (symbol, variant): own Strategy + own paper account + own exit rules."""

    def __init__(self, symbol, tag, vkey, vcfg, log_trades=True, log_fn=None):
        self.symbol, self.tag, self.vkey = symbol, tag, vkey
        safe = symbol.replace("/", "")
        self.file_tag = f"{vkey}_{safe}"
        params = dict(config.params_for(symbol))
        params.update(vcfg.get("overrides", {}))
        self.params = params
        self.strat = Strategy(params)
        self.eng = PaperEngine(
            log_fn=log_fn or (lambda m: log_line(f"[{vkey}·{tag}] {m}")),
            symbol=symbol, params=params, file_tag=self.file_tag,
            exit_on_flip=vcfg["exit_on_flip"], arm_flip=vcfg["arm_flip"],
            log_trades=log_trades)

    def on_chart_bar(self, o, h, l, c):
        # OHLC reconciliation FIRST, using the stop as it stood before this
        # bar: check_stop_intrabar() only sees prices at ~POLL_SECONDS
        # granularity between bars, so a touch that happened and reverted
        # within one polling gap could otherwise be missed entirely.
        if self.eng.pos is not None:
            self.eng.check_stop_bar(o, h, l)
        snap = self.strat.feed_chart_bar(o, h, l, c)   # unconditional indicator update
        if self.eng.pos is not None:
            self.eng.manage_on_bar(snap)               # ratchet TSL + Supertrend flip, only if still open
        self.eng.on_signals(snap)                       # entries when flat (guard-gated)
        return snap

    def snapshot(self, price):
        s, e = self.strat, self.eng
        pos = None
        if e.pos:
            p = e.pos
            pos = {"side": p.side, "qty": p.qty, "entry": p.entry_price,
                   "stop": p.tsl.sl_price, "unrealized": p.unrealized(price),
                   "reason": p.reason, "entry_time": p.entry_time,
                   "flip_armed": p.flip_armed, "sl_pct": p.tsl.sl_pct,
                   "updated_entry": p.tsl.updated_entry, "entry_fee": p.entry_fee}
        return {
            "symbol": self.symbol, "price": price,
            "equity": e.equity, "cash": e.cash, "realized": e.realized,
            "start_capital": config.START_CAPITAL,
            "return_pct": (e.equity / config.START_CAPITAL - 1) * 100,
            "n_trades": e.n_trades, "wins": e.wins, "position": pos,
            "peak_equity": e.peak_equity, "consecutive_losses": e.consecutive_losses,
            "trading_day": e.trading_day, "daily_start_equity": e.daily_start_equity,
            "reconciliation_failed": e.reconciliation_failed,
            "market": {"trend": s.trend, "fz_trend": s.fz_trend,
                       "z15": s.z15, "z30": s.z30, "z70": s.z70, "z85": s.z85,
                       "nw_lower": s.nw_lower, "nw_upper": s.nw_upper,
                       "st_dir": s.st_dir, "st_value": s.st_value}}

    def load_state(self, d):
        if not d:
            return
        e = self.eng
        e.cash = d.get("cash", e.cash)
        e.realized = d.get("realized", 0.0)
        e.n_trades = d.get("n_trades", 0)
        e.wins = d.get("wins", 0)
        e.equity = d.get("equity", e.cash)
        e.peak_equity = d.get("peak_equity", e.equity)
        e.consecutive_losses = d.get("consecutive_losses", 0)
        e.trading_day = d.get("trading_day")
        e.daily_start_equity = d.get("daily_start_equity", e.equity)
        e.reconciliation_failed = d.get("reconciliation_failed", False)
        p = d.get("position")
        if p:
            pos = Position(p["side"], p["entry"], p["qty"], p["entry_time"],
                           p["reason"], self.params["ST_INIT_STOP"],
                           entry_fee=p.get("entry_fee", 0.0))
            pos.flip_armed = p.get("flip_armed", False)
            pos.tsl.sl_price = p["stop"]
            pos.tsl.sl_pct = p.get("sl_pct", pos.tsl.sl_pct)
            pos.tsl.updated_entry = p.get("updated_entry", p["entry"])
            e.pos = pos


class SymbolGroup:
    """One symbol, shared candle feed, fanned out to every variant's strategy."""

    def __init__(self, symbol, feed, log_trades=True, log_fn=None):
        self.symbol = symbol
        self.tag = symbol.split("/")[0]
        self.safe = symbol.replace("/", "")
        self.feed = feed
        self.last_chart_ts = 0
        self.last_struct_ts = 0
        self.last_price = None
        self.log_fn = log_fn or log_line   # this group's own status/diagnostic messages
        self.variants = {vk: VariantTrader(symbol, self.tag, vk, vc,
                                            log_trades=log_trades, log_fn=log_fn)
                         for vk, vc in config.VARIANTS.items()}

    def _events(self, struct, chart):
        ev = []
        for _, r in struct.iterrows():
            ev.append((int(r.ts) + TF_MS[config.STRUCTURE_TF], 0, "S", r))
        for _, r in chart.iterrows():
            ev.append((int(r.ts) + TF_MS[config.CHART_TF], 1, "C", r))
        ev.sort(key=lambda e: (e[0], e[1]))
        return ev

    def warmup(self, resume_chart_ts=None):
        """Rebuild indicator state from recent history. If resume_chart_ts is
        given (the chart-bar cursor from before an outage/restart), any bar
        newer than it is ALSO replayed through check_stop_bar() then
        manage_on_bar() -- but NEVER on_signals() -- for variants that already
        have an open position restored (via load_state(), called before
        this). check_stop_bar() runs first each bar, using the stop as it
        stood before that bar (OHLC-aware: gap vs. ordinary touch), so a
        missed stop-out is detected even if the bar's close alone wouldn't
        show it; manage_on_bar() then ratchets/flip-checks only if the
        position is still open after that. If the position closes mid-replay,
        indicator state continues to be rebuilt for all subsequent bars, but
        no entry is ever created from a signal that fired while offline.

        Gap-integrity: if resume_chart_ts predates the earliest bar this
        fetch can see, some missed bars are permanently unreconcilable for
        any variant with an open position -- rather than silently resuming,
        that engine's reconciliation_failed flag is set, which blocks new
        entries (but not managing the existing position) until it closes."""
        struct = self.feed.retry(self.feed.closed_candles,
                                 self.symbol, config.STRUCTURE_TF, config.WARMUP_STRUCT)
        chart = self.feed.retry(self.feed.closed_candles,
                                self.symbol, config.CHART_TF, config.WARMUP_CHART)

        if resume_chart_ts is not None and len(chart) > 0:
            earliest_available = int(chart.iloc[0].ts)
            if resume_chart_ts < earliest_available:
                for vk, vt in self.variants.items():
                    if vt.eng.pos is not None:
                        vt.eng.reconciliation_failed = True
                        self.log_fn(f"[{vk}·{self.tag}] !! GAP INTEGRITY FAILURE: saved cursor "
                                    f"predates available history by "
                                    f"{(earliest_available - resume_chart_ts) // 60_000} min -- "
                                    f"open position's missed bars cannot be fully reconciled; "
                                    f"new entries blocked until this position closes.")

        caught_up = 0
        for _, _, kind, r in self._events(struct, chart):
            for vt in self.variants.values():
                if kind == "S":
                    vt.strat.feed_struct_bar(r.open, r.high, r.low, r.close)
                    continue
                is_catchup = resume_chart_ts is not None and int(r.ts) > resume_chart_ts
                if is_catchup and vt.eng.pos is not None:
                    vt.eng.check_stop_bar(r.open, r.high, r.low)
                snap = vt.strat.feed_chart_bar(r.open, r.high, r.low, r.close)
                if is_catchup and vt.eng.pos is not None:
                    vt.eng.manage_on_bar(snap)   # exit management only, never on_signals
                    caught_up += 1
        self.last_struct_ts = int(struct.iloc[-1].ts)
        self.last_chart_ts = int(chart.iloc[-1].ts)
        self.last_price = self.feed.retry(self.feed.last_price, self.symbol)
        suffix = f", caught up {caught_up} missed management bar(s)" if caught_up else ""
        self.log_fn(f"[{self.tag}] warmed @ {self.last_price:,.2f} "
                    f"({len(self.variants)} variants){suffix}")

    def process_new_bars(self):
        struct = self.feed.retry(self.feed.closed_candles, self.symbol, config.STRUCTURE_TF, 40)
        chart = self.feed.retry(self.feed.closed_candles, self.symbol, config.CHART_TF, 40)
        new = []
        for _, r in struct.iterrows():
            if int(r.ts) > self.last_struct_ts:
                new.append((int(r.ts) + TF_MS[config.STRUCTURE_TF], 0, "S", r))
        for _, r in chart.iterrows():
            if int(r.ts) > self.last_chart_ts:
                new.append((int(r.ts) + TF_MS[config.CHART_TF], 1, "C", r))
        new.sort(key=lambda e: (e[0], e[1]))
        for _, _, kind, r in new:
            if kind == "S":
                for vt in self.variants.values():
                    vt.strat.feed_struct_bar(r.open, r.high, r.low, r.close)
                self.last_struct_ts = int(r.ts)
            else:
                for vt in self.variants.values():
                    snap = vt.on_chart_bar(r.open, r.high, r.low, r.close)
                    fired = [k for k in ("sig_buy_prime", "sig_buy",
                                         "sig_sell_prime", "sig_sell") if snap[k]]
                    if fired:
                        self.log_fn(f"[{vt.vkey}·{self.tag}] bar {r['dt']:%H:%M} "
                                    f"{', '.join(fired)} @ {snap['close']:,.2f}")
                self.last_chart_ts = int(r.ts)

    def on_tick(self, price):
        self.last_price = price
        for vt in self.variants.values():
            vt.eng.check_stop_intrabar(price)
            vt.eng.mark(price)


class Runner:
    def __init__(self):
        self.feed = Feed()
        self.groups = [SymbolGroup(s, self.feed) for s in config.SYMBOLS]
        self.cursors = {}          # symbol -> {"chart_ts": int}, from the last state.json

    def warmup(self):
        vs = ", ".join(f"{k}={v['label']}" for k, v in config.VARIANTS.items())
        log_line(f"Warming up {len(self.groups)} symbols × {len(config.VARIANTS)} variants  ({vs})…")
        for g in self.groups:
            resume_ts = self.cursors.get(g.symbol, {}).get("chart_ts")
            g.warmup(resume_chart_ts=resume_ts)

    def load_state(self):
        """Restores cash/realized/position/cursors from state.json. Must be
        called BEFORE warmup() so that any restored open position is present
        when warmup() replays missed bars through manage_on_bar()."""
        path = os.path.join(HERE, config.STATE_FILE)
        if not os.path.exists(path):
            return
        try:
            with open(path) as f:
                d = json.load(f)
        except Exception:
            return
        variants = d.get("variants", {})
        for vk in config.VARIANTS:
            syms = variants.get(vk, {}).get("symbols", {})
            for g in self.groups:
                g.variants[vk].load_state(syms.get(g.symbol))
        self.cursors = d.get("cursors", {})   # {} on an old/pre-upgrade state.json -- safe default

    def write_state(self):
        variants_out = {}
        for vk, vc in config.VARIANTS.items():
            syms = {g.symbol: g.variants[vk].snapshot(g.last_price) for g in self.groups}
            tot_eq = sum(s["equity"] for s in syms.values())
            start = config.START_CAPITAL * len(self.groups)
            variants_out[vk] = {
                "label": vc["label"], "symbols": syms,
                "totals": {"equity": tot_eq, "start_capital": start,
                           "return_pct": (tot_eq / start - 1) * 100,
                           "realized": sum(s["realized"] for s in syms.values()),
                           "n_trades": sum(s["n_trades"] for s in syms.values())}}
        cursors = {g.symbol: {"chart_ts": g.last_chart_ts} for g in self.groups}
        state = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                 "variants": variants_out, "cursors": cursors}
        path = os.path.join(HERE, config.STATE_FILE)
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f)
        os.replace(tmp, path)

    def loop(self):
        log_line("── LIVE — A/B variants on Binance public prices (no keys) ──")
        last_status = 0
        while True:
            try:
                for g in self.groups:
                    g.process_new_bars()
                    price = self.feed.retry(self.feed.last_price, g.symbol)
                    g.on_tick(price)
                self.write_state()

                now = time.time()
                if now - last_status >= 60:
                    for vk in config.VARIANTS:
                        eq = sum(g.variants[vk].eng.equity for g in self.groups)
                        ntr = sum(g.variants[vk].eng.n_trades for g in self.groups)
                        opn = sum(1 for g in self.groups if g.variants[vk].eng.pos)
                        start = config.START_CAPITAL * len(self.groups)
                        log_line(f"[{vk}] eq=${eq:,.0f} ret={ (eq/start-1)*100:+.2f}% "
                                 f"closed={ntr} open={opn}")
                        for g in self.groups:
                            with open(os.path.join(HERE, f"equity_{vk}_{g.safe}.csv"), "a") as f:
                                f.write(f"{int(now)},{g.variants[vk].eng.equity:.2f}\n")
                    last_status = now
                time.sleep(config.POLL_SECONDS)
            except KeyboardInterrupt:
                return
            except Exception as e:
                log_line(f"[loop] {type(e).__name__}: {str(e)[:120]} — continuing")
                time.sleep(config.POLL_SECONDS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", action="store_true")
    args = ap.parse_args()
    r = Runner()
    r.load_state()      # restore cash/realized/position/cursors BEFORE warmup, so any
                         # restored open position gets missed-bar management catch-up
    r.warmup()
    if args.warmup:
        log_line("warmup-only: exiting.")
        return
    r.write_state()
    r.loop()


if __name__ == "__main__":
    main()
