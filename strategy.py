"""
Fused strategy engine.

  ENTRY  = Prime Strategy Swing
             - structure (swings + BOS/CHoCH -> trend) on STRUCTURE_TF (30m)
             - fixed zone with Fibonacci 15/30/50/70/85 levels
             - Nadaraya-Watson band filter on CHART_TF (5m)
             - signals: BUY / BUY PRIME (long), SELL / SELL PRIME (short)
  EXIT   = Supertrend ATR with Trailing Stop Loss (computed on CHART_TF)
             - per-position ratcheting trailing stop (initial 3%)
             - close on trailing-stop hit OR Supertrend direction flip against the position

State is updated incrementally, one closed candle at a time, so results do not
repaint. Session/weekday/daily-cap gates from the Pine source are intentionally
dropped (24/7 crypto).
"""
import math
from collections import deque

import numpy as np

import config


def _is_pivot_high(highs, c, n):
    """Strict pivot high at index c with n bars either side (Pine ta.pivothigh)."""
    v = highs[c]
    for j in range(c - n, c + n + 1):
        if j != c and highs[j] >= v:
            return False
    return True


def _is_pivot_low(lows, c, n):
    v = lows[c]
    for j in range(c - n, c + n + 1):
        if j != c and lows[j] <= v:
            return False
    return True


class TrailingStop:
    """Supertrend-ATR trailing stop loss, ported per-position (long or short).

    Pine fidelity: the source's `stopLossPercent` is declared WITHOUT `var`,
    so Pine resets it to the initial configured percentage on every bar;
    only `updatedEntryPrice`/`stopLossPrice` are `var` (persistent across
    bars). `base_stop_pct` below is that immutable initial percentage, used
    fresh on every update() call. `sl_pct` is retained as an attribute (last
    computed value) only for state-file/display compatibility with run.py's
    state.json serialization -- it is never read back into the ratchet math.
    """

    def __init__(self, side, entry_price, init_stop):
        self.side = side
        self.updated_entry = entry_price
        self.base_stop_pct = -init_stop if side == "long" else init_stop
        self.sl_pct = self.base_stop_pct
        self.sl_price = entry_price * (1 + self.sl_pct / 100.0)

    def update(self, close):
        pct = (close - self.updated_entry) / self.updated_entry * 100.0
        if self.side == "long" and pct > 1:
            temp_sl_pct = self.base_stop_pct + pct - 1.0
            candidate = self.updated_entry * (1 + temp_sl_pct / 100.0)
            self.sl_price = max(self.sl_price, candidate)
            self.updated_entry = self.sl_price
            self.sl_pct = temp_sl_pct
        elif self.side == "short" and pct < -1:
            temp_sl_pct = self.base_stop_pct + pct + 1.0
            candidate = self.updated_entry * (1 + temp_sl_pct / 100.0)
            self.sl_price = min(self.sl_price, candidate)
            self.updated_entry = self.sl_price
            self.sl_pct = temp_sl_pct

    def hit(self, price):
        return price <= self.sl_price if self.side == "long" else price >= self.sl_price


class Strategy:
    def __init__(self, params=None):
        # params may differ per symbol; default to current config globals
        self.p = params or {k: getattr(config, k) for k in (
            "SWING_LEN", "NW_BANDWIDTH", "NW_MULT", "NW_WINDOW",
            "ST_ATR_PERIOD", "ST_ATR_MULT", "ST_INIT_STOP")}
        # ── structure-TF (30m) state ──
        self.s_high, self.s_low, self.s_close, self.s_open = [], [], [], []
        self.pivot_checked = -1            # last structure index evaluated for a pivot
        self.zone_swing_h = None
        self.zone_swing_l = None
        self.htf_last_sh = None            # last swing high for BOS/CHoCH
        self.htf_last_sl = None
        self.trend = 0                     # +1 bull / -1 bear / 0 neutral
        self.choch_dir = 0
        self.zone_needs_update = True

        # ── fixed zone + fib levels ──
        self.fz_high = None
        self.fz_low = None
        self.fz_trend = 0
        self.z15 = self.z30 = self.zmid = self.z70 = self.z85 = None

        # ── chart-TF (5m) state ──
        self.c_closes = []                 # full close history (for NW)
        self.nw_abs_dev = deque(maxlen=self.p["NW_WINDOW"] - 1)
        self.nw_lower = None
        self.nw_upper = None
        self.nw_w = np.array([math.exp(-(i * i) / (self.p["NW_BANDWIDTH"] ** 2 * 2.0))
                              for i in range(self.p["NW_WINDOW"])])
        self.nw_den = self.nw_w.sum()

        # signal latches
        self.in_buy_prime = self.in_buy = self.in_sell_prime = self.in_sell = False

        # ── supertrend (5m) state ──
        self.st_long_prev = None
        self.st_short_prev = None
        self.st_dir = 1
        self.st_atr_prev = None
        self.st_close_prev = None
        self.st_value = None               # active stop line (long or short)

    # ──────────────────────────────────────────────────────────────
    # STRUCTURE  (30m)
    # ──────────────────────────────────────────────────────────────
    def feed_struct_bar(self, o, h, l, c):
        self.s_open.append(o); self.s_high.append(h)
        self.s_low.append(l);  self.s_close.append(c)
        n = self.p["SWING_LEN"]
        idx = len(self.s_high) - 1

        # confirm any pivots whose right-side window just completed
        cand = idx - n
        if cand > self.pivot_checked and cand - n >= 0:
            for c_i in range(self.pivot_checked + 1, cand + 1):
                if c_i - n < 0:
                    continue
                if _is_pivot_high(self.s_high, c_i, n):
                    self.zone_swing_h = self.s_high[c_i]
                    self.htf_last_sh = self.s_high[c_i]
                if _is_pivot_low(self.s_low, c_i, n):
                    self.zone_swing_l = self.s_low[c_i]
                    self.htf_last_sl = self.s_low[c_i]
            self.pivot_checked = cand

        # BOS / CHoCH on this confirmed 30m bar
        bull_body = self.htf_last_sh is not None and c > self.htf_last_sh
        bull_wick = self.htf_last_sh is not None and h > self.htf_last_sh and c <= self.htf_last_sh
        bear_body = self.htf_last_sl is not None and c < self.htf_last_sl
        bear_wick = self.htf_last_sl is not None and l < self.htf_last_sl and c >= self.htf_last_sl
        bull_break = bull_body or bull_wick
        bear_break = bear_body or bear_wick

        if bull_break:
            if self.trend != 1:            # CHoCH
                self.trend = 1
                self.choch_dir = 1
                self.zone_needs_update = True
            self.htf_last_sh = None        # BOS otherwise; clear either way
        if bear_break:
            if self.trend != -1:           # CHoCH
                self.trend = -1
                self.choch_dir = -1
                self.zone_needs_update = True
            self.htf_last_sl = None

    # ──────────────────────────────────────────────────────────────
    # CHART  (5m)  — returns a snapshot dict incl. fired signals
    # ──────────────────────────────────────────────────────────────
    def feed_chart_bar(self, o, h, l, c):
        # zone invalidation (Pine: confirmed-bar close beyond the zone)
        if self.fz_trend == -1 and self.fz_low is not None and c < self.fz_low and c < o:
            self.zone_needs_update = True
        if self.fz_trend == 1 and self.fz_high is not None and c > self.fz_high and c > o:
            self.zone_needs_update = True

        # (re)build fixed zone + fib levels
        if (self.zone_needs_update and self.zone_swing_h is not None
                and self.zone_swing_l is not None and self.trend != 0):
            self.fz_high = max(self.zone_swing_h, self.zone_swing_l)
            self.fz_low = min(self.zone_swing_h, self.zone_swing_l)
            self.fz_trend = self.trend
            self.zone_needs_update = False
        if self.fz_high is not None and self.fz_low is not None:
            rng = self.fz_high - self.fz_low
            self.z15 = self.fz_low + rng * 0.15
            self.z30 = self.fz_low + rng * 0.30
            self.zmid = self.fz_low + rng * 0.50
            self.z70 = self.fz_low + rng * 0.70
            self.z85 = self.fz_low + rng * 0.85

        # Nadaraya-Watson bands
        self.c_closes.append(c)
        self._update_nw()

        # supertrend
        self._update_supertrend(h, l, c)

        # signals (latched, gates dropped)
        z = self.fz_trend
        buy_prime = z == 1 and self.z15 is not None and l <= self.z15
        buy_zone = z == 1 and self.z30 is not None and l <= self.z30 and l > self.z15
        sell_prime = z == -1 and self.z85 is not None and h >= self.z85
        sell_zone = z == -1 and self.z70 is not None and h >= self.z70 and h < self.z85
        nw_buy_ok = self.nw_lower is not None and l <= self.nw_lower
        nw_sell_ok = self.nw_upper is not None and h >= self.nw_upper

        if not buy_prime:  self.in_buy_prime = False
        if not buy_zone:   self.in_buy = False
        if not sell_prime: self.in_sell_prime = False
        if not sell_zone:  self.in_sell = False

        sig_buy_prime = buy_prime and not self.in_buy_prime
        sig_buy = buy_zone and nw_buy_ok and not self.in_buy
        sig_sell_prime = sell_prime and not self.in_sell_prime
        sig_sell = sell_zone and nw_sell_ok and not self.in_sell

        if sig_buy_prime:  self.in_buy_prime = True
        if sig_buy:        self.in_buy = True
        if sig_sell_prime: self.in_sell_prime = True
        if sig_sell:       self.in_sell = True

        return {
            "close": c, "high": h, "low": l,
            "sig_buy_prime": sig_buy_prime, "sig_buy": sig_buy,
            "sig_sell_prime": sig_sell_prime, "sig_sell": sig_sell,
            "trend": self.trend, "fz_trend": self.fz_trend,
            "z15": self.z15, "z30": self.z30, "z70": self.z70, "z85": self.z85,
            "nw_lower": self.nw_lower, "nw_upper": self.nw_upper,
            "st_dir": self.st_dir, "st_value": self.st_value,
        }

    def _update_nw(self):
        if len(self.c_closes) < self.p["NW_WINDOW"]:
            return
        window = np.array(self.c_closes[-self.p["NW_WINDOW"]:])[::-1]  # idx 0 = current
        nw_out = float(np.dot(window, self.nw_w) / self.nw_den)
        self.nw_abs_dev.append(abs(self.c_closes[-1] - nw_out))
        mae = (sum(self.nw_abs_dev) / len(self.nw_abs_dev)) * self.p["NW_MULT"]
        self.nw_lower = nw_out - mae
        self.nw_upper = nw_out + mae

    def _update_supertrend(self, h, l, c):
        hl2 = (h + l) / 2.0
        # ATR (Wilder); period 1 -> ATR == TR
        if self.st_close_prev is None:
            tr = h - l
        else:
            tr = max(h - l, abs(h - self.st_close_prev), abs(l - self.st_close_prev))
        p = self.p["ST_ATR_PERIOD"]
        if self.st_atr_prev is None:
            atr = tr
        else:
            atr = (self.st_atr_prev * (p - 1) + tr) / p
        a = self.p["ST_ATR_MULT"] * atr

        long_stop = hl2 - a
        if self.st_long_prev is not None and self.st_close_prev is not None \
                and self.st_close_prev > self.st_long_prev:
            long_stop = max(long_stop, self.st_long_prev)
        short_stop = hl2 + a
        if self.st_short_prev is not None and self.st_close_prev is not None \
                and self.st_close_prev < self.st_short_prev:
            short_stop = min(short_stop, self.st_short_prev)

        d = self.st_dir
        if self.st_short_prev is not None and d == -1 and c > self.st_short_prev:
            d = 1
        elif self.st_long_prev is not None and d == 1 and c < self.st_long_prev:
            d = -1
        self.st_dir = d
        self.st_value = long_stop if d == 1 else short_stop

        self.st_atr_prev = atr
        self.st_close_prev = c
        self.st_long_prev = long_stop
        self.st_short_prev = short_stop
