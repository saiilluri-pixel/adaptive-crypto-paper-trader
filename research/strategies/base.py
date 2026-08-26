"""
Research strategy interface (Crypto Strategy Research V2, Phase 2).

Deliberately separate from strategy.py / PaperEngine.on_signals(), which are
tightly coupled to Prime-Swing's specific 4-way signal vocabulary
(sig_buy_prime/sig_buy/sig_sell_prime/sig_sell). New candidates don't need
to speak that vocabulary or touch the live runner at all -- they implement
this narrower interface, and research/generic_runner.py drives the exact
same PaperEngine (fees, slippage, risk-based sizing, risk guards, the
corrected trailing-stop math and stop-fill logic) directly via its generic
enter()/check_stop_bar()/manage_on_bar() methods.

Contract:
  - on_bar() is called once per closed bar, in chronological order, and may
    only use the bar just passed in plus whatever state the strategy chose
    to remember from EARLIER calls. No lookahead is possible by construction
    -- the runner never passes future bars, and there is no mechanism to
    request one.
  - Returns "long", "short", or None. The runner only acts on a signal when
    flat (one position at a time, same as production).
  - Exits are handled by the shared, already-validated trailing-stop
    machinery (Position/TrailingStop/check_stop_bar), not reimplemented per
    strategy -- every candidate family gets identical, correctness-tested
    exit execution. A strategy expresses its desired stop distance via
    params["INIT_STOP_PCT"], read once per backtest run (see generic_runner
    docstring for why this is a per-run, not per-trade, approximation for
    "ATR-based stop" designs).
"""


class ResearchStrategy:
    def __init__(self, params):
        self.p = params

    def on_bar(self, o, h, l, c, ts_ms, htf_trend=None, funding_rate=None):
        """o/h/l/c: this bar's OHLC. ts_ms: this bar's close timestamp (int,
        matches the chart cache's 'ts' + one bar). htf_trend: optional +1/0/-1
        higher-timeframe trend context, precomputed by the runner (see
        generic_runner.compute_htf_trend). funding_rate: optional current
        Binance perpetual funding rate (as a fraction, e.g. 0.0001 = 0.01%),
        forward-filled onto this bar (see generic_runner.align_funding) --
        the real, most-recently-known rate as of this bar, never a future
        one. Both are purely informational context; a strategy may ignore
        either. Must return 'long', 'short', or None."""
        raise NotImplementedError
