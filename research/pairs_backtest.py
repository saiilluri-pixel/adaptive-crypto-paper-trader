"""
Hypothesis D: RELATIVE VALUE / PAIRS (BTC/ETH)

Dedicated two-leg backtester -- PaperEngine is single-instrument, a spread
trade needs both legs modeled explicitly (each leg's own fee, both legs'
PnL). Rolling/causal only: hedge ratio and z-score are both computed from a
trailing window ending at the CURRENT bar, never the full dataset.
"""
import numpy as np
import pandas as pd

import config


def rolling_hedge_ratio(log_a, log_b, window):
    """Causal OLS beta of log_a on log_b, trailing `window` bars ending at
    each point (uses .shift(1) implicitly via rolling -- only bars up to and
    including the current one, no future data)."""
    cov = log_a.rolling(window).cov(log_b)
    var = log_b.rolling(window).var()
    beta = cov / var
    return beta


def run_pairs_backtest(df_a, df_b, hedge_window=60, z_window=20, entry_z=2.0, exit_z=0.5,
                        stop_z=4.0, start_capital=10_000.0, fee=None, slippage=None):
    """df_a/df_b: daily OHLCV (or any single TF) with 'ts'/'close', ALIGNED
    1:1 by row (same dates). Returns (trades_df, equity_curve)."""
    fee = config.TAKER_FEE if fee is None else fee
    slippage = config.SLIPPAGE if slippage is None else slippage

    df = pd.DataFrame({"ts": df_a["ts"].values, "a": df_a["close"].values, "b": df_b["close"].values})
    log_a = np.log(df["a"])
    log_b = np.log(df["b"])
    beta = rolling_hedge_ratio(log_a, log_b, hedge_window)
    spread = log_a - beta * log_b
    spread_mean = spread.rolling(z_window).mean()
    spread_std = spread.rolling(z_window).std()
    z = (spread - spread_mean) / spread_std

    cash = start_capital
    equity_curve = []
    pos = None   # dict: side ("long_spread"/"short_spread"), entry a/b price, beta, notional, entry_ts
    trades = []

    for i in range(len(df)):
        if pd.isna(z.iloc[i]) or pd.isna(beta.iloc[i]):
            equity_curve.append((int(df["ts"].iloc[i]), cash))
            continue
        price_a, price_b, zi, bi = df["a"].iloc[i], df["b"].iloc[i], z.iloc[i], beta.iloc[i]

        if pos is not None:
            # mark-to-market unrealized for equity curve
            ret_a = price_a / pos["entry_a"] - 1
            ret_b = price_b / pos["entry_b"] - 1
            leg_notional = pos["notional"]
            if pos["side"] == "long_spread":     # long A, short beta*B
                unreal = leg_notional * ret_a - leg_notional * pos["beta"] * ret_b
            else:                                 # short A, long beta*B
                unreal = -leg_notional * ret_a + leg_notional * pos["beta"] * ret_b
            equity = cash + unreal

            exit_now, reason = False, None
            if pos["side"] == "long_spread" and (zi >= -exit_z or zi <= -stop_z):
                exit_now, reason = True, ("reverted" if zi >= -exit_z else "stop")
            elif pos["side"] == "short_spread" and (zi <= exit_z or zi >= stop_z):
                exit_now, reason = True, ("reverted" if zi <= exit_z else "stop")

            if exit_now:
                gross = unreal
                exit_fee = leg_notional * fee * 2   # both legs, exit side
                exit_slip = leg_notional * slippage * 2
                net = gross - exit_fee - exit_slip
                cash += net
                trades.append({"entry_ts": pos["entry_ts"], "exit_ts": int(df["ts"].iloc[i]),
                                "side": pos["side"], "beta": pos["beta"], "gross_pnl": gross,
                                "fees": pos["entry_fee"] + exit_fee + exit_slip, "net_pnl": net - pos["entry_fee"],
                                "exit_reason": reason})
                pos = None
                equity = cash
        else:
            equity = cash
            if abs(zi) < stop_z:   # don't enter into an already-broken relationship
                notional = cash * 0.5   # 50% of equity per leg -- conservative for a 2-leg position
                entry_fee = notional * fee * 2
                entry_slip = notional * slippage * 2
                if zi <= -entry_z:
                    pos = {"side": "long_spread", "entry_a": price_a, "entry_b": price_b, "beta": bi,
                           "notional": notional, "entry_ts": int(df["ts"].iloc[i]),
                           "entry_fee": entry_fee + entry_slip}
                    cash -= (entry_fee + entry_slip)
                elif zi >= entry_z:
                    pos = {"side": "short_spread", "entry_a": price_a, "entry_b": price_b, "beta": bi,
                           "notional": notional, "entry_ts": int(df["ts"].iloc[i]),
                           "entry_fee": entry_fee + entry_slip}
                    cash -= (entry_fee + entry_slip)

        equity_curve.append((int(df["ts"].iloc[i]), equity))

    return pd.DataFrame(trades), equity_curve
