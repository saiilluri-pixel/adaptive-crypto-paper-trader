"""
File I/O for the SOL Shock Continuation forward shadow: state.json, paper.log,
trades.csv, equity.csv, events.csv, config_snapshot.json.

`runtime_dir` is always an explicit constructor argument, never a module-level
constant -- this is what lets tests point a Ledger at a tmp directory and
assert the real shadow runtime directory gained zero files (rule 6/13:
"No test may write into these production-shadow files").
"""
import csv
import json
import os
from datetime import datetime, timezone

EVENTS_HEADER = [
    "market_ts_ms", "market_dt_utc", "local_observation_dt_utc", "phase",
    "price", "signal", "direction_label", "atr_norm_shock", "volume_mult",
    "return_pct", "range", "body_frac", "upper_wick_frac", "lower_wick_frac",
    "eligible", "position_state_before", "entry_decision", "rejection_reason",
]

TRADES_HEADER = [
    "signal_ts_ms", "signal_dt_utc", "entry_ts_ms", "entry_dt_utc",
    "exit_ts_ms", "exit_dt_utc", "side", "entry_ref_px", "entry_fill_px",
    "exit_ref_px", "exit_fill_px", "qty", "risk_amount_usdt", "entry_fee",
    "exit_fee", "slippage_cost_usdt", "gross_pnl", "net_pnl", "mfe_pct",
    "mae_pct", "holding_hours", "exit_reason", "equity_after", "phase",
    "local_observation_entry_dt", "local_observation_exit_dt",
]


def iso(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).isoformat(timespec="seconds")


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Ledger:
    def __init__(self, runtime_dir):
        self.dir = runtime_dir
        os.makedirs(self.dir, exist_ok=True)
        self.events_path = os.path.join(self.dir, "events.csv")
        self.trades_path = os.path.join(self.dir, "trades.csv")
        self.equity_path = os.path.join(self.dir, "equity.csv")
        self.log_path = os.path.join(self.dir, "paper.log")
        self.state_path = os.path.join(self.dir, "state.json")
        self.config_snapshot_path = os.path.join(self.dir, "config_snapshot.json")
        self._init_csv(self.events_path, EVENTS_HEADER)
        self._init_csv(self.trades_path, TRADES_HEADER)
        self._init_csv(self.equity_path, ["unix_ts", "dt_utc", "equity"])

    @staticmethod
    def _init_csv(path, header):
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.writer(f).writerow(header)

    def log(self, msg):
        line = f"{now_iso()}  {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def write_event(self, row):
        with open(self.events_path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(k, "") for k in EVENTS_HEADER])

    def write_trade(self, row):
        with open(self.trades_path, "a", newline="") as f:
            csv.writer(f).writerow([row.get(k, "") for k in TRADES_HEADER])

    def write_equity(self, ts_ms, equity):
        with open(self.equity_path, "a", newline="") as f:
            csv.writer(f).writerow([int(ts_ms // 1000), iso(ts_ms), f"{equity:.4f}"])

    def load_state(self):
        if not os.path.exists(self.state_path):
            return None
        try:
            with open(self.state_path) as f:
                return json.load(f)
        except Exception:
            return None

    def write_state(self, state):
        tmp = self.state_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2)
        os.replace(tmp, self.state_path)

    def write_config_snapshot(self, snapshot):
        with open(self.config_snapshot_path, "w") as f:
            json.dump(snapshot, f, indent=2)
