"""
Standalone, periodically-scheduled research process (adaptive spec
sections 2-3, service label com.btcpaper.adaptive.research). Runs as a
completely separate OS process from adaptive/runner.py -- never blocks
market-data polling, position management, stop management, exits, or
state persistence, because it IS a separate process, not a thread inside
the execution loop.

Owns its own ccxt exchange instance (public, read-only), its own
ResearchCache (adaptive/research_cache.py), and its own AdaptationEngine
used purely as the RULES engine for evaluate_promotion/consider_promotion.
Champion state itself is persisted through adaptive/champion_store.py's
ChampionStore -- the single shared, versioned, hash-verified file that
adaptive/runner.py reads (read-only, with corruption fallback) and this
process is the ONLY writer of. This is the entire research<->execution
interaction surface (spec section 15): no other file this process writes
is ever read by the execution process, and it never touches state.json,
trades.csv, or decisions.jsonl (all execution-owned).

Cadence (spec section 3): a light cycle every LIGHT_CYCLE_SECONDS (default
6h) that only extends the research cache with new closed candles; a full
challenger-evaluation cycle every FULL_CYCLE_SECONDS (default 24h) that
runs the TRAIN->VALIDATION->TEST walk-forward and may promote. Promotion
frequency is separately capped by adaptation.py's own
MIN_PROMOTION_INTERVAL_SEC (24h) regardless of how often this scheduler
wakes up, so even a full cycle running more often than intended cannot
promote more often than that floor allows.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Optional

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from adaptive.market_data import SYMBOLS, build_exchange  # noqa: E402
from adaptive.research_cache import ResearchCache  # noqa: E402
from adaptive.adaptation import AdaptationEngine  # noqa: E402
from adaptive.champion_store import ChampionStore, ChampionCorruptionError  # noqa: E402
from adaptive.research_worker import run_research_cycle_walkforward  # noqa: E402
from adaptive.runner import DEFAULT_PARAMS, FEE_RATE, SLIPPAGE  # noqa: E402

LIGHT_CYCLE_SECONDS = 6 * 3600
FULL_CYCLE_SECONDS = 24 * 3600
SCHEDULER_POLL_SECONDS = 600  # check every 10 min whether a cycle is due
RESEARCH_TIMEFRAME = "1h"

# Automatic rollback thresholds (spec section 12): "severe statistically
# meaningful degradation", never a rollback for one or two losing trades.
ROLLBACK_MIN_SAMPLE = 15
ROLLBACK_PF_FLOOR = 0.5


def now_iso():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _iso_or_none(ts):
    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(timespec="seconds") if ts else None


class ResearchService:
    def __init__(self, runtime_dir: str, exchange=None):
        self.runtime_dir = runtime_dir
        os.makedirs(runtime_dir, exist_ok=True)
        self.ex = exchange if exchange is not None else build_exchange()
        self.cache = ResearchCache(os.path.join(runtime_dir, "research_cache"), self.ex)
        self.champion_store = ChampionStore(os.path.join(runtime_dir, "champion_state.json"))
        self.adaptation = AdaptationEngine(os.path.join(runtime_dir, "adaptation_log.jsonl"))
        self.status_path = os.path.join(runtime_dir, "research_status.json")
        self.log_path = os.path.join(runtime_dir, "research.log")
        self.last_light_ts: Optional[float] = None
        self.last_full_ts: Optional[float] = None

    def log(self, msg: str):
        line = f"{now_iso()}  {msg}"
        print(line, flush=True)
        with open(self.log_path, "a") as f:
            f.write(line + "\n")

    def bootstrap(self):
        records = self.champion_store.ensure_baseline(DEFAULT_PARAMS)
        self.adaptation.champions = {s: r.parameters for s, r in records.items()}
        status = self._load_status()
        if status:
            self.last_light_ts = status.get("last_light_run_ts")
            self.last_full_ts = status.get("last_full_run_ts")
            self.adaptation.last_promotion_ts = status.get("last_promotion_ts") or {}
        self.log(f"bootstrap complete: champions={ {s: r.version for s, r in records.items()} }")
        self._write_status()

    def _load_status(self) -> Optional[dict]:
        if not os.path.exists(self.status_path):
            return None
        try:
            with open(self.status_path) as f:
                return json.load(f)
        except Exception:
            return None

    def _write_status(self, last_result: Optional[dict] = None):
        try:
            records = self.champion_store.read()
            champions_summary = {s: {"version": r.version, "status": r.status, "hash": r.hash}
                                  for s, r in records.items()}
        except Exception:
            champions_summary = {}
        state = {
            "written_at_iso": now_iso(),
            "last_light_run_ts": self.last_light_ts, "last_light_run_iso": _iso_or_none(self.last_light_ts),
            "next_light_run_iso": _iso_or_none(self.last_light_ts + LIGHT_CYCLE_SECONDS) if self.last_light_ts else "due now",
            "last_full_run_ts": self.last_full_ts, "last_full_run_iso": _iso_or_none(self.last_full_ts),
            "next_full_run_iso": _iso_or_none(self.last_full_ts + FULL_CYCLE_SECONDS) if self.last_full_ts else "due now",
            "last_promotion_ts": self.adaptation.last_promotion_ts,
            "champions": champions_summary,
            "last_result": last_result,
        }
        tmp = self.status_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(state, f, indent=2, default=str)
        os.replace(tmp, self.status_path)

    def run_light_cycle(self):
        self.log("light cycle: extending research cache")
        gaps = []
        for sym in SYMBOLS:
            for tf in ("5m", "15m", "1h", "4h"):
                _, gap = self.cache.extend(sym, tf)
                if gap:
                    gaps.append(f"{sym}/{tf}")
        if gaps:
            self.log(f"  gaps detected while extending research cache: {gaps}")
        self.last_light_ts = time.time()
        self._write_status({"type": "light", "gaps": gaps})

    def check_for_rollback(self) -> list:
        """Automatic rollback on severe, statistically meaningful live-paper
        degradation (spec section 9/12). Reads adaptive_runtime/state.json
        (the execution process's own file) READ-ONLY for monitoring --
        this is a read, not a write, so it does not violate section 15's
        isolation rule (which constrains this process from WRITING
        anything execution-owned; it says nothing against reading for
        monitoring purposes, and section 9 explicitly asks for live-paper
        results as a degradation input).

        Known scoping simplification, disclosed rather than hidden:
        evaluates each strategy's OVERALL rolling live stats (all recorded
        trades in stats_store for that strategy, across regimes), not
        trades strictly isolated to the time since the CURRENT promotion --
        stats_store does not currently track per-promotion-epoch
        attribution. Only strategies with a PROMOTED (non-baseline)
        champion are eligible; a BASELINE champion is never "rolled back"
        (there is nothing before it). Never triggers on a small sample --
        requires ROLLBACK_MIN_SAMPLE trades minimum."""
        state_path = os.path.join(self.runtime_dir, "state.json")
        if not os.path.exists(state_path):
            return []
        try:
            with open(state_path) as f:
                exec_state = json.load(f)
        except Exception:
            return []
        stats_raw = exec_state.get("stats_store", {})

        try:
            records = self.champion_store.read()
        except ChampionCorruptionError:
            return []

        rolled_back = []
        for strategy, record in records.items():
            if record.status != "PROMOTED":
                continue
            trades = []
            for key, cell in stats_raw.items():
                parts = key.split("|", 2)
                if len(parts) == 3 and parts[1] == strategy:
                    trades.extend(cell.get("trades", []))
            if len(trades) < ROLLBACK_MIN_SAMPLE:
                continue
            wins = sum(t for t in trades if t > 0)
            losses = sum(abs(t) for t in trades if t <= 0)
            pf = (wins / losses) if losses > 0 else (float("inf") if wins > 0 else 0.0)
            if pf < ROLLBACK_PF_FLOOR:
                restored = self.champion_store.rollback(strategy)
                if restored:
                    self.log(f"!! AUTOMATIC ROLLBACK: {strategy} live PF={pf:.3f} < floor "
                             f"{ROLLBACK_PF_FLOOR} (n={len(trades)} trades) -- restored v{restored.version}")
                    self.adaptation._log(strategy, "rollback", {
                        "reason": f"live PF {pf:.3f} < floor {ROLLBACK_PF_FLOOR} over {len(trades)} live trades",
                        "restored_version": restored.version, "restored_hash": restored.hash,
                        "rolled_back_from_version": record.version, "rolled_back_from_hash": record.hash,
                    })
                    rolled_back.append(strategy)
        return rolled_back

    def run_full_cycle(self):
        self.log("full challenger-evaluation cycle starting")
        rolled_back = self.check_for_rollback()
        records = self.champion_store.ensure_baseline(DEFAULT_PARAMS)
        self.adaptation.champions = {s: r.parameters for s, r in records.items()}

        historical_data_by_symbol = {}
        for sym in SYMBOLS:
            df, gap = self.cache.extend(sym, RESEARCH_TIMEFRAME)
            if gap:
                self.log(f"  !! gap in {sym}/{RESEARCH_TIMEFRAME} research cache -- "
                         f"proceeding with available contiguous history only")
            historical_data_by_symbol[sym] = df

        results = run_research_cycle_walkforward(
            historical_data_by_symbol=historical_data_by_symbol, adaptation_engine=self.adaptation,
            default_params=DEFAULT_PARAMS, fee_rate=FEE_RATE, slippage=SLIPPAGE, now_ts=time.time())

        # persist any promotions into the shared, versioned champion file --
        # the ONLY write path into live-affecting state this process has
        current_records = self.champion_store.ensure_baseline(DEFAULT_PARAMS)
        for strategy, new_params in self.adaptation.champions.items():
            current = current_records.get(strategy)
            if current is None or current.parameters != new_params:
                current_records[strategy] = self.champion_store.promote(strategy, new_params, current)
        self.champion_store.write(current_records)

        self.last_full_ts = time.time()
        n_promoted = sum(1 for r in results if r.get("promoted"))
        self.log(f"full cycle complete: {len(results)} challengers evaluated, {n_promoted} promoted")
        self._write_status({"type": "full", "n_evaluated": len(results), "n_promoted": n_promoted,
                             "results": results})

    def run_once(self):
        now = time.time()
        if self.last_full_ts is None or now - self.last_full_ts >= FULL_CYCLE_SECONDS:
            self.run_full_cycle()
        elif self.last_light_ts is None or now - self.last_light_ts >= LIGHT_CYCLE_SECONDS:
            self.run_light_cycle()

    def run_forever(self):
        self.log(f"── ADAPTIVE RESEARCH SERVICE — light cadence {LIGHT_CYCLE_SECONDS/3600:.0f}h, "
                 f"full cadence {FULL_CYCLE_SECONDS/3600:.0f}h ──")
        while True:
            try:
                self.run_once()
            except Exception as e:
                self.log(f"[research loop] {type(e).__name__}: {str(e)[:200]} -- continuing")
            time.sleep(SCHEDULER_POLL_SECONDS)


def main():
    runtime_dir = os.path.join(ROOT, "adaptive_runtime")
    service = ResearchService(runtime_dir)
    service.bootstrap()
    service.run_forever()


if __name__ == "__main__":
    main()
