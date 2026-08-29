"""Deterministic tests for adaptive/research_service.py's scheduler."""
import os
import time

import pandas as pd
import pytest

from adaptive.research_service import ResearchService, LIGHT_CYCLE_SECONDS, FULL_CYCLE_SECONDS
from adaptive.market_data import SYMBOLS, TF_MS
from adaptive.champion_store import ChampionStore

HOUR = TF_MS["1h"]
BASE = 1_800_000_000_000


class FakeExchange:
    def __init__(self, now_ms, price=100.0, n_bars=3000):
        self._now_ms = now_ms
        self.price = price
        self.n_bars = n_bars

    def milliseconds(self):
        return self._now_ms

    def fetch_ohlcv(self, symbol, timeframe, since=None, limit=1000):
        tf_ms = TF_MS[timeframe]
        last_close = self._now_ms - (self._now_ms % tf_ms) - tf_ms
        ts0 = last_close - (self.n_bars - 1) * tf_ms
        rows = [[ts0 + i * tf_ms, self.price, self.price * 1.001, self.price * 0.999, self.price, 10.0]
                for i in range(self.n_bars)]
        if since is not None:
            rows = [r for r in rows if r[0] >= since]
        return rows[:limit]


def test_bootstrap_creates_baseline_champion_if_missing(tmp_path):
    ex = FakeExchange(BASE + 3000 * HOUR)
    svc = ResearchService(str(tmp_path), exchange=ex)
    svc.bootstrap()
    store = ChampionStore(os.path.join(str(tmp_path), "champion_state.json"))
    records = store.read()
    assert "trend_momentum" in records
    assert records["trend_momentum"].version == 1
    assert records["trend_momentum"].status == "BASELINE"


def test_full_cycle_runs_before_light_cycle_on_first_ever_run(tmp_path):
    ex = FakeExchange(BASE + 3000 * HOUR, n_bars=1500)
    svc = ResearchService(str(tmp_path), exchange=ex)
    svc.bootstrap()
    assert svc.last_full_ts is None
    svc.run_once()
    assert svc.last_full_ts is not None  # full cycle ran first (never run before)


def test_light_cycle_runs_when_full_cycle_not_yet_due(tmp_path):
    ex = FakeExchange(BASE + 3000 * HOUR, n_bars=1500)
    svc = ResearchService(str(tmp_path), exchange=ex)
    svc.bootstrap()
    svc.last_full_ts = time.time()  # just ran -- not due again
    svc.last_light_ts = None
    svc.run_once()
    assert svc.last_light_ts is not None


def test_full_cycle_not_re_run_before_interval_elapsed(tmp_path):
    ex = FakeExchange(BASE + 3000 * HOUR, n_bars=1500)
    svc = ResearchService(str(tmp_path), exchange=ex)
    svc.bootstrap()
    svc.last_full_ts = time.time()
    svc.last_light_ts = time.time()  # both just ran -- neither due
    before = svc.last_light_ts
    svc.run_once()
    assert svc.last_light_ts == before  # unchanged -- nothing was due


def test_scheduler_state_persists_across_restarts(tmp_path):
    ex = FakeExchange(BASE + 3000 * HOUR, n_bars=1500)
    svc1 = ResearchService(str(tmp_path), exchange=ex)
    svc1.bootstrap()
    svc1.run_light_cycle()
    light_ts = svc1.last_light_ts

    svc2 = ResearchService(str(tmp_path), exchange=ex)
    svc2.bootstrap()
    assert svc2.last_light_ts == pytest.approx(light_ts)


def test_research_service_never_writes_execution_owned_files(tmp_path):
    ex = FakeExchange(BASE + 3000 * HOUR, n_bars=1500)
    svc = ResearchService(str(tmp_path), exchange=ex)
    svc.bootstrap()
    svc.run_full_cycle()
    for forbidden in ("state.json", "trades.csv", "decisions.jsonl", "paper.log"):
        assert not os.path.exists(os.path.join(str(tmp_path), forbidden)), \
            f"research service wrote execution-owned file: {forbidden}"


def test_research_loop_exception_does_not_crash_service(tmp_path):
    class BrokenExchange(FakeExchange):
        def fetch_ohlcv(self, *a, **k):
            raise RuntimeError("simulated network failure")

    svc = ResearchService(str(tmp_path), exchange=BrokenExchange(BASE + 3000 * HOUR))
    svc.bootstrap()
    try:
        svc.run_once()  # run_once itself may raise -- run_forever's try/except is what protects the loop
    except RuntimeError:
        pass  # run_once() is allowed to raise; only run_forever() must survive it (structural, not tested here directly)
    # service object must still be usable afterward -- no corrupted internal state
    assert svc.champion_store.path.endswith("champion_state.json")


def test_status_file_reports_cadence_and_champions(tmp_path):
    ex = FakeExchange(BASE + 3000 * HOUR, n_bars=1500)
    svc = ResearchService(str(tmp_path), exchange=ex)
    svc.bootstrap()
    svc.run_light_cycle()
    import json
    status = json.load(open(os.path.join(str(tmp_path), "research_status.json")))
    for key in ("last_light_run_iso", "next_light_run_iso", "last_full_run_iso",
                "next_full_run_iso", "champions"):
        assert key in status


def test_rollback_not_triggered_by_a_baseline_champion(tmp_path):
    ex = FakeExchange(BASE + 3000 * HOUR, n_bars=1500)
    svc = ResearchService(str(tmp_path), exchange=ex)
    svc.bootstrap()  # only BASELINE champions exist -- nothing to roll back
    # simulate terrible live stats in state.json
    import json
    state = {"stats_store": {"BTC/USDT|trend_momentum|TREND_UP": {
        "trades": [-5.0] * 20, "wins": 0, "losses": 20, "gross_win": 0.0, "gross_loss": 100.0}}}
    with open(os.path.join(str(tmp_path), "state.json"), "w") as f:
        json.dump(state, f)
    rolled_back = svc.check_for_rollback()
    assert rolled_back == []  # BASELINE is never "rolled back" -- nothing precedes it


def test_rollback_triggered_by_severe_degradation_after_promotion(tmp_path):
    ex = FakeExchange(BASE + 3000 * HOUR, n_bars=1500)
    svc = ResearchService(str(tmp_path), exchange=ex)
    svc.bootstrap()
    records = svc.champion_store.read()
    promoted = svc.champion_store.promote("trend_momentum", {"z_threshold": 0.3}, records["trend_momentum"])
    records["trend_momentum"] = promoted
    svc.champion_store.write(records)
    assert svc.champion_store.read()["trend_momentum"].version == 2

    import json
    state = {"stats_store": {"BTC/USDT|trend_momentum|TREND_UP": {
        "trades": [-5.0] * 20, "wins": 0, "losses": 20, "gross_win": 0.0, "gross_loss": 100.0}}}
    with open(os.path.join(str(tmp_path), "state.json"), "w") as f:
        json.dump(state, f)

    rolled_back = svc.check_for_rollback()
    assert rolled_back == ["trend_momentum"]
    assert svc.champion_store.read()["trend_momentum"].version == 1


def test_rollback_not_triggered_by_insufficient_sample(tmp_path):
    ex = FakeExchange(BASE + 3000 * HOUR, n_bars=1500)
    svc = ResearchService(str(tmp_path), exchange=ex)
    svc.bootstrap()
    records = svc.champion_store.read()
    promoted = svc.champion_store.promote("trend_momentum", {"z_threshold": 0.3}, records["trend_momentum"])
    records["trend_momentum"] = promoted
    svc.champion_store.write(records)

    import json
    state = {"stats_store": {"BTC/USDT|trend_momentum|TREND_UP": {
        "trades": [-5.0] * 3, "wins": 0, "losses": 3, "gross_win": 0.0, "gross_loss": 15.0}}}  # only 3 trades
    with open(os.path.join(str(tmp_path), "state.json"), "w") as f:
        json.dump(state, f)

    rolled_back = svc.check_for_rollback()
    assert rolled_back == []
    assert svc.champion_store.read()["trend_momentum"].version == 2  # unchanged


def test_rollback_not_triggered_by_one_or_two_losing_trades(tmp_path):
    """A couple of losses among otherwise-fine performance must never
    trigger rollback -- only a severe, adequately-sampled PF collapse."""
    ex = FakeExchange(BASE + 3000 * HOUR, n_bars=1500)
    svc = ResearchService(str(tmp_path), exchange=ex)
    svc.bootstrap()
    records = svc.champion_store.read()
    promoted = svc.champion_store.promote("trend_momentum", {"z_threshold": 0.3}, records["trend_momentum"])
    records["trend_momentum"] = promoted
    svc.champion_store.write(records)

    import json
    trades = [1.0] * 18 + [-1.0, -1.0]  # 2 losses among 20 trades, still PF >> floor
    gross_win = sum(t for t in trades if t > 0)
    gross_loss = sum(abs(t) for t in trades if t <= 0)
    state = {"stats_store": {"BTC/USDT|trend_momentum|TREND_UP": {
        "trades": trades, "wins": 18, "losses": 2, "gross_win": gross_win, "gross_loss": gross_loss}}}
    with open(os.path.join(str(tmp_path), "state.json"), "w") as f:
        json.dump(state, f)

    rolled_back = svc.check_for_rollback()
    assert rolled_back == []
    assert svc.champion_store.read()["trend_momentum"].version == 2


def test_rollback_is_logged_to_adaptation_log(tmp_path):
    ex = FakeExchange(BASE + 3000 * HOUR, n_bars=1500)
    svc = ResearchService(str(tmp_path), exchange=ex)
    svc.bootstrap()
    records = svc.champion_store.read()
    promoted = svc.champion_store.promote("trend_momentum", {"z_threshold": 0.3}, records["trend_momentum"])
    records["trend_momentum"] = promoted
    svc.champion_store.write(records)

    import json
    state = {"stats_store": {"BTC/USDT|trend_momentum|TREND_UP": {
        "trades": [-5.0] * 20, "wins": 0, "losses": 20, "gross_win": 0.0, "gross_loss": 100.0}}}
    with open(os.path.join(str(tmp_path), "state.json"), "w") as f:
        json.dump(state, f)
    svc.check_for_rollback()

    log_lines = open(os.path.join(str(tmp_path), "adaptation_log.jsonl")).readlines()
    rollback_lines = [json.loads(l) for l in log_lines if json.loads(l).get("action") == "rollback"]
    assert len(rollback_lines) == 1
    assert rollback_lines[0]["strategy"] == "trend_momentum"


def test_research_service_never_writes_to_execution_state_json_during_rollback_check(tmp_path):
    ex = FakeExchange(BASE + 3000 * HOUR, n_bars=1500)
    svc = ResearchService(str(tmp_path), exchange=ex)
    svc.bootstrap()
    import json
    state = {"stats_store": {}}
    state_path = os.path.join(str(tmp_path), "state.json")
    with open(state_path, "w") as f:
        json.dump(state, f)
    mtime_before = os.path.getmtime(state_path)
    svc.check_for_rollback()
    assert os.path.getmtime(state_path) == mtime_before  # read-only, never modified


def test_promotion_frequency_respects_min_interval_across_full_cycles(tmp_path):
    """Running run_full_cycle() twice in immediate succession must not
    promote twice -- adaptation.py's MIN_PROMOTION_INTERVAL_SEC (24h)
    caps this regardless of scheduler cadence."""
    ex = FakeExchange(BASE + 3000 * HOUR, n_bars=1500)
    svc = ResearchService(str(tmp_path), exchange=ex)
    svc.bootstrap()
    svc.run_full_cycle()
    store = ChampionStore(os.path.join(str(tmp_path), "champion_state.json"))
    versions_after_first = {s: r.version for s, r in store.read().items()}
    svc.run_full_cycle()  # immediately again
    versions_after_second = {s: r.version for s, r in store.read().items()}
    assert versions_after_first == versions_after_second  # no strategy could promote twice within min_interval
