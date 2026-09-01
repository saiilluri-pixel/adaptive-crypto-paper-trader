"""Deterministic tests for adaptive/champion_store.py."""
import json
import os

import pytest

from adaptive.champion_store import ChampionStore, ChampionRecord, ChampionCorruptionError, BASELINE, PROMOTED

DEFAULTS = {"trend_momentum": {"z_threshold": 1.0}, "volatility_breakout": {"n": 20, "volume_percentile": 0.7}}


def test_initial_champion_exists_after_ensure_baseline(tmp_path):
    store = ChampionStore(str(tmp_path / "champion_state.json"))
    records = store.ensure_baseline(DEFAULTS)
    assert "trend_momentum" in records
    assert records["trend_momentum"].version == 1
    assert records["trend_momentum"].status == BASELINE


def test_initial_champion_version_persisted(tmp_path):
    path = str(tmp_path / "champion_state.json")
    store = ChampionStore(path)
    store.ensure_baseline(DEFAULTS)
    store2 = ChampionStore(path)
    records = store2.read()
    assert records["trend_momentum"].version == 1
    assert records["trend_momentum"].parameters == {"z_threshold": 1.0}


def test_ensure_baseline_is_idempotent_does_not_overwrite_promotion(tmp_path):
    store = ChampionStore(str(tmp_path / "champion_state.json"))
    records = store.ensure_baseline(DEFAULTS)
    promoted = store.promote("trend_momentum", {"z_threshold": 0.8}, records["trend_momentum"])
    records["trend_momentum"] = promoted
    store.write(records)

    reloaded = store.ensure_baseline(DEFAULTS)  # must NOT reset back to v1
    assert reloaded["trend_momentum"].version == 2
    assert reloaded["trend_momentum"].status == PROMOTED


def test_version_increments_correctly_across_multiple_promotions(tmp_path):
    store = ChampionStore(str(tmp_path / "champion_state.json"))
    records = store.ensure_baseline(DEFAULTS)
    for i in range(3):
        new_params = {"z_threshold": 1.0 - 0.1 * (i + 1)}
        promoted = store.promote("trend_momentum", new_params, records["trend_momentum"])
        records["trend_momentum"] = promoted
        store.write(records)
    final = store.read()
    assert final["trend_momentum"].version == 4  # 1 baseline + 3 promotions


def test_hash_changes_when_parameters_change(tmp_path):
    store = ChampionStore(str(tmp_path / "x.json"))
    records = store.ensure_baseline(DEFAULTS)
    promoted = store.promote("trend_momentum", {"z_threshold": 0.5}, records["trend_momentum"])
    assert promoted.hash != records["trend_momentum"].hash


def test_record_verify_detects_tampered_parameters():
    rec = ChampionRecord(strategy="trend_momentum", version=1, parameters={"z_threshold": 1.0},
                          created_at_iso="2026-01-01T00:00:00Z", status=BASELINE)
    assert rec.verify() is True
    rec.parameters["z_threshold"] = 999.0  # tamper after hash was computed
    assert rec.verify() is False


def test_corrupted_champion_file_raises_and_is_detectable(tmp_path):
    path = tmp_path / "champion_state.json"
    path.write_text("{ this is not valid json")
    store = ChampionStore(str(path))
    with pytest.raises(ChampionCorruptionError):
        store.read()


def test_corrupted_file_hash_mismatch_detected(tmp_path):
    path = tmp_path / "champion_state.json"
    store = ChampionStore(str(path))
    records = store.ensure_baseline(DEFAULTS)
    # tamper the on-disk file directly, bypassing the store's own write()
    raw = json.loads(path.read_text())
    raw["records"]["trend_momentum"]["parameters"]["z_threshold"] = 999.0
    path.write_text(json.dumps(raw))
    with pytest.raises(ChampionCorruptionError):
        store.read()


def test_missing_file_raises_corruption_error(tmp_path):
    store = ChampionStore(str(tmp_path / "does_not_exist.json"))
    with pytest.raises(ChampionCorruptionError):
        store.read()


def test_write_auto_tracks_history_on_change(tmp_path):
    store = ChampionStore(str(tmp_path / "champion_state.json"))
    records = store.ensure_baseline(DEFAULTS)
    promoted = store.promote("trend_momentum", {"z_threshold": 0.8}, records["trend_momentum"])
    records["trend_momentum"] = promoted
    store.write(records)
    history = store.read_history()
    assert len(history["trend_momentum"]) == 1
    assert history["trend_momentum"][0].version == 1
    assert history["trend_momentum"][0].parameters == {"z_threshold": 1.0}


def test_rollback_restores_previous_champion(tmp_path):
    store = ChampionStore(str(tmp_path / "champion_state.json"))
    records = store.ensure_baseline(DEFAULTS)
    promoted = store.promote("trend_momentum", {"z_threshold": 0.8}, records["trend_momentum"])
    records["trend_momentum"] = promoted
    store.write(records)
    assert store.read()["trend_momentum"].version == 2

    restored = store.rollback("trend_momentum")
    assert restored is not None
    assert restored.version == 1
    assert store.read()["trend_momentum"].version == 1
    assert store.read()["trend_momentum"].parameters == {"z_threshold": 1.0}


def test_rollback_with_no_history_returns_none(tmp_path):
    store = ChampionStore(str(tmp_path / "champion_state.json"))
    store.ensure_baseline(DEFAULTS)
    assert store.rollback("trend_momentum") is None


def test_rollback_persists_across_store_instances(tmp_path):
    path = str(tmp_path / "champion_state.json")
    store1 = ChampionStore(path)
    records = store1.ensure_baseline(DEFAULTS)
    promoted = store1.promote("trend_momentum", {"z_threshold": 0.8}, records["trend_momentum"])
    records["trend_momentum"] = promoted
    store1.write(records)
    store1.rollback("trend_momentum")

    store2 = ChampionStore(path)
    assert store2.read()["trend_momentum"].version == 1


def test_rollback_is_logged_via_history_shrinking(tmp_path):
    """No dedicated rollback log in champion_store.py itself (the research
    service logs rollbacks to adaptation_log.jsonl) -- but the persisted
    history list must shrink by exactly one entry per rollback, proving
    each rollback consumes exactly one promotion step, not more."""
    store = ChampionStore(str(tmp_path / "champion_state.json"))
    records = store.ensure_baseline(DEFAULTS)
    for i in range(3):
        promoted = store.promote("trend_momentum", {"z_threshold": 1.0 - 0.1 * (i + 1)}, records["trend_momentum"])
        records["trend_momentum"] = promoted
        store.write(records)
    assert len(store.read_history()["trend_momentum"]) == 3
    store.rollback("trend_momentum")
    assert len(store.read_history()["trend_momentum"]) == 2
    assert store.read()["trend_momentum"].version == 3  # one step back from v4


def test_ensure_baseline_adds_new_strategy_to_existing_file(tmp_path):
    """A new strategy added to the system after the champion file already
    exists on a running deployment must get seeded in additively -- never
    by resetting or dropping the strategies already there."""
    store = ChampionStore(str(tmp_path / "champion_state.json"))
    original = store.ensure_baseline(DEFAULTS)
    promoted = store.promote("trend_momentum", {"z_threshold": 0.7}, original["trend_momentum"])
    records = dict(original)
    records["trend_momentum"] = promoted
    store.write(records)

    expanded_defaults = dict(DEFAULTS)
    expanded_defaults["atr_trailing_stop"] = {"atr_period": 5, "hhv_period": 10, "mult": 2.5}
    merged = store.ensure_baseline(expanded_defaults)

    assert "atr_trailing_stop" in merged
    assert merged["atr_trailing_stop"].version == 1
    assert merged["atr_trailing_stop"].status == BASELINE
    assert merged["trend_momentum"].version == 2  # untouched -- real promotion preserved
    assert merged["volatility_breakout"].version == 1  # untouched


def test_ensure_baseline_no_write_when_nothing_missing(tmp_path):
    path = tmp_path / "champion_state.json"
    store = ChampionStore(str(path))
    store.ensure_baseline(DEFAULTS)
    mtime_before = path.stat().st_mtime_ns
    store.ensure_baseline(DEFAULTS)  # nothing new -- must not rewrite
    assert path.stat().st_mtime_ns == mtime_before


def test_write_is_atomic_no_partial_file_visible(tmp_path):
    path = tmp_path / "champion_state.json"
    store = ChampionStore(str(path))
    records = store.ensure_baseline(DEFAULTS)
    store.write(records)
    assert not os.path.exists(str(path) + ".tmp")
    assert path.exists()
