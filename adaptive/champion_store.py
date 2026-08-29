"""
Shared, versioned, hash-verified champion file -- the ONLY approved
interaction between the research process (adaptive/research_service.py,
the sole writer) and the execution process (adaptive/runner.py, read-only)
per adaptive spec section 15. Atomic writes (tmp + os.replace). The
execution process must detect an invalid/corrupt file and retain its last
successfully loaded copy rather than crash or silently trade on garbage.

One ChampionRecord per adaptive strategy (trend_momentum,
volatility_breakout, mean_reversion). shock_continuation never appears
here -- it is permanently excluded from adaptation (see adaptive/adaptation.py).

version starts at 1 ("v1") for every strategy's initial BASELINE record and
increments by exactly 1 on each successful promotion. Never claims a
BASELINE record is historically profitable -- it is the current documented
default parameter set, nothing more.
"""
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

BASELINE = "BASELINE"
PROMOTED = "PROMOTED"


def _hash_params(strategy: str, version: int, parameters: dict) -> str:
    payload = json.dumps({"strategy": strategy, "version": version, "parameters": parameters},
                          sort_keys=True)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


@dataclass
class ChampionRecord:
    strategy: str
    version: int
    parameters: dict
    created_at_iso: str
    status: str  # BASELINE | PROMOTED
    hash: str = field(default="")

    def __post_init__(self):
        if not self.hash:
            self.hash = _hash_params(self.strategy, self.version, self.parameters)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ChampionRecord":
        return cls(strategy=d["strategy"], version=d["version"], parameters=d["parameters"],
                    created_at_iso=d["created_at_iso"], status=d["status"], hash=d["hash"])

    def verify(self) -> bool:
        return self.hash == _hash_params(self.strategy, self.version, self.parameters)


class ChampionCorruptionError(Exception):
    pass


class ChampionStore:
    def __init__(self, path: str):
        self.path = path

    def exists(self) -> bool:
        return os.path.exists(self.path)

    def read(self) -> Dict[str, ChampionRecord]:
        """Raises ChampionCorruptionError on any structural or hash-integrity
        problem -- callers (the execution process) MUST catch this and fall
        back to the last successfully loaded copy, never crash and never
        trade on an unverified file."""
        return self.read_full()[0]

    def read_history(self) -> Dict[str, List[ChampionRecord]]:
        return self.read_full()[1]

    def read_full(self) -> Tuple[Dict[str, ChampionRecord], Dict[str, List[ChampionRecord]]]:
        if not os.path.exists(self.path):
            raise ChampionCorruptionError(f"{self.path} does not exist")
        try:
            with open(self.path) as f:
                raw = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            raise ChampionCorruptionError(f"failed to read/parse {self.path}: {e}")

        file_hash = raw.get("file_hash")
        records_raw = raw.get("records", {})
        expected_file_hash = hashlib.sha256(
            json.dumps(records_raw, sort_keys=True).encode()).hexdigest()[:16]
        if file_hash != expected_file_hash:
            raise ChampionCorruptionError(
                f"{self.path} file_hash mismatch (stored={file_hash}, expected={expected_file_hash})")

        records = {}
        for strategy, rec_dict in records_raw.items():
            rec = ChampionRecord.from_dict(rec_dict)
            if not rec.verify():
                raise ChampionCorruptionError(f"{self.path}: {strategy} record hash mismatch")
            records[strategy] = rec

        history_raw = raw.get("history", {})
        history = {s: [ChampionRecord.from_dict(d) for d in lst] for s, lst in history_raw.items()}
        return records, history

    def write(self, records: Dict[str, ChampionRecord],
              history: Optional[Dict[str, List[ChampionRecord]]] = None):
        """If `history` is omitted, it is computed automatically: whatever
        history is currently on disk is preserved, and for any strategy
        whose record is CHANGING (different hash), the old on-disk record
        is appended to that strategy's history before being overwritten.
        This means callers never need to manage rollback history
        bookkeeping themselves -- every write() call is automatically
        rollback-safe."""
        if history is None:
            try:
                prev_records, history = self.read_full()
            except ChampionCorruptionError:
                prev_records, history = {}, {}
            history = {s: list(v) for s, v in history.items()}
            for strategy, new_rec in records.items():
                old_rec = prev_records.get(strategy)
                if old_rec is not None and old_rec.hash != new_rec.hash:
                    history.setdefault(strategy, []).append(old_rec)

        records_raw = {s: r.to_dict() for s, r in records.items()}
        history_raw = {s: [r.to_dict() for r in lst] for s, lst in history.items()}
        file_hash = hashlib.sha256(json.dumps(records_raw, sort_keys=True).encode()).hexdigest()[:16]
        payload = {"written_at_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                   "file_hash": file_hash, "records": records_raw, "history": history_raw}
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, self.path)

    def rollback(self, strategy: str) -> Optional[ChampionRecord]:
        """Restores the immediately-previous record for `strategy` from
        persisted history. Returns the restored record, or None if there
        is no history to roll back to (never raises for that case)."""
        try:
            records, history = self.read_full()
        except ChampionCorruptionError:
            return None
        strategy_history = history.get(strategy, [])
        if not strategy_history:
            return None
        restored = strategy_history.pop()
        records[strategy] = restored
        self.write(records, history=history)
        return restored

    def ensure_baseline(self, default_params: Dict[str, dict]) -> Dict[str, ChampionRecord]:
        """Idempotent: if the file already exists (even partially), returns
        its current contents unchanged. Only creates v1 BASELINE records
        when the file is entirely absent -- this lets either the execution
        process or the research service safely call it on first startup
        without a race condition overwriting real promotion history."""
        if self.exists():
            try:
                return self.read()
            except ChampionCorruptionError:
                pass  # fall through and re-seed a fresh baseline below
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        records = {strategy: ChampionRecord(strategy=strategy, version=1, parameters=dict(params),
                                             created_at_iso=now, status=BASELINE)
                   for strategy, params in default_params.items()}
        self.write(records)
        return records

    def promote(self, strategy: str, new_parameters: dict, current: Optional[ChampionRecord]) -> ChampionRecord:
        new_version = (current.version + 1) if current else 1
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        return ChampionRecord(strategy=strategy, version=new_version, parameters=dict(new_parameters),
                               created_at_iso=now, status=PROMOTED)
