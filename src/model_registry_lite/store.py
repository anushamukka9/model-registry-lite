"""SQLite-backed persistence for the model registry.

One table stores whole model-version documents as JSON so the schema never
needs migrations when metadata fields evolve; a second table records the
audit trail of stage transitions.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import List, Optional, Union

from .models import ModelVersion, Stage, StageTransition

PathLike = Union[str, Path]

_SCHEMA = """
CREATE TABLE IF NOT EXISTS model_versions (
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    document TEXT NOT NULL,
    PRIMARY KEY (name, version)
);
CREATE TABLE IF NOT EXISTS transitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    version TEXT NOT NULL,
    from_stage TEXT NOT NULL,
    to_stage TEXT NOT NULL,
    approved_by TEXT NOT NULL,
    note TEXT NOT NULL,
    at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_transitions_model ON transitions (name, version);
"""


class SqliteStore:
    """Low-level SQLite persistence for model-version documents."""

    def __init__(self, path: PathLike = "model_registry.db") -> None:
        self.path = Path(path)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.row_factory = sqlite3.Row
        with self._conn:
            self._conn.executescript(_SCHEMA)

    # -- writes ---------------------------------------------------------
    def upsert(self, model: ModelVersion) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO model_versions (name, version, document) VALUES (?, ?, ?)"
                " ON CONFLICT(name, version) DO UPDATE SET document = excluded.document",
                (model.name, model.version, model.to_json()),
            )

    def record_transition(self, name: str, version: str, transition: StageTransition) -> None:
        with self._conn:
            self._conn.execute(
                "INSERT INTO transitions (name, version, from_stage, to_stage, approved_by, note, at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    name,
                    version,
                    transition.from_stage,
                    transition.to_stage,
                    transition.approved_by,
                    transition.note,
                    transition.at,
                ),
            )

    def delete(self, name: str, version: str) -> bool:
        with self._conn:
            cur = self._conn.execute(
                "DELETE FROM model_versions WHERE name = ? AND version = ?", (name, version)
            )
            self._conn.execute(
                "DELETE FROM transitions WHERE name = ? AND version = ?", (name, version)
            )
        return cur.rowcount > 0

    # -- reads ----------------------------------------------------------
    def get(self, name: str, version: str) -> Optional[ModelVersion]:
        row = self._conn.execute(
            "SELECT document FROM model_versions WHERE name = ? AND version = ?",
            (name, version),
        ).fetchone()
        if row is None:
            return None
        return ModelVersion.from_json(row["document"])

    def list_all(self) -> List[ModelVersion]:
        rows = self._conn.execute(
            "SELECT document FROM model_versions ORDER BY name, version"
        ).fetchall()
        return [ModelVersion.from_json(r["document"]) for r in rows]

    def list_versions(self, name: str) -> List[ModelVersion]:
        rows = self._conn.execute(
            "SELECT document FROM model_versions WHERE name = ? ORDER BY version",
            (name,),
        ).fetchall()
        return [ModelVersion.from_json(r["document"]) for r in rows]

    def list_names(self) -> List[str]:
        rows = self._conn.execute("SELECT DISTINCT name FROM model_versions ORDER BY name").fetchall()
        return [r["name"] for r in rows]

    def transition_history(self, name: str, version: str) -> List[StageTransition]:
        rows = self._conn.execute(
            "SELECT from_stage, to_stage, approved_by, note, at FROM transitions"
            " WHERE name = ? AND version = ? ORDER BY id",
            (name, version),
        ).fetchall()
        return [StageTransition.from_dict(dict(r)) for r in rows]

    def metric_values(self, metric: str) -> List[float]:
        """All recorded values of one metric across every registered version."""
        values: List[float] = []
        for model in self.list_all():
            value = model.metrics.get(metric)
            if isinstance(value, (int, float)):
                values.append(float(value))
        return values

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SqliteStore":  # pragma: no cover - convenience
        return self

    def __exit__(self, *exc: object) -> None:  # pragma: no cover - convenience
        self.close()


def _validate_store_roundtrip(path: PathLike) -> None:
    """Sanity check that a store file round-trips through JSON export."""
    import json as _json

    store = SqliteStore(path)
    try:
        docs = [json.loads(m.to_json()) for m in store.list_all()]
        _json.dumps({"model_versions": docs, "exported": True})
    finally:
        store.close()
