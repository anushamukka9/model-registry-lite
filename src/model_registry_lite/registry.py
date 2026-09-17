"""High-level registry API: register, promote, search, export, import."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .models import ModelVersion, Stage
from .store import SqliteStore

PathLike = Union[str, Path]

# (operator suffix, predicate)
_METRIC_OPERATORS = {
    "gte": lambda value, bound: value >= bound,
    "lte": lambda value, bound: value <= bound,
    "gt": lambda value, bound: value > bound,
    "lt": lambda value, bound: value < bound,
}


class RegistryError(Exception):
    """Raised for invalid registry operations (missing models, bad transitions, ...)."""


def parse_metric_filter(expression: str) -> Tuple[str, str, float]:
    """Parse a metric filter like ``accuracy>=0.9`` or ``loss < 0.5``.

    Returns (metric_name, operator, threshold). Raises RegistryError on bad input.
    """
    for op in ("gte", "lte", "gt", "lt", ">=", "<=", ">", "<"):
        if op in expression:
            metric, _, raw = expression.partition(op)
            metric = metric.strip()
            raw = raw.strip()
            if not metric:
                raise RegistryError(f"Metric filter {expression!r} has no metric name.")
            try:
                threshold = float(raw)
            except ValueError:
                raise RegistryError(f"Metric filter {expression!r} has a non-numeric threshold.")
            canonical = {">=": "gte", "<=": "lte", ">": "gt", "<": "lt"}.get(op, op)
            return metric, canonical, threshold
    raise RegistryError(
        f"Metric filter {expression!r} needs an operator: one of >=, <=, >, < (or gte/lte/gt/lt)."
    )


class ModelRegistry:
    """Facade over the SQLite store exposing registry workflows."""

    def __init__(self, db_path: PathLike = "model_registry.db") -> None:
        self.store = SqliteStore(db_path)

    # -- registration ---------------------------------------------------
    def register(
        self,
        name: str,
        version: str,
        framework: str = "",
        description: str = "",
        artifact_path: str = "",
        dataset_hash: str = "",
        metrics: Optional[Dict[str, float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        tags: Optional[List[str]] = None,
        overwrite: bool = False,
    ) -> ModelVersion:
        if not name.strip():
            raise RegistryError("Model name must not be empty.")
        if not version.strip():
            raise RegistryError("Model version must not be empty.")
        existing = self.store.get(name, version)
        if existing is not None and not overwrite:
            raise RegistryError(
                f"Model {name!r} version {version!r} already exists. "
                "Use overwrite=True to re-register."
            )
        model = ModelVersion(
            name=name,
            version=version,
            framework=framework,
            description=description,
            artifact_path=artifact_path,
            dataset_hash=dataset_hash,
            metrics=dict(metrics or {}),
            hyperparameters=dict(hyperparameters or {}),
            tags=list(tags or []),
        )
        self.store.upsert(model)
        return model

    def update_metadata(self, name: str, version: str, **fields: Any) -> ModelVersion:
        """Update mutable metadata fields (metrics, hyperparameters, tags, ...)."""
        model = self.get(name, version)
        editable = {
            "framework",
            "description",
            "artifact_path",
            "dataset_hash",
            "metrics",
            "hyperparameters",
            "tags",
        }
        for key, value in fields.items():
            if key not in editable:
                raise RegistryError(f"Field {key!r} is not editable; editable fields: {sorted(editable)}.")
            setattr(model, key, value)
        model.touch()
        self.store.upsert(model)
        return model

    def deregister(self, name: str, version: str) -> None:
        if not self.store.delete(name, version):
            raise RegistryError(f"Model {name!r} version {version!r} not found.")

    # -- reads ----------------------------------------------------------
    def get(self, name: str, version: str) -> ModelVersion:
        model = self.store.get(name, version)
        if model is None:
            raise RegistryError(f"Model {name!r} version {version!r} not found.")
        return model

    def list_models(self, stage: Optional[str] = None) -> List[ModelVersion]:
        models = self.store.list_all()
        if stage is not None:
            try:
                wanted = Stage(stage)
            except ValueError:
                raise RegistryError(f"Unknown stage {stage!r}; valid: {[s.value for s in Stage]}.")
            models = [m for m in models if m.stage == wanted.value]
        return models

    def list_versions(self, name: str) -> List[ModelVersion]:
        versions = self.store.list_versions(name)
        if not versions:
            raise RegistryError(f"No versions registered for model {name!r}.")
        return versions

    def latest(self, name: str, stage: Optional[str] = None) -> ModelVersion:
        """Newest version of a model (by version string), optionally filtered by stage."""
        versions = self.list_versions(name)
        if stage is not None:
            try:
                wanted = Stage(stage)
            except ValueError:
                raise RegistryError(f"Unknown stage {stage!r}.")
            versions = [v for v in versions if v.stage == wanted.value]
            if not versions:
                raise RegistryError(f"No version of {name!r} is in stage {stage!r}.")
        return sorted(versions, key=lambda v: v.version)[-1]

    # -- lifecycle ------------------------------------------------------
    def transition(
        self, name: str, version: str, to_stage: str, approved_by: str, note: str = ""
    ) -> ModelVersion:
        try:
            target = Stage(to_stage)
        except ValueError:
            raise RegistryError(f"Unknown stage {to_stage!r}; valid: {[s.value for s in Stage]}.")
        model = self.get(name, version)
        try:
            transition = model.record_transition(target, approved_by=approved_by, note=note)
        except ValueError as exc:
            raise RegistryError(str(exc)) from exc
        self.store.upsert(model)
        self.store.record_transition(name, version, transition)
        return model

    def promote(self, name: str, version: str, approved_by: str, note: str = "") -> ModelVersion:
        """Move a version from staging to production."""
        return self.transition(name, version, Stage.PRODUCTION.value, approved_by, note)

    def archive(self, name: str, version: str, approved_by: str, note: str = "") -> ModelVersion:
        """Move a version (staging or production) to archived."""
        return self.transition(name, version, Stage.ARCHIVED.value, approved_by, note)

    # -- search ---------------------------------------------------------
    def search(
        self,
        name_contains: Optional[str] = None,
        framework: Optional[str] = None,
        stage: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metric_filters: Optional[List[str]] = None,
    ) -> List[ModelVersion]:
        """Filter registered versions by metadata and metric thresholds.

        ``metric_filters`` is a list of expressions like ``["accuracy>=0.9", "loss<0.5"]``;
        all filters must match.
        """
        models = self.store.list_all()

        if name_contains:
            needle = name_contains.lower()
            models = [m for m in models if needle in m.name.lower()]
        if framework:
            models = [m for m in models if m.framework.lower() == framework.lower()]
        if stage:
            try:
                wanted = Stage(stage)
            except ValueError:
                raise RegistryError(f"Unknown stage {stage!r}.")
            models = [m for m in models if m.stage == wanted.value]
        if tags:
            wanted_tags = {t.lower() for t in tags}
            models = [
                m
                for m in models
                if wanted_tags.issubset({t.lower() for t in m.tags})
            ]

        parsed: List[Tuple[str, str, float]] = []
        for expression in metric_filters or []:
            parsed.append(parse_metric_filter(expression))
        for metric, operator, threshold in parsed:
            predicate = _METRIC_OPERATORS[operator]
            kept = []
            for model in models:
                value = model.metrics.get(metric)
                if isinstance(value, (int, float)) and predicate(float(value), threshold):
                    kept.append(model)
            models = kept

        return models

    # -- export / import ------------------------------------------------
    def export_json(self) -> str:
        payload = {
            "format": "model-registry-lite/v1",
            "models": [m.to_dict() for m in self.store.list_all()],
        }
        return json.dumps(payload, indent=2, sort_keys=True)

    def export_to_file(self, path: PathLike) -> Path:
        target = Path(path)
        target.write_text(self.export_json(), encoding="utf-8")
        return target

    def import_json(self, payload: str, overwrite: bool = False) -> Tuple[int, int]:
        """Import a previously exported registry document.

        Returns (imported, skipped). Raises RegistryError on a malformed payload.
        """
        try:
            data = json.loads(payload)
        except json.JSONDecodeError as exc:
            raise RegistryError(f"Invalid JSON payload: {exc}") from exc
        entries = data.get("models") if isinstance(data, dict) else None
        if not isinstance(entries, list):
            raise RegistryError("Import payload must be an object with a 'models' list.")
        imported = skipped = 0
        for entry in entries:
            model = ModelVersion.from_dict(entry)
            if self.store.get(model.name, model.version) is not None and not overwrite:
                skipped += 1
                continue
            self.store.upsert(model)
            imported += 1
        return imported, skipped

    def import_from_file(self, path: PathLike, overwrite: bool = False) -> Tuple[int, int]:
        return self.import_json(Path(path).read_text(encoding="utf-8"), overwrite=overwrite)

    def close(self) -> None:
        self.store.close()

    def __enter__(self) -> "ModelRegistry":  # pragma: no cover - convenience
        return self

    def __exit__(self, *exc: object) -> None:  # pragma: no cover - convenience
        self.close()
