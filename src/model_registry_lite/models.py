"""Domain objects: stages, model versions, and stage-transition records."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List


class Stage(str, Enum):
    """Lifecycle stages a model version can occupy."""

    STAGING = "staging"
    PRODUCTION = "production"
    ARCHIVED = "archived"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


#: Allowed forward (and archival) transitions in the lifecycle.
ALLOWED_TRANSITIONS: Dict[Stage, List[Stage]] = {
    Stage.STAGING: [Stage.PRODUCTION, Stage.ARCHIVED],
    Stage.PRODUCTION: [Stage.ARCHIVED],
    Stage.ARCHIVED: [],
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass
class StageTransition:
    """A record of one lifecycle move, including who approved it and why."""

    from_stage: str
    to_stage: str
    approved_by: str
    note: str = ""
    at: str = field(default_factory=_utcnow_iso)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StageTransition":
        return cls(
            from_stage=data["from_stage"],
            to_stage=data["to_stage"],
            approved_by=data["approved_by"],
            note=data.get("note", ""),
            at=data.get("at", _utcnow_iso()),
        )


@dataclass
class ModelVersion:
    """One registered version of a model, with full metadata."""

    name: str
    version: str
    framework: str = ""
    stage: str = Stage.STAGING.value
    description: str = ""
    artifact_path: str = ""
    dataset_hash: str = ""
    metrics: Dict[str, float] = field(default_factory=dict)
    hyperparameters: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=_utcnow_iso)
    updated_at: str = field(default_factory=_utcnow_iso)
    history: List[StageTransition] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["history"] = [h.to_dict() for h in self.history]
        return data

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelVersion":
        data = dict(data)
        history = [StageTransition.from_dict(h) for h in data.pop("history", [])]
        return cls(history=history, **data)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "ModelVersion":
        return cls.from_dict(json.loads(payload))

    def touch(self) -> None:
        self.updated_at = _utcnow_iso()

    def can_transition_to(self, target: Stage) -> bool:
        return target in ALLOWED_TRANSITIONS[Stage(self.stage)]

    def record_transition(self, target: Stage, approved_by: str, note: str = "") -> StageTransition:
        if not self.can_transition_to(target):
            raise ValueError(
                f"Illegal transition from {self.stage!r} to {target.value!r}. "
                f"Allowed: {[s.value for s in ALLOWED_TRANSITIONS[Stage(self.stage)]]}"
            )
        if not approved_by.strip():
            raise ValueError("An approver name is required for a stage transition.")
        transition = StageTransition(
            from_stage=self.stage, to_stage=target.value, approved_by=approved_by, note=note
        )
        self.history.append(transition)
        self.stage = target.value
        self.touch()
        return transition
