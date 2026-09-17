"""Tests for the domain objects (stages, transitions, serialization)."""

import pytest

from model_registry_lite.models import ModelVersion, Stage, StageTransition


def test_model_version_defaults_to_staging():
    model = ModelVersion(name="churn", version="1.0.0")
    assert model.stage == Stage.STAGING.value
    assert model.metrics == {}
    assert model.history == []


def test_valid_transition_records_audit_entry():
    model = ModelVersion(name="churn", version="1.0.0")
    transition = model.record_transition(Stage.PRODUCTION, approved_by="Anusha Mukka", note="beat baseline")
    assert model.stage == Stage.PRODUCTION.value
    assert transition.from_stage == "staging"
    assert transition.to_stage == "production"
    assert transition.approved_by == "Anusha Mukka"
    assert len(model.history) == 1


def test_archived_is_terminal():
    model = ModelVersion(name="churn", version="1.0.0")
    model.record_transition(Stage.ARCHIVED, approved_by="Anusha Mukka")
    with pytest.raises(ValueError, match="Illegal transition"):
        model.record_transition(Stage.PRODUCTION, approved_by="Anusha Mukka")


def test_transition_requires_approver():
    model = ModelVersion(name="churn", version="1.0.0")
    with pytest.raises(ValueError, match="approver"):
        model.record_transition(Stage.PRODUCTION, approved_by="   ")


def test_json_round_trip_preserves_everything():
    model = ModelVersion(
        name="churn",
        version="1.0.0",
        framework="sklearn",
        metrics={"f1": 0.91},
        hyperparameters={"max_depth": 8},
        tags=["tabular"],
        dataset_hash="sha256:abc123",
    )
    model.record_transition(Stage.PRODUCTION, approved_by="Anusha Mukka", note="ok")
    restored = ModelVersion.from_json(model.to_json())
    assert restored == model
    assert restored.history[0].approved_by == "Anusha Mukka"
