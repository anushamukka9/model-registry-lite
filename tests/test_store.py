"""Tests for the SQLite store (persistence across process lifetimes)."""

import pytest

from model_registry_lite.models import ModelVersion
from model_registry_lite.store import SqliteStore


@pytest.fixture()
def store(tmp_path):
    st = SqliteStore(tmp_path / "store.db")
    yield st
    st.close()


def test_upsert_and_get(store):
    model = ModelVersion(name="a", version="1.0.0", framework="sklearn",
                         metrics={"f1": 0.9})
    store.upsert(model)
    assert store.get("a", "1.0.0") == model
    assert store.get("a", "9.9.9") is None


def test_upsert_replaces_existing(store):
    store.upsert(ModelVersion(name="a", version="1.0.0", framework="sklearn"))
    store.upsert(ModelVersion(name="a", version="1.0.0", framework="pytorch"))
    assert store.get("a", "1.0.0").framework == "pytorch"
    assert len(store.list_all()) == 1


def test_data_survives_reopen(tmp_path):
    path = tmp_path / "persist.db"
    first = SqliteStore(path)
    first.upsert(ModelVersion(name="a", version="1.0.0", metrics={"f1": 0.9}))
    first.close()

    second = SqliteStore(path)
    try:
        assert second.get("a", "1.0.0").metrics["f1"] == 0.9
    finally:
        second.close()


def test_delete_removes_model_and_history(store):
    from model_registry_lite.models import StageTransition

    store.upsert(ModelVersion(name="a", version="1.0.0"))
    store.record_transition("a", "1.0.0",
                            StageTransition("staging", "production", "Anusha Mukka"))
    assert store.delete("a", "1.0.0") is True
    assert store.delete("a", "1.0.0") is False
    assert store.get("a", "1.0.0") is None
    assert store.transition_history("a", "1.0.0") == []


def test_list_versions_and_names(store):
    store.upsert(ModelVersion(name="b", version="2.0.0"))
    store.upsert(ModelVersion(name="a", version="1.0.0"))
    store.upsert(ModelVersion(name="a", version="1.1.0"))
    assert [m.version for m in store.list_versions("a")] == ["1.0.0", "1.1.0"]
    assert store.list_versions("missing") == []
    assert store.list_names() == ["a", "b"]


def test_metric_values_aggregates(store):
    store.upsert(ModelVersion(name="a", version="1.0.0", metrics={"f1": 0.9}))
    store.upsert(ModelVersion(name="b", version="1.0.0", metrics={"f1": 0.7}))
    store.upsert(ModelVersion(name="c", version="1.0.0"))  # no f1 recorded
    assert sorted(store.metric_values("f1")) == [0.7, 0.9]
    assert store.metric_values("never-recorded") == []
