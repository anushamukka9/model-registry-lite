"""Tests for the registry API (registration, lifecycle, search, export/import)."""

import pytest

from model_registry_lite.registry import ModelRegistry, RegistryError, parse_metric_filter


@pytest.fixture()
def registry(tmp_path):
    reg = ModelRegistry(tmp_path / "test.db")
    yield reg
    reg.close()


def test_register_and_get_round_trip(registry):
    model = registry.register(
        "churn-model", "1.0.0", framework="sklearn",
        metrics={"f1": 0.91, "auc": 0.96},
        hyperparameters={"max_depth": 8},
        tags=["tabular", "baseline"],
        dataset_hash="sha256:deadbeef",
    )
    fetched = registry.get("churn-model", "1.0.0")
    assert fetched.metrics["f1"] == 0.91
    assert fetched.hyperparameters["max_depth"] == 8
    assert "baseline" in fetched.tags
    assert fetched.dataset_hash == "sha256:deadbeef"


def test_duplicate_registration_is_rejected(registry):
    registry.register("churn-model", "1.0.0")
    with pytest.raises(RegistryError, match="already exists"):
        registry.register("churn-model", "1.0.0")
    # overwrite=True replaces it
    model = registry.register("churn-model", "1.0.0", framework="xgboost", overwrite=True)
    assert registry.get("churn-model", "1.0.0").framework == "xgboost"


def test_empty_name_or_version_rejected(registry):
    with pytest.raises(RegistryError, match="name"):
        registry.register("  ", "1.0.0")
    with pytest.raises(RegistryError, match="version"):
        registry.register("churn-model", "  ")


def test_get_missing_model_raises(registry):
    with pytest.raises(RegistryError, match="not found"):
        registry.get("nope", "0.0.1")


def test_promote_and_archive_flow(registry):
    registry.register("churn-model", "1.0.0")
    model = registry.promote("churn-model", "1.0.0", approved_by="Anusha Mukka", note="QA pass")
    assert model.stage == "production"
    # promoting again (already in production) is illegal
    with pytest.raises(RegistryError, match="Illegal transition"):
        registry.promote("churn-model", "1.0.0", approved_by="Anusha Mukka")
    model = registry.archive("churn-model", "1.0.0", approved_by="Anusha Mukka")
    assert model.stage == "archived"


def test_illegal_transition_is_rejected(registry):
    registry.register("churn-model", "1.0.0")
    registry.archive("churn-model", "1.0.0", approved_by="Anusha Mukka")
    with pytest.raises(RegistryError, match="Illegal transition"):
        registry.transition("churn-model", "1.0.0", "production", approved_by="Anusha Mukka")


def test_transition_history_is_recorded(registry):
    registry.register("churn-model", "1.0.0")
    registry.promote("churn-model", "1.0.0", approved_by="Anusha Mukka", note="ship it")
    entries = registry.store.transition_history("churn-model", "1.0.0")
    assert len(entries) == 1
    assert entries[0].to_stage == "production"
    assert entries[0].approved_by == "Anusha Mukka"


def test_list_models_filters_by_stage(registry):
    registry.register("a", "1.0.0")
    registry.register("b", "1.0.0")
    registry.promote("a", "1.0.0", approved_by="Anusha Mukka")
    assert {m.name for m in registry.list_models(stage="production")} == {"a"}
    assert {m.name for m in registry.list_models(stage="staging")} == {"b"}
    with pytest.raises(RegistryError, match="Unknown stage"):
        registry.list_models(stage="moon")


def test_latest_returns_newest_version(registry):
    registry.register("churn-model", "1.0.0")
    registry.register("churn-model", "2.0.0")
    assert registry.latest("churn-model").version == "2.0.0"
    registry.promote("churn-model", "1.0.0", approved_by="Anusha Mukka")
    assert registry.latest("churn-model", stage="production").version == "1.0.0"


def test_search_by_metric_thresholds(registry):
    registry.register("good", "1.0.0", metrics={"f1": 0.94, "loss": 0.2})
    registry.register("bad", "1.0.0", metrics={"f1": 0.70, "loss": 0.6})
    registry.register("no-metrics", "1.0.0")

    hits = registry.search(metric_filters=["f1>=0.9"])
    assert [m.name for m in hits] == ["good"]

    hits = registry.search(metric_filters=["f1 >= 0.9", "loss < 0.5"])
    assert [m.name for m in hits] == ["good"]

    hits = registry.search(metric_filters=["f1>0.9", "f1<0.95"])
    assert [m.name for m in hits] == ["good"]


def test_search_combines_metadata_filters(registry):
    registry.register("churn-model", "1.0.0", framework="sklearn", tags=["tabular"],
                      metrics={"f1": 0.91})
    registry.register("vision-net", "1.0.0", framework="pytorch", tags=["images"],
                      metrics={"f1": 0.95})
    assert [m.name for m in registry.search(framework="PyTorch")] == ["vision-net"]
    assert [m.name for m in registry.search(tags=["TABULAR"])] == ["churn-model"]
    assert [m.name for m in registry.search(name_contains="churn")] == ["churn-model"]
    assert registry.search(name_contains="churn", metric_filters=["f1>=0.99"]) == []


def test_parse_metric_filter_rejects_bad_input():
    with pytest.raises(RegistryError, match="operator"):
        parse_metric_filter("f1")
    with pytest.raises(RegistryError, match="non-numeric"):
        parse_metric_filter("f1>=high")
    with pytest.raises(RegistryError, match="no metric name"):
        parse_metric_filter(">=0.9")
    assert parse_metric_filter("accuracy>=0.9") == ("accuracy", "gte", 0.9)


def test_update_metadata(registry):
    registry.register("churn-model", "1.0.0", metrics={"f1": 0.80})
    model = registry.update_metadata("churn-model", "1.0.0", metrics={"f1": 0.91},
                                     tags=["tuned"], description="retrained")
    assert model.metrics["f1"] == 0.91
    assert model.tags == ["tuned"]
    with pytest.raises(RegistryError, match="not editable"):
        registry.update_metadata("churn-model", "1.0.0", stage="production")


def test_export_import_round_trip(tmp_path, registry):
    registry.register("churn-model", "1.0.0", metrics={"f1": 0.91}, tags=["tabular"])
    registry.promote("churn-model", "1.0.0", approved_by="Anusha Mukka", note="ok")

    other = ModelRegistry(tmp_path / "other.db")
    try:
        imported, skipped = other.import_json(registry.export_json())
        assert (imported, skipped) == (1, 0)
        fetched = other.get("churn-model", "1.0.0")
        assert fetched.stage == "production"
        assert fetched.metrics["f1"] == 0.91

        imported, skipped = other.import_json(registry.export_json())
        assert (imported, skipped) == (0, 1)
        imported, skipped = other.import_json(registry.export_json(), overwrite=True)
        assert (imported, skipped) == (1, 0)
    finally:
        other.close()


def test_import_rejects_malformed_payload(registry):
    with pytest.raises(RegistryError, match="Invalid JSON"):
        registry.import_json("{not json")
    with pytest.raises(RegistryError, match="'models' list"):
        registry.import_json('{"foo": 1}')


def test_deregister(registry):
    registry.register("churn-model", "1.0.0")
    registry.deregister("churn-model", "1.0.0")
    with pytest.raises(RegistryError, match="not found"):
        registry.get("churn-model", "1.0.0")
    with pytest.raises(RegistryError, match="not found"):
        registry.deregister("churn-model", "1.0.0")
