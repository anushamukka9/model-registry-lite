# model-registry-lite

A minimal, dependency-free **model registry** for ML teams that don't need a
full MLOps platform. Register versioned model artifacts with rich metadata,
move them through lifecycle stages with approval records, search by metric
thresholds, and back everything up as JSON. The whole registry lives in a
single SQLite file you can check into version control.

Built by [Anusha Mukka](https://anushamukka.com).

## Features

- **Versioned registration** — name, version, framework, description, artifact
  path, training-dataset hash, metrics, hyperparameters, tags.
- **Lifecycle stages** — `staging` → `production` → `archived`, with an
  approver name + note recorded on every transition and a full audit trail.
- **Metric search** — filter by thresholds like `f1>=0.9`, combined with
  metadata filters (framework, stage, tags, name substring).
- **SQLite store** — zero dependencies beyond the standard library; documents
  stored as JSON so metadata evolves without migrations.
- **Export / import** — the whole registry as one portable JSON document.
- **CLI + Python API** — script it or click it.

## Install

```bash
pip install model-registry-lite
```

Requires Python 3.9+. No runtime dependencies.

## Quickstart

```bash
# Register two versions
model-registry register --name churn-predictor --version 1.1.0 \
    --framework scikit-learn \
    --metrics '{"f1": 0.92, "auc": 0.96}' \
    --hyperparameters '{"n_estimators": 500, "max_depth": 8}' \
    --dataset-hash sha256:d71be804 \
    --tags tabular,candidate

# Promote with an approval record
model-registry promote --name churn-predictor --version 1.1.0 \
    --by "Anusha Mukka" --note "F1 0.92 on holdout; beats baseline"

# Find production models with f1 >= 0.9
model-registry search --stage production --metric "f1>=0.9"
```

Python API:

```python
from model_registry_lite import ModelRegistry

with ModelRegistry("models.db") as registry:
    registry.register("churn-predictor", "1.1.0",
                      metrics={"f1": 0.92}, tags=["tabular"])
    registry.promote("churn-predictor", "1.1.0",
                     approved_by="Anusha Mukka", note="QA pass")
    for m in registry.search(metric_filters=["f1>=0.9"]):
        print(m.name, m.version, m.stage)
```

See [docs/usage.md](docs/usage.md) for the full guide and
[examples/quickstart.py](examples/quickstart.py) for a runnable walkthrough.

## API reference

- `ModelRegistry(db_path)` — main entry point; also a context manager.
  - `register(name, version, framework=..., description=..., artifact_path=...,
    dataset_hash=..., metrics=..., hyperparameters=..., tags=..., overwrite=False)`
  - `get(name, version)` / `list_models(stage=None)` / `list_versions(name)` /
    `latest(name, stage=None)`
  - `promote(name, version, approved_by, note="")` /
    `archive(name, version, approved_by, note="")` /
    `transition(name, version, to_stage, approved_by, note="")`
  - `search(name_contains=None, framework=None, stage=None, tags=None,
    metric_filters=None)` — metric filters like `"f1>=0.9"`, `"loss<0.5"`
  - `update_metadata(name, version, **fields)` / `deregister(name, version)`
  - `export_json()` / `export_to_file(path)` / `import_json(payload, overwrite=False)` /
    `import_from_file(path, overwrite=False)` → `(imported, skipped)`
- `ModelVersion` — dataclass holding a version's metadata and transition history.
- `Stage` — `STAGING` / `PRODUCTION` / `ARCHIVED`.
- `RegistryError` — raised on invalid operations.

## Architecture

```
src/model_registry_lite/
├── __init__.py      # public API surface
├── __main__.py      # `python -m model_registry_lite`
├── models.py        # Stage, ModelVersion, StageTransition (dataclasses, JSON serde)
├── store.py         # SqliteStore: two tables (model docs as JSON, transition audit log)
├── registry.py      # ModelRegistry: registration, lifecycle, search, export/import
└── cli.py           # argparse CLI (`model-registry`)
```

`store.py` persists whole version documents as JSON, so adding metadata fields
never requires a schema migration; the separate `transitions` table keeps the
approval audit trail queryable even if a document is replaced. All stage rules
live in `models.py` (`ALLOWED_TRANSITIONS`), enforced by `record_transition`,
and surfaced through `registry.transition` / the CLI.

## Development

```bash
pip install -e ".[dev]"
pytest -q
python examples/quickstart.py
```

CI runs the test suite on Python 3.9–3.12 plus a smoke run of the example.

## License

MIT — see [LICENSE](LICENSE). Copyright © 2026 Anusha Mukka.
