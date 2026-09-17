# model-registry-lite usage guide

A minimal, dependency-free model registry for ML teams that don't need a
full MLOps platform. Everything lives in one SQLite file you can check into
version control, back up, or email.

## Installation

```bash
pip install model-registry-lite
# or, from a checkout:
pip install -e .
```

The `model-registry` command becomes available, and the Python API can be
imported as `model_registry_lite`.

## Concepts

- **Model version** — one trained artifact plus metadata: name, version,
  framework, description, artifact path, training-dataset hash, metrics,
  hyperparameters, and tags.
- **Stage** — a version's lifecycle position: `staging` -> `production` ->
  `archived`. Archived is terminal. Every move needs an approver name and
  can carry a note; moves are kept in an audit trail (`history`).
- **Store** — a single SQLite file (`model_registry.db` by default). Model
  documents are stored as JSON so metadata fields can evolve without
  schema migrations.

## Python API

```python
from model_registry_lite import ModelRegistry

registry = ModelRegistry("models.db")   # creates the file if missing

registry.register(
    name="churn-predictor",
    version="1.1.0",
    framework="scikit-learn",
    description="Retrained on the March data refresh.",
    artifact_path="s3://models/churn/1.1.0/model.pkl",
    dataset_hash="sha256:d71be804",
    metrics={"f1": 0.92, "auc": 0.96},
    hyperparameters={"n_estimators": 500, "max_depth": 8},
    tags=["tabular", "candidate"],
)

# Lifecycle: promote with an approval record.
registry.promote("churn-predictor", "1.1.0",
                 approved_by="Anusha Mukka",
                 note="F1 0.92 on holdout; beats baseline.")

# Search by metric thresholds (all filters must match).
hits = registry.search(stage="production", metric_filters=["f1>=0.9", "auc>0.95"])

# Newest version overall, or newest in a stage.
latest = registry.latest("churn-predictor")
prod = registry.latest("churn-predictor", stage="production")

# Update mutable metadata (metrics, hyperparameters, tags, description...).
registry.update_metadata("churn-predictor", "1.1.0", metrics={"f1": 0.925})

# Export the whole registry as JSON; import into another registry.
registry.export_to_file("backup.json")
registry.import_from_file("backup.json")          # skips existing unless overwrite=True

registry.close()
```

`ModelRegistry` also works as a context manager.

### Search syntax

Metric filters are strings of the form `<metric><operator><threshold>`:

- operators: `>=`, `<=`, `>`, `<` (or `gte`, `lte`, `gt`, `lt`)
- examples: `"f1>=0.9"`, `"loss < 0.5"`, `"latency_ms<=120"`

Metadata filters: `name_contains` (substring), `framework` (exact,
case-insensitive), `stage`, and `tags` (all listed tags must be present).

## CLI

Every registry operation has a CLI equivalent. The default database is
`./model_registry.db`; override with `--db <path>` before the subcommand.

```bash
# Register
model-registry register --name churn-predictor --version 1.1.0 \
    --framework scikit-learn \
    --metrics '{"f1": 0.92, "auc": 0.96}' \
    --hyperparameters '{"n_estimators": 500}' \
    --dataset-hash sha256:d71be804 \
    --tags tabular,candidate

# Promote / archive (approver is recorded in the audit trail)
model-registry promote --name churn-predictor --version 1.1.0 \
    --by "Anusha Mukka" --note "QA pass on holdout"
model-registry archive --name churn-predictor --version 1.0.0 \
    --by "Anusha Mukka" --note "Superseded by 1.1.0"

# Inspect
model-registry list --stage production
model-registry versions --name churn-predictor
model-registry show --name churn-predictor --version 1.1.0
model-registry history --name churn-predictor --version 1.1.0

# Search
model-registry search --metric "f1>=0.9" --framework scikit-learn --tag tabular

# Export / import
model-registry export --out backup.json
model-registry import --in backup.json          # add --overwrite to replace
```

## Workflow suggestions

1. **Experiment tracking**: register every training run in `staging` with
   its dataset hash, metrics, and hyperparameters — the hash makes results
   reproducible.
2. **Promotion gate**: only `promote` after an offline evaluation; the
   required approver name + note gives you a paper trail for audits.
3. **Cleanup**: `archive` stale candidates instead of deleting them, so the
   audit trail survives; `deregister` only for true mistakes.
4. **Sharing**: `export` produces a single JSON document the whole team can
   diff, review, and `import` into their own registry files.
