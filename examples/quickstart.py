"""Quickstart example: register two model versions, promote one, search by metric.

Run from the repo root:

    python examples/quickstart.py
"""

import os
import tempfile
from pathlib import Path

from model_registry_lite import ModelRegistry


def main() -> None:
    db_path = Path(tempfile.mkdtemp(prefix="mrl-")) / "quickstart.db"
    print(f"Using throwaway registry: {db_path}")

    registry = ModelRegistry(db_path)

    # 1. Register two candidate versions of the same model.
    registry.register(
        name="churn-predictor",
        version="1.0.0",
        framework="scikit-learn",
        description="Gradient boosting baseline on the churn dataset.",
        dataset_hash="sha256:9f2c41ab",
        metrics={"f1": 0.87, "auc": 0.93},
        hyperparameters={"n_estimators": 300, "max_depth": 6},
        tags=["tabular", "baseline"],
    )
    registry.register(
        name="churn-predictor",
        version="1.1.0",
        framework="scikit-learn",
        description="Retrained with the March data refresh.",
        dataset_hash="sha256:d71be804",
        metrics={"f1": 0.92, "auc": 0.96},
        hyperparameters={"n_estimators": 500, "max_depth": 8},
        tags=["tabular", "candidate"],
    )
    print("Registered churn-predictor 1.0.0 and 1.1.0 (both staging).")

    # 2. Promote the better one to production with an approval record.
    registry.promote(
        "churn-predictor", "1.1.0",
        approved_by="Anusha Mukka",
        note="F1 0.92 on holdout; beats the 1.0.0 baseline.",
    )
    print("Promoted churn-predictor 1.1.0 -> production.")

    # 3. Search for production-ready candidates by metric threshold.
    hits = registry.search(stage="production", metric_filters=["f1>=0.9"])
    print(f"Production models with f1 >= 0.9: {[f'{m.name}:{m.version}' for m in hits]}")

    # 4. Export the registry to JSON (portable backup / sharing).
    export_path = Path(tempfile.mkdtemp(prefix="mrl-")) / "registry-export.json"
    registry.export_to_file(export_path)
    print(f"Exported registry to {export_path} ({os.path.getsize(export_path)} bytes).")

    # 5. Import it into a second, empty registry.
    second = ModelRegistry(Path(tempfile.mkdtemp(prefix="mrl-")) / "clone.db")
    imported, skipped = second.import_json(export_path.read_text())
    print(f"Imported into a fresh registry: {imported} model(s), {skipped} skipped.")

    latest_prod = second.latest("churn-predictor", stage="production")
    print(f"Latest production churn-predictor in clone: {latest_prod.version}")

    registry.close()
    second.close()
    print("Quickstart complete.")


if __name__ == "__main__":
    main()
