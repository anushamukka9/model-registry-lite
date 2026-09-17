"""model-registry-lite: a minimal local model registry.

Track versioned machine-learning model artifacts with rich metadata, move them
through lifecycle stages with approval records, search by metric thresholds,
and export/import the whole registry as JSON.

The default store is a SQLite database in the current working directory
(``model_registry.db``); every operation works against a single file that can
be checked into version control or shipped alongside a project.
"""

from .models import ModelVersion, Stage, StageTransition
from .registry import ModelRegistry, RegistryError

__all__ = ["ModelVersion", "Stage", "StageTransition", "ModelRegistry", "RegistryError"]
__version__ = "0.1.0"
__author__ = "Anusha Mukka"
