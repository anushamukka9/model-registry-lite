"""Artifact checksum example: catch corrupted or swapped model files.

Registers a model version while hashing its artifact file, verifies it,
then tampers with the file and verifies again to show the mismatch being
caught. Also shows the "no checksum recorded" and missing-file cases.

Run from the repo root:

    python examples/artifact_checksums.py
"""

import tempfile
from pathlib import Path

from model_registry_lite import ModelRegistry, sha256_of_file, short_digest


def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="mrl-checksum-"))
    registry = ModelRegistry(workdir / "registry.db")
    print(f"Using throwaway registry: {workdir / 'registry.db'}\n")

    # 1. Write a fake artifact and register it with a computed checksum.
    artifact = workdir / "churn-1.1.0.pkl"
    artifact.write_bytes(b"fake-model-bytes-v1.1.0" * 4096)
    registry.register(
        name="churn-predictor",
        version="1.1.0",
        framework="scikit-learn",
        artifact_path=str(artifact),
        compute_checksum=True,
        metrics={"f1": 0.92},
    )
    stored = registry.get("churn-predictor", "1.1.0").artifact_sha256
    print(f"1. Registered with checksum {short_digest(stored)}")
    print(f"   (sha256_of_file agrees: {sha256_of_file(artifact) == stored})")

    # 2. Verify: file untouched, so this passes.
    ok, detail = registry.verify_artifact("churn-predictor", "1.1.0")
    print(f"2. Verify untouched file: {detail}")

    # 3. Tamper with the file (append one byte) and verify again.
    with artifact.open("ab") as handle:
        handle.write(b"!")
    ok, detail = registry.verify_artifact("churn-predictor", "1.1.0")
    print(f"3. Verify tampered file: {detail} (ok={ok})")

    # 4. A version registered without a checksum reports UNKNOWN, not a pass.
    registry.register(name="churn-predictor", version="1.0.0",
                      artifact_path=str(artifact))
    ok, detail = registry.verify_artifact("churn-predictor", "1.0.0")
    print(f"4. No checksum recorded: {detail} (ok={ok})")

    # 5. Deleting the file is reported as a failure, not a silent skip.
    artifact.unlink()
    ok, detail = registry.verify_artifact("churn-predictor", "1.1.0")
    print(f"5. Missing file: {detail} (ok={ok})")

    registry.close()


if __name__ == "__main__":
    main()
