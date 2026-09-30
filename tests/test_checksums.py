"""Tests for artifact checksum tracking: hashing, registration, verification."""

import hashlib
import pytest

from model_registry_lite import ModelRegistry, RegistryError, sha256_of_file, short_digest
from model_registry_lite.cli import main as cli_main


@pytest.fixture()
def registry(tmp_path):
    reg = ModelRegistry(tmp_path / "test.db")
    yield reg
    reg.close()


def test_sha256_of_file_matches_hashlib(tmp_path):
    target = tmp_path / "artifact.bin"
    target.write_bytes(b"model-bytes" * 1000)
    assert sha256_of_file(target) == hashlib.sha256(b"model-bytes" * 1000).hexdigest()


def test_sha256_of_file_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        sha256_of_file(tmp_path / "nope.bin")


def test_short_digest_format():
    assert short_digest("abcdef1234567890") == "sha256:abcdef123456"


def test_register_with_compute_checksum(registry, tmp_path):
    artifact = tmp_path / "m.pkl"
    artifact.write_bytes(b"weights")
    model = registry.register(
        "m", "1.0", artifact_path=str(artifact), compute_checksum=True
    )
    assert model.artifact_sha256 == hashlib.sha256(b"weights").hexdigest()


def test_register_compute_checksum_requires_path(registry):
    with pytest.raises(RegistryError, match="requires artifact_path"):
        registry.register("m", "1.0", compute_checksum=True)


def test_register_compute_checksum_missing_file(registry, tmp_path):
    with pytest.raises(RegistryError, match="Cannot hash artifact"):
        registry.register(
            "m", "1.0",
            artifact_path=str(tmp_path / "gone.bin"),
            compute_checksum=True,
        )


def test_register_with_explicit_checksum(registry):
    model = registry.register("m", "1.0", artifact_sha256="deadbeef" * 8)
    assert model.artifact_sha256 == "deadbeef" * 8


def test_verify_artifact_ok(registry, tmp_path):
    artifact = tmp_path / "m.pkl"
    artifact.write_bytes(b"weights")
    registry.register("m", "1.0", artifact_path=str(artifact), compute_checksum=True)
    ok, detail = registry.verify_artifact("m", "1.0")
    assert ok is True
    assert "matches" in detail


def test_verify_artifact_mismatch(registry, tmp_path):
    artifact = tmp_path / "m.pkl"
    artifact.write_bytes(b"weights")
    registry.register("m", "1.0", artifact_path=str(artifact), compute_checksum=True)
    artifact.write_bytes(b"different-weights")
    ok, detail = registry.verify_artifact("m", "1.0")
    assert ok is False
    assert "mismatch" in detail


def test_verify_artifact_no_checksum_recorded(registry):
    registry.register("m", "1.0")
    ok, detail = registry.verify_artifact("m", "1.0")
    assert ok is None
    assert "no checksum" in detail


def test_verify_artifact_missing_file(registry, tmp_path):
    missing = tmp_path / "gone.bin"
    registry.register("m", "1.0", artifact_path=str(missing),
                      artifact_sha256="ab" * 32)
    ok, detail = registry.verify_artifact("m", "1.0")
    assert ok is False
    assert "not found" in detail


def test_verify_artifact_path_override(registry, tmp_path):
    original = tmp_path / "orig.pkl"
    original.write_bytes(b"weights")
    moved = tmp_path / "moved.pkl"
    moved.write_bytes(b"weights")
    registry.register("m", "1.0", artifact_path=str(original), compute_checksum=True)
    ok, _ = registry.verify_artifact("m", "1.0", path=moved)
    assert ok is True


def test_checksum_survives_export_import(registry, tmp_path):
    artifact = tmp_path / "m.pkl"
    artifact.write_bytes(b"weights")
    registry.register("m", "1.0", artifact_path=str(artifact), compute_checksum=True)
    payload = registry.export_json()
    other = ModelRegistry(tmp_path / "other.db")
    try:
        other.import_json(payload)
        assert other.get("m", "1.0").artifact_sha256 == registry.get("m", "1.0").artifact_sha256
    finally:
        other.close()


def test_update_metadata_can_set_checksum(registry):
    registry.register("m", "1.0")
    updated = registry.update_metadata("m", "1.0", artifact_sha256="ff" * 32)
    assert updated.artifact_sha256 == "ff" * 32


def test_cli_verify_ok(tmp_path, capsys):
    db = tmp_path / "cli.db"
    artifact = tmp_path / "m.pkl"
    artifact.write_bytes(b"weights")
    rc = cli_main([
        "--db", str(db), "register", "--name", "m", "--version", "1.0",
        "--artifact-path", str(artifact), "--hash-artifact",
    ])
    assert rc == 0
    rc = cli_main(["--db", str(db), "verify", "--name", "m", "--version", "1.0"])
    assert rc == 0
    assert "OK" in capsys.readouterr().out


def test_cli_verify_failed(tmp_path, capsys):
    db = tmp_path / "cli.db"
    artifact = tmp_path / "m.pkl"
    artifact.write_bytes(b"weights")
    cli_main([
        "--db", str(db), "register", "--name", "m", "--version", "1.0",
        "--artifact-path", str(artifact), "--artifact-sha256", "00" * 32,
    ])
    rc = cli_main(["--db", str(db), "verify", "--name", "m", "--version", "1.0"])
    assert rc == 1
    assert "FAILED" in capsys.readouterr().out


def test_cli_verify_unknown_without_checksum(tmp_path, capsys):
    db = tmp_path / "cli.db"
    cli_main(["--db", str(db), "register", "--name", "m", "--version", "1.0"])
    rc = cli_main(["--db", str(db), "verify", "--name", "m", "--version", "1.0"])
    assert rc == 1
    assert "UNKNOWN" in capsys.readouterr().out
