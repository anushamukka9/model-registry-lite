"""Artifact checksum helpers: integrity tracking for registered model files.

A checksum recorded at registration time lets you verify later that the
artifact file on disk is the same one that was registered - catching
accidental overwrites, truncated downloads, or silent corruption before a
stale or wrong artifact reaches production.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]

_CHUNK_SIZE = 1024 * 1024  # 1 MiB


def sha256_of_file(path: PathLike) -> str:
    """Return the hex SHA-256 digest of the file at ``path``.

    Reads in 1 MiB chunks, so multi-gigabyte model files are hashed without
    loading them into memory. Raises ``FileNotFoundError`` when the file does
    not exist.
    """
    digest_target = Path(path)
    digest = hashlib.sha256()
    with digest_target.open("rb") as handle:
        while True:
            chunk = handle.read(_CHUNK_SIZE)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def short_digest(digest: str, length: int = 12) -> str:
    """Format a hex digest as ``sha256:<prefix>`` for compact display."""
    return f"sha256:{digest[:length]}"
