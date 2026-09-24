"""Content hashing and code versioning for dataset provenance (§4.5).

§9 predicts one RobotSpec revision at Stage 9. Datasets recorded before it stay
on disk and stay trainable, and become subtly wrong in a way that degrades a
policy rather than raising. A hash stamped into every episode is what makes
that detectable instead of silent.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from enum import Enum
from pathlib import Path
from typing import Any

HASH_LENGTH = 16

_code_version_cache: str | None = None


def _canonical(value: Any) -> Any:
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: _canonical(getattr(value, field.name))
            for field in dataclasses.fields(value)
            if field.metadata.get("hash", True)
        }
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): _canonical(v) for k, v in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        # Keep hashes stable across trivial float representation differences.
        return round(value, 12)
    return value


def content_hash(value: Any) -> str:
    """Stable short hash of a dataclass or plain structure.

    Fields marked ``metadata={"hash": False}`` are excluded, so machine-local
    details such as where a spec file happened to live do not change the hash.
    """
    payload = json.dumps(_canonical(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:HASH_LENGTH]


def code_version() -> str:
    """Short git revision, or ``"unknown"`` outside a repository."""
    global _code_version_cache
    if _code_version_cache is not None:
        return _code_version_cache

    version = "unknown"
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if result.returncode == 0 and result.stdout.strip():
            version = result.stdout.strip()
            if _git_is_dirty():
                version += "-dirty"
    except (OSError, subprocess.SubprocessError):
        pass

    _code_version_cache = version
    return version


def _git_is_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True,
        text=True,
        timeout=2.0,
        check=False,
    )
    return bool(result.stdout.strip())
