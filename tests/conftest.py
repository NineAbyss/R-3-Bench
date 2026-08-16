"""Shared paths for tests of the installed-style public package."""

from __future__ import annotations

from pathlib import Path

import pytest

from r3bench.resource_paths import resource_root


@pytest.fixture(scope="session")
def release_root() -> Path:
    return Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def resources() -> Path:
    return resource_root()
