"""Resolve bundled resources without depending on a source checkout."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path


_RESOURCE_PREFIXES = frozenset({"configs", "examples", "prompts"})


def resource_root() -> Path:
    """Return the installed package-resource directory.

    Wheels are installed as ordinary directories by pip, so converting the
    Traversable returned by :mod:`importlib.resources` to ``Path`` is valid for
    every supported installation mode in this release.
    """

    return Path(str(files("r3bench.resources")))


def resource_path(*parts: str) -> Path:
    """Return a path below the bundled resource directory."""

    return resource_root().joinpath(*parts)


def resolve_path(value: str | Path) -> Path:
    """Resolve a user path or a release-logical bundled resource path.

    Explicit absolute paths and existing paths relative to the current working
    directory remain user-controlled. Logical ``configs/...``, ``prompts/...``
    and ``examples/...`` paths resolve inside the installed package.
    """

    candidate = Path(value).expanduser()
    if candidate.is_absolute() or candidate.exists():
        return candidate
    if candidate.parts and candidate.parts[0] in _RESOURCE_PREFIXES:
        return resource_root().joinpath(*candidate.parts)
    return candidate


__all__ = ["resolve_path", "resource_path", "resource_root"]
