"""R3Bench public evaluator package."""

from importlib.metadata import PackageNotFoundError, version

from r3bench.common.schema import ContestSuite, ProblemRecord

__all__ = ["ContestSuite", "ProblemRecord"]

try:
    __version__ = version("r3bench")
except PackageNotFoundError:  # pragma: no cover - source tree without installation
    __version__ = "0+unknown"
