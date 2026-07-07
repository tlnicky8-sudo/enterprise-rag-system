from typing import Any

__all__ = ["IngestPipeline"]


def __getattr__(name: str) -> Any:
    if name == "IngestPipeline":
        from core.ingest.pipeline import IngestPipeline

        return IngestPipeline
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
