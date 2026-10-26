from __future__ import annotations

from collections.abc import Iterable


RESOURCE_PATHS = (
    "app.py",
    "logos",
    "corn/components/document_selector/frontend",
    "examples/demo",
)


def validate_resource_manifest(paths: Iterable[str]) -> None:
    allowed = set(RESOURCE_PATHS)
    for path in paths:
        normalized = path.replace("\\", "/").strip("/")
        if normalized not in allowed:
            raise ValueError(f"Unsafe packaged resource: {path}")
