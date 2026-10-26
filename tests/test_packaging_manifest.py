from __future__ import annotations

import pytest

from corn.packaging_manifest import RESOURCE_PATHS, validate_resource_manifest


def test_packaging_manifest_contains_only_required_project_resources() -> None:
    assert RESOURCE_PATHS == (
        "app.py",
        "logos",
        "corn/components/document_selector/frontend",
        "examples/demo",
    )
    validate_resource_manifest(RESOURCE_PATHS)


@pytest.mark.parametrize(
    "unsafe_path",
    [
        "data",
        ".env",
        ".venv/lib",
        ".git/config",
        "projects/research.sqlite",
        "annotations/llm_events.jsonl",
        "provider_settings.json",
        "API-key.md",
    ],
)
def test_packaging_manifest_rejects_sensitive_or_mutable_paths(
    unsafe_path: str,
) -> None:
    with pytest.raises(ValueError, match="Unsafe packaged resource"):
        validate_resource_manifest((*RESOURCE_PATHS, unsafe_path))
