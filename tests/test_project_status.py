from __future__ import annotations

from pathlib import Path

from corn.ingestion import initialize_runtime_database, write_cache_metadata
from corn.projects import create_project_structure, get_project_status, resolve_source_path


def test_get_project_status_returns_ready_when_source_and_cache_are_current(tmp_path: Path) -> None:
    source_path = tmp_path / "dataset.csv"
    source_path.write_text("doc_id,body\n1,Hello world\n", encoding="utf-8")
    project_path = tmp_path / "projects" / "eu-qual"
    config = _project_config(source_path)
    stored_config = create_project_structure(project_path, config)

    initialize_runtime_database(project_path / "derived" / "runtime.sqlite", stored_config)
    write_cache_metadata(
        project_path / "derived" / "cache_metadata.json",
        resolve_source_path(project_path, stored_config),
    )

    assert get_project_status(project_path) == "ready"


def test_get_project_status_returns_source_missing_when_source_path_is_gone(tmp_path: Path) -> None:
    source_path = tmp_path / "dataset.csv"
    source_path.write_text("doc_id,body\n1,Hello world\n", encoding="utf-8")
    project_path = tmp_path / "projects" / "eu-qual"
    config = _project_config(source_path, mode="reference")
    create_project_structure(project_path, config)
    source_path.unlink()

    assert get_project_status(project_path) == "source_missing"


def test_get_project_status_returns_config_invalid_when_project_config_cannot_validate(tmp_path: Path) -> None:
    project_path = tmp_path / "projects" / "eu-qual"
    project_path.mkdir(parents=True, exist_ok=True)
    (project_path / "project.yaml").write_text("project:\n  slug: eu-qual\n", encoding="utf-8")

    assert get_project_status(project_path) == "config_invalid"


def test_get_project_status_returns_cache_stale_when_source_changes_after_cache_build(tmp_path: Path) -> None:
    source_path = tmp_path / "dataset.csv"
    source_path.write_text("doc_id,body\n1,Hello world\n", encoding="utf-8")
    project_path = tmp_path / "projects" / "eu-qual"
    config = _project_config(source_path)
    stored_config = create_project_structure(project_path, config)

    initialize_runtime_database(project_path / "derived" / "runtime.sqlite", stored_config)
    stored_source_path = resolve_source_path(project_path, stored_config)
    write_cache_metadata(project_path / "derived" / "cache_metadata.json", stored_source_path)
    stored_source_path.write_text("doc_id,body\n1,Changed\n", encoding="utf-8")

    assert get_project_status(project_path) == "cache_stale"


def _project_config(source_path: Path, *, mode: str = "copy") -> dict[str, object]:
    return {
        "project": {"name": "EU Qual", "slug": "eu-qual"},
        "source": {"type": "csv", "mode": mode, "path": str(source_path)},
        "document": {"id_field": "doc_id", "text_field": "body"},
        "metadata_fields": [],
        "filters": [],
        "tie_schema": [{"label": "Policy", "description": "Policy tie"}],
    }
