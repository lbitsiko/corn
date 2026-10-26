from __future__ import annotations

from pathlib import Path

from corn.project_config import load_project_config
from corn.projects import create_project_structure, resolve_source_path


def test_create_project_structure_bootstraps_expected_directories(tmp_path: Path) -> None:
    source_path = tmp_path / "dataset.csv"
    source_path.write_text("doc_id,body\n1,Hello world\n", encoding="utf-8")
    project_path = tmp_path / "projects" / "eu-qual"

    create_project_structure(project_path, _csv_project_config(source_path, mode="copy"))

    assert (project_path / "project.yaml").exists()
    assert (project_path / "source").is_dir()
    assert (project_path / "derived").is_dir()
    assert (project_path / "annotations").is_dir()


def test_create_project_structure_copies_source_files_for_copy_mode(tmp_path: Path) -> None:
    source_path = tmp_path / "dataset.csv"
    source_path.write_text("doc_id,body\n1,Hello world\n", encoding="utf-8")
    project_path = tmp_path / "projects" / "eu-qual"

    create_project_structure(project_path, _csv_project_config(source_path, mode="copy"))
    stored_config = load_project_config(project_path / "project.yaml")

    assert stored_config["source"]["path"] == "source/dataset.csv"
    assert resolve_source_path(project_path, stored_config) == project_path / "source" / "dataset.csv"
    assert (project_path / "source" / "dataset.csv").read_text(encoding="utf-8") == source_path.read_text(
        encoding="utf-8"
    )


def test_create_project_structure_preserves_external_source_for_reference_mode(tmp_path: Path) -> None:
    source_path = tmp_path / "dataset.csv"
    source_path.write_text("doc_id,body\n1,Hello world\n", encoding="utf-8")
    project_path = tmp_path / "projects" / "eu-qual"

    create_project_structure(project_path, _csv_project_config(source_path, mode="reference"))
    stored_config = load_project_config(project_path / "project.yaml")

    assert stored_config["source"]["path"] == str(source_path)
    assert resolve_source_path(project_path, stored_config) == source_path
    assert list((project_path / "source").iterdir()) == []


def _csv_project_config(source_path: Path, *, mode: str) -> dict[str, object]:
    return {
        "project": {"name": "EU Qual", "slug": "eu-qual"},
        "source": {"type": "csv", "mode": mode, "path": str(source_path)},
        "document": {"id_field": "doc_id", "text_field": "body"},
        "metadata_fields": [{"name": "committee", "source_field": "committee"}],
        "filters": [{"field": "committee", "label": "Committee", "type": "select"}],
        "tie_schema": [{"label": "Policy", "description": "Policy tie"}],
    }
