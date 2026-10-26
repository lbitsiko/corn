from __future__ import annotations

from pathlib import Path

from app import _project_runtime_context
from corn.projects import create_project_structure


def test_project_runtime_context_uses_project_scoped_paths_and_config(tmp_path: Path) -> None:
    source_path = tmp_path / "toy.csv"
    source_path.write_text("id,comment\n1,Hello world\n", encoding="utf-8")
    project_path = tmp_path / "projects" / "toy"
    create_project_structure(
        project_path,
        {
            "project": {"name": "Toy Project", "slug": "toy"},
            "source": {"type": "csv", "mode": "reference", "path": str(source_path)},
            "document": {"id_field": "id", "text_field": "comment"},
            "metadata_fields": [{"name": "sentiment", "source_field": "sentiment"}],
            "filters": [{"field": "sentiment", "label": "Sentiment", "type": "select"}],
            "tie_schema": [{"label": "support", "description": "Supportive relationship"}],
        },
    )

    context = _project_runtime_context(
        {
            "name": "Toy Project",
            "slug": "toy",
            "project_path": str(project_path),
            "archived": False,
        }
    )

    assert context["runtime_db_path"] == project_path / "derived" / "runtime.sqlite"
    assert context["annotations_dir"] == project_path / "annotations"
    assert context["annotations_db_path"] == project_path / "annotations" / "annotations.sqlite"
    assert context["filter_schema"] == [{"field": "sentiment", "label": "Sentiment", "type": "select"}]
    assert context["config"]["project"]["slug"] == "toy"
