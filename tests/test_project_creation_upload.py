from __future__ import annotations

import json
from pathlib import Path

import pytest

from app import _create_project_from_uploaded_config
from corn.project_config import load_project_config
from corn.registry import load_registry


def test_create_project_from_uploaded_yaml_config(tmp_path: Path) -> None:
    source_path = tmp_path / "dataset.csv"
    source_path.write_text("doc_id,body\n1,Hello world\n", encoding="utf-8")

    entry = _create_project_from_uploaded_config(
        projects_root=tmp_path / "projects",
        registry_path=tmp_path / "project_registry.yaml",
        filename="project.yaml",
        raw_content=f"""
project:
  name: EU Qual
  slug: eu-qual
source:
  type: csv
  mode: copy
  path: {source_path}
document:
  id_field: doc_id
  text_field: body
metadata_fields: []
filters: []
tie_schema:
  - label: Policy
    description: Policy tie
""".strip(),
    )

    project_path = tmp_path / "projects" / "eu-qual"
    stored_config = load_project_config(project_path / "project.yaml")

    assert entry["slug"] == "eu-qual"
    assert stored_config["source"]["path"] == "source/dataset.csv"
    assert load_registry(tmp_path / "project_registry.yaml") == [entry]


def test_create_project_from_uploaded_json_config(tmp_path: Path) -> None:
    source_path = tmp_path / "dataset.csv"
    source_path.write_text("doc_id,body\n1,Hello world\n", encoding="utf-8")

    entry = _create_project_from_uploaded_config(
        projects_root=tmp_path / "projects",
        registry_path=tmp_path / "project_registry.yaml",
        filename="project.json",
        raw_content=json.dumps(
            {
                "project": {"name": "EU Qual", "slug": "eu-qual"},
                "source": {"type": "csv", "mode": "reference", "path": str(source_path)},
                "document": {"id_field": "doc_id", "text_field": "body"},
                "metadata_fields": [],
                "filters": [],
                "tie_schema": [{"label": "Policy", "description": "Policy tie"}],
            }
        ),
    )

    project_path = tmp_path / "projects" / "eu-qual"
    stored_config = load_project_config(project_path / "project.yaml")

    assert entry["project_path"] == str(project_path)
    assert stored_config["source"]["path"] == str(source_path)


def test_create_project_from_uploaded_config_rejects_invalid_config(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="id_field"):
        _create_project_from_uploaded_config(
            projects_root=tmp_path / "projects",
            registry_path=tmp_path / "project_registry.yaml",
            filename="project.yaml",
            raw_content="""
project:
  name: EU Qual
  slug: eu-qual
source:
  type: csv
  mode: copy
  path: /tmp/missing.csv
document:
  text_field: body
metadata_fields: []
filters: []
tie_schema:
  - label: Policy
    description: Policy tie
""".strip(),
        )
