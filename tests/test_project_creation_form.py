from __future__ import annotations

import pytest

from corn.source_inspection import build_guided_project_config, validate_project_slug


def test_validate_project_slug_accepts_safe_folder_name() -> None:
    assert validate_project_slug("toy_relationships-2026") == "toy_relationships-2026"


@pytest.mark.parametrize("slug", ["", "Toy Project", "../toy", "toy/project", "."])
def test_validate_project_slug_rejects_unsafe_folder_name(slug: str) -> None:
    with pytest.raises(ValueError):
        validate_project_slug(slug)


def test_build_guided_project_config_builds_canonical_csv_config() -> None:
    config = build_guided_project_config(
        project_name="Toy Relationship Dataset",
        project_slug="toy-relationships",
        source_type="csv",
        source_mode="copy",
        source_path="/tmp/toy.csv",
        id_field="id",
        text_field="comment",
        title_field="subreddit",
        metadata_fields=["person_a", "person_b", "sentiment"],
        filter_types_by_field={
            "sentiment": "select",
            "person_a": "text_contains",
        },
        tie_schema=[
            {"label": "support", "description": "Supportive relationship"},
            {"label": "conflict", "description": "Conflict relationship"},
        ],
        tie_attribute_schema=[
            {
                "name": "valence",
                "type": "categorical",
                "description": "Character of the tie",
                "options": ["positive", "negative"],
            },
            {
                "name": "strength",
                "type": "numeric",
                "description": "Intensity of the tie",
                "range": [0, 10],
            },
            {
                "name": "period",
                "type": "temporal",
                "description": "When the tie occurred",
            },
        ],
    )

    assert config == {
        "project": {"name": "Toy Relationship Dataset", "slug": "toy-relationships"},
        "source": {"type": "csv", "mode": "copy", "path": "/tmp/toy.csv"},
        "document": {"id_field": "id", "text_field": "comment", "title_field": "subreddit"},
        "metadata_fields": [
            {"name": "person_a", "source_field": "person_a"},
            {"name": "person_b", "source_field": "person_b"},
            {"name": "sentiment", "source_field": "sentiment"},
        ],
        "filters": [
            {"field": "person_a", "label": "Person A", "type": "text_contains"},
            {"field": "sentiment", "label": "Sentiment", "type": "select"},
        ],
        "tie_schema": [
            {"label": "support", "description": "Supportive relationship"},
            {"label": "conflict", "description": "Conflict relationship"},
        ],
        "tie_attribute_schema": [
            {
                "name": "valence",
                "type": "categorical",
                "description": "Character of the tie",
                "options": ["positive", "negative"],
            },
            {
                "name": "strength",
                "type": "numeric",
                "description": "Intensity of the tie",
                "range": [0, 10],
            },
            {
                "name": "period",
                "type": "temporal",
                "description": "When the tie occurred",
            },
        ],
    }


def test_build_guided_project_config_builds_sqlite_config_with_selected_table() -> None:
    config = build_guided_project_config(
        project_name="Workspace Notes",
        project_slug="workspace-notes",
        source_type="sqlite",
        source_mode="reference",
        source_path="/tmp/notes.sqlite",
        table_or_view="documents_view",
        id_field="doc_id",
        text_field="body",
        title_field=None,
        metadata_fields=["team", "priority"],
        filter_types_by_field={
            "team": "select",
        },
        tie_schema=[
            {"label": "coordination", "description": "Shared work coordination"},
        ],
    )

    assert config["source"]["table_or_view"] == "documents_view"
    assert config["filters"] == [
        {"field": "team", "label": "Team", "type": "select"},
    ]
    assert config["tie_attribute_schema"] == []
