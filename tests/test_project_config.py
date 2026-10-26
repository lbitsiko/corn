from __future__ import annotations

import json
from pathlib import Path

import pytest

from corn.project_config import load_project_config, validate_project_config


def test_validate_project_config_accepts_valid_csv_config() -> None:
    config = {
        "project": {"name": "EU Qual", "slug": "eu-qual"},
        "source": {"type": "csv", "mode": "copy", "path": "/tmp/eu-qual.csv"},
        "document": {"id_field": "doc_id", "text_field": "body", "title_field": "title"},
        "metadata_fields": [
            {"name": "committee", "source_field": "committee_code"},
        ],
        "filters": [
            {"field": "committee", "label": "Committee", "type": "select"},
        ],
        "tie_schema": [
            {"label": "Policy", "description": "Policy tie"},
        ],
    }

    validated = validate_project_config(config)

    assert validated["project"]["slug"] == "eu-qual"
    assert validated["document"]["text_field"] == "body"


def test_load_project_config_accepts_valid_sqlite_json_config(tmp_path: Path) -> None:
    config_path = tmp_path / "project.json"
    config_path.write_text(
        json.dumps(
            {
                "project": {"name": "EU Qual", "slug": "eu-qual"},
                "source": {
                    "type": "sqlite",
                    "mode": "reference",
                    "path": "/tmp/eu-qual.sqlite",
                    "table_or_view": "documents_view",
                },
                "document": {"id_field": "doc_id", "text_field": "body"},
                "metadata_fields": [],
                "filters": [
                    {"field": "committee", "label": "Committee", "type": "multi_select"},
                    {"field": "is_coded", "label": "Coded", "type": "boolean"},
                ],
                "tie_schema": [
                    {"label": "Policy", "description": "Policy tie"},
                    {"label": "Funding", "description": "Funding tie"},
                ],
            }
        ),
        encoding="utf-8",
    )

    loaded = load_project_config(config_path)

    assert loaded["source"]["table_or_view"] == "documents_view"
    assert loaded["filters"][1]["type"] == "boolean"


def test_validate_project_config_requires_id_field() -> None:
    config = {
        "project": {"name": "EU Qual", "slug": "eu-qual"},
        "source": {"type": "csv", "mode": "copy", "path": "/tmp/eu-qual.csv"},
        "document": {"text_field": "body"},
        "metadata_fields": [],
        "filters": [],
        "tie_schema": [{"label": "Policy", "description": "Policy tie"}],
    }

    with pytest.raises(ValueError, match="id_field"):
        validate_project_config(config)


def test_validate_project_config_requires_text_field() -> None:
    config = {
        "project": {"name": "EU Qual", "slug": "eu-qual"},
        "source": {"type": "csv", "mode": "copy", "path": "/tmp/eu-qual.csv"},
        "document": {"id_field": "doc_id"},
        "metadata_fields": [],
        "filters": [],
        "tie_schema": [{"label": "Policy", "description": "Policy tie"}],
    }

    with pytest.raises(ValueError, match="text_field"):
        validate_project_config(config)


def test_validate_project_config_requires_sqlite_table_or_view() -> None:
    config = {
        "project": {"name": "EU Qual", "slug": "eu-qual"},
        "source": {"type": "sqlite", "mode": "reference", "path": "/tmp/eu-qual.sqlite"},
        "document": {"id_field": "doc_id", "text_field": "body"},
        "metadata_fields": [],
        "filters": [],
        "tie_schema": [{"label": "Policy", "description": "Policy tie"}],
    }

    with pytest.raises(ValueError, match="table_or_view"):
        validate_project_config(config)


def test_validate_project_config_rejects_duplicate_tie_labels() -> None:
    config = {
        "project": {"name": "EU Qual", "slug": "eu-qual"},
        "source": {"type": "csv", "mode": "copy", "path": "/tmp/eu-qual.csv"},
        "document": {"id_field": "doc_id", "text_field": "body"},
        "metadata_fields": [],
        "filters": [],
        "tie_schema": [
            {"label": "Policy", "description": "First label"},
            {"label": "Policy", "description": "Duplicate label"},
        ],
    }

    with pytest.raises(ValueError, match="Duplicate tie label"):
        validate_project_config(config)


def _config_with_tie_attribute_schema(tie_attribute_schema: object) -> dict:
    return {
        "project": {"name": "EU Qual", "slug": "eu-qual"},
        "source": {"type": "csv", "mode": "copy", "path": "/tmp/eu-qual.csv"},
        "document": {"id_field": "doc_id", "text_field": "body"},
        "metadata_fields": [],
        "filters": [],
        "tie_schema": [{"label": "Policy", "description": "Policy tie"}],
        "tie_attribute_schema": tie_attribute_schema,
    }


def test_tie_attribute_schema_defaults_to_empty() -> None:
    config = _config_with_tie_attribute_schema([])
    del config["tie_attribute_schema"]

    assert validate_project_config(config)["tie_attribute_schema"] == []


def test_tie_attribute_schema_valid_types_normalize() -> None:
    config = _config_with_tie_attribute_schema(
        [
            {
                "name": "valence",
                "type": "categorical",
                "description": "d",
                "options": ["positive", "negative"],
            },
            {"name": "sentiment", "type": "numeric", "description": "d", "range": [0, 1]},
            {"name": "time", "type": "temporal", "description": "d"},
        ]
    )

    normalized = validate_project_config(config)["tie_attribute_schema"]

    assert [entry["name"] for entry in normalized] == ["valence", "sentiment", "time"]
    assert normalized[1]["range"] == [0.0, 1.0]


@pytest.mark.parametrize(
    "bad_schema",
    [
        [
            {"name": "a", "type": "categorical", "description": "d", "options": ["x"]},
            {"name": "a", "type": "temporal", "description": "d"},
        ],  # duplicate name
        [{"name": "a", "type": "text", "description": "d"}],  # unknown type
        [{"name": "a", "type": "categorical", "description": "d"}],  # missing options
        [{"name": "a", "type": "categorical", "description": "d", "options": []}],
        [{"name": "a", "type": "categorical", "description": "d", "options": ["x", "x"]}],
        [{"name": "a", "type": "numeric", "description": "d"}],  # missing range
        [{"name": "a", "type": "numeric", "description": "d", "range": [1, 0]}],  # min >= max
        [{"name": "a", "type": "numeric", "description": "d", "range": [1, 1]}],  # min == max
        [{"name": "a", "type": "numeric", "description": "d", "range": [True, False]}],  # bool bounds
        [{"name": "a", "type": "numeric", "description": "d", "range": [0, float("nan")]}],  # NaN bound
        [{"name": "a", "type": "numeric", "description": "d", "range": [0]}],
        [{"name": "a", "type": "categorical", "description": "d", "options": ["x", 1]}],  # non-string option
        [{"name": "a", "type": "categorical", "description": "d", "options": ["(unset)"]}],  # reserved editor sentinel
        [["x"]],  # non-mapping schema entry
        "not-a-list",
    ],
)
def test_tie_attribute_schema_rejections(bad_schema: object) -> None:
    with pytest.raises(ValueError):
        validate_project_config(_config_with_tie_attribute_schema(bad_schema))


def test_validate_project_config_rejects_invalid_filter_type() -> None:
    config = {
        "project": {"name": "EU Qual", "slug": "eu-qual"},
        "source": {"type": "csv", "mode": "copy", "path": "/tmp/eu-qual.csv"},
        "document": {"id_field": "doc_id", "text_field": "body"},
        "metadata_fields": [],
        "filters": [{"field": "committee", "label": "Committee", "type": "slider"}],
        "tie_schema": [{"label": "Policy", "description": "Policy tie"}],
    }

    with pytest.raises(ValueError, match="Invalid filter type"):
        validate_project_config(config)
