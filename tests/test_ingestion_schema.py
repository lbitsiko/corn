from __future__ import annotations

import sqlite3
from pathlib import Path

from corn.ingestion import CANONICAL_DOCUMENTS_TABLE, initialize_runtime_database


def test_initialize_runtime_database_creates_canonical_table_with_title_and_metadata(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime.sqlite"

    initialize_runtime_database(db_path, _project_config(include_title=True))

    connection = sqlite3.connect(db_path)
    try:
        columns = {
            row[1]: row[2]
            for row in connection.execute(f"PRAGMA table_info({CANONICAL_DOCUMENTS_TABLE})").fetchall()
        }
    finally:
        connection.close()

    assert columns == {
        "document_id": "TEXT",
        "document_text": "TEXT",
        "document_title": "TEXT",
        "committee": "TEXT",
        "time_period": "TEXT",
    }


def test_initialize_runtime_database_omits_document_title_when_unmapped(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime.sqlite"

    initialize_runtime_database(db_path, _project_config(include_title=False))

    connection = sqlite3.connect(db_path)
    try:
        columns = [row[1] for row in connection.execute(f"PRAGMA table_info({CANONICAL_DOCUMENTS_TABLE})").fetchall()]
    finally:
        connection.close()

    assert columns == ["document_id", "document_text", "committee", "time_period"]


def _project_config(*, include_title: bool) -> dict[str, object]:
    document = {"id_field": "doc_id", "text_field": "body"}
    if include_title:
        document["title_field"] = "title"

    return {
        "project": {"name": "EU Qual", "slug": "eu-qual"},
        "source": {"type": "csv", "mode": "copy", "path": "/tmp/input.csv"},
        "document": document,
        "metadata_fields": [
            {"name": "committee", "source_field": "committee_code"},
            {"name": "time_period", "source_field": "period"},
        ],
        "filters": [],
        "tie_schema": [{"label": "Policy", "description": "Policy tie"}],
    }
