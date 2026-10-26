from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from corn.ingestion import CANONICAL_DOCUMENTS_TABLE, ingest_sqlite_source


def test_ingest_sqlite_source_normalizes_rows_from_configured_view(tmp_path: Path) -> None:
    source_db_path = tmp_path / "source.sqlite"
    _create_source_database(source_db_path)
    runtime_db_path = tmp_path / "runtime.sqlite"

    ingest_sqlite_source(source_db_path, runtime_db_path, _sqlite_config(table_or_view="documents_view"))

    connection = sqlite3.connect(runtime_db_path)
    try:
        rows = connection.execute(
            f"""
            SELECT document_id, document_text, document_title, committee
            FROM {CANONICAL_DOCUMENTS_TABLE}
            ORDER BY document_id
            """
        ).fetchall()
    finally:
        connection.close()

    assert rows == [
        ("doc-1", "Alpha text", "Alpha", "AFET"),
        ("doc-2", "Beta text", "Beta", "IMCO"),
    ]


def test_ingest_sqlite_source_rejects_missing_table_or_view(tmp_path: Path) -> None:
    source_db_path = tmp_path / "source.sqlite"
    _create_source_database(source_db_path)

    with pytest.raises(ValueError, match="Missing SQLite table or view"):
        ingest_sqlite_source(
            source_db_path,
            tmp_path / "runtime.sqlite",
            _sqlite_config(table_or_view="missing_view"),
        )


def test_ingest_sqlite_source_rejects_missing_mapped_columns(tmp_path: Path) -> None:
    source_db_path = tmp_path / "source.sqlite"
    _create_source_database(source_db_path)

    with pytest.raises(ValueError, match="Missing source fields"):
        ingest_sqlite_source(
            source_db_path,
            tmp_path / "runtime.sqlite",
            _sqlite_config(table_or_view="documents_view", text_field="missing_body"),
        )


def _create_source_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE source_documents (
                doc_id TEXT NOT NULL,
                body TEXT NOT NULL,
                title TEXT,
                committee_code TEXT
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO source_documents (doc_id, body, title, committee_code)
            VALUES (?, ?, ?, ?)
            """,
            [
                ("doc-1", "Alpha text", "Alpha", "AFET"),
                ("doc-2", "Beta text", "Beta", "IMCO"),
            ],
        )
        connection.execute("CREATE VIEW documents_view AS SELECT * FROM source_documents")
        connection.commit()
    finally:
        connection.close()


def _sqlite_config(*, table_or_view: str, text_field: str = "body") -> dict[str, object]:
    return {
        "project": {"name": "EU Qual", "slug": "eu-qual"},
        "source": {
            "type": "sqlite",
            "mode": "reference",
            "path": "/tmp/source.sqlite",
            "table_or_view": table_or_view,
        },
        "document": {"id_field": "doc_id", "text_field": text_field, "title_field": "title"},
        "metadata_fields": [{"name": "committee", "source_field": "committee_code"}],
        "filters": [],
        "tie_schema": [{"label": "Policy", "description": "Policy tie"}],
    }
