from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

import pytest

from corn.ingestion import CANONICAL_DOCUMENTS_TABLE, ingest_csv_source


def test_ingest_csv_source_normalizes_rows_into_runtime_database(tmp_path: Path) -> None:
    csv_path = tmp_path / "documents.csv"
    _write_csv(
        csv_path,
        [
            {"doc_id": "doc-1", "body": "Alpha text", "title": "Alpha", "committee_code": "AFET"},
            {"doc_id": "doc-2", "body": "Beta text", "title": "Beta", "committee_code": "IMCO"},
        ],
    )
    runtime_db_path = tmp_path / "runtime.sqlite"

    ingest_csv_source(csv_path, runtime_db_path, _csv_config())

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


def test_ingest_csv_source_rejects_missing_required_headers(tmp_path: Path) -> None:
    csv_path = tmp_path / "documents.csv"
    _write_csv(csv_path, [{"doc_id": "doc-1", "title": "Alpha"}])

    with pytest.raises(ValueError, match="Missing source fields"):
        ingest_csv_source(csv_path, tmp_path / "runtime.sqlite", _csv_config())


def test_ingest_csv_source_rejects_invalid_metadata_field_mapping(tmp_path: Path) -> None:
    csv_path = tmp_path / "documents.csv"
    _write_csv(
        csv_path,
        [{"doc_id": "doc-1", "body": "Alpha text", "title": "Alpha", "wrong_header": "AFET"}],
    )

    with pytest.raises(ValueError, match="Missing source fields"):
        ingest_csv_source(csv_path, tmp_path / "runtime.sqlite", _csv_config())


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _csv_config() -> dict[str, object]:
    return {
        "project": {"name": "EU Qual", "slug": "eu-qual"},
        "source": {"type": "csv", "mode": "copy", "path": "/tmp/input.csv"},
        "document": {"id_field": "doc_id", "text_field": "body", "title_field": "title"},
        "metadata_fields": [{"name": "committee", "source_field": "committee_code"}],
        "filters": [],
        "tie_schema": [{"label": "Policy", "description": "Policy tie"}],
    }
