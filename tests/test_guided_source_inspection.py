from __future__ import annotations

import csv
import sqlite3
from pathlib import Path

from corn.source_inspection import (
    discover_source_files,
    inspect_csv_columns,
    inspect_sqlite_columns,
    inspect_sqlite_tables,
)


def test_discover_source_files_returns_supported_server_side_sources(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    ignored_root = data_root / "projects"
    included_csv = data_root / "toy.csv"
    included_sqlite = data_root / "warehouse.sqlite"
    ignored_sqlite = ignored_root / "derived" / "runtime.sqlite"
    ignored_file = data_root / "annotations.sqlite"

    included_csv.parent.mkdir(parents=True, exist_ok=True)
    ignored_sqlite.parent.mkdir(parents=True, exist_ok=True)
    included_csv.write_text("id,text\n1,hello\n", encoding="utf-8")
    included_sqlite.write_bytes(b"sqlite")
    ignored_sqlite.write_bytes(b"sqlite")
    ignored_file.write_bytes(b"sqlite")

    discovered = discover_source_files(
        [data_root],
        ignored_roots=[ignored_root],
        ignored_files=[ignored_file],
    )

    assert discovered == [included_csv.resolve(), included_sqlite.resolve()]


def test_inspect_csv_columns_reads_header_row(tmp_path: Path) -> None:
    csv_path = tmp_path / "documents.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["doc_id", "body", "topic"])
        writer.writeheader()
        writer.writerow({"doc_id": "1", "body": "hello", "topic": "policy"})

    assert inspect_csv_columns(csv_path) == ["doc_id", "body", "topic"]


def test_inspect_sqlite_tables_and_columns_reads_available_schema(tmp_path: Path) -> None:
    db_path = tmp_path / "source.sqlite"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute("CREATE TABLE documents (doc_id TEXT, body TEXT, topic TEXT)")
        connection.execute("CREATE VIEW documents_view AS SELECT doc_id, body FROM documents")
        connection.commit()
    finally:
        connection.close()

    assert inspect_sqlite_tables(db_path) == ["documents", "documents_view"]
    assert inspect_sqlite_columns(db_path, "documents") == ["doc_id", "body", "topic"]
