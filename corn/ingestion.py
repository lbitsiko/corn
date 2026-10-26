from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path
from typing import Any, Mapping

from corn.project_config import validate_project_config

CANONICAL_DOCUMENTS_TABLE = "documents"


def initialize_runtime_database(db_path: Path, config: Mapping[str, Any]) -> None:
    """Create the canonical runtime database schema for one project."""
    validated_config = validate_project_config(config)
    columns = _canonical_columns(validated_config)

    db_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(f"DROP TABLE IF EXISTS {CANONICAL_DOCUMENTS_TABLE}")
        column_sql = ", ".join(f'{_quote_identifier(name)} TEXT' for name in columns)
        connection.execute(f"CREATE TABLE {CANONICAL_DOCUMENTS_TABLE} ({column_sql})")
        connection.commit()
    finally:
        connection.close()


def ingest_csv_source(csv_path: Path, runtime_db_path: Path, config: Mapping[str, Any]) -> None:
    """Load CSV source rows into the canonical runtime database."""
    validated_config = validate_project_config(config)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
        _validate_source_fields(headers, validated_config)

        initialize_runtime_database(runtime_db_path, validated_config)
        rows = [_normalized_row_from_source_row(row, validated_config) for row in reader]

    _insert_rows(runtime_db_path, validated_config, rows)


def ingest_sqlite_source(
    source_db_path: Path,
    runtime_db_path: Path,
    config: Mapping[str, Any],
) -> None:
    """Load rows from a configured SQLite table or view into the runtime database."""
    validated_config = validate_project_config(config)
    table_or_view = validated_config["source"]["table_or_view"]

    connection = sqlite3.connect(source_db_path)
    connection.row_factory = sqlite3.Row
    try:
        _validate_table_or_view_exists(connection, table_or_view)
        rows = connection.execute(f"SELECT * FROM {_quote_identifier(table_or_view)}").fetchall()
        available_fields = list(rows[0].keys()) if rows else _fetch_column_names(connection, table_or_view)
        _validate_source_fields(available_fields, validated_config)

        initialize_runtime_database(runtime_db_path, validated_config)
        normalized_rows = [_normalized_row_from_source_row(dict(row), validated_config) for row in rows]
    finally:
        connection.close()

    _insert_rows(runtime_db_path, validated_config, normalized_rows)


def write_cache_metadata(metadata_path: Path, source_path: Path) -> None:
    """Persist source snapshot metadata for cache freshness checks."""
    snapshot = _source_snapshot(source_path)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(snapshot, sort_keys=True), encoding="utf-8")


def _canonical_columns(config: Mapping[str, Any]) -> list[str]:
    columns = ["document_id", "document_text"]
    if "title_field" in config["document"]:
        columns.append("document_title")
    columns.extend(field["name"] for field in config["metadata_fields"])
    return columns


def _insert_rows(
    runtime_db_path: Path,
    config: Mapping[str, Any],
    rows: list[dict[str, str | None]],
) -> None:
    columns = _canonical_columns(config)
    placeholders = ", ".join("?" for _ in columns)
    column_sql = ", ".join(_quote_identifier(column) for column in columns)
    values = [tuple(row[column] for column in columns) for row in rows]

    connection = sqlite3.connect(runtime_db_path)
    try:
        connection.executemany(
            f"INSERT INTO {CANONICAL_DOCUMENTS_TABLE} ({column_sql}) VALUES ({placeholders})",
            values,
        )
        connection.commit()
    finally:
        connection.close()


def _normalized_row_from_source_row(
    row: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, str | None]:
    normalized = {
        "document_id": _stringify_value(row[config["document"]["id_field"]]),
        "document_text": _stringify_value(row[config["document"]["text_field"]]),
    }
    title_field = config["document"].get("title_field")
    if title_field is not None:
        normalized["document_title"] = _stringify_value(row[title_field])
    for metadata_field in config["metadata_fields"]:
        normalized[metadata_field["name"]] = _stringify_value(row[metadata_field["source_field"]])
    return normalized


def _required_source_fields(config: Mapping[str, Any]) -> list[str]:
    fields = [
        config["document"]["id_field"],
        config["document"]["text_field"],
    ]
    title_field = config["document"].get("title_field")
    if title_field is not None:
        fields.append(title_field)
    fields.extend(field["source_field"] for field in config["metadata_fields"])
    return fields


def _validate_source_fields(available_fields: list[str], config: Mapping[str, Any]) -> None:
    missing_fields = sorted(set(_required_source_fields(config)) - set(available_fields))
    if missing_fields:
        raise ValueError(f"Missing source fields: {', '.join(missing_fields)}")


def _validate_table_or_view_exists(connection: sqlite3.Connection, table_or_view: str) -> None:
    row = connection.execute(
        """
        SELECT name
        FROM sqlite_master
        WHERE type IN ('table', 'view') AND name = ?
        """,
        (table_or_view,),
    ).fetchone()
    if row is None:
        raise ValueError(f"Missing SQLite table or view: {table_or_view}")


def _fetch_column_names(connection: sqlite3.Connection, table_or_view: str) -> list[str]:
    rows = connection.execute(f"PRAGMA table_info({_quote_identifier(table_or_view)})").fetchall()
    return [row[1] for row in rows]


def _source_snapshot(source_path: Path) -> dict[str, int]:
    stat = source_path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _stringify_value(value: Any) -> str | None:
    if value is None:
        return None
    return str(value)


def _quote_identifier(identifier: str) -> str:
    escaped_identifier = identifier.replace('"', '""')
    return f'"{escaped_identifier}"'
