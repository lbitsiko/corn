from __future__ import annotations

import random
import sqlite3
from pathlib import Path
from typing import Any

from corn.llm_storage import get_llm_annotated_document_ids
from corn.storage import has_annotation


def _row_to_dict(cursor: sqlite3.Cursor, row: tuple) -> dict:
    return {description[0]: value for description, value in zip(cursor.description, row)}


def fetch_project_filter_options(
    runtime_db_path: Path,
    filter_schema: list[dict[str, str]],
) -> dict[str, list[str]]:
    """Return distinct filter values for project-configured select-style filters."""
    options: dict[str, list[str]] = {}
    with sqlite3.connect(runtime_db_path) as connection:
        for filter_config in filter_schema:
            if filter_config["type"] not in {"select", "multi_select"}:
                continue
            field = filter_config["field"]
            rows = connection.execute(
                f"""
                SELECT DISTINCT {_quote_identifier(field)}
                FROM documents
                WHERE {_quote_identifier(field)} IS NOT NULL AND {_quote_identifier(field)} != ''
                ORDER BY {_quote_identifier(field)}
                """
            ).fetchall()
            options[field] = [row[0] for row in rows]
    return options


def build_filter_clauses(
    filter_schema: list[dict[str, str]],
    filter_values: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Translate dynamic filter schema and values into SQL clauses."""
    clauses: list[str] = []
    parameters: list[str] = []

    for filter_config in filter_schema:
        field = filter_config["field"]
        filter_type = filter_config["type"]
        value = filter_values.get(field)

        if filter_type == "select":
            if value:
                clauses.append(f'{_quote_identifier(field)} = ?')
                parameters.append(str(value))
        elif filter_type == "multi_select":
            if value:
                values = [str(item) for item in value if item]
                if values:
                    placeholders = ", ".join("?" for _ in values)
                    clauses.append(f'{_quote_identifier(field)} IN ({placeholders})')
                    parameters.extend(values)
        elif filter_type == "text_contains":
            if value:
                clauses.append(f'LOWER({_quote_identifier(field)}) LIKE ?')
                parameters.append(f"%{str(value).lower()}%")
        elif filter_type == "boolean":
            if value is not None:
                clauses.append(f'{_quote_identifier(field)} = ?')
                parameters.append("1" if bool(value) else "0")

    return clauses, parameters


def query_project_documents(
    runtime_db_path: Path,
    filter_schema: list[dict[str, str]],
    filter_values: dict[str, Any],
    sample_size: int | None,
    annotations_dir: Path | None = None,
    annotations_db_path: Path | None = None,
    annotator_id: str | None = None,
    workspace_id: str | None = None,
    annotation_status: str = "unannotated",
    document_id_filter: str | None = None,
    llm_annotation_status: str = "all",
    random_seed: int = 42,
) -> list[dict]:
    """Query a canonical project runtime database using dynamic filter schema."""
    clauses, parameters = build_filter_clauses(filter_schema, filter_values)
    if document_id_filter:
        doc_id_lines = [line.strip() for line in document_id_filter.splitlines() if line.strip()]
        if len(doc_id_lines) == 1:
            clauses.append("document_id = ?")
            parameters.append(doc_id_lines[0])
        elif len(doc_id_lines) > 1:
            placeholders = ",".join("?" * len(doc_id_lines))
            clauses.append(f"document_id IN ({placeholders})")
            parameters.extend(doc_id_lines)
    with sqlite3.connect(runtime_db_path) as connection:
        columns = _document_table_columns(connection)
        title_select = "document_title AS title" if "document_title" in columns else "NULL AS title"
        query = f"""
            SELECT
                rowid AS id,
                document_id AS link,
                {title_select},
                document_text AS content,
                *
            FROM documents
        """
        if clauses:
            query = f"{query} WHERE {' AND '.join(clauses)}"
        query = f"{query} ORDER BY id"
        cursor = connection.execute(query, parameters)
        rows = [_row_to_dict(cursor, row) for row in cursor.fetchall()]

    if (annotations_db_path or annotations_dir) and annotation_status in {"annotated", "unannotated"}:
        filtered_rows: list[dict] = []
        for row in rows:
            is_annotated = False
            if annotations_db_path and annotator_id and workspace_id:
                is_annotated = has_annotation(
                    db_path=annotations_db_path,
                    workspace_id=workspace_id,
                    annotator_id=annotator_id,
                    document_id=str(row["link"]),
                )
            elif annotations_dir:
                annotation_path = annotations_dir / f"{sanitize_filename(str(row['link']))}.json"
                is_annotated = annotation_path.exists()
            if annotation_status == "annotated" and is_annotated:
                filtered_rows.append(row)
            if annotation_status == "unannotated" and not is_annotated:
                filtered_rows.append(row)
        rows = filtered_rows

    if llm_annotation_status in {"has_llm", "no_llm"} and annotations_db_path:
        llm_doc_ids = get_llm_annotated_document_ids(annotations_db_path)
        if llm_annotation_status == "has_llm":
            rows = [r for r in rows if str(r["id"]) in llm_doc_ids]
        else:
            rows = [r for r in rows if str(r["id"]) not in llm_doc_ids]

    if sample_size and sample_size > 0 and sample_size < len(rows):
        return random.Random(random_seed).sample(rows, sample_size)
    return rows


def get_project_document_by_id(runtime_db_path: Path, internal_id: int) -> dict | None:
    """Fetch a single document row from the canonical runtime database."""
    with sqlite3.connect(runtime_db_path) as connection:
        columns = _document_table_columns(connection)
        title_select = "document_title AS title" if "document_title" in columns else "NULL AS title"
        cursor = connection.execute(
            f"""
            SELECT
                rowid AS id,
                document_id AS link,
                {title_select},
                document_text AS content,
                *
            FROM documents
            WHERE rowid = ?
            """,
            (internal_id,),
        )
        row = cursor.fetchone()
    if row is None:
        return None
    return _row_to_dict(cursor, row)


def _quote_identifier(identifier: str) -> str:
    escaped_identifier = identifier.replace('"', '""')
    return f'"{escaped_identifier}"'


def _document_table_columns(connection: sqlite3.Connection) -> set[str]:
    rows = connection.execute("PRAGMA table_info(documents)").fetchall()
    return {row[1] for row in rows}
