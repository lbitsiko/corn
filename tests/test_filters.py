from __future__ import annotations

import sqlite3
from pathlib import Path

from corn.querying import build_filter_clauses, fetch_project_filter_options, query_project_documents


def test_build_filter_clauses_supports_select_multi_select_text_and_boolean() -> None:
    schema = [
        {"field": "sentiment", "label": "Sentiment", "type": "select"},
        {"field": "topic", "label": "Topic", "type": "multi_select"},
        {"field": "content_hint", "label": "Hint", "type": "text_contains"},
        {"field": "is_reviewed", "label": "Reviewed", "type": "boolean"},
    ]
    values = {
        "sentiment": "positive",
        "topic": ["care", "trust"],
        "content_hint": "Alpha",
        "is_reviewed": True,
    }

    clauses, parameters = build_filter_clauses(schema, values)

    assert clauses == [
        '"sentiment" = ?',
        '"topic" IN (?, ?)',
        'LOWER("content_hint") LIKE ?',
        '"is_reviewed" = ?',
    ]
    assert parameters == ["positive", "care", "trust", "%alpha%", "1"]


def test_fetch_project_filter_options_returns_distinct_values_from_schema_fields(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime.sqlite"
    _create_runtime_db(db_path)

    options = fetch_project_filter_options(
        db_path,
        [
            {"field": "sentiment", "label": "Sentiment", "type": "select"},
            {"field": "topic", "label": "Topic", "type": "multi_select"},
            {"field": "content_hint", "label": "Hint", "type": "text_contains"},
        ],
    )

    assert options == {
        "sentiment": ["negative", "positive"],
        "topic": ["care", "trust"],
    }


def test_query_project_documents_applies_dynamic_filters(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime.sqlite"
    _create_runtime_db(db_path)

    rows = query_project_documents(
        runtime_db_path=db_path,
        filter_schema=[
            {"field": "sentiment", "label": "Sentiment", "type": "select"},
            {"field": "topic", "label": "Topic", "type": "multi_select"},
            {"field": "content_hint", "label": "Hint", "type": "text_contains"},
        ],
        filter_values={
            "sentiment": "positive",
            "topic": ["trust"],
            "content_hint": "Alpha",
        },
        sample_size=0,
    )

    assert [row["link"] for row in rows] == ["doc-1"]


def test_query_project_documents_handles_runtime_without_document_title(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime.sqlite"
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(
            """
            CREATE TABLE documents (
                document_id TEXT NOT NULL,
                document_text TEXT NOT NULL,
                sentiment TEXT
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO documents (document_id, document_text, sentiment)
            VALUES (?, ?, ?)
            """,
            [
                ("doc-1", "Alpha text", "positive"),
                ("doc-2", "Beta text", "negative"),
            ],
        )
        connection.commit()
    finally:
        connection.close()

    rows = query_project_documents(
        runtime_db_path=db_path,
        filter_schema=[{"field": "sentiment", "label": "Sentiment", "type": "select"}],
        filter_values={"sentiment": "positive"},
        sample_size=0,
    )

    assert rows == [
        {
            "id": 1,
            "link": "doc-1",
            "title": None,
            "content": "Alpha text",
            "document_id": "doc-1",
            "document_text": "Alpha text",
            "sentiment": "positive",
        }
    ]


def test_query_project_documents_filters_by_multiple_exact_document_ids(tmp_path: Path) -> None:
    db_path = tmp_path / "runtime.sqlite"
    _create_runtime_db(db_path)

    rows = query_project_documents(
        runtime_db_path=db_path,
        filter_schema=[],
        filter_values={},
        sample_size=0,
        document_id_filter="doc-2\ndoc-1\nmissing-doc",
    )

    assert [row["link"] for row in rows] == ["doc-1", "doc-2"]


def _create_runtime_db(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(
            """
            CREATE TABLE documents (
                document_id TEXT NOT NULL,
                document_text TEXT NOT NULL,
                document_title TEXT,
                sentiment TEXT,
                topic TEXT,
                content_hint TEXT,
                is_reviewed TEXT
            )
            """
        )
        connection.executemany(
            """
            INSERT INTO documents (
                document_id,
                document_text,
                document_title,
                sentiment,
                topic,
                content_hint,
                is_reviewed
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            [
                ("doc-1", "Alpha trust text", "Doc 1", "positive", "trust", "Alpha", "1"),
                ("doc-2", "Beta care text", "Doc 2", "negative", "care", "Beta", "0"),
            ],
        )
        connection.commit()
    finally:
        connection.close()
