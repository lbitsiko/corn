from __future__ import annotations

import csv
import io
import json
import sqlite3
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from corn.utils import ensure_directory, sanitize_filename


def initialize_annotations_db(db_path: Path) -> Path:
    """Create the on-disk annotations SQLite database if it does not exist."""
    ensure_directory(db_path.parent)
    with sqlite3.connect(db_path) as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS annotations (
                workspace_id TEXT NOT NULL,
                annotator_id TEXT NOT NULL,
                document_id TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                json_path TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                PRIMARY KEY (workspace_id, annotator_id, document_id)
            )
            """
        )
        connection.commit()
    return db_path


def _normalized_json_path(db_path: Path, json_path: Path) -> str:
    try:
        return json_path.relative_to(db_path.parent).as_posix()
    except ValueError:
        return json_path.as_posix()


def save_annotation_record(db_path: Path, payload: dict, json_path: Path) -> None:
    """Insert or update an annotation payload in the SQLite index."""
    initialize_annotations_db(db_path)
    json_path_value = _normalized_json_path(db_path, json_path)

    with sqlite3.connect(db_path) as connection:
        existing_row = connection.execute(
            """
            SELECT payload_json
            FROM annotations
            WHERE workspace_id = ? AND annotator_id = ? AND document_id = ?
            """,
            (
                payload["workspace_id"],
                payload["annotator_id"],
                payload["document_id"],
            ),
        ).fetchone()
        if existing_row is not None:
            existing_payload = json.loads(existing_row[0])
            payload["created_at"] = existing_payload["created_at"]

        connection.execute(
            """
            INSERT INTO annotations (
                workspace_id,
                annotator_id,
                document_id,
                created_at,
                updated_at,
                json_path,
                payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(workspace_id, annotator_id, document_id) DO UPDATE SET
                created_at = excluded.created_at,
                updated_at = excluded.updated_at,
                json_path = excluded.json_path,
                payload_json = excluded.payload_json
            """,
            (
                payload["workspace_id"],
                payload["annotator_id"],
                payload["document_id"],
                payload["created_at"],
                payload["updated_at"],
                json_path_value,
                json.dumps(payload),
            ),
        )
        connection.commit()


def _row_to_record(row: tuple) -> dict:
    payload = json.loads(row[6])
    return {
        "workspace_id": row[0],
        "annotator_id": row[1],
        "document_id": row[2],
        "created_at": row[3],
        "updated_at": row[4],
        "json_path": row[5],
        "payload": payload,
    }


def get_annotation(
    db_path: Path,
    workspace_id: str,
    annotator_id: str,
    document_id: str,
) -> dict | None:
    """Fetch a single annotation record by its scoped identity."""
    initialize_annotations_db(db_path)
    with sqlite3.connect(db_path) as connection:
        row = connection.execute(
            """
            SELECT
                workspace_id,
                annotator_id,
                document_id,
                created_at,
                updated_at,
                json_path,
                payload_json
            FROM annotations
            WHERE workspace_id = ? AND annotator_id = ? AND document_id = ?
            """,
            (workspace_id, annotator_id, document_id),
        ).fetchone()
    if row is None:
        return None
    return _row_to_record(row)


def has_annotation(
    db_path: Path,
    workspace_id: str,
    annotator_id: str,
    document_id: str,
) -> bool:
    """Return whether a scoped annotation exists."""
    return get_annotation(
        db_path=db_path,
        workspace_id=workspace_id,
        annotator_id=annotator_id,
        document_id=document_id,
    ) is not None


def list_annotations(
    db_path: Path,
    annotator_id: str | None = None,
    workspace_id: str | None = None,
    annotation_source: str | None = None,
) -> list[dict]:
    """List saved annotations with optional annotator, workspace, or source filters."""
    initialize_annotations_db(db_path)
    clauses: list[str] = []
    parameters: list[str] = []
    if annotator_id:
        clauses.append("annotator_id = ?")
        parameters.append(annotator_id)
    if workspace_id:
        clauses.append("workspace_id = ?")
        parameters.append(workspace_id)
    if annotation_source:
        clauses.append("json_extract(payload_json, '$.annotation_source') = ?")
        parameters.append(annotation_source)

    query = """
        SELECT
            workspace_id,
            annotator_id,
            document_id,
            created_at,
            updated_at,
            json_path,
            payload_json
        FROM annotations
    """
    if clauses:
        query = f"{query} WHERE {' AND '.join(clauses)}"
    query = f"{query} ORDER BY workspace_id, annotator_id, document_id"

    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(query, parameters).fetchall()
    return [_row_to_record(row) for row in rows]


def list_annotation_scopes(db_path: Path) -> list[tuple[str, str]]:
    """List distinct (workspace_id, annotator_id) pairs present in the index."""
    initialize_annotations_db(db_path)
    with sqlite3.connect(db_path) as connection:
        rows = connection.execute(
            """
            SELECT DISTINCT workspace_id, annotator_id
            FROM annotations
            ORDER BY workspace_id, annotator_id
            """
        ).fetchall()
    return [(row[0], row[1]) for row in rows]


def export_annotations_edge_list_csv(records: list[dict]) -> bytes:
    """Export annotation ties as a flat edge-list CSV."""
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=[
            "workspace_id",
            "annotator_id",
            "document_id",
            "source",
            "target",
            "annotation_source",
            "tie_type",
            "reasoning_text",
            "reasoning_text_span",
            "reasoning_text_note",
            "reasoning_start",
            "reasoning_end",
            "normalized_annotation",
            "created_at",
            "updated_at",
        ],
    )
    writer.writeheader()
    for record in records:
        payload = record["payload"]
        annotation_source = str(payload.get("annotation_source", "human") or "human")
        exported_annotation_source = annotation_source.replace("_", "-")
        for entry in payload.get("network_data", []):
            for tie in entry.get("ties", []):
                normalized_annotation = {
                    "annotation_source": exported_annotation_source,
                    "source": entry.get("source", ""),
                    "target": entry.get("target", ""),
                    "tie": {
                        "type": tie.get("type", ""),
                        "reasoning_text": tie.get("reasoning_text", ""),
                        "reasoning_text_span": tie.get("reasoning_text_span", ""),
                        "reasoning_text_note": tie.get("reasoning_text_note", ""),
                        "reasoning_start": tie.get("reasoning_start", ""),
                        "reasoning_end": tie.get("reasoning_end", ""),
                    },
                }
                writer.writerow(
                    {
                        "workspace_id": record["workspace_id"],
                        "annotator_id": record["annotator_id"],
                        "document_id": record["document_id"],
                        "source": entry.get("source", ""),
                        "target": entry.get("target", ""),
                        "annotation_source": exported_annotation_source,
                        "tie_type": tie.get("type", ""),
                        "reasoning_text": tie.get("reasoning_text", ""),
                        "reasoning_text_span": tie.get("reasoning_text_span", ""),
                        "reasoning_text_note": tie.get("reasoning_text_note", ""),
                        "reasoning_start": tie.get("reasoning_start", ""),
                        "reasoning_end": tie.get("reasoning_end", ""),
                        "normalized_annotation": json.dumps(
                            normalized_annotation,
                            ensure_ascii=True,
                            sort_keys=True,
                        ),
                        "created_at": record["created_at"],
                        "updated_at": record["updated_at"],
                    }
                )
    return output.getvalue().encode("utf-8")


def export_annotations_json_archive(records: list[dict]) -> bytes:
    """Export annotation payloads as a ZIP archive of JSON files."""
    buffer = io.BytesIO()
    with ZipFile(buffer, mode="w", compression=ZIP_DEFLATED) as archive:
        for record in records:
            json_path = str(record.get("json_path") or "").strip()
            if not json_path:
                json_path = (
                    Path(sanitize_filename(str(record.get("workspace_id", "annotations"))))
                    / sanitize_filename(str(record.get("annotator_id", "unknown")))
                    / f"{sanitize_filename(str(record.get('document_id', 'annotation')))}.json"
                ).as_posix()
            archive.writestr(
                json_path,
                json.dumps(record["payload"], indent=2),
            )
    return buffer.getvalue()
