from __future__ import annotations

import sqlite3
from pathlib import Path

from corn.attribute_validation import validate_tie_attributes
from corn.storage import get_annotation, save_annotation_record
from corn.annotations import save_annotation
from corn.utils import sanitize_filename


def export_document_annotation(
    annotations_db_path: Path,
    workspace_id: str,
    annotator_id: str,
    document_id: str,
) -> dict:
    """Export a single scoped annotation payload from the project-local index."""
    record = get_annotation(
        db_path=annotations_db_path,
        workspace_id=workspace_id,
        annotator_id=annotator_id,
        document_id=document_id,
    )
    if record is None:
        raise ValueError(f"No annotation found for document: {document_id}")
    return record["payload"]


def import_document_annotation(
    runtime_db_path: Path,
    annotations_dir: Path,
    annotations_db_path: Path,
    payload: dict,
    valid_tie_labels: set[str],
    conflict_mode: str,
    tie_attribute_schema: list[dict] | None = None,
) -> dict:
    """Import a single annotation payload into the active project with strict validation."""
    if conflict_mode not in {"replace", "abort"}:
        raise ValueError(f"Unsupported conflict mode: {conflict_mode}")

    document_id = str(payload["document_id"])
    if not _document_exists(runtime_db_path, document_id):
        raise ValueError(f"Document ID not found in target project: {document_id}")

    imported_tie_labels = _payload_tie_labels(payload)
    unknown_labels = sorted(imported_tie_labels - valid_tie_labels)
    if unknown_labels:
        raise ValueError(f"Unknown tie labels for target project: {', '.join(unknown_labels)}")

    # Gate validity only: imported payloads are stored verbatim, never normalized.
    for entry in payload.get("network_data", []):
        for tie in entry.get("ties", []):
            try:
                validate_tie_attributes(
                    tie.get("attributes"),
                    tie_attribute_schema=tie_attribute_schema or [],
                )
            except ValueError as error:
                raise ValueError(
                    f"{error} (tie: {entry.get('source')} → {entry.get('target')})"
                ) from error

    existing = get_annotation(
        db_path=annotations_db_path,
        workspace_id=payload["workspace_id"],
        annotator_id=payload["annotator_id"],
        document_id=document_id,
    )
    if existing is not None and conflict_mode == "abort":
        raise ValueError(f"Annotation conflict for document: {document_id}")

    relative_path = (
        Path(sanitize_filename(payload["workspace_id"]))
        / sanitize_filename(payload["annotator_id"])
        / f"{sanitize_filename(document_id)}.json"
    )
    output_path = save_annotation(
        output_dir=annotations_dir,
        relative_path=relative_path,
        payload=payload,
    )
    save_annotation_record(
        db_path=annotations_db_path,
        payload=payload,
        json_path=output_path,
    )
    return payload


def _document_exists(runtime_db_path: Path, document_id: str) -> bool:
    with sqlite3.connect(runtime_db_path) as connection:
        row = connection.execute(
            "SELECT 1 FROM documents WHERE document_id = ? LIMIT 1",
            (document_id,),
        ).fetchone()
    return row is not None


def _payload_tie_labels(payload: dict) -> set[str]:
    labels: set[str] = set()
    for entry in payload.get("network_data", []):
        for tie in entry.get("ties", []):
            tie_type = tie.get("type")
            if tie_type:
                labels.add(str(tie_type))
    return labels
