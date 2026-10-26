from __future__ import annotations

import csv
import re
import sqlite3
from pathlib import Path
from typing import Any, Iterable, Mapping

SAFE_SLUG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9_-]*$")
DISCOVERABLE_SOURCE_SUFFIXES = {".csv", ".sqlite", ".db"}


def validate_project_slug(slug: str) -> str:
    """Validate that a project slug is safe to use as a folder name."""
    normalized = slug.strip()
    if not normalized:
        raise ValueError("Project slug is required")
    if normalized in {".", ".."}:
        raise ValueError("Project slug cannot be . or ..")
    if not SAFE_SLUG_PATTERN.fullmatch(normalized):
        raise ValueError(
            "Project slug must use lowercase letters, numbers, hyphens, or underscores"
        )
    return normalized


def discover_source_files(
    search_roots: Iterable[Path],
    *,
    ignored_roots: Iterable[Path] = (),
    ignored_files: Iterable[Path] = (),
) -> list[Path]:
    """Return discoverable CSV/SQLite files under the given roots."""
    ignored = {path.resolve() for path in ignored_roots}
    ignored_file_paths = {path.resolve() for path in ignored_files}
    discovered: list[Path] = []
    seen: set[Path] = set()

    for root in search_roots:
        if not root.exists():
            continue
        resolved_root = root.resolve()
        if resolved_root in ignored:
            continue

        for path in root.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in DISCOVERABLE_SOURCE_SUFFIXES:
                continue
            resolved_path = path.resolve()
            if resolved_path in ignored_file_paths:
                continue
            if any(parent == resolved_path or parent in resolved_path.parents for parent in ignored):
                continue
            if resolved_path in seen:
                continue
            seen.add(resolved_path)
            discovered.append(resolved_path)

    return sorted(discovered, key=lambda path: path.as_posix())


def inspect_csv_columns(csv_path: Path) -> list[str]:
    """Read the header row from a CSV file."""
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        headers = reader.fieldnames or []
    if not headers:
        raise ValueError(f"CSV has no header row: {csv_path}")
    return [header for header in headers if header]


def inspect_sqlite_tables(source_db_path: Path) -> list[str]:
    """List available SQLite tables and views."""
    connection = sqlite3.connect(source_db_path)
    try:
        rows = connection.execute(
            """
            SELECT name
            FROM sqlite_master
            WHERE type IN ('table', 'view')
              AND name NOT LIKE 'sqlite_%'
            ORDER BY name
            """
        ).fetchall()
    finally:
        connection.close()

    return [row[0] for row in rows]


def inspect_sqlite_columns(source_db_path: Path, table_or_view: str) -> list[str]:
    """Return column names for a SQLite table or view."""
    connection = sqlite3.connect(source_db_path)
    try:
        rows = connection.execute(
            f'PRAGMA table_info("{table_or_view.replace(chr(34), chr(34) * 2)}")'
        ).fetchall()
    finally:
        connection.close()

    if not rows:
        raise ValueError(f"Missing SQLite table or view: {table_or_view}")
    return [row[1] for row in rows]


def build_guided_project_config(
    *,
    project_name: str,
    project_slug: str,
    source_type: str,
    source_mode: str,
    source_path: str,
    id_field: str,
    text_field: str,
    title_field: str | None,
    metadata_fields: list[str],
    filter_types_by_field: Mapping[str, str],
    tie_schema: list[Mapping[str, str]],
    table_or_view: str | None = None,
    tie_attribute_schema: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build a canonical project config from guided wizard selections."""
    config: dict[str, Any] = {
        "project": {
            "name": project_name.strip(),
            "slug": validate_project_slug(project_slug),
        },
        "source": {
            "type": source_type,
            "mode": source_mode,
            "path": source_path,
        },
        "document": {
            "id_field": id_field,
            "text_field": text_field,
        },
        "metadata_fields": [
            {
                "name": field_name,
                "source_field": field_name,
            }
            for field_name in metadata_fields
        ],
        "filters": [
            {
                "field": field_name,
                "label": _humanize_field_label(field_name),
                "type": filter_types_by_field[field_name],
            }
            for field_name in metadata_fields
            if field_name in filter_types_by_field
        ],
        "tie_schema": [
            {
                "label": item["label"].strip(),
                "description": item["description"].strip(),
            }
            for item in tie_schema
        ],
        "tie_attribute_schema": [dict(item) for item in tie_attribute_schema or []],
    }
    if title_field:
        config["document"]["title_field"] = title_field
    if source_type == "sqlite":
        if not table_or_view:
            raise ValueError("SQLite guided setup requires table_or_view")
        config["source"]["table_or_view"] = table_or_view
    return config


def _humanize_field_label(field_name: str) -> str:
    return field_name.replace("_", " ").strip().title()
