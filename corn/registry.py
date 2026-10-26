from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

RegistryEntry = dict[str, Any]


def load_registry(registry_path: Path) -> list[RegistryEntry]:
    """Load project registry entries from disk."""
    if not registry_path.exists():
        return []

    content = yaml.safe_load(registry_path.read_text(encoding="utf-8")) or {}
    projects = content.get("projects", [])
    if not isinstance(projects, list):
        raise ValueError(f"Invalid registry payload in {registry_path}")
    return [_normalize_entry(entry) for entry in projects]


def save_registry(registry_path: Path, entries: list[RegistryEntry]) -> None:
    """Persist project registry entries to disk."""
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"projects": [_normalize_entry(entry) for entry in entries]}
    registry_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def register_project(
    registry_path: Path,
    *,
    name: str,
    slug: str,
    project_path: Path,
) -> RegistryEntry:
    """Append a new project entry to the registry."""
    entries = load_registry(registry_path)
    if any(entry["slug"] == slug for entry in entries):
        raise ValueError(f"Project slug already registered: {slug}")

    entry = {
        "name": name,
        "slug": slug,
        "project_path": str(project_path),
        "archived": False,
    }
    entries.append(entry)
    save_registry(registry_path, entries)
    return entry


def archive_project(registry_path: Path, slug: str) -> RegistryEntry:
    """Mark a project as archived."""
    entries = load_registry(registry_path)
    for entry in entries:
        if entry["slug"] == slug:
            entry["archived"] = True
            save_registry(registry_path, entries)
            return entry
    raise ValueError(f"Unknown project slug: {slug}")


def delete_project(registry_path: Path, slug: str) -> None:
    """Remove a project entry from the registry."""
    entries = load_registry(registry_path)
    filtered_entries = [entry for entry in entries if entry["slug"] != slug]
    if len(filtered_entries) == len(entries):
        raise ValueError(f"Unknown project slug: {slug}")
    save_registry(registry_path, filtered_entries)


def list_projects(registry_path: Path, *, include_archived: bool = False) -> list[RegistryEntry]:
    """Return registry entries, optionally including archived projects."""
    entries = load_registry(registry_path)
    if include_archived:
        return entries
    return [entry for entry in entries if not entry["archived"]]


def _normalize_entry(entry: RegistryEntry) -> RegistryEntry:
    """Validate and normalize an individual registry entry."""
    if not isinstance(entry, dict):
        raise ValueError("Registry entries must be mappings")

    try:
        name = entry["name"]
        slug = entry["slug"]
        project_path = entry["project_path"]
    except KeyError as exc:
        raise ValueError(f"Missing registry field: {exc.args[0]}") from exc

    archived = entry.get("archived", False)
    if not isinstance(name, str) or not isinstance(slug, str) or not isinstance(project_path, str):
        raise ValueError("Registry name, slug, and project_path must be strings")
    if not isinstance(archived, bool):
        raise ValueError("Registry archived flag must be a boolean")

    return {
        "name": name,
        "slug": slug,
        "project_path": project_path,
        "archived": archived,
    }
