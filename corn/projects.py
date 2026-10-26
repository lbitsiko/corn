from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import yaml

from corn.ingestion import ingest_csv_source, ingest_sqlite_source, write_cache_metadata
from corn.project_config import ProjectConfig, load_project_config, validate_project_config
from corn.registry import register_project

PROJECT_CONFIG_FILENAME = "project.yaml"
DERIVED_DB_FILENAME = "runtime.sqlite"
CACHE_METADATA_FILENAME = "cache_metadata.json"
def create_project_structure(project_path: Path, config: Mapping[str, Any]) -> ProjectConfig:
    """Create a project directory and persist its normalized config."""
    normalized_config = validate_project_config(config)

    project_path.mkdir(parents=True, exist_ok=True)
    source_dir = _ensure_directory(project_path / "source")
    _ensure_directory(project_path / "derived")
    _ensure_directory(project_path / "annotations")

    stored_config = dict(normalized_config)
    stored_config["source"] = _prepare_source_config(source_dir, normalized_config["source"])
    _write_project_config(project_path / PROJECT_CONFIG_FILENAME, stored_config)
    return stored_config


def create_and_register_project(
    projects_root: Path,
    registry_path: Path,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a project directory from config and register it globally."""
    normalized_config = validate_project_config(config)
    slug = normalized_config["project"]["slug"]
    project_path = projects_root / slug
    if project_path.exists():
        raise ValueError(f"Project already exists: {slug}")

    create_project_structure(project_path, normalized_config)
    build_project_runtime(project_path)
    return register_project(
        registry_path=registry_path,
        name=normalized_config["project"]["name"],
        slug=slug,
        project_path=project_path,
    )


def resolve_source_path(project_path: Path, config: Mapping[str, Any]) -> Path:
    """Resolve a stored project source path against the project root when needed."""
    source = config.get("source")
    if not isinstance(source, Mapping):
        raise ValueError("source must be a mapping")

    source_path = source.get("path")
    if not isinstance(source_path, str) or not source_path:
        raise ValueError("source.path must be a non-empty string")

    candidate = Path(source_path)
    if candidate.is_absolute():
        return candidate
    return project_path / candidate


def get_project_status(project_path: Path) -> str:
    """Return the current lifecycle status for a project directory."""
    config_path = project_path / PROJECT_CONFIG_FILENAME
    try:
        config = load_project_config(config_path)
    except (FileNotFoundError, ValueError, json.JSONDecodeError, yaml.YAMLError):
        return "config_invalid"

    source_path = resolve_source_path(project_path, config)
    if not source_path.exists():
        return "source_missing"

    runtime_db_path = project_path / "derived" / DERIVED_DB_FILENAME
    cache_metadata_path = project_path / "derived" / CACHE_METADATA_FILENAME
    if not runtime_db_path.exists() or not cache_metadata_path.exists():
        return "cache_stale"

    try:
        cached_snapshot = json.loads(cache_metadata_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "cache_stale"

    current_snapshot = _source_snapshot(source_path)
    if cached_snapshot != current_snapshot:
        return "cache_stale"
    return "ready"


def build_project_runtime(project_path: Path) -> Path:
    """Build or rebuild the project-local runtime database from project config."""
    config = load_project_config(project_path / PROJECT_CONFIG_FILENAME)
    source_path = resolve_source_path(project_path, config)
    runtime_db_path = project_path / "derived" / DERIVED_DB_FILENAME

    if config["source"]["type"] == "csv":
        ingest_csv_source(source_path, runtime_db_path, config)
    elif config["source"]["type"] == "sqlite":
        ingest_sqlite_source(source_path, runtime_db_path, config)
    else:
        raise ValueError(f"Unsupported project source type: {config['source']['type']}")

    write_cache_metadata(project_path / "derived" / CACHE_METADATA_FILENAME, source_path)
    return runtime_db_path


def _prepare_source_config(source_dir: Path, source_config: Mapping[str, Any]) -> ProjectConfig:
    source_path = Path(source_config["path"])
    normalized = dict(source_config)

    if source_config["mode"] == "copy":
        if not source_path.exists():
            raise FileNotFoundError(source_path)
        copied_path = source_dir / source_path.name
        shutil.copy2(source_path, copied_path)
        normalized["path"] = copied_path.relative_to(source_dir.parent).as_posix()

    return normalized


def _write_project_config(config_path: Path, config: Mapping[str, Any]) -> None:
    config_path.write_text(yaml.safe_dump(dict(config), sort_keys=False), encoding="utf-8")


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _source_snapshot(source_path: Path) -> dict[str, int]:
    stat = source_path.stat()
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
