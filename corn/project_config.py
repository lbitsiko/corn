from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import yaml

from corn.source_inspection import validate_project_slug

VALID_FILTER_TYPES = {"select", "multi_select", "text_contains", "boolean"}
VALID_SOURCE_TYPES = {"csv", "sqlite"}
VALID_SOURCE_MODES = {"copy", "reference"}
VALID_TIE_ATTRIBUTE_TYPES = {"categorical", "numeric", "temporal"}

ProjectConfig = dict[str, Any]


def load_project_config(config_path: Path) -> ProjectConfig:
    """Load and validate a project config from JSON or YAML."""
    return load_project_config_content(
        filename=config_path.name,
        raw_content=config_path.read_text(encoding="utf-8"),
    )


def load_project_config_content(filename: str, raw_content: str | bytes) -> ProjectConfig:
    """Load and validate a project config from uploaded or in-memory content."""
    suffix = Path(filename).suffix.lower()
    text = raw_content.decode("utf-8") if isinstance(raw_content, bytes) else raw_content

    if suffix == ".json":
        content = json.loads(text)
    elif suffix in {".yaml", ".yml"}:
        content = yaml.safe_load(text)
    else:
        raise ValueError(f"Unsupported config format: {filename}")

    if not isinstance(content, dict):
        raise ValueError(f"Invalid config payload in {filename}")
    return validate_project_config(content)


def validate_project_config(config: Mapping[str, Any]) -> ProjectConfig:
    """Validate and normalize a project configuration mapping."""
    project = _validate_project(_require_mapping(config, "project"))
    source = _validate_source(_require_mapping(config, "source"))
    document = _validate_document(_require_mapping(config, "document"))
    metadata_fields = _validate_metadata_fields(config.get("metadata_fields", []))
    filters = _validate_filters(config.get("filters", []))
    tie_schema = _validate_tie_schema(config.get("tie_schema", []))
    tie_attribute_schema = _validate_tie_attribute_schema(config.get("tie_attribute_schema", []))

    return {
        "project": project,
        "source": source,
        "document": document,
        "metadata_fields": metadata_fields,
        "filters": filters,
        "tie_schema": tie_schema,
        "tie_attribute_schema": tie_attribute_schema,
    }


def _validate_project(project: Mapping[str, Any]) -> ProjectConfig:
    return {
        "name": _require_string(project, "name"),
        "slug": validate_project_slug(_require_string(project, "slug")),
    }


def _validate_source(source: Mapping[str, Any]) -> ProjectConfig:
    source_type = _require_string(source, "type")
    if source_type not in VALID_SOURCE_TYPES:
        raise ValueError(f"Invalid source type: {source_type}")

    source_mode = _require_string(source, "mode")
    if source_mode not in VALID_SOURCE_MODES:
        raise ValueError(f"Invalid source mode: {source_mode}")

    normalized = {
        "type": source_type,
        "mode": source_mode,
        "path": _require_string(source, "path"),
    }
    if source_type == "sqlite":
        normalized["table_or_view"] = _require_string(source, "table_or_view")

    return normalized


def _validate_document(document: Mapping[str, Any]) -> ProjectConfig:
    normalized = {
        "id_field": _require_string(document, "id_field"),
        "text_field": _require_string(document, "text_field"),
    }
    title_field = document.get("title_field")
    if title_field is not None:
        if not isinstance(title_field, str) or not title_field:
            raise ValueError("document.title_field must be a non-empty string when provided")
        normalized["title_field"] = title_field
    return normalized


def _validate_metadata_fields(metadata_fields: Any) -> list[ProjectConfig]:
    if not isinstance(metadata_fields, list):
        raise ValueError("metadata_fields must be a list")

    normalized: list[ProjectConfig] = []
    for field in metadata_fields:
        mapping = _coerce_mapping(field, "metadata field")
        normalized.append(
            {
                "name": _require_string(mapping, "name"),
                "source_field": _require_string(mapping, "source_field"),
            }
        )
    return normalized


def _validate_filters(filters: Any) -> list[ProjectConfig]:
    if not isinstance(filters, list):
        raise ValueError("filters must be a list")

    normalized: list[ProjectConfig] = []
    for filter_config in filters:
        mapping = _coerce_mapping(filter_config, "filter")
        filter_type = _require_string(mapping, "type")
        if filter_type not in VALID_FILTER_TYPES:
            raise ValueError(f"Invalid filter type: {filter_type}")
        normalized.append(
            {
                "field": _require_string(mapping, "field"),
                "label": _require_string(mapping, "label"),
                "type": filter_type,
            }
        )
    return normalized


def _validate_tie_schema(tie_schema: Any) -> list[ProjectConfig]:
    if not isinstance(tie_schema, list):
        raise ValueError("tie_schema must be a list")

    labels: set[str] = set()
    normalized: list[ProjectConfig] = []
    for tie in tie_schema:
        mapping = _coerce_mapping(tie, "tie schema entry")
        label = _require_string(mapping, "label")
        if label in labels:
            raise ValueError(f"Duplicate tie label: {label}")
        labels.add(label)
        normalized.append(
            {
                "label": label,
                "description": _require_string(mapping, "description"),
            }
        )
    return normalized


def _validate_tie_attribute_schema(tie_attribute_schema: Any) -> list[ProjectConfig]:
    if not isinstance(tie_attribute_schema, list):
        raise ValueError("tie_attribute_schema must be a list")

    names: set[str] = set()
    normalized: list[ProjectConfig] = []
    for attribute in tie_attribute_schema:
        mapping = _coerce_mapping(attribute, "tie attribute schema entry")
        name = _require_string(mapping, "name")
        if name in names:
            raise ValueError(f"Duplicate tie attribute name: {name}")
        names.add(name)
        attribute_type = _require_string(mapping, "type")
        if attribute_type not in VALID_TIE_ATTRIBUTE_TYPES:
            raise ValueError(f"Invalid tie attribute type: {attribute_type}")
        entry: ProjectConfig = {
            "name": name,
            "type": attribute_type,
            "description": _require_string(mapping, "description"),
        }
        if attribute_type == "categorical":
            entry["options"] = _validate_attribute_options(name, mapping.get("options"))
        elif attribute_type == "numeric":
            entry["range"] = _validate_attribute_range(name, mapping.get("range"))
        normalized.append(entry)
    return normalized


def _validate_attribute_options(name: str, options: Any) -> list[str]:
    if not isinstance(options, list) or not options:
        raise ValueError(f"Tie attribute {name} requires a non-empty options list")
    normalized: list[str] = []
    for option in options:
        if not isinstance(option, str) or not option:
            raise ValueError(f"Tie attribute {name} options must be non-empty strings")
        if option == "(unset)":
            # "(unset)" is the annotation editor's reserved no-value sentinel.
            raise ValueError(
                f'Tie attribute {name} options must not use the reserved label "(unset)"'
            )
        if option in normalized:
            raise ValueError(f"Duplicate option for tie attribute {name}: {option}")
        normalized.append(option)
    return normalized


def _validate_attribute_range(name: str, range_value: Any) -> list[float]:
    if (
        not isinstance(range_value, list)
        or len(range_value) != 2
        or any(
            isinstance(bound, bool)
            or not isinstance(bound, (int, float))
            or math.isnan(bound)
            for bound in range_value
        )
    ):
        raise ValueError(f"Tie attribute {name} requires range [min, max]")
    minimum, maximum = float(range_value[0]), float(range_value[1])
    if minimum >= maximum:
        raise ValueError(f"Tie attribute {name} range must have min < max")
    return [minimum, maximum]


def _require_mapping(container: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    value = container.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{key} must be a mapping")
    return value


def _coerce_mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping")
    return value


def _require_string(container: Mapping[str, Any], key: str) -> str:
    value = container.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"{key} must be a non-empty string")
    return value
