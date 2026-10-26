from __future__ import annotations

import json
import re
from typing import Any


SUPPORTED_PROMPT_PLACEHOLDERS = (
    "{{project_name}}",
    "{{document_id}}",
    "{{document_title}}",
    "{{document_text}}",
    "{{metadata_json}}",
    "{{tie_schema_json}}",
    "{{tie_attribute_schema_json}}",
    "{{annotation_schema_json}}",
)

PLACEHOLDER_PATTERN = re.compile(r"\{\{[^{}]+\}\}")


def build_prompt_context(
    *,
    project_name: str,
    document: dict[str, Any],
    tie_schema: list[dict[str, str]],
    tie_attribute_schema: list[dict[str, Any]] | None = None,
) -> dict[str, str]:
    metadata = {
        key: value
        for key, value in document.items()
        if key not in {"id", "link", "title", "document_id", "document_title", "content"}
    }
    return {
        "{{project_name}}": project_name,
        "{{document_id}}": str(document.get("document_id") or document.get("link") or ""),
        "{{document_title}}": str(document.get("document_title") or document.get("title") or ""),
        "{{document_text}}": str(document.get("content") or ""),
        "{{metadata_json}}": json.dumps(metadata, indent=2, sort_keys=True),
        "{{tie_schema_json}}": json.dumps(tie_schema, indent=2, sort_keys=True),
        "{{tie_attribute_schema_json}}": json.dumps(
            tie_attribute_schema or [], indent=2, sort_keys=True
        ),
        "{{annotation_schema_json}}": _annotation_schema_json(tie_attribute_schema or []),
    }


def render_prompt_template(template_text: str, context: dict[str, str]) -> str:
    supported = set(SUPPORTED_PROMPT_PLACEHOLDERS)
    discovered = set(PLACEHOLDER_PATTERN.findall(template_text))
    unsupported = sorted(discovered - supported)
    if unsupported:
        raise ValueError(f"Unsupported prompt placeholders: {', '.join(unsupported)}")

    rendered = template_text
    for placeholder in SUPPORTED_PROMPT_PLACEHOLDERS:
        rendered = rendered.replace(placeholder, context.get(placeholder, ""))
    return rendered


def _annotation_schema_json(tie_attribute_schema: list[dict[str, Any]]) -> str:
    tie_example: dict[str, Any] = {
        "type": "string",
        "reasoning_text": "string",
    }
    if tie_attribute_schema:
        attributes_example: dict[str, Any] = {}
        for entry in tie_attribute_schema:
            if entry["type"] == "categorical":
                attributes_example[entry["name"]] = " | ".join(
                    f'"{option}"' for option in entry["options"]
                )
            elif entry["type"] == "numeric":
                minimum, maximum = entry["range"]
                attributes_example[entry["name"]] = f"number between {minimum} and {maximum}"
            else:
                attributes_example[entry["name"]] = {
                    "start": "YYYY or YYYY-MM or YYYY-MM-DD",
                    "end": "YYYY or YYYY-MM or YYYY-MM-DD",
                }
        tie_example["attributes"] = attributes_example
    return json.dumps(
        {"relationships": [{"source": "string", "target": "string", "ties": [tie_example]}]},
        indent=2,
        sort_keys=True,
    )
