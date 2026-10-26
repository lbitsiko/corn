from __future__ import annotations

import json
from typing import Any

from corn.annotations import find_reasoning_spans
from corn.attribute_validation import validate_tie_attributes

INTERMEDIATE_SCHEMA_EXAMPLE = json.dumps(
    {
        "relationships": [
            {
                "source": "string",
                "target": "string",
                "ties": [
                    {
                        "type": "string",
                        "reasoning_text": "string",
                    }
                ],
            }
        ]
    },
    indent=2,
    sort_keys=True,
)


def validate_llm_intermediate_output(
    output: dict[str, Any],
    *,
    valid_tie_labels: set[str],
    tie_attribute_schema: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    if not isinstance(output, dict):
        raise ValueError("LLM output must be a JSON object")

    relationships = output.get("relationships")
    if relationships is None:
        raise ValueError("relationships is required")
    if not isinstance(relationships, list):
        raise ValueError("relationships must be a list")

    normalized_relationships: list[dict[str, Any]] = []
    for relationship in relationships:
        if not isinstance(relationship, dict):
            raise ValueError("each relationship must be an object")
        source = _required_string(relationship, "source")
        target = _required_string(relationship, "target")
        ties = relationship.get("ties")
        if not isinstance(ties, list) or not ties:
            raise ValueError("ties is required and must be a non-empty list")

        normalized_ties: list[dict[str, Any]] = []
        for tie in ties:
            if not isinstance(tie, dict):
                raise ValueError("each tie must be an object")
            tie_type = _required_string(tie, "type")
            if tie_type not in valid_tie_labels:
                raise ValueError(f"Unknown tie label: {tie_type}")
            reasoning_text = _required_string(tie, "reasoning_text")
            normalized_tie = {
                "type": tie_type,
                "reasoning_text": reasoning_text,
            }
            attributes = validate_tie_attributes(
                tie.get("attributes"),
                tie_attribute_schema=tie_attribute_schema or [],
            )
            if attributes:
                normalized_tie["attributes"] = attributes
            normalized_ties.append(normalized_tie)

        normalized_relationships.append(
            {
                "source": source,
                "target": target,
                "ties": normalized_ties,
            }
        )

    return {"relationships": normalized_relationships}


def transform_llm_output_to_payload(
    *,
    validated_output: dict[str, Any],
    document: dict[str, Any],
    workspace_id: str,
    annotator_id: str,
    created_at: str,
    updated_at: str,
) -> dict[str, Any]:
    content = str(document.get("content") or "")
    payload = {
        "workspace_id": workspace_id,
        "annotator_id": annotator_id,
        "annotation_source": "llm",
        "document_id": str(document.get("document_id") or document.get("link") or ""),
        "created_at": created_at,
        "updated_at": updated_at,
        "network_data": [],
    }
    for relationship in validated_output["relationships"]:
        entry = {
            "source": relationship["source"],
            "target": relationship["target"],
            "ties": [],
        }
        for tie in relationship["ties"]:
            reasoning_text = tie["reasoning_text"]
            matches = find_reasoning_spans(content, reasoning_text)
            if len(matches) == 1:
                reasoning_start, reasoning_end = matches[0]
            else:
                reasoning_start = None
                reasoning_end = None
                reasoning_text_span = ""
                reasoning_text_note = reasoning_text
            if len(matches) == 1:
                reasoning_text_span = reasoning_text
                reasoning_text_note = ""
            tie_payload = {
                "type": tie["type"],
                "reasoning_text": reasoning_text,
                "reasoning_text_span": reasoning_text_span,
                "reasoning_text_note": reasoning_text_note,
                "reasoning_start": reasoning_start,
                "reasoning_end": reasoning_end,
            }
            if tie.get("attributes"):
                tie_payload["attributes"] = tie["attributes"]
            entry["ties"].append(tie_payload)
        payload["network_data"].append(entry)
    return payload


def _required_string(mapping: dict[str, Any], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} is required")
    return value.strip()
