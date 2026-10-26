from __future__ import annotations

import datetime
import math
import re
from typing import Any

ISO_PARTIAL_DATE_PATTERN = re.compile(r"^\d{4}(-\d{2}(-\d{2})?)?$")


def validate_tie_attributes(
    attributes: Any,
    *,
    tie_attribute_schema: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate and normalize one tie's attributes dict against the project schema."""
    if attributes is None:
        return {}
    if not isinstance(attributes, dict):
        raise ValueError("attributes must be an object")

    schema_by_name = {entry["name"]: entry for entry in tie_attribute_schema}
    unknown = sorted(set(attributes) - set(schema_by_name))
    if unknown:
        raise ValueError(f"Unknown tie attributes: {', '.join(unknown)}")

    normalized: dict[str, Any] = {}
    for name, value in attributes.items():
        if value is None:
            continue
        entry = schema_by_name[name]
        if entry["type"] == "categorical":
            normalized[name] = _validate_categorical(name, value, entry["options"])
        elif entry["type"] == "numeric":
            normalized[name] = _validate_numeric(name, value, entry["range"])
        elif entry["type"] == "temporal":
            normalized[name] = _validate_temporal(name, value)
        else:
            raise ValueError(f"Unsupported tie attribute type: {entry['type']}")
    return normalized


def _validate_categorical(name: str, value: Any, options: list[str]) -> str:
    if not isinstance(value, str) or value not in options:
        raise ValueError(f"Tie attribute {name} must be one of: {', '.join(options)}")
    return value


def _validate_numeric(name: str, value: Any, value_range: list[float]) -> float:
    if isinstance(value, bool):
        raise ValueError(f"Tie attribute {name} must be a number")
    if isinstance(value, str):
        try:
            value = float(value.strip())
        except ValueError:
            raise ValueError(f"Tie attribute {name} must be a number") from None
    if not isinstance(value, (int, float)):
        raise ValueError(f"Tie attribute {name} must be a number")
    if not math.isfinite(float(value)):
        raise ValueError(f"Tie attribute {name} must be a number")
    minimum, maximum = value_range
    if not minimum <= float(value) <= maximum:
        raise ValueError(f"Tie attribute {name} must be between {minimum} and {maximum}")
    return float(value)


def _validate_temporal(name: str, value: Any) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ValueError(f"Tie attribute {name} must be an object with start and optional end")
    unknown = sorted(set(value) - {"start", "end"})
    if unknown:
        raise ValueError(f"Tie attribute {name} has unknown keys: {', '.join(unknown)}")
    start = _validate_partial_date(name, value.get("start"))
    normalized = {"start": start}
    end = value.get("end")
    if end is not None and str(end).strip():
        end = _validate_partial_date(name, end)
        # Lexicographic comparison works for zero-padded ISO partials; note it is
        # strict for mixed granularity (e.g. end "2019" is rejected against
        # start "2019-07" even though the year contains the month).
        if end < start:
            raise ValueError(f"Tie attribute {name} end must not precede start")
        normalized["end"] = end
    return normalized


def _validate_partial_date(name: str, value: Any) -> str:
    if not isinstance(value, str) or not ISO_PARTIAL_DATE_PATTERN.match(value.strip()):
        raise ValueError(
            f"Tie attribute {name} dates must be ISO-partial strings: YYYY, YYYY-MM, or YYYY-MM-DD"
        )
    normalized = value.strip()
    parts = [int(part) for part in normalized.split("-")]
    if len(parts) >= 2 and not 1 <= parts[1] <= 12:
        raise ValueError(f"Tie attribute {name} dates must use a month between 01 and 12")
    if len(parts) == 3:
        try:
            datetime.date(*parts)
        except ValueError:
            raise ValueError(f"Tie attribute {name} dates must be valid calendar dates") from None
    return normalized
