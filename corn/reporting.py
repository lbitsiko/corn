from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Literal

from corn.attribute_validation import validate_tie_attributes


NONE_LABEL = "none"


TupleValue = tuple[str, str, str]
PairValue = tuple[str, str]
RelationshipDirection = Literal["directed", "undirected"]
HumanDocumentCoverage = Literal[
    "include_all_human_annotated_docs",
    "exclude_human_docs_without_valid_llm_runs",
]


def extract_document_tuples(
    payload: dict,
    *,
    relationship_direction: RelationshipDirection = "directed",
) -> set[TupleValue]:
    """Return unique (source, target, tie_label) tuples for one annotation payload."""
    tuples: set[TupleValue] = set()
    for entry in payload.get("network_data", []):
        source = str(entry.get("source", "")).strip()
        target = str(entry.get("target", "")).strip()
        if not source or not target:
            continue
        source, target = _normalize_pair(
            source,
            target,
            relationship_direction=relationship_direction,
        )
        for tie in entry.get("ties", []):
            tie_label = str(tie.get("type", "")).strip()
            if not tie_label:
                continue
            tuples.add((source, target, tie_label))
    return tuples


def extract_document_tuple_attributes(
    payload: dict,
    *,
    relationship_direction: RelationshipDirection = "directed",
) -> dict[TupleValue, dict]:
    """Return attributes per (source, target, tie_label); first occurrence wins."""
    attributes_by_tuple: dict[TupleValue, dict] = {}
    for entry in payload.get("network_data", []):
        source = str(entry.get("source", "")).strip()
        target = str(entry.get("target", "")).strip()
        if not source or not target:
            continue
        source, target = _normalize_pair(
            source,
            target,
            relationship_direction=relationship_direction,
        )
        for tie in entry.get("ties", []):
            tie_label = str(tie.get("type", "")).strip()
            if not tie_label:
                continue
            key = (source, target, tie_label)
            if key not in attributes_by_tuple:
                raw_attributes = tie.get("attributes")
                attributes_by_tuple[key] = (
                    dict(raw_attributes) if isinstance(raw_attributes, dict) else {}
                )
    return attributes_by_tuple


def build_pairwise_llm_report(
    *,
    human_records: list[dict],
    llm_runs: list[dict],
    tie_labels: Iterable[str] | None = None,
    relationship_direction: RelationshipDirection = "directed",
    human_document_coverage: HumanDocumentCoverage = "include_all_human_annotated_docs",
    tie_attribute_schema: list[dict] | None = None,
    numeric_tolerances: dict[str, float] | None = None,
) -> dict:
    """Build a pairwise comparison report for one human scope and one LLM scope."""
    human_documents = _normalize_scope_documents(
        human_records,
        payload_key="payload",
        relationship_direction=relationship_direction,
    )
    llm_documents = _normalize_scope_documents(
        llm_runs,
        payload_key="normalized_payload",
        relationship_direction=relationship_direction,
    )
    human_documents = _apply_human_document_coverage(
        human_documents,
        llm_documents,
        human_document_coverage=human_document_coverage,
    )
    llm_documents = _restrict_llm_documents_to_human_documents(
        llm_documents,
        human_documents,
    )

    labels = _collect_labels(human_documents, llm_documents, tie_labels=tie_labels)
    matrix_labels = [*labels, NONE_LABEL]
    rows = {
        truth_label: {predicted_label: 0 for predicted_label in matrix_labels}
        for truth_label in matrix_labels
    }

    comparison_document_ids = sorted(human_documents)
    for document_id in comparison_document_ids:
        _compare_document(
            rows,
            human_documents.get(document_id, set()),
            llm_documents.get(document_id, set()),
        )

    overall = _build_overall_metrics(rows, labels)
    per_label = {label: _build_label_metrics(rows, label, labels) for label in labels}

    report = {
        "relationship_direction": relationship_direction,
        "human_document_coverage": human_document_coverage,
        "labels": labels,
        "confusion_matrix": {
            "labels": matrix_labels,
            "rows": rows,
        },
        "overall": overall,
        "per_label": per_label,
        "document_coverage": {
            "human_documents": len(human_documents),
            "llm_documents": len(llm_documents),
            "union_documents": len(comparison_document_ids),
        },
    }
    if tie_attribute_schema:
        report["attribute_agreement"] = _build_attribute_agreement(
            human_records=human_records,
            llm_runs=llm_runs,
            human_documents=human_documents,
            llm_documents=llm_documents,
            tie_attribute_schema=tie_attribute_schema,
            numeric_tolerances=numeric_tolerances or {},
            relationship_direction=relationship_direction,
        )
    return report


def _normalize_scope_documents(
    records: list[dict],
    *,
    payload_key: str,
    relationship_direction: RelationshipDirection,
) -> dict[str, set[TupleValue]]:
    documents: dict[str, set[TupleValue]] = defaultdict(set)
    for record in records:
        payload = record.get(payload_key) if payload_key in record else record
        if payload is None:
            continue
        document_id = str(payload.get("document_id", "")).strip()
        if not document_id:
            continue
        if document_id in documents:
            scope_name = "LLM run" if payload_key == "normalized_payload" else "human record"
            raise ValueError(f"Duplicate {scope_name} for document_id: {document_id}")
        documents[document_id].update(
            extract_document_tuples(
                payload,
                relationship_direction=relationship_direction,
            )
        )
    return dict(documents)


def _apply_human_document_coverage(
    human_documents: dict[str, set[TupleValue]],
    llm_documents: dict[str, set[TupleValue]],
    *,
    human_document_coverage: HumanDocumentCoverage,
) -> dict[str, set[TupleValue]]:
    if human_document_coverage == "include_all_human_annotated_docs":
        return human_documents
    if human_document_coverage == "exclude_human_docs_without_valid_llm_runs":
        llm_document_ids = set(llm_documents)
        return {
            document_id: tuples
            for document_id, tuples in human_documents.items()
            if document_id in llm_document_ids
        }
    raise ValueError(f"Unsupported human_document_coverage: {human_document_coverage}")


def _restrict_llm_documents_to_human_documents(
    llm_documents: dict[str, set[TupleValue]],
    human_documents: dict[str, set[TupleValue]],
) -> dict[str, set[TupleValue]]:
    human_document_ids = set(human_documents)
    return {
        document_id: tuples
        for document_id, tuples in llm_documents.items()
        if document_id in human_document_ids
    }


def _normalize_pair(
    source: str,
    target: str,
    *,
    relationship_direction: RelationshipDirection,
) -> PairValue:
    if relationship_direction == "directed":
        return source, target
    if relationship_direction == "undirected":
        return tuple(sorted((source, target)))
    raise ValueError(f"Unsupported relationship_direction: {relationship_direction}")


def _collect_labels(
    human_documents: dict[str, set[TupleValue]],
    llm_documents: dict[str, set[TupleValue]],
    *,
    tie_labels: Iterable[str] | None,
) -> list[str]:
    labels = set(tie_labels or [])
    for tuples in [*human_documents.values(), *llm_documents.values()]:
        for _, _, tie_label in tuples:
            labels.add(tie_label)
    return sorted(label for label in labels if label and label != NONE_LABEL)


def _compare_document(
    rows: dict[str, dict[str, int]],
    human_tuples: set[TupleValue],
    llm_tuples: set[TupleValue],
) -> None:
    human_pairs = _group_labels_by_pair(human_tuples)
    llm_pairs = _group_labels_by_pair(llm_tuples)

    for pair in sorted(set(human_pairs) | set(llm_pairs)):
        human_labels = set(human_pairs.get(pair, set()))
        llm_labels = set(llm_pairs.get(pair, set()))

        exact_matches = human_labels & llm_labels
        for label in sorted(exact_matches):
            rows[label][label] += 1

        remaining_human = sorted(human_labels - exact_matches)
        remaining_llm = sorted(llm_labels - exact_matches)
        if len(remaining_human) == 1 and len(remaining_llm) == 1:
            rows[remaining_human[0]][remaining_llm[0]] += 1
            continue

        for label in remaining_human:
            rows[label][NONE_LABEL] += 1

        for label in remaining_llm:
            rows[NONE_LABEL][label] += 1


def _group_labels_by_pair(tuples: set[TupleValue]) -> dict[PairValue, set[str]]:
    grouped: dict[PairValue, set[str]] = defaultdict(set)
    for source, target, label in tuples:
        grouped[(source, target)].add(label)
    return dict(grouped)


def _build_overall_metrics(rows: dict[str, dict[str, int]], labels: list[str]) -> dict:
    true_positives = sum(rows[label][label] for label in labels)
    false_positives = sum(
        count
        for truth_label, predicted_counts in rows.items()
        for predicted_label, count in predicted_counts.items()
        if predicted_label != NONE_LABEL and truth_label != predicted_label
    )
    false_negatives = sum(
        count
        for truth_label, predicted_counts in rows.items()
        if truth_label != NONE_LABEL
        for predicted_label, count in predicted_counts.items()
        if predicted_label != truth_label
    )
    precision = _safe_divide(true_positives, true_positives + false_positives)
    recall = _safe_divide(true_positives, true_positives + false_negatives)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    label_f1_values = [_build_label_metrics(rows, label, labels)["f1"] for label in labels]
    macro_f1 = _safe_divide(sum(label_f1_values), len(label_f1_values))

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "macro_f1": macro_f1,
    }


def _build_label_metrics(
    rows: dict[str, dict[str, int]],
    label: str,
    labels: list[str],
) -> dict:
    _ = labels
    true_positives = rows[label][label]
    false_positives = sum(
        rows[truth_label][label] for truth_label in rows if truth_label != label
    )
    false_negatives = sum(
        count for predicted_label, count in rows[label].items() if predicted_label != label
    )
    precision = _safe_divide(true_positives, true_positives + false_positives)
    recall = _safe_divide(true_positives, true_positives + false_negatives)
    f1 = _safe_divide(2 * precision * recall, precision + recall)
    support = sum(rows[label].values())
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
    }


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 4)


def _build_attribute_agreement(
    *,
    human_records: list[dict],
    llm_runs: list[dict],
    human_documents: dict[str, set[TupleValue]],
    llm_documents: dict[str, set[TupleValue]],
    tie_attribute_schema: list[dict],
    numeric_tolerances: dict[str, float],
    relationship_direction: RelationshipDirection,
) -> dict:
    human_attributes = _attributes_by_document(
        human_records,
        payload_key="payload",
        relationship_direction=relationship_direction,
    )
    llm_attributes = _attributes_by_document(
        llm_runs,
        payload_key="normalized_payload",
        relationship_direction=relationship_direction,
    )

    matched_pairs: list[tuple[dict, dict]] = []
    for document_id in sorted(human_documents):
        common = human_documents[document_id] & llm_documents.get(document_id, set())
        for key in sorted(common):
            matched_pairs.append((
                human_attributes.get(document_id, {}).get(key, {}),
                llm_attributes.get(document_id, {}).get(key, {}),
            ))

    attributes_report: dict[str, dict] = {}
    for entry in tie_attribute_schema:
        name = entry["name"]
        coverage = {"both": 0, "human_only": 0, "llm_only": 0, "neither": 0}
        value_pairs: list[tuple] = []
        for human_tie_attributes, llm_tie_attributes in matched_pairs:
            human_value = _lenient_attribute_value(entry, human_tie_attributes.get(name))
            llm_value = _lenient_attribute_value(entry, llm_tie_attributes.get(name))
            if human_value is not None and llm_value is not None:
                coverage["both"] += 1
                value_pairs.append((human_value, llm_value))
            elif human_value is not None:
                coverage["human_only"] += 1
            elif llm_value is not None:
                coverage["llm_only"] += 1
            else:
                coverage["neither"] += 1

        attribute_report: dict = {"coverage": coverage}
        if entry["type"] == "categorical":
            attribute_report.update(_categorical_agreement(value_pairs, entry["options"]))
        elif entry["type"] == "numeric":
            minimum, maximum = entry["range"]
            tolerance = numeric_tolerances.get(name, round(0.1 * (maximum - minimum), 4))
            attribute_report.update(_numeric_agreement(value_pairs, tolerance))
        elif entry["type"] == "temporal":
            attribute_report.update(_temporal_agreement(value_pairs))
        else:
            raise ValueError(f"Unsupported tie attribute type: {entry['type']}")
        attributes_report[name] = attribute_report

    return {"matched_ties": len(matched_pairs), "attributes": attributes_report}


def _attributes_by_document(
    records: list[dict],
    *,
    payload_key: str,
    relationship_direction: RelationshipDirection,
) -> dict[str, dict[TupleValue, dict]]:
    documents: dict[str, dict[TupleValue, dict]] = {}
    for record in records:
        payload = record.get(payload_key) if payload_key in record else record
        if payload is None:
            continue
        document_id = str(payload.get("document_id", "")).strip()
        if not document_id:
            continue
        documents[document_id] = extract_document_tuple_attributes(
            payload,
            relationship_direction=relationship_direction,
        )
    return documents


def _lenient_attribute_value(entry: dict, value: Any) -> Any:
    """Validate one attribute value against its schema entry; invalid values count as absent."""
    if value is None:
        return None
    try:
        normalized = validate_tie_attributes(
            {entry["name"]: value},
            tie_attribute_schema=[entry],
        )
    except ValueError:
        return None
    return normalized.get(entry["name"])


def _categorical_agreement(value_pairs: list[tuple], options: list[str]) -> dict:
    observed = sorted({value for pair in value_pairs for value in pair} | set(options))
    confusion = {truth: {predicted: 0 for predicted in observed} for truth in observed}
    agreements = 0
    for human_value, llm_value in value_pairs:
        confusion[human_value][llm_value] += 1
        if human_value == llm_value:
            agreements += 1
    return {
        "accuracy": _safe_divide(agreements, len(value_pairs)) if value_pairs else None,
        "confusion_matrix": confusion,
    }


def _numeric_agreement(value_pairs: list[tuple], tolerance: float) -> dict:
    deviations = [round(abs(float(human) - float(llm)), 4) for human, llm in value_pairs]
    within = sum(1 for deviation in deviations if deviation <= tolerance)
    return {
        "mae": _safe_divide(sum(deviations), len(deviations)) if deviations else None,
        "tolerance": tolerance,
        "within_tolerance": _safe_divide(within, len(deviations)) if deviations else None,
    }


def _temporal_agreement(value_pairs: list[tuple]) -> dict:
    matches = sum(1 for human, llm in value_pairs if _temporal_values_match(human, llm))
    return {
        "match_at_common_granularity": (
            _safe_divide(matches, len(value_pairs)) if value_pairs else None
        ),
    }


def _temporal_values_match(human_value: dict, llm_value: dict) -> bool:
    if not _partial_dates_match(str(human_value.get("start", "")), str(llm_value.get("start", ""))):
        return False
    human_end = str(human_value.get("end") or human_value.get("start", ""))
    llm_end = str(llm_value.get("end") or llm_value.get("start", ""))
    return _partial_dates_match(human_end, llm_end)


def _partial_dates_match(left: str, right: str) -> bool:
    """Compare ISO-partial dates at the coarsest common granularity."""
    common = min(len(left), len(right))
    return bool(common) and left[:common] == right[:common]
