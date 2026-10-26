from __future__ import annotations

import pytest

from corn.reporting import build_pairwise_llm_report, extract_document_tuples


def _annotation_payload(
    *,
    document_id: str,
    tuples: list[tuple[str, str, str]],
    workspace_id: str = "workspace-1",
    annotator_id: str = "annotator-a",
    annotation_source: str = "human",
) -> dict:
    return {
        "workspace_id": workspace_id,
        "annotator_id": annotator_id,
        "annotation_source": annotation_source,
        "document_id": document_id,
        "created_at": "2026-06-06T12:00:00+00:00",
        "updated_at": "2026-06-06T12:00:00+00:00",
        "network_data": [
            {
                "source": source,
                "target": target,
                "ties": [{"type": tie_label, "reasoning_text": f"{source}-{target}-{tie_label}"}],
            }
            for source, target, tie_label in tuples
        ],
    }


def _human_record(*, document_id: str, tuples: list[tuple[str, str, str]]) -> dict:
    payload = _annotation_payload(document_id=document_id, tuples=tuples)
    return {
        "workspace_id": payload["workspace_id"],
        "annotator_id": payload["annotator_id"],
        "document_id": document_id,
        "created_at": payload["created_at"],
        "updated_at": payload["updated_at"],
        "json_path": f"{document_id}.json",
        "payload": payload,
    }


def _llm_run(*, document_id: str, tuples: list[tuple[str, str, str]]) -> dict:
    return {
        "run_id": f"run-{document_id}",
        "document_id": document_id,
        "provider_id": "openrouter",
        "model_id": "anthropic/claude-3.5-sonnet",
        "prompt_id": "prompt-1",
        "validation_status": "valid",
        "created_at": "2026-06-06T12:00:00+00:00",
        "normalized_payload": _annotation_payload(
            document_id=document_id,
            tuples=tuples,
            workspace_id="llm",
            annotator_id="openrouter/anthropic/claude-3.5-sonnet",
            annotation_source="llm",
        ),
    }


def test_extract_document_tuples_deduplicates_identical_tuples_within_one_document() -> None:
    payload = _annotation_payload(
        document_id="doc-1",
        tuples=[
            ("Org A", "Org B", "Policy"),
            ("Org A", "Org B", "Policy"),
            ("Org A", "Org B", "Coordination"),
        ],
    )

    assert extract_document_tuples(payload) == {
        ("Org A", "Org B", "Coordination"),
        ("Org A", "Org B", "Policy"),
    }


def test_build_pairwise_llm_report_keeps_the_same_tuple_in_separate_documents() -> None:
    report = build_pairwise_llm_report(
        human_records=[
            _human_record(document_id="doc-1", tuples=[("Org A", "Org B", "Policy")]),
            _human_record(document_id="doc-2", tuples=[("Org A", "Org B", "Policy")]),
        ],
        llm_runs=[
            _llm_run(document_id="doc-1", tuples=[("Org A", "Org B", "Policy")]),
            _llm_run(document_id="doc-2", tuples=[("Org A", "Org B", "Policy")]),
        ],
    )

    assert report["overall"]["true_positives"] == 2
    assert report["confusion_matrix"]["rows"]["Policy"]["Policy"] == 2


def test_build_pairwise_llm_report_scores_exact_matches_and_ignores_llm_only_documents() -> None:
    report = build_pairwise_llm_report(
        human_records=[
            _human_record(document_id="doc-match", tuples=[("Org A", "Org B", "Policy")]),
            _human_record(document_id="doc-missing", tuples=[("Org C", "Org D", "Coordination")]),
        ],
        llm_runs=[
            _llm_run(document_id="doc-match", tuples=[("Org A", "Org B", "Policy")]),
            _llm_run(document_id="doc-extra", tuples=[("Org E", "Org F", "Conflict")]),
        ],
    )

    assert report["confusion_matrix"]["rows"]["Policy"]["Policy"] == 1
    assert report["confusion_matrix"]["rows"]["Coordination"]["none"] == 1
    assert report["overall"] == {
        "true_positives": 1,
        "false_positives": 0,
        "false_negatives": 1,
        "precision": 1.0,
        "recall": 0.5,
        "f1": 0.6667,
        "macro_f1": 0.5,
    }
    assert report["document_coverage"] == {
        "human_documents": 2,
        "llm_documents": 1,
        "union_documents": 2,
    }


def test_build_pairwise_llm_report_records_wrong_label_confusion_for_the_same_pair() -> None:
    report = build_pairwise_llm_report(
        human_records=[
            _human_record(document_id="doc-1", tuples=[("Org A", "Org B", "Policy")]),
        ],
        llm_runs=[
            _llm_run(document_id="doc-1", tuples=[("Org A", "Org B", "Coordination")]),
        ],
    )

    assert report["confusion_matrix"]["rows"]["Policy"]["Coordination"] == 1
    assert report["overall"]["true_positives"] == 0
    assert report["overall"]["false_positives"] == 1
    assert report["overall"]["false_negatives"] == 1
    assert report["per_label"]["Policy"] == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 1,
    }
    assert report["per_label"]["Coordination"] == {
        "precision": 0.0,
        "recall": 0.0,
        "f1": 0.0,
        "support": 0,
    }


def test_build_pairwise_llm_report_can_treat_reversed_pairs_as_undirected_matches() -> None:
    directed_report = build_pairwise_llm_report(
        human_records=[
            _human_record(document_id="doc-1", tuples=[("Org A", "Org B", "Policy")]),
        ],
        llm_runs=[
            _llm_run(document_id="doc-1", tuples=[("Org B", "Org A", "Policy")]),
        ],
    )

    undirected_report = build_pairwise_llm_report(
        human_records=[
            _human_record(document_id="doc-1", tuples=[("Org A", "Org B", "Policy")]),
        ],
        llm_runs=[
            _llm_run(document_id="doc-1", tuples=[("Org B", "Org A", "Policy")]),
        ],
        relationship_direction="undirected",
    )

    assert directed_report["confusion_matrix"]["rows"]["Policy"]["none"] == 1
    assert directed_report["confusion_matrix"]["rows"]["none"]["Policy"] == 1
    assert directed_report["overall"]["true_positives"] == 0

    assert undirected_report["confusion_matrix"]["rows"]["Policy"]["Policy"] == 1
    assert undirected_report["overall"]["true_positives"] == 1
    assert undirected_report["overall"]["false_positives"] == 0
    assert undirected_report["overall"]["false_negatives"] == 0


def test_build_pairwise_llm_report_can_exclude_human_only_documents_from_metrics() -> None:
    default_report = build_pairwise_llm_report(
        human_records=[
            _human_record(document_id="doc-match", tuples=[("Org A", "Org B", "Policy")]),
            _human_record(document_id="doc-human-only", tuples=[("Org C", "Org D", "Conflict")]),
        ],
        llm_runs=[
            _llm_run(document_id="doc-match", tuples=[("Org A", "Org B", "Policy")]),
        ],
    )

    filtered_report = build_pairwise_llm_report(
        human_records=[
            _human_record(document_id="doc-match", tuples=[("Org A", "Org B", "Policy")]),
            _human_record(document_id="doc-human-only", tuples=[("Org C", "Org D", "Conflict")]),
        ],
        llm_runs=[
            _llm_run(document_id="doc-match", tuples=[("Org A", "Org B", "Policy")]),
        ],
        human_document_coverage="exclude_human_docs_without_valid_llm_runs",
    )

    assert default_report["confusion_matrix"]["rows"]["Conflict"]["none"] == 1
    assert default_report["overall"]["false_negatives"] == 1
    assert default_report["document_coverage"] == {
        "human_documents": 2,
        "llm_documents": 1,
        "union_documents": 2,
    }

    assert "Conflict" not in filtered_report["confusion_matrix"]["rows"]
    assert filtered_report["confusion_matrix"]["rows"]["Policy"]["Policy"] == 1
    assert filtered_report["overall"]["true_positives"] == 1
    assert filtered_report["overall"]["false_positives"] == 0
    assert filtered_report["overall"]["false_negatives"] == 0
    assert filtered_report["document_coverage"] == {
        "human_documents": 1,
        "llm_documents": 1,
        "union_documents": 1,
    }


def test_build_pairwise_llm_report_ignores_llm_only_documents_outside_human_eval_subset() -> None:
    report = build_pairwise_llm_report(
        human_records=[
            _human_record(document_id="doc-eval", tuples=[("Org A", "Org B", "Policy")]),
        ],
        llm_runs=[
            _llm_run(document_id="doc-eval", tuples=[("Org A", "Org B", "Policy")]),
            _llm_run(document_id="doc-outside-eval", tuples=[("Org X", "Org Y", "Conflict")]),
        ],
    )

    assert report["confusion_matrix"]["rows"]["Policy"]["Policy"] == 1
    assert report["overall"] == {
        "true_positives": 1,
        "false_positives": 0,
        "false_negatives": 0,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "macro_f1": 1.0,
    }
    assert report["document_coverage"] == {
        "human_documents": 1,
        "llm_documents": 1,
        "union_documents": 1,
    }


def test_build_pairwise_llm_report_rejects_duplicate_inputs_for_the_same_document() -> None:
    with pytest.raises(ValueError, match="Duplicate human record for document_id: doc-1"):
        build_pairwise_llm_report(
            human_records=[
                _human_record(document_id="doc-1", tuples=[("Org A", "Org B", "Policy")]),
                _human_record(document_id="doc-1", tuples=[("Org A", "Org B", "Coordination")]),
            ],
            llm_runs=[],
        )

    with pytest.raises(ValueError, match="Duplicate LLM run for document_id: doc-1"):
        build_pairwise_llm_report(
            human_records=[],
            llm_runs=[
                _llm_run(document_id="doc-1", tuples=[("Org A", "Org B", "Policy")]),
                _llm_run(document_id="doc-1", tuples=[("Org A", "Org B", "Coordination")]),
            ],
        )


ATTRIBUTE_SCHEMA = [
    {"name": "valence", "type": "categorical", "description": "d", "options": ["positive", "negative"]},
    {"name": "sentiment", "type": "numeric", "description": "d", "range": [0.0, 1.0]},
    {"name": "time", "type": "temporal", "description": "d"},
]


def _attribute_payload(document_id, ties_attributes, *, source="A", target="B"):
    """One doc, one A->B pair, one 'interaction' tie carrying given attributes."""
    return {
        "document_id": document_id,
        "network_data": [{
            "source": source, "target": target,
            "ties": [{"type": "interaction", "reasoning_text": "t", "attributes": ties_attributes}],
        }],
    }


def _attribute_human_record(document_id, ties_attributes, *, source="A", target="B") -> dict:
    return {
        "document_id": document_id,
        "payload": _attribute_payload(document_id, ties_attributes, source=source, target=target),
    }


def _attribute_llm_run(document_id, ties_attributes, *, source="A", target="B") -> dict:
    return {
        "document_id": document_id,
        "normalized_payload": _attribute_payload(
            document_id, ties_attributes, source=source, target=target
        ),
    }


def test_attribute_agreement_absent_without_schema() -> None:
    report = build_pairwise_llm_report(
        human_records=[_attribute_human_record("doc-1", {"valence": "positive"})],
        llm_runs=[_attribute_llm_run("doc-1", {"valence": "positive"})],
    )

    assert "attribute_agreement" not in report


def test_attribute_agreement_categorical_accuracy_and_coverage() -> None:
    report = build_pairwise_llm_report(
        human_records=[_attribute_human_record("doc-1", {"valence": "positive"})],
        llm_runs=[_attribute_llm_run("doc-1", {"valence": "negative"})],
        tie_attribute_schema=ATTRIBUTE_SCHEMA,
    )

    agreement = report["attribute_agreement"]
    assert agreement["matched_ties"] == 1
    valence = agreement["attributes"]["valence"]
    assert valence["coverage"] == {"both": 1, "human_only": 0, "llm_only": 0, "neither": 0}
    assert valence["accuracy"] == 0.0
    assert valence["confusion_matrix"]["positive"]["negative"] == 1
    assert agreement["attributes"]["sentiment"]["coverage"]["neither"] == 1
    assert agreement["attributes"]["time"]["coverage"]["neither"] == 1


def test_attribute_agreement_numeric_mae_and_tolerance() -> None:
    report = build_pairwise_llm_report(
        human_records=[_attribute_human_record("doc-1", {"sentiment": 0.8})],
        llm_runs=[_attribute_llm_run("doc-1", {"sentiment": 0.7})],
        tie_attribute_schema=ATTRIBUTE_SCHEMA,
    )

    sentiment = report["attribute_agreement"]["attributes"]["sentiment"]
    assert sentiment["mae"] == 0.1
    assert sentiment["tolerance"] == 0.1
    assert sentiment["within_tolerance"] == 1.0


def test_attribute_agreement_numeric_tolerance_override() -> None:
    report = build_pairwise_llm_report(
        human_records=[_attribute_human_record("doc-1", {"sentiment": 0.8})],
        llm_runs=[_attribute_llm_run("doc-1", {"sentiment": 0.7})],
        tie_attribute_schema=ATTRIBUTE_SCHEMA,
        numeric_tolerances={"sentiment": 0.05},
    )

    sentiment = report["attribute_agreement"]["attributes"]["sentiment"]
    assert sentiment["tolerance"] == 0.05
    assert sentiment["within_tolerance"] == 0.0


def test_attribute_agreement_temporal_common_granularity() -> None:
    report = build_pairwise_llm_report(
        human_records=[_attribute_human_record("doc-1", {"time": {"start": "2019"}})],
        llm_runs=[_attribute_llm_run("doc-1", {"time": {"start": "2019-06"}})],
        tie_attribute_schema=ATTRIBUTE_SCHEMA,
    )

    time_report = report["attribute_agreement"]["attributes"]["time"]
    assert time_report["match_at_common_granularity"] == 1.0


def test_attribute_agreement_temporal_mismatch() -> None:
    report = build_pairwise_llm_report(
        human_records=[_attribute_human_record("doc-1", {"time": {"start": "2019-06"}})],
        llm_runs=[_attribute_llm_run("doc-1", {"time": {"start": "2019-08"}})],
        tie_attribute_schema=ATTRIBUTE_SCHEMA,
    )

    time_report = report["attribute_agreement"]["attributes"]["time"]
    assert time_report["match_at_common_granularity"] == 0.0


def test_attribute_agreement_temporal_period_end_must_match() -> None:
    report = build_pairwise_llm_report(
        human_records=[
            _attribute_human_record("doc-1", {"time": {"start": "2019", "end": "2020"}}),
        ],
        llm_runs=[_attribute_llm_run("doc-1", {"time": {"start": "2019-03"}})],
        tie_attribute_schema=ATTRIBUTE_SCHEMA,
    )

    time_report = report["attribute_agreement"]["attributes"]["time"]
    assert time_report["match_at_common_granularity"] == 0.0


def test_attribute_agreement_only_over_matched_ties() -> None:
    human_records = [_attribute_human_record("doc-1", {"valence": "positive"})]
    llm_runs = [_attribute_llm_run("doc-1", {"valence": "negative"}, target="C")]

    report_without_schema = build_pairwise_llm_report(
        human_records=human_records,
        llm_runs=llm_runs,
    )
    report_with_schema = build_pairwise_llm_report(
        human_records=human_records,
        llm_runs=llm_runs,
        tie_attribute_schema=ATTRIBUTE_SCHEMA,
    )

    assert report_with_schema["attribute_agreement"]["matched_ties"] == 0
    assert report_with_schema["overall"] == report_without_schema["overall"]


def test_attribute_agreement_human_only_coverage() -> None:
    report = build_pairwise_llm_report(
        human_records=[_attribute_human_record("doc-1", {"valence": "positive"})],
        llm_runs=[_attribute_llm_run("doc-1", {})],
        tie_attribute_schema=ATTRIBUTE_SCHEMA,
    )

    valence = report["attribute_agreement"]["attributes"]["valence"]
    assert valence["coverage"] == {"both": 0, "human_only": 1, "llm_only": 0, "neither": 0}
    assert valence["accuracy"] is None


def test_attribute_agreement_duplicate_tuple_first_wins() -> None:
    human_payload = {
        "document_id": "doc-1",
        "network_data": [
            {
                "source": "A", "target": "B",
                "ties": [
                    {"type": "interaction", "reasoning_text": "t", "attributes": {"valence": "positive"}},
                ],
            },
            {
                "source": "A", "target": "B",
                "ties": [
                    {"type": "interaction", "reasoning_text": "t", "attributes": {"valence": "negative"}},
                ],
            },
        ],
    }

    report = build_pairwise_llm_report(
        human_records=[{"document_id": "doc-1", "payload": human_payload}],
        llm_runs=[_attribute_llm_run("doc-1", {"valence": "positive"})],
        tie_attribute_schema=ATTRIBUTE_SCHEMA,
    )

    valence = report["attribute_agreement"]["attributes"]["valence"]
    assert valence["accuracy"] == 1.0
    assert valence["confusion_matrix"]["positive"]["positive"] == 1


def test_attribute_agreement_llm_only_coverage() -> None:
    report = build_pairwise_llm_report(
        human_records=[_attribute_human_record("doc-1", {})],
        llm_runs=[_attribute_llm_run("doc-1", {"valence": "positive"})],
        tie_attribute_schema=ATTRIBUTE_SCHEMA,
    )

    valence = report["attribute_agreement"]["attributes"]["valence"]
    assert valence["coverage"] == {"both": 0, "human_only": 0, "llm_only": 1, "neither": 0}
    assert valence["accuracy"] is None


def test_attribute_agreement_treats_string_temporal_value_as_absent() -> None:
    report = build_pairwise_llm_report(
        human_records=[_attribute_human_record("doc-1", {"time": "2019"})],
        llm_runs=[_attribute_llm_run("doc-1", {})],
        tie_attribute_schema=ATTRIBUTE_SCHEMA,
    )

    time_report = report["attribute_agreement"]["attributes"]["time"]
    assert time_report["coverage"] == {"both": 0, "human_only": 0, "llm_only": 0, "neither": 1}


def test_attribute_agreement_treats_non_numeric_value_as_absent() -> None:
    report = build_pairwise_llm_report(
        human_records=[_attribute_human_record("doc-1", {"sentiment": "high"})],
        llm_runs=[_attribute_llm_run("doc-1", {"sentiment": 0.7})],
        tie_attribute_schema=ATTRIBUTE_SCHEMA,
    )

    sentiment = report["attribute_agreement"]["attributes"]["sentiment"]
    assert sentiment["coverage"] == {"both": 0, "human_only": 0, "llm_only": 1, "neither": 0}


def test_attribute_agreement_treats_non_string_categorical_value_as_absent() -> None:
    report = build_pairwise_llm_report(
        human_records=[_attribute_human_record("doc-1", {"valence": 1})],
        llm_runs=[_attribute_llm_run("doc-1", {"valence": "positive"})],
        tie_attribute_schema=ATTRIBUTE_SCHEMA,
    )

    valence = report["attribute_agreement"]["attributes"]["valence"]
    assert valence["coverage"] == {"both": 0, "human_only": 0, "llm_only": 1, "neither": 0}


def test_attribute_agreement_treats_string_attributes_container_as_empty() -> None:
    report = build_pairwise_llm_report(
        human_records=[_attribute_human_record("doc-1", "positive")],
        llm_runs=[_attribute_llm_run("doc-1", {})],
        tie_attribute_schema=ATTRIBUTE_SCHEMA,
    )

    valence = report["attribute_agreement"]["attributes"]["valence"]
    assert valence["coverage"] == {"both": 0, "human_only": 0, "llm_only": 0, "neither": 1}


def test_attribute_agreement_treats_list_attributes_container_as_empty() -> None:
    report = build_pairwise_llm_report(
        human_records=[_attribute_human_record("doc-1", ["positive"])],
        llm_runs=[_attribute_llm_run("doc-1", {})],
        tie_attribute_schema=ATTRIBUTE_SCHEMA,
    )

    valence = report["attribute_agreement"]["attributes"]["valence"]
    assert valence["coverage"] == {"both": 0, "human_only": 0, "llm_only": 0, "neither": 1}


def test_attribute_agreement_numeric_metrics_are_none_without_comparable_pairs() -> None:
    report = build_pairwise_llm_report(
        human_records=[_attribute_human_record("doc-1", {"sentiment": 0.8})],
        llm_runs=[_attribute_llm_run("doc-1", {})],
        tie_attribute_schema=ATTRIBUTE_SCHEMA,
    )

    sentiment = report["attribute_agreement"]["attributes"]["sentiment"]
    assert sentiment["mae"] is None
    assert sentiment["within_tolerance"] is None
    assert sentiment["tolerance"] == 0.1


def test_attribute_agreement_temporal_metric_is_none_without_comparable_pairs() -> None:
    report = build_pairwise_llm_report(
        human_records=[_attribute_human_record("doc-1", {"time": {"start": "2019"}})],
        llm_runs=[_attribute_llm_run("doc-1", {})],
        tie_attribute_schema=ATTRIBUTE_SCHEMA,
    )

    time_report = report["attribute_agreement"]["attributes"]["time"]
    assert time_report["match_at_common_granularity"] is None


def test_attribute_agreement_rejects_unsupported_attribute_type() -> None:
    with pytest.raises(ValueError, match="Unsupported tie attribute type: boolean"):
        build_pairwise_llm_report(
            human_records=[_attribute_human_record("doc-1", {})],
            llm_runs=[_attribute_llm_run("doc-1", {})],
            tie_attribute_schema=[
                {"name": "flag", "type": "boolean", "description": "d"},
            ],
        )


def test_build_pairwise_llm_report_avoids_arbitrary_pairing_for_multi_label_mismatches() -> None:
    report = build_pairwise_llm_report(
        human_records=[
            _human_record(
                document_id="doc-1",
                tuples=[
                    ("Org A", "Org B", "Conflict"),
                    ("Org A", "Org B", "Policy"),
                ],
            ),
        ],
        llm_runs=[
            _llm_run(
                document_id="doc-1",
                tuples=[
                    ("Org A", "Org B", "Coordination"),
                    ("Org A", "Org B", "Support"),
                ],
            ),
        ],
    )

    assert report["confusion_matrix"]["rows"]["Conflict"]["none"] == 1
    assert report["confusion_matrix"]["rows"]["Policy"]["none"] == 1
    assert report["confusion_matrix"]["rows"]["none"]["Coordination"] == 1
    assert report["confusion_matrix"]["rows"]["none"]["Support"] == 1
    assert report["confusion_matrix"]["rows"]["Conflict"]["Coordination"] == 0
    assert report["confusion_matrix"]["rows"]["Policy"]["Support"] == 0
