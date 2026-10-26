from __future__ import annotations

import pytest

from corn.llm_validation import (
    INTERMEDIATE_SCHEMA_EXAMPLE,
    transform_llm_output_to_payload,
    validate_llm_intermediate_output,
)


def test_validate_llm_intermediate_output_accepts_minimal_valid_shape() -> None:
    validated = validate_llm_intermediate_output(
        {
            "relationships": [
                {
                    "source": "Org A",
                    "target": "Org B",
                    "ties": [
                        {
                            "type": "support",
                            "reasoning_text": "shared policy agenda",
                        }
                    ],
                }
            ]
        },
        valid_tie_labels={"support"},
    )

    assert validated["relationships"][0]["ties"][0]["type"] == "support"


def test_validate_llm_intermediate_output_rejects_missing_relationships() -> None:
    with pytest.raises(ValueError, match="relationships is required"):
        validate_llm_intermediate_output({"foo": []}, valid_tie_labels={"support"})


def test_validate_llm_intermediate_output_rejects_unknown_tie_type() -> None:
    with pytest.raises(ValueError, match="Unknown tie label"):
        validate_llm_intermediate_output(
            {
                "relationships": [
                    {
                        "source": "Org A",
                        "target": "Org B",
                        "ties": [{"type": "conflict", "reasoning_text": "evidence"}],
                    }
                ]
            },
            valid_tie_labels={"support"},
        )


def test_transform_llm_output_to_payload_matches_exact_reasoning_span_when_unique() -> None:
    payload = transform_llm_output_to_payload(
        validated_output={
            "relationships": [
                {
                    "source": "Org A",
                    "target": "Org B",
                    "ties": [{"type": "support", "reasoning_text": "shared policy agenda"}],
                }
            ]
        },
        document={
            "link": "doc-1",
            "content": "Alpha noted a shared policy agenda in the meeting.",
        },
        workspace_id="llm",
        annotator_id="openrouter/claude-3.5-sonnet",
        created_at="2026-05-31T10:00:00+00:00",
        updated_at="2026-05-31T10:00:00+00:00",
    )

    assert payload["document_id"] == "doc-1"
    assert payload["annotation_source"] == "llm"
    assert payload["network_data"][0]["ties"][0]["reasoning_text_span"] == "shared policy agenda"
    assert payload["network_data"][0]["ties"][0]["reasoning_text_note"] == ""
    assert payload["network_data"][0]["ties"][0]["reasoning_start"] == 14
    assert payload["network_data"][0]["ties"][0]["reasoning_end"] == 34


def test_transform_llm_output_to_payload_leaves_span_empty_for_ambiguous_match() -> None:
    payload = transform_llm_output_to_payload(
        validated_output={
            "relationships": [
                {
                    "source": "Org A",
                    "target": "Org B",
                    "ties": [{"type": "support", "reasoning_text": "Beta"}],
                }
            ]
        },
        document={
            "link": "doc-1",
            "content": "Alpha supports Beta. Beta responds.",
        },
        workspace_id="llm",
        annotator_id="openrouter/claude-3.5-sonnet",
        created_at="2026-05-31T10:00:00+00:00",
        updated_at="2026-05-31T10:00:00+00:00",
    )

    assert payload["network_data"][0]["ties"][0]["reasoning_text_span"] == ""
    assert payload["network_data"][0]["ties"][0]["reasoning_text_note"] == "Beta"
    assert payload["network_data"][0]["ties"][0]["reasoning_start"] is None
    assert payload["network_data"][0]["ties"][0]["reasoning_end"] is None


def test_intermediate_schema_example_is_relationship_based() -> None:
    assert '"relationships"' in INTERMEDIATE_SCHEMA_EXAMPLE


ATTRIBUTE_SCHEMA = [
    {
        "name": "valence",
        "type": "categorical",
        "description": "d",
        "options": ["positive", "negative"],
    },
]


def test_tie_attributes_validated_and_normalized() -> None:
    output = {
        "relationships": [
            {
                "source": "A",
                "target": "B",
                "ties": [
                    {
                        "type": "interaction",
                        "reasoning_text": "text",
                        "attributes": {"valence": "positive"},
                    }
                ],
            }
        ]
    }
    validated = validate_llm_intermediate_output(
        output,
        valid_tie_labels={"interaction"},
        tie_attribute_schema=ATTRIBUTE_SCHEMA,
    )
    assert validated["relationships"][0]["ties"][0]["attributes"] == {"valence": "positive"}


def test_unknown_tie_attribute_rejected() -> None:
    output = {
        "relationships": [
            {
                "source": "A",
                "target": "B",
                "ties": [
                    {
                        "type": "interaction",
                        "reasoning_text": "text",
                        "attributes": {"bogus": "x"},
                    }
                ],
            }
        ]
    }
    with pytest.raises(ValueError, match="Unknown tie attributes"):
        validate_llm_intermediate_output(
            output,
            valid_tie_labels={"interaction"},
            tie_attribute_schema=ATTRIBUTE_SCHEMA,
        )


def test_attributes_rejected_when_no_schema() -> None:
    # attributes present but project defines no schema -> unknown keys -> ValueError
    output = {
        "relationships": [
            {
                "source": "A",
                "target": "B",
                "ties": [
                    {
                        "type": "interaction",
                        "reasoning_text": "text",
                        "attributes": {"valence": "positive"},
                    }
                ],
            }
        ]
    }
    with pytest.raises(ValueError, match="Unknown tie attributes"):
        validate_llm_intermediate_output(output, valid_tie_labels={"interaction"})


def test_missing_attributes_key_is_valid_and_absent() -> None:
    output = {
        "relationships": [
            {
                "source": "A",
                "target": "B",
                "ties": [{"type": "interaction", "reasoning_text": "text"}],
            }
        ]
    }
    validated = validate_llm_intermediate_output(
        output,
        valid_tie_labels={"interaction"},
        tie_attribute_schema=ATTRIBUTE_SCHEMA,
    )
    assert "attributes" not in validated["relationships"][0]["ties"][0]


def test_numeric_attribute_normalized_to_float_through_validation_and_transform() -> None:
    numeric_schema = [
        {
            "name": "sentiment",
            "type": "numeric",
            "description": "Sentiment score",
            "range": [0, 1],
        },
    ]
    validated = validate_llm_intermediate_output(
        {
            "relationships": [
                {
                    "source": "Org A",
                    "target": "Org B",
                    "ties": [
                        {
                            "type": "support",
                            "reasoning_text": "shared policy agenda",
                            "attributes": {"sentiment": "0.7"},
                        }
                    ],
                }
            ]
        },
        valid_tie_labels={"support"},
        tie_attribute_schema=numeric_schema,
    )
    payload = transform_llm_output_to_payload(
        validated_output=validated,
        document={
            "link": "doc-1",
            "content": "Alpha noted a shared policy agenda in the meeting.",
        },
        workspace_id="llm",
        annotator_id="openrouter/claude-3.5-sonnet",
        created_at="2026-05-31T10:00:00+00:00",
        updated_at="2026-05-31T10:00:00+00:00",
    )

    persisted = payload["network_data"][0]["ties"][0]["attributes"]["sentiment"]
    assert isinstance(persisted, float)
    assert persisted == 0.7


def test_transform_llm_output_to_payload_carries_tie_attributes() -> None:
    payload = transform_llm_output_to_payload(
        validated_output={
            "relationships": [
                {
                    "source": "Org A",
                    "target": "Org B",
                    "ties": [
                        {
                            "type": "support",
                            "reasoning_text": "shared policy agenda",
                            "attributes": {"valence": "positive"},
                        },
                        {"type": "support", "reasoning_text": "Beta"},
                    ],
                }
            ]
        },
        document={
            "link": "doc-1",
            "content": "Alpha noted a shared policy agenda in the meeting.",
        },
        workspace_id="llm",
        annotator_id="openrouter/claude-3.5-sonnet",
        created_at="2026-05-31T10:00:00+00:00",
        updated_at="2026-05-31T10:00:00+00:00",
    )

    ties = payload["network_data"][0]["ties"]
    assert ties[0]["attributes"] == {"valence": "positive"}
    assert "attributes" not in ties[1]
