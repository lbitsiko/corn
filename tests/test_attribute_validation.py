import pytest

from corn.attribute_validation import validate_tie_attributes

SCHEMA = [
    {"name": "valence", "type": "categorical", "description": "d", "options": ["positive", "negative"]},
    {"name": "sentiment", "type": "numeric", "description": "d", "range": [0.0, 1.0]},
    {"name": "time", "type": "temporal", "description": "d"},
]


def test_none_and_empty_attributes_normalize_to_empty():
    assert validate_tie_attributes(None, tie_attribute_schema=SCHEMA) == {}
    assert validate_tie_attributes({}, tie_attribute_schema=SCHEMA) == {}


def test_valid_attributes_normalize():
    result = validate_tie_attributes(
        {"valence": "positive", "sentiment": "0.2", "time": {"start": "2019-06", "end": "2019-07"}},
        tie_attribute_schema=SCHEMA,
    )
    assert result == {
        "valence": "positive",
        "sentiment": 0.2,
        "time": {"start": "2019-06", "end": "2019-07"},
    }


def test_temporal_point_in_time_omits_end():
    result = validate_tie_attributes({"time": {"start": "2019"}}, tie_attribute_schema=SCHEMA)
    assert result == {"time": {"start": "2019"}}


def test_none_values_are_dropped():
    assert validate_tie_attributes({"valence": None}, tie_attribute_schema=SCHEMA) == {}


def test_temporal_dates_are_whitespace_stripped():
    result = validate_tie_attributes({"time": {"start": " 2019 "}}, tie_attribute_schema=SCHEMA)
    assert result == {"time": {"start": "2019"}}


def test_temporal_empty_string_end_is_dropped():
    result = validate_tie_attributes(
        {"time": {"start": "2019", "end": ""}}, tie_attribute_schema=SCHEMA
    )
    assert result == {"time": {"start": "2019"}}


def test_numeric_int_input_accepted():
    result = validate_tie_attributes({"sentiment": 1}, tie_attribute_schema=SCHEMA)
    assert result == {"sentiment": 1.0}


def test_infinite_value_rejected_even_with_infinite_range_bound():
    schema = [{"name": "score", "type": "numeric", "description": "d", "range": [0.0, float("inf")]}]
    with pytest.raises(ValueError):
        validate_tie_attributes({"score": float("inf")}, tie_attribute_schema=schema)


def test_unsupported_schema_type_raises():
    schema = [{"name": "weird", "type": "spatial", "description": "d"}]
    with pytest.raises(ValueError):
        validate_tie_attributes({"weird": "x"}, tie_attribute_schema=schema)


@pytest.mark.parametrize("attributes", [
    {"unknown": "x"},
    {"valence": "neutral"},           # not in options
    {"valence": 1},                   # wrong type
    {"sentiment": 1.5},               # out of range
    {"sentiment": "abc"},             # not coercible
    {"sentiment": True},              # bool is not a number
    {"sentiment": float("nan")},      # NaN is not comparable to the range
    {"sentiment": -0.1},              # below minimum
    {"time": "2019"},                 # not an object
    {"time": {"end": "2019"}},        # missing start
    {"time": {"start": ""}},          # empty start
    {"time": {"start": "June 2019"}}, # not ISO-partial
    {"time": {"start": "2019-6"}},    # month not zero-padded
    {"time": {"start": "2019-13"}},   # month out of range
    {"time": {"start": "2019-00"}},   # month zero
    {"time": {"start": "2019-02-32"}},  # day out of range
    {"time": {"start": "2019-02-30"}},  # not a real calendar date
    {"time": {"start": "2019", "until": "2020"}},  # unknown temporal key
    {"time": {"start": "2019-07", "end": "2019-06"}},  # end before start
    "not-a-dict",
])
def test_invalid_attributes_rejected(attributes):
    with pytest.raises(ValueError):
        validate_tie_attributes(attributes, tie_attribute_schema=SCHEMA)
