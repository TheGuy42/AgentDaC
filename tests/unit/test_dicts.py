"""Nested-dict helpers (`src.utils.dicts`)."""

from __future__ import annotations

import pytest

from src.utils.dicts import get_dict_value, set_dict_value


# --------------------------------------------------------------------------------------------------
# get_dict_value
# --------------------------------------------------------------------------------------------------
def test_get_nested_value_found():
    d = {"a": {"b": {"c": 42}}}
    value, found = get_dict_value(d, "a", "b", "c")
    assert value == 42
    assert found is True


def test_get_missing_returns_flag_when_not_raising():
    value, found = get_dict_value({"a": {}}, "a", "missing", raise_missing=False)
    assert value is None
    assert found is False


def test_get_missing_raises_by_default():
    with pytest.raises(ValueError):
        get_dict_value({"a": {}}, "a", "missing")


def test_get_through_intermediate_non_dict_raises():
    # traversing *through* a non-dict intermediate is guarded with a ValueError
    with pytest.raises(ValueError):
        get_dict_value({"a": 5}, "a", "b", "c")


# --------------------------------------------------------------------------------------------------
# set_dict_value
# --------------------------------------------------------------------------------------------------
def test_set_creates_intermediate_dicts():
    d: dict = {}
    set_dict_value(d, "a", "b", "c", value=7)
    assert d == {"a": {"b": {"c": 7}}}


def test_set_overwrites_existing():
    d = {"a": {"b": 1}}
    set_dict_value(d, "a", "b", value=2)
    assert d["a"]["b"] == 2


def test_set_default_does_not_overwrite_existing():
    d = {"a": {"b": 1}}
    returned = set_dict_value(d, "a", "b", value=99, set_default=True)
    assert d["a"]["b"] == 1
    assert returned == 1


def test_set_default_sets_when_absent():
    d: dict = {"a": {}}
    returned = set_dict_value(d, "a", "b", value=5, set_default=True)
    assert d["a"]["b"] == 5
    assert returned == 5
