from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PACKAGE))

from core import ContractError, SEED, schema_map
from materialize import _select_template
from run_inference import normalize_response
from score import calls_to_map, preference_exact, prf, project, value_or_counts


PREF = {"D": ["s", "t"], "E": ["t"]}


@pytest.mark.parametrize(
    ("gt_calls", "pred_calls", "expected"),
    [
        ([{"name": "D", "arguments": {"s": "a"}}], [{"name": "D", "arguments": {"s": "a"}}], (1, 0, 0)),
        ([{"name": "D", "arguments": {"s": "a"}}], [], (0, 0, 1)),
        ([{"name": "D", "arguments": {"s": "a"}}], [{"name": "E", "arguments": {"t": "x"}}], (0, 1, 1)),
        ([{"name": "D", "arguments": {"s": "a"}}], [{"name": "D", "arguments": {"s": "x"}}], (0, 1, 1)),
        ([{"name": "D", "arguments": {"s": "a"}}], [{"name": "D", "arguments": {"s": "a", "city": "c"}}], (1, 0, 0)),
        ([{"name": "D", "arguments": {"s": "a"}}], [{"name": "D", "arguments": {"city": "c"}}], (0, 0, 1)),
        ([{"name": "D", "arguments": {"s": "a"}}], [{"name": "D", "arguments": {"s": "a", "t": "x"}}], (1, 1, 0)),
        ([{"name": "D", "arguments": {"s": "a"}}, {"name": "D", "arguments": {"s": "b"}}], [{"name": "D", "arguments": {"s": "b"}}], (1, 0, 0)),
        ([{"name": "D", "arguments": {"s": "a"}}, {"name": "D", "arguments": {"s": "b"}}], [{"name": "D", "arguments": {"s": "b"}}, {"name": "D", "arguments": {"s": "x"}}], (1, 0, 0)),
        ([{"name": "D", "arguments": {"s": "a"}}], [{"name": "D", "arguments": {"s": "a"}}, {"name": "D", "arguments": {"s": "a"}}], (1, 0, 0)),
        ([{"name": "D", "arguments": {"s": "a", "t": "b"}}], [{"name": "D", "arguments": {"s": "a"}}], (1, 0, 1)),
        ([{"name": "D", "arguments": {"s": "a"}}], [], (0, 0, 1)),
    ],
)
def test_literal_primary_oracles(gt_calls, pred_calls, expected):
    gt = calls_to_map(gt_calls)
    pred = project(calls_to_map(pred_calls), PREF, True)
    assert value_or_counts(gt, pred) == expected


def test_strict_preference_exact_rejects_extra_same_key_value():
    gt = calls_to_map([{"name": "D", "arguments": {"s": "a"}}, {"name": "D", "arguments": {"s": "b"}}])
    pred = calls_to_map([{"name": "D", "arguments": {"s": "b"}}, {"name": "D", "arguments": {"s": "x"}}])
    assert value_or_counts(gt, pred) == (1, 0, 0)
    assert preference_exact(gt, pred) == 0


def test_literal_secondary_table():
    rows = [
        ((1, 0, 0), (2, 0, 0), (1, 0, 0), 1, 1, 1),
        ((1, 0, 0), (1, 0, 1), (0, 0, 1), 1, 0, 1),
        ((0, 0, 1), (1, 0, 1), (1, 0, 0), 0, 0, 1),
        ((1, 0, 0), (2, 1, 0), (1, 1, 0), 1, 0, 1),
        ((0, 1, 1), (0, 1, 2), (0, 0, 1), 0, 0, 0),
        ((0, 0, 1), (0, 0, 2), (0, 0, 1), 0, 0, 0),
    ]
    assert rows[0] == ((1, 0, 0), (2, 0, 0), (1, 0, 0), 1, 1, 1)
    assert prf(*rows[3][1]) == {"tp": 2, "fp": 1, "fn": 0, "precision": 2 / 3, "recall": 1.0, "f1": 0.8}
    assert 1 / 7 == pytest.approx(0.14285714285714285)
    assert 6 / 7 == pytest.approx(0.8571428571428571)


def test_bootstrap_seed_and_delta_oracle_resets():
    expected_indices = [1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1]
    for _label in ("pooled", "easy", "medium", "hard"):
        rng = random.Random(SEED)
        assert [rng.randrange(2) for _ in range(16)] == expected_indices
    assert [1.0, -0.3, -1.0, -0.3, -1.0, -0.3, -0.3, -0.3] == [1.0, -0.3, -1.0, -0.3, -1.0, -0.3, -0.3, -0.3]


def test_template_positive_overlap_and_lexical_tie():
    templates = [
        {"query_id": "b", "target": [{"slot": "s"}]},
        {"query_id": "a", "target": [{"slot": "s"}]},
    ]
    assert _select_template(templates, {"s"})["query_id"] == "a"
    with pytest.raises(ContractError):
        _select_template(templates, {"z"})


def test_response_normalization_and_failure_classes():
    tools = [{"type": "function", "function": {"name": "D", "parameters": {"type": "object", "properties": {"s": {"type": "string"}}}}}]
    schemas = schema_map(tools)
    valid = {"choices": [{"message": {"tool_calls": [{"function": {"name": "D", "arguments": json.dumps({"s": "a"})}}]}}]}
    assert normalize_response(valid, schemas) == ("success", [{"name": "D", "arguments": {"s": "a"}}], None)
    no_tool = {"choices": [{"message": {"content": "plain"}}]}
    assert normalize_response(no_tool, schemas)[0] == "model_no_tool"
    unknown = {"choices": [{"message": {"tool_calls": [{"function": {"name": "X", "arguments": "{}"}}]}}]}
    assert normalize_response(unknown, schemas) == ("success", [{"name": "X", "arguments": {}}], None)


def test_response_normalization_preserves_schema_adherence_errors():
    tools = [{"type": "function", "function": {"name": "D", "parameters": {"type": "object", "properties": {"s": {"type": "string", "enum": ["Seattle"]}}}}}]
    response = {"choices": [{"message": {"tool_calls": [{"function": {"name": "D", "arguments": json.dumps({"s": "Berkeley"})}}]}}]}
    assert normalize_response(response, schema_map(tools)) == (
        "success",
        [{"name": "D", "arguments": {"s": "Berkeley"}}],
        None,
    )
