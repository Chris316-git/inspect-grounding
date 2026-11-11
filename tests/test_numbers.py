from decimal import Decimal

import pytest

from inspect_grounding import extract_numbers, ungrounded_numbers


def texts(s: str) -> list[str]:
    return [m.text for m in extract_numbers(s)]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("HRV was 62 ms, up from 58.", ["62", "58"]),
        ("It ended at 42.", ["42"]),
        ("1,200 users and 3.14 pies", ["1,200", "3.14"]),
        ("up 7.5% this week", ["7.5%"]),
        ("about 12k steps", ["12k"]),
        ("a range of 1.8-2.3x", ["1.8", "2.3"]),
        ("a 3-day plan", ["3"]),
        ("temperature of -3 degrees", ["-3"]),
        ("at 07:30", ["07", "30"]),
        ("costs .5 dollars", [".5"]),
    ],
)
def test_extracts_numbers(text: str, expected: list[str]) -> None:
    assert texts(text) == expected


@pytest.mark.parametrize(
    "text",
    ["gpt-4.1 and claude-3", "COVID-19", "evidence E12 and v2", "H2O", "snake_case_9"],
)
def test_skips_identifiers(text: str) -> None:
    assert texts(text) == []


def test_full_width_digits_are_normalized() -> None:
    [m] = extract_numbers("４２，０００円")
    assert m.value == Decimal(42000)


def test_values_and_candidates() -> None:
    pct, k = extract_numbers("45% of 12k")
    assert pct.candidates == (Decimal(45), Decimal("0.45"))
    assert k.value == Decimal(12000)
    assert Decimal(12) in k.candidates


def test_ignore_patterns_remove_tokens() -> None:
    assert texts("see [3] and [12] for 40 runs") == ["3", "12", "40"]
    assert [m.text for m in extract_numbers("see [3] and [12] for 40 runs", [r"\[\d+\]"])] == ["40"]


def bad(text: str, evidence: list[str], **kwargs: object) -> list[str]:
    return [m.text for m in ungrounded_numbers(text, evidence, **kwargs)]  # type: ignore[arg-type]


def test_grounded_answer_passes() -> None:
    assert bad("Recovery 62%, HRV 58 ms", ["recovery_score: 62, hrv_ms: 58.0"]) == []


def test_invented_number_is_flagged() -> None:
    assert bad("HRV rose to 71 ms", ["hrv_ms: 58"]) == ["71"]


def test_formatting_differences_are_grounded() -> None:
    assert bad("1,200 steps", ["steps=1200"]) == []
    assert bad("1200 steps", ["1,200 steps"]) == []
    assert bad("about 1.2k steps", ["1,200 steps"]) == []


def test_percent_and_fraction_ground_each_other() -> None:
    assert bad("45% of nights", ["share: 0.45"]) == []
    assert bad("share 0.45", ["45% of nights"]) == []
    assert bad("7.5%", ["0.075"]) == []


def test_rounding() -> None:
    assert bad("slept 7.8 h", ["7.849"]) == []
    assert bad("slept 8 h", ["7.849"]) == []
    assert bad("slept 7.85 h", ["7.849"]) == []
    # more precise than the evidence
    assert bad("slept 7.849 h", ["7.85"]) == ["7.849"]
    # rounding off
    assert bad("slept 7.8 h", ["7.849"], rounding=False) == ["7.8"]


def test_rounding_does_not_ground_far_values() -> None:
    assert bad("slept 9 h", ["7.849"]) == ["9"]


def test_negative_needs_negative_evidence() -> None:
    assert bad("change of -3", ["change: 3"]) == ["-3"]
    assert bad("change of -3", ["change: -3"]) == []


def test_ignore_and_ignore_below() -> None:
    assert bad("1. walk 25 min", ["walk"], ignore=[1, 25]) == []
    assert bad("try 2 things for 25 min", ["25"], ignore_below=10) == []
    assert bad("try 2 things for 26 min", ["25"], ignore_below=10) == ["26"]


def test_no_numbers_no_problems() -> None:
    assert bad("Take it easy today.", ["anything 1 2 3"]) == []
