import pytest

from experiments.saturn.rewards import answer_reward, calc_sat_value, parse_assignment


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (r"The final answer is \boxed{0011}.", "0011"),
        ("After considering 1010, the final answer is 0011.", "0011"),
        ("I considered 1010 and 0101; therefore 0011", "0011"),
        (r"The final answer is $0000$.", "0000"),
        (r"The final answer is \boxed{11}.", "0011"),
    ],
)
def test_parse_assignment_preserves_selected_raw_bitstring(
    text: str, expected: str
) -> None:
    assert parse_assignment(text, k=len(expected)) == expected


@pytest.mark.parametrize(
    "text",
    [
        "I considered 0011, but the final answer is 1201.",
        "The final answer is -0011.",
        "The final answer is 0011.0.",
        "The final answer is 11111.",
    ],
)
def test_parse_assignment_rejects_invalid_final_token(text: str) -> None:
    with pytest.raises(ValueError):
        parse_assignment(text, k=4)


def test_calc_sat_value_accepts_valid_assignment() -> None:
    assert calc_sat_value("!A & !B & C & D", "0011")


@pytest.mark.parametrize("solution", ["", "0121", "abc"])
def test_calc_sat_value_rejects_nonbinary_assignment(solution: str) -> None:
    with pytest.raises(ValueError):
        calc_sat_value("A", solution)


def test_calc_sat_value_rejects_missing_variable_value() -> None:
    with pytest.raises(ValueError, match="Variable 'C'"):
        calc_sat_value("C", "01")


def test_answer_reward_preserves_leading_zero_assignment() -> None:
    sample = {"k": 4, "clause": "!A & !B & C & D"}

    assert answer_reward(sample, r"The final answer is \boxed{0011}.") == (1.0, True)
