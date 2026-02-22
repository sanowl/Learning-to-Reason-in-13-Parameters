from __future__ import annotations

from ltr13.reward import answers_match, exact_match_reward, extract_gsm8k_answer


def test_extract_gsm8k_answer() -> None:
    text = "Some chain of thought. #### 42"
    assert extract_gsm8k_answer(text) == "42"


def test_answers_match_numeric_and_boxed() -> None:
    assert answers_match("Final: 0.5", "1/2")
    assert answers_match("We get \\boxed{17}", "17")
    assert not answers_match("16", "17")


def test_exact_match_reward_vectorized() -> None:
    rewards = exact_match_reward(["10", "11"], ["10", "12"])
    assert rewards == [1.0, 0.0]
