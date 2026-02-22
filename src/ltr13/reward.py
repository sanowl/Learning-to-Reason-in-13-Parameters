from __future__ import annotations

import math
import re
from fractions import Fraction

_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
_NUMBER_RE = re.compile(r"[-+]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?")


def extract_boxed(text: str) -> str | None:
    matches = _BOXED_RE.findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def extract_last_number(text: str) -> str | None:
    matches = _NUMBER_RE.findall(text)
    if not matches:
        return None
    return matches[-1].replace(",", "")


def extract_gsm8k_answer(answer_text: str) -> str:
    if "####" in answer_text:
        return answer_text.split("####")[-1].strip()
    boxed = extract_boxed(answer_text)
    if boxed is not None:
        return boxed
    number = extract_last_number(answer_text)
    return number if number is not None else answer_text.strip()


def normalize_answer(text: str) -> str:
    cleaned = text.strip()
    cleaned = cleaned.replace("$", "")
    cleaned = cleaned.replace(",", "")
    cleaned = cleaned.strip().lower()
    return cleaned


def to_number(text: str) -> float | None:
    normalized = normalize_answer(text)
    if not normalized:
        return None
    try:
        if "/" in normalized and normalized.count("/") == 1:
            return float(Fraction(normalized))
        return float(normalized)
    except (ValueError, ZeroDivisionError):
        return None


def answers_match(prediction: str, reference: str, tol: float = 1e-6) -> bool:
    pred_boxed = extract_boxed(prediction)
    ref_boxed = extract_boxed(reference)

    pred_candidate = pred_boxed or extract_last_number(prediction) or prediction
    ref_candidate = ref_boxed or extract_last_number(reference) or reference

    pred_num = to_number(pred_candidate)
    ref_num = to_number(ref_candidate)
    if pred_num is not None and ref_num is not None:
        return math.isclose(pred_num, ref_num, rel_tol=tol, abs_tol=tol)

    return normalize_answer(pred_candidate) == normalize_answer(ref_candidate)


def exact_match_reward(predictions: list[str], references: list[str]) -> list[float]:
    if len(predictions) != len(references):
        raise ValueError("predictions and references must have identical length")
    return [1.0 if answers_match(pred, ref) else 0.0 for pred, ref in zip(predictions, references)]
