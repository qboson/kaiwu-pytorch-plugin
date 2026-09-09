"""NeMo-compatible final-answer extraction and equivalence checking."""

from __future__ import annotations

import re
from importlib.metadata import version

from latex2sympy2_extended import NormalizationConfig, normalize_latex
from math_verify import (
    LatexExtractionConfig,
    StringExtractionConfig,
    parse,
    verify,
)


def last_boxed_content(text: str) -> str | None:
    """Extract the last ``\\boxed{...}``, including nested LaTeX braces.

    Args:
        text: Model output text to scan.

    Returns:
        Content of the last complete ``\\boxed{...}`` group with surrounding
        whitespace stripped, or ``None`` when no complete group exists.
    """
    marker = r"\boxed{"
    start = 0
    matches: list[str] = []
    while True:
        marker_index = text.find(marker, start)
        if marker_index < 0:
            break
        content_start = marker_index + len(marker)
        depth = 1
        index = content_start
        while index < len(text) and depth:
            if text[index] == "{" and (index == 0 or text[index - 1] != "\\"):
                depth += 1
            elif text[index] == "}" and (index == 0 or text[index - 1] != "\\"):
                depth -= 1
            index += 1
        if depth == 0:
            matches.append(text[content_start : index - 1].strip())
            start = index
        else:
            start = content_start
    return matches[-1] if matches else None


def require_math_verify() -> str:
    """Fail before a run if the authoritative scorer is unavailable.

    Returns:
        Installed ``math-verify`` distribution version string.

    Raises:
        PackageNotFoundError: If ``math-verify`` is not installed.
    """
    return version("math-verify")


def _additional_math_verify_normalize(value: str) -> str:
    """Strips percent signs and trailing periods from a plain answer.

    Args:
        value: Raw answer text.

    Returns:
        Normalized answer used before the exact-match fallback.
    """
    percentage = re.fullmatch(r"(\d+\.?\d*)(?:\\%|%)", value)
    if percentage:
        value = percentage.group(1)
    return value.rstrip(".\\")


def math_equivalent(prediction_text: str, gold_answer: str) -> bool:
    """Judge the final boxed answer with NeMo-compatible math verification.

    Args:
        prediction_text: Full model output containing a ``\\boxed{}`` answer.
        gold_answer: Reference answer, either an option letter or LaTeX.

    Returns:
        ``True`` when the predicted boxed answer matches ``gold_answer``.
    """
    predicted_answer = last_boxed_content(prediction_text)
    if predicted_answer is None:
        return False
    mcq_options = tuple("ABCDEFGHIJ")
    if gold_answer.strip() in mcq_options:
        gold_mcq = parse(
            gold_answer,
            [StringExtractionConfig(strings=mcq_options)],
        )
        prediction_mcq = parse(
            predicted_answer,
            [StringExtractionConfig(strings=mcq_options)],
        )
        if verify(gold_mcq, prediction_mcq):
            return True

    gold_answer = _additional_math_verify_normalize(str(gold_answer))
    predicted_answer = _additional_math_verify_normalize(predicted_answer)
    normalization_config = NormalizationConfig()
    normalized_gold = normalize_latex(gold_answer, normalization_config)
    normalized_prediction = normalize_latex(
        predicted_answer,
        normalization_config,
    )
    if normalized_gold.replace(" ", "") == normalized_prediction.replace(" ", ""):
        return True

    text_literal = r"[a-zA-Z ,]+"
    if re.fullmatch(text_literal, normalized_gold) and re.fullmatch(
        text_literal,
        normalized_prediction,
    ):
        return False

    latex_environment = r"\$.*\$|\\\(.*\\\)|\\\[.*\\\]|\\boxed\{"
    if not re.search(latex_environment, gold_answer, re.DOTALL):
        gold_answer = f"${gold_answer}$"
    if not re.search(latex_environment, predicted_answer, re.DOTALL):
        predicted_answer = f"${predicted_answer}$"
    gold = parse(gold_answer, [LatexExtractionConfig()])
    prediction = parse(predicted_answer, [LatexExtractionConfig()])
    return bool(verify(gold, prediction))
