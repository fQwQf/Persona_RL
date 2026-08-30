"""Deterministic metrics used by offline evaluation."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def mean(values: Sequence[float]) -> float:
    """Return the arithmetic mean, rejecting an empty sequence."""
    if not values:
        raise ValueError("mean requires at least one value")
    return sum(values) / len(values)


def variance(values: Sequence[float]) -> float:
    """Return population variance for repeated model samples."""
    average = mean(values)
    return sum((value - average) ** 2 for value in values) / len(values)


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Return Pearson correlation for paired observations."""
    if len(xs) != len(ys) or len(xs) < 2:
        raise ValueError("correlation requires equal sequences with at least two values")
    x_mean, y_mean = mean(xs), mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys, strict=True))
    x_sum = sum((x - x_mean) ** 2 for x in xs)
    y_sum = sum((y - y_mean) ** 2 for y in ys)
    denominator = math.sqrt(x_sum * y_sum)
    return 0.0 if denominator == 0 else numerator / denominator


def agreement(predictions: Sequence[int], labels: Sequence[int]) -> float:
    """Return exact-match agreement for binary or ordinal outputs."""
    if len(predictions) != len(labels) or not predictions:
        raise ValueError("agreement requires equally sized non-empty sequences")
    return sum(
        prediction == label for prediction, label in zip(predictions, labels, strict=True)
    ) / len(labels)


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires at least one value")
    position = (len(sorted_values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight


def bootstrap_ci(
    values: Sequence[float],
    *,
    rounds: int = 1000,
    confidence: float = 0.95,
    seed: int = 7,
) -> tuple[float, float]:
    """Return a deterministic percentile bootstrap interval for a sample mean."""
    if not values:
        raise ValueError("bootstrap_ci requires at least one value")
    if rounds < 100:
        raise ValueError("bootstrap_ci requires at least 100 rounds")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between zero and one")
    rng = random.Random(seed)
    observations = tuple(float(value) for value in values)
    estimates = sorted(
        sum(observations[rng.randrange(len(observations))] for _ in observations)
        / len(observations)
        for _ in range(rounds)
    )
    alpha = (1 - confidence) / 2
    return _quantile(estimates, alpha), _quantile(estimates, 1 - alpha)


def paired_bootstrap_delta_ci(
    left: Sequence[float],
    right: Sequence[float],
    *,
    rounds: int = 1000,
    confidence: float = 0.95,
    seed: int = 7,
) -> tuple[float, float]:
    """Return a percentile bootstrap interval for paired right-minus-left deltas."""
    if len(left) != len(right) or not left:
        raise ValueError("paired bootstrap requires equally sized non-empty sequences")
    deltas = [
        float(right_value) - float(left_value)
        for left_value, right_value in zip(left, right, strict=True)
    ]
    return bootstrap_ci(deltas, rounds=rounds, confidence=confidence, seed=seed)


def icc_consistency(matrix: Sequence[Sequence[float]]) -> float:
    """Compute ICC(2,1) for cases rated by multiple prompt variants."""
    if len(matrix) < 2:
        raise ValueError("ICC requires at least two cases")
    width = len(matrix[0]) if matrix else 0
    if width < 2 or any(len(row) != width for row in matrix):
        raise ValueError("ICC requires a rectangular matrix with at least two variants")
    n, k = len(matrix), width
    row_means = [mean(row) for row in matrix]
    column_means = [mean([matrix[row][column] for row in range(n)]) for column in range(k)]
    grand_mean = mean([value for row in matrix for value in row])
    ms_cases = k * sum((value - grand_mean) ** 2 for value in row_means) / (n - 1)
    ms_variants = n * sum((value - grand_mean) ** 2 for value in column_means) / (k - 1)
    residual = sum(
        (matrix[row][column] - row_means[row] - column_means[column] + grand_mean) ** 2
        for row in range(n)
        for column in range(k)
    ) / ((n - 1) * (k - 1))
    denominator = ms_cases + (k - 1) * residual + k * (ms_variants - residual) / n
    if denominator == 0:
        return 1.0
    return max(-1.0, min(1.0, (ms_cases - residual) / denominator))
