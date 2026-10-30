"""Retrieval metrics and the paired bootstrap, as tested functions.

These are the same formulas notebook 06 computes inline (so the notebook stays readable).
flags: list of 1/0 for the top-k results of one question, in rank order (1 = labelled relevant).
"""
import numpy as np


def hit_at_k(flags):
    """1 if any of the top-k results is relevant."""
    return 1 if sum(flags) > 0 else 0


def precision_at_k(flags, k):
    """Relevant results in the top k, divided by k (a short result list counts as misses)."""
    return sum(flags[:k]) / k


def reciprocal_rank(flags):
    """1 / rank of the first relevant result, or 0 if none."""
    for position, flag in enumerate(flags):
        if flag == 1:
            return 1 / (position + 1)
    return 0


def ndcg_at_k(flags, total_relevant, k):
    """DCG of the top-k flags divided by the best possible DCG given how many relevant results exist."""
    dcg = sum(flag / np.log2(i + 2) for i, flag in enumerate(flags[:k]))
    ideal = sum(1 / np.log2(i + 2) for i in range(min(total_relevant, k)))
    return dcg / ideal if ideal > 0 else 0


def paired_bootstrap(first, second, repeats=2000, seed=42):
    """Mean difference (first minus second) per question, with a 95% range from re-drawing questions with replacement.

    first, second: equal-length lists with one score per question (same question order).
    Returns (mean difference, low, high). If low <= 0 <= high the difference is not clear.
    """
    first, second = np.asarray(first, dtype=float), np.asarray(second, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(first)
    differences = []
    for _ in range(repeats):
        picked = rng.integers(0, n, n)
        differences.append((first[picked] - second[picked]).mean())
    low, high = np.percentile(differences, [2.5, 97.5])
    return float((first - second).mean()), float(low), float(high)
