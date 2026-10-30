import math

from shopsense.evaluation.metrics import hit_at_k, ndcg_at_k, paired_bootstrap, precision_at_k, reciprocal_rank


def test_hit_and_precision():
    assert hit_at_k([0, 0, 1, 0, 0]) == 1
    assert hit_at_k([0, 0, 0, 0, 0]) == 0
    assert precision_at_k([1, 0, 1, 0, 0], 5) == 0.4


def test_reciprocal_rank():
    assert reciprocal_rank([0, 0, 1]) == 1 / 3
    assert reciprocal_rank([0, 0, 0]) == 0


def test_ndcg_known_value():
    # one relevant result at rank 2, two relevant results exist
    expected = (1 / math.log2(3)) / (1 + 1 / math.log2(3))
    assert math.isclose(ndcg_at_k([0, 1, 0, 0, 0], total_relevant=2, k=5), expected)


def test_ndcg_perfect_and_empty():
    assert math.isclose(ndcg_at_k([1, 1, 0, 0, 0], total_relevant=2, k=5), 1.0)
    assert ndcg_at_k([0, 0, 0, 0, 0], total_relevant=0, k=5) == 0


def test_bootstrap_identical_scores_have_no_difference():
    mean, low, high = paired_bootstrap([0.5, 0.7, 0.2, 0.9], [0.5, 0.7, 0.2, 0.9])
    assert mean == low == high == 0


def test_bootstrap_clear_difference_excludes_zero():
    mean, low, high = paired_bootstrap([1.0] * 20, [0.0] * 20)
    assert low > 0 and mean == 1.0
