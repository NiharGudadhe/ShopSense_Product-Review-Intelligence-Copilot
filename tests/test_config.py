from pathlib import Path

import yaml

from shopsense.llm import answer
from shopsense.retrieval import hybrid

CONFIG = yaml.safe_load((Path(__file__).resolve().parents[1] / "configs" / "config.yaml").read_text())


def test_retrieval_settings_match_the_code():
    assert CONFIG["retrieval"]["embedding_model"] == hybrid.EMBED_MODEL
    assert CONFIG["retrieval"]["candidates_per_method"] == hybrid.CANDIDATES
    assert CONFIG["retrieval"]["rrf_k"] == hybrid.RRF_K


def test_llm_settings_match_the_code():
    assert CONFIG["llm"]["model"] == answer.MODEL
    assert CONFIG["llm"]["refusal_text"] == answer.NO_ANSWER
    assert CONFIG["llm"]["min_reviews_for_product_answer"] == answer.MIN_REVIEWS_PRODUCT
    assert CONFIG["llm"]["max_answer_tokens"] == answer.MAX_ANSWER_TOKENS
