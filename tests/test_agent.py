import pytest

from shopsense.agent.graph import NO_PRODUCT, build_agent, classify
from shopsense.agent.tools import rating_stats, share_mentioning, star_range, topic_for


def test_rating_stats(reviews):
    stats = rating_stats(reviews[reviews["parent_asin"] == "A"])
    assert stats["count"] == 12 and stats["mean"] == 3.08


def test_star_range_reading():
    assert star_range("share of the 1-2 star reviews") == (1, 2)
    assert star_range("how many of the 1-star reviews") == (1, 1)
    assert star_range("what percent of negative reviews") == (1, 2)
    assert star_range("the one to two star reviews") == (1, 2)


def test_topic_cues():
    assert topic_for("mention shipping or packaging")[0] == "shipping or packaging"
    assert topic_for("complain about the smell")[0] == "scent"
    assert topic_for("something unrelated") is None


def test_share_mentioning_counts_keywords(reviews):
    result = share_mentioning(reviews[reviews["parent_asin"] == "A"], r"ship|damag|leak", 1, 2)
    assert result["n_reviews"] == 5 and result["n_match"] == 3 and result["percent"] == 60.0


@pytest.mark.parametrize("question,asin,asin_b,route", [
    ("What is the average rating of <product>?", "A", "", "rating"),
    ("What share of the 1-2 star reviews for <product> mention shipping or packaging?", "A", "", "share"),
    ("What do reviewers say about the scent of <product>?", "A", "", "reviews"),
    ("Compare <product> and <product> on quality.", "A", "B", "compare"),
    ("Which stock should I buy?", "", "", "refuse"),
])
def test_routes(question, asin, asin_b, route):
    assert classify(question, asin, asin_b) == route


def test_agent_end_to_end(reviews, products, fake_answer):
    agent = build_agent(reviews, products, fake_answer)
    rating = agent.invoke({"question": "What is the average rating of Alpha Shampoo Bottle?", "asin": "A"})
    assert rating["route"] == "rating" and rating["value"] == 3.08

    share = agent.invoke({"question": "What share of the 1-2 star reviews mention shipping or packaging?", "asin": "A"})
    assert share["route"] == "share" and share["value"] == 60.0 and "keyword match" in share["answer"]

    named = agent.invoke({"question": "What is the average rating of Alpha Shampoo Bottle?"})   # product found from the title
    assert named["asin"] == "A"

    refused = agent.invoke({"question": "Write me a poem about the ocean."})
    assert refused["answer"] == NO_PRODUCT

    compare = agent.invoke({"question": "Compare Alpha Shampoo Bottle and Beta Conditioner Jar on quality.", "asin": "A", "asin_b": "B"})
    assert compare["route"] == "compare" and "Product A" in compare["answer"] and "Product B" in compare["answer"]


def test_product_title_cannot_trigger_a_route(reviews, products, fake_answer):
    products.loc["A", "title"] = "Compare Bottle versus Jar Average Rating Set"
    agent = build_agent(reviews, products, fake_answer)
    state = agent.invoke({"question": "What do reviewers say about the scent of Compare Bottle versus Jar Average Rating Set?", "asin": "A"})
    assert state["route"] == "reviews"
