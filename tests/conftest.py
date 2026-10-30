import pandas as pd
import pytest


@pytest.fixture
def reviews():
    """12 reviews of product A (ratings 1 to 5) with a few shipping words, 3 reviews of product B."""
    rows = [("A", 5, "great"), ("A", 1, "box arrived damaged"), ("A", 2, "ok"), ("A", 1, "late shipping"),
            ("A", 4, "fine"), ("A", 2, "smells bad"), ("A", 5, "nice"), ("A", 1, "leaked everywhere"),
            ("A", 3, "meh"), ("A", 4, "good"), ("A", 4, "works"), ("A", 5, "love it"),
            ("B", 5, "great"), ("B", 1, "broke"), ("B", 3, "fine")]
    return pd.DataFrame(rows, columns=["parent_asin", "rating", "text"])


@pytest.fixture
def products():
    return pd.DataFrame({"title": ["Alpha Shampoo Bottle", "Beta Conditioner Jar"], "clean_reviews": [12, 3]}, index=["A", "B"])


@pytest.fixture
def fake_answer():
    """Stands in for the search + language-model step."""
    return lambda question, asin: {"answer": f"Model answer for {asin} [1].", "hits": pd.DataFrame({"text": ["an excerpt"]})}
