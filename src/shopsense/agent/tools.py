"""Read-only tools for the ShopSense agent. Plain pandas, no model, no SQL built from user text.

The user's question is never run as code or as a query. It only picks WHICH fixed tool runs and with which
product and which keyword list from the table below.
"""
import re

# keyword lists a 'share of reviews that mention X' question can use (the same lists notebook 05 used)
SHIPPING_PATTERN = r"ship|deliver|arriv|packag|box|seal|leak|damag"
TOPICS = {
    "shipping or packaging": SHIPPING_PATTERN,
    "scent": r"smell|scent|fragrance|odor|odour|perfume|aroma",
    "price or value": r"price|worth|money|value|expensive|affordable",
    "quality or durability": r"quality|well made|sturdy|cheap|flimsy|broke|durable|lasts?\b",
    "skin reaction": r"irritat|rash|break ?out|allerg|sensitive|burn|redness|itch",
}
# words in a question that point to a topic
TOPIC_CUES = [
    ("shipping or packaging", r"shipping|packag|deliver"),
    ("scent", r"scent|smell|fragrance"),
    ("price or value", r"price|value|worth|expensive"),
    ("quality or durability", r"quality|durab"),
    ("skin reaction", r"skin|irritat|rash|allerg"),
]
NUMBER_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}


def rating_stats(reviews):
    """Average rating, number of reviews and the share of each star level for one product's clean reviews."""
    count = len(reviews)
    if count == 0:
        return {"count": 0, "mean": None, "share_by_star": {}}
    shares = (reviews["rating"].round().value_counts(normalize=True) * 100).round(1).to_dict()
    return {"count": count, "mean": round(float(reviews["rating"].mean()), 2),
            "share_by_star": {int(star): shares.get(star, 0.0) for star in [1, 2, 3, 4, 5]}}


def topic_for(question):
    """Return (topic name, keyword pattern) for the first topic cue found in the question, or None."""
    for name, cue in TOPIC_CUES:
        if re.search(cue, question, re.I):
            return name, TOPICS[name]
    return None


def star_range(question):
    """Which star levels the question means: '1-2 star' -> (1, 2), '1 star' -> (1, 1), 'negative' -> (1, 2)."""
    text = question.lower()
    for word, number in NUMBER_WORDS.items():
        text = re.sub(rf"\b{word}\b", str(number), text)
    both = re.search(r"([1-5])\s*(?:-|to|or|and)\s*([1-5])[\s-]*star", text)
    if both:
        return min(int(both.group(1)), int(both.group(2))), max(int(both.group(1)), int(both.group(2)))
    single = re.search(r"([1-5])[\s-]*star", text)
    if single:
        return int(single.group(1)), int(single.group(1))
    return 1, 2


def share_mentioning(reviews, pattern, low, high):
    """Of the reviews with stars from low to high, how many contain a keyword. Returns counts and the percentage."""
    group = reviews[(reviews["rating"] >= low) & (reviews["rating"] <= high)]
    matched = group["text"].str.contains(pattern, case=False, regex=True)
    total = len(group)
    return {"n_reviews": total, "n_match": int(matched.sum()),
            "percent": round(100 * float(matched.mean()), 1) if total else None, "matched_index": group.index[matched].tolist()}
