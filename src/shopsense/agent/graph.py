"""The ShopSense agent: a small LangGraph that sends each question to the right tool.

    question -> classify -> one of:  rating | share | compare | reviews | refuse -> answer

The routing is plain rules (no model), so it is fast, free and easy to test. The language model is used only for the
'reviews' and 'compare' routes, where it writes an answer from review excerpts. Every tool is read-only.
"""
import re
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from shopsense.agent.tools import rating_stats, share_mentioning, star_range, topic_for

NO_PRODUCT = "Please choose a product (or name it in the question) so I know which reviews to use."


class State(TypedDict, total=False):
    question: str
    asin: str
    asin_b: str
    route: str
    answer: str
    value: float
    details: dict


def find_products(question, products, minimum_length=8):
    """ASINs of products whose full title appears in the question, in the order they appear."""
    lowered = question.lower()
    found = []
    for asin, title in products["title"].items():
        title_lower = str(title).lower()
        if len(title_lower) >= minimum_length and title_lower in lowered:
            found.append((lowered.index(title_lower), asin, title_lower))
    found.sort()
    return [(asin, title) for _, asin, title in found]


def classify(question, asin, asin_b):
    """Pick the route from the wording. Product names are removed first so a title cannot trigger a route."""
    text = question.lower()
    if not asin:
        return "refuse"
    if asin_b and re.search(r"\bcompare\b|\bvs\.?\b|\bversus\b|difference between", text):
        return "compare"
    if (re.search(r"\b(share|percent|percentage|proportion|fraction|how many)\b", text)
            and re.search(r"star|negative|low-rated|bad review", text) and topic_for(text)):
        return "share"
    if re.search(r"average rating|mean rating|average score|\bavg\b|how many stars|star rating|rated on average", text):
        return "rating"
    return "reviews"


def build_agent(reviews, products, answer_fn):
    """Build the graph.

    reviews: clean reviews with columns parent_asin, rating, text.  products: indexed by parent_asin with a 'title'.
    answer_fn(question, asin) -> dict with 'answer' and optional 'hits': the retrieval + language-model step.
    """
    reviews_of = {asin: group for asin, group in reviews.groupby("parent_asin")}

    def prepare(state):
        asin, asin_b = state.get("asin") or "", state.get("asin_b") or ""
        question = state["question"]
        titles = []
        if not asin:
            found = find_products(question, products)
            asin = found[0][0] if found else ""
            asin_b = found[1][0] if len(found) > 1 else asin_b
            titles = [t for _, t in found]
        else:
            titles = [str(products.loc[a, "title"]).lower() for a in (asin, asin_b) if a]
        stripped = question.lower()
        for title in titles:
            stripped = stripped.replace(title, "<product>")
        return {"asin": asin, "asin_b": asin_b, "route": classify(stripped, asin, asin_b),
                "details": {"question_without_titles": stripped}}

    def pick(state):
        return state["route"]

    def do_rating(state):
        stats = rating_stats(reviews_of.get(state["asin"], reviews.iloc[0:0]))
        if not stats["count"]:
            return {"answer": "There are no reviews for this product.", "value": None, "details": stats}
        stars = ", ".join(f"{s} star {p}%" for s, p in stats["share_by_star"].items())
        return {"answer": f"The average rating is {stats['mean']} from {stats['count']} reviews ({stars}).",
                "value": stats["mean"], "details": stats}

    def do_share(state):
        text = state["details"]["question_without_titles"]
        topic, pattern = topic_for(text)
        low, high = star_range(text)
        result = share_mentioning(reviews_of.get(state["asin"], reviews.iloc[0:0]), pattern, low, high)
        stars = f"{low} to {high} stars" if low != high else f"{low} star" + ("" if low == 1 else "s")
        if not result["n_reviews"]:
            return {"answer": f"This product has no reviews with {stars}.", "value": None, "details": result}
        return {"answer": (f"{result['percent']}% of the {result['n_reviews']} reviews with {stars} ({result['n_match']} reviews) "
                           f"contain a word about {topic}. This is a keyword match, not a reading of each review."),
                "value": result["percent"], "details": {**result, "topic": topic, "stars": [low, high]}}

    def do_reviews(state):
        result = answer_fn(state["question"], state["asin"])
        sources = result["hits"]["text"].tolist() if result.get("hits") is not None else []
        return {"answer": result["answer"], "details": {"sources": sources}}

    def do_compare(state):
        parts, sources = [], {}
        for label, asin in (("A", state["asin"]), ("B", state["asin_b"])):
            title = str(products.loc[asin, "title"])
            stats = rating_stats(reviews_of.get(asin, reviews.iloc[0:0]))
            result = answer_fn(f"What do reviewers say about the quality and durability of {title}?", asin)
            parts.append(f"Product {label}: {title}\nAverage rating {stats['mean']} from {stats['count']} reviews.\n{result['answer']}")
            sources[label] = result["hits"]["text"].tolist() if result.get("hits") is not None else []
        return {"answer": "\n\n".join(parts), "details": {"sources": sources}}

    def do_refuse(state):
        return {"answer": NO_PRODUCT, "details": {"reason": "no_product"}}

    graph = StateGraph(State)
    graph.add_node("prepare", prepare)
    for name, function in [("rating", do_rating), ("share", do_share), ("reviews", do_reviews),
                           ("compare", do_compare), ("refuse", do_refuse)]:
        graph.add_node(name, function)
        graph.add_edge(name, END)
    graph.add_edge(START, "prepare")
    graph.add_conditional_edges("prepare", pick, ["rating", "share", "reviews", "compare", "refuse"])
    return graph.compile()
