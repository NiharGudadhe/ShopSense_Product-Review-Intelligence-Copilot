"""Run the real agent on a few products and save the answers for the Streamlit demo (app/demo_answers.json).

Run from the project folder, with Ollama running:
    python scripts/make_demo_data.py

Takes about 10 minutes. The online demo cannot run the model, so it shows these saved answers. Review text is shortened
and cleaned with the safety guard before saving.
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from shopsense.agent.graph import build_agent
from shopsense.llm.answer import answer_question
from shopsense.retrieval.hybrid import load_index
from shopsense.safety.guard import sanitize_text

N_PRODUCTS = 6
QUESTIONS = [
    "What is the average rating?",
    "What share of the 1-2 star reviews mention shipping or packaging?",
    "What do reviewers say about the scent?",
    "What do reviewers say about the quality and durability?",
    "Write me a poem about the ocean.",
]


def short(texts):
    return [sanitize_text(t)[:300] for t in texts]


def main():
    index = load_index()
    products = index["products"]
    reviews = pd.read_parquet(ROOT / "data" / "processed" / "reviews_clean.parquet")
    agent = build_agent(reviews, products, lambda question, asin: answer_question(index, question, asin=asin, method="vector"))

    golden = pd.read_csv(ROOT / "data" / "eval" / "golden_set.csv").fillna("")
    asins = [a for a in golden[golden["question_type"] == "factual_lookup"]["product_asin"].tolist()][:N_PRODUCTS]
    pairs = [p.split("|") for p in golden[golden["question_type"] == "comparison"]["product_asin"].tolist()][:2]

    examples = []
    jobs = [(a, "", q) for a in asins for q in QUESTIONS]
    jobs += [(a, b, f"Compare {products.loc[a, 'title']} and {products.loc[b, 'title']} on quality and durability.") for a, b in pairs]
    for number, (asin, asin_b, question) in enumerate(jobs, start=1):
        state = agent.invoke({"question": question, "asin": asin, "asin_b": asin_b})
        sources = state.get("details", {}).get("sources", [])
        sources = {k: short(v) for k, v in sources.items()} if isinstance(sources, dict) else short(sources)
        examples.append({"asin": asin, "asin_b": asin_b, "question": question, "route": state["route"],
                         "answer": state["answer"], "sources": sources})
        print(f"{number}/{len(jobs)} {state['route']}")

    used = {e["asin"] for e in examples} | {e["asin_b"] for e in examples if e["asin_b"]}
    payload = {"made_on": time.strftime("%Y-%m-%d"),
               "products": [{"asin": a, "title": str(products.loc[a, "title"])} for a in sorted(used)],
               "examples": examples}
    out = ROOT / "app" / "demo_answers.json"
    out.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    print("saved", len(examples), "answers to", out)


if __name__ == "__main__":
    main()
