"""ShopSense API (FastAPI).
Every request is appended to data/logs/requests.jsonl (question, route, seconds, refused) for monitoring.
"""
import json
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[3]
LOG_FILE = ROOT / "data" / "logs" / "requests.jsonl"
MIN_REVIEWS = 10


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    asin: str = ""
    asin_b: str = ""


def write_log(record, log_file=LOG_FILE):
    log_file.parent.mkdir(parents=True, exist_ok=True)
    with open(log_file, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


def create_app(agent, products, log_file=LOG_FILE):
    """Build the API around an agent (see shopsense.agent.graph.build_agent) and a products table (index parent_asin)."""
    api = FastAPI(title="ShopSense", version="1.0")
    allowed = products[products["clean_reviews"] >= MIN_REVIEWS]

    @api.get("/health")
    def health():
        return {"status": "ok", "products": int(len(allowed))}

    @api.get("/products")
    def list_products(query: str = "", limit: int = 20):
        titles = allowed["title"].astype(str)
        shown = allowed[titles.str.contains(query, case=False, regex=False)].head(min(limit, 50))
        return [{"asin": asin, "title": row["title"], "reviews": int(row["clean_reviews"])} for asin, row in shown.iterrows()]

    @api.post("/ask")
    def ask(request: AskRequest):
        for asin in (request.asin, request.asin_b):
            if asin and asin not in products.index:
                raise HTTPException(status_code=422, detail=f"Unknown product id: {asin}")
        start = time.perf_counter()
        state = agent.invoke({"question": request.question, "asin": request.asin, "asin_b": request.asin_b})
        seconds = round(time.perf_counter() - start, 1)
        details = state.get("details", {})
        sources = details.get("sources", [])
        write_log({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "question": request.question, "asin": state.get("asin", ""),
                   "route": state["route"], "seconds": seconds,
                   "refused": state["answer"].startswith(("The reviews do not say", "Please choose a product"))}, log_file)
        return {"route": state["route"], "answer": state["answer"], "value": state.get("value"),
                "sources": sources, "seconds": seconds}

    return api


def create_app_from_disk():
    """Load the real index, reviews and agent (about 30 seconds) and build the API."""
    import pandas as pd

    from shopsense.agent.graph import build_agent
    from shopsense.llm.answer import answer_question
    from shopsense.retrieval.hybrid import load_index

    index = load_index()
    reviews = pd.read_parquet(ROOT / "data" / "processed" / "reviews_clean.parquet")
    agent = build_agent(reviews, index["products"],
                        lambda question, asin: answer_question(index, question, asin=asin, method="vector"))
    return create_app(agent, index["products"])


class LazyApp:
    """Uvicorn needs a module-level `app`. This builds the real one on the first request, so importing the module is cheap."""

    def __init__(self):
        self._app = None

    async def __call__(self, scope, receive, send):
        if self._app is None:
            self._app = create_app_from_disk()
        await self._app(scope, receive, send)


app = LazyApp()
