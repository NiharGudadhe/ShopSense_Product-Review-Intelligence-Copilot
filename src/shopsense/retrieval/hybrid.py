"""Hybrid search over review passages: BM25 (keywords) + vectors (meaning), merged with reciprocal rank fusion.

Use:
    index = load_index()                                   # once (about 20 seconds)
    hits = search(index, "does it irritate skin", asin=None, top_k=6)
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
PASSAGES_FILE = ROOT / "data" / "processed" / "passages.parquet"
PRODUCTS_FILE = ROOT / "data" / "processed" / "products_clean.parquet"
LANCE_DIR = ROOT / "data" / "warehouse" / "lancedb"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TOKEN = re.compile(r"[a-z0-9']+")
CANDIDATES = 50      # results taken from each method before merging
RRF_K = 60           # reciprocal rank fusion constant


def load_index():
    """Load passages, the LanceDB table, the BM25 index and the embedding model into one dictionary."""
    import lancedb                                            # imported here so importing this module stays light
    from rank_bm25 import BM25Okapi                           # (tests and the CI run without these packages)
    from sentence_transformers import SentenceTransformer

    passages = pd.read_parquet(PASSAGES_FILE).reset_index(drop=True)
    products = pd.read_parquet(PRODUCTS_FILE)
    table = lancedb.connect(str(LANCE_DIR)).open_table("passages")
    bm25 = BM25Okapi([TOKEN.findall(t.lower()) for t in passages["search_text"]])
    embedder = SentenceTransformer(EMBED_MODEL)
    embedder.max_seq_length = 256
    return {
        "passages": passages,
        "products": products.set_index("parent_asin"),
        "table": table,
        "bm25": bm25,
        "rows_of_product": passages.groupby("parent_asin").indices,
        "embedder": embedder,
    }


def search(index, question, asin=None, top_k=6, skip_flagged=False, method="hybrid"):
    """Return the best passages for a question as a DataFrame (best first).

    asin: search only this product's reviews. skip_flagged: leave out reviews with links or instruction-like text.
    Columns include passage_id, review_id, parent_asin, product_title, rating, text, found_by ('both', 'vector', 'bm25').
    """
    passages = index["passages"]

    vector = index["embedder"].encode([question], normalize_embeddings=True)[0]
    query = index["table"].search(vector).metric("cosine").limit(CANDIDATES)
    if asin is not None:
        query = query.where(f"parent_asin = '{asin}'", prefilter=True)
    vector_ids = query.to_pandas()["passage_id"].tolist()

    scores = index["bm25"].get_scores(TOKEN.findall(question.lower()))
    if asin is None:
        order = np.argsort(-scores)[:CANDIDATES]
    else:
        rows = index["rows_of_product"].get(asin, np.array([], dtype=int))
        order = rows[np.argsort(-scores[rows])][:CANDIDATES]
    bm25_ids = [passages.at[i, "passage_id"] for i in order if scores[i] > 0]

    fused = {}
    for rank, passage_id in enumerate(vector_ids):
        fused[passage_id] = fused.get(passage_id, 0) + 1 / (RRF_K + rank + 1)
    for rank, passage_id in enumerate(bm25_ids):
        fused[passage_id] = fused.get(passage_id, 0) + 1 / (RRF_K + rank + 1)
    if method == "vector":
        ranked = list(vector_ids)
    elif method == "bm25":
        ranked = list(bm25_ids)
    else:
        ranked = sorted(fused, key=fused.get, reverse=True)

    if skip_flagged:
        flagged = passages.set_index("passage_id")[["has_contact_or_link", "looks_like_injection"]].any(axis=1)
        ranked = [p for p in ranked if not flagged[p]]
    ranked = ranked[:top_k]

    result = passages.set_index("passage_id").loc[ranked].reset_index()
    result["score"] = [fused[p] for p in ranked]
    result["found_by"] = ["both" if (p in vector_ids and p in bm25_ids) else "vector" if p in vector_ids else "bm25" for p in ranked]
    return result
