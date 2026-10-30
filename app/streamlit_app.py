import json
import os
from pathlib import Path

import requests
import streamlit as st

HERE = Path(__file__).parent
API_URL = os.environ.get("SHOPSENSE_API", "http://localhost:8000")

ROUTE_TEXT = {
    "rating": "Calculated by code from the review ratings. No language model.",
    "share": "Counted by code with a keyword list. This is a keyword match, not a reading of each review.",
    "reviews": "Search found the most relevant reviews, then a small local model summarised them with citations.",
    "compare": "The review-summary step ran for each product, plus both average ratings.",
    "refuse": "No product was chosen, so no reviews were used.",
}

st.set_page_config(page_title="ShopSense", page_icon="🛍️", layout="wide")
st.title("ShopSense: ask questions about product reviews")


@st.cache_data(ttl=30)
def api_is_up():
    try:
        return requests.get(API_URL + "/health", timeout=1).ok
    except requests.RequestException:
        return False


@st.cache_data
def load_demo():
    path = HERE / "demo_answers.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def show_result(route, answer, sources, seconds=None):
    st.caption(f"Route: **{route}**. {ROUTE_TEXT.get(route, '')}")
    st.markdown(answer.replace("\n", "  \n"))
    if seconds is not None:
        st.caption(f"Answered in {seconds} s")
    if sources:
        with st.expander("Review excerpts used (numbers match the [1], [2] in the answer)"):
            groups = sources.items() if isinstance(sources, dict) else [("", sources)]
            for label, texts in groups:
                if label:
                    st.markdown(f"**Product {label}**")
                for number, text in enumerate(texts, start=1):
                    st.markdown(f"[{number}] {text}")


live = api_is_up()
st.sidebar.markdown("### Mode")
if live:
    st.sidebar.success("LIVE: using the local API and model")
else:
    st.sidebar.warning("DEMO: showing answers saved from real runs on my laptop")
st.sidebar.markdown("The live system runs a small language model (Ollama) on a laptop, so the online demo cannot run it. "
                    "Reviews are from the public Amazon Reviews 2023 dataset (All Beauty).")

if live:
    query = st.text_input("Search for a product", key="live_query")
    found = requests.get(API_URL + "/products", params={"query": query}, timeout=10).json() if query else []
    labels = {f"{p['title'][:90]} ({p['reviews']} reviews)": p["asin"] for p in found}
    choice = st.selectbox("Product", list(labels), key="live_product") if labels else None
    question = st.text_input("Your question", "What do reviewers say about the quality and durability?", key="live_question")
    if st.button("Ask", key="live_ask", disabled=choice is None):
        with st.spinner("Thinking (a local model can take 20 to 40 seconds)..."):
            reply = requests.post(API_URL + "/ask", json={"question": question, "asin": labels[choice]}, timeout=300)
        if reply.ok:
            data = reply.json()
            show_result(data["route"], data["answer"], data["sources"], data["seconds"])
        else:
            st.error(reply.text)
else:
    demo = load_demo()
    if demo is None:
        st.error("No saved answers found. Run `python scripts/make_demo_data.py` once on the laptop with Ollama running.")
        st.stop()
    titles = {p["asin"]: p["title"] for p in demo["products"]}
    single = [e for e in demo["examples"] if not e["asin_b"]]
    asin = st.selectbox("Product", list(dict.fromkeys(e["asin"] for e in single)),
                        format_func=lambda a: titles[a][:100], key="demo_product")
    options = [e for e in single if e["asin"] == asin]
    pick = st.selectbox("Question", range(len(options)), format_func=lambda i: options[i]["question"], key="demo_question")
    st.caption(f"Saved answers from {demo['made_on']}. Free-text questions need the live system.")
    show_result(options[pick]["route"], options[pick]["answer"], options[pick]["sources"])

    compares = [e for e in demo["examples"] if e["asin_b"]]
    if compares:
        with st.expander("Saved product comparisons"):
            for number, example in enumerate(compares):
                st.markdown(f"**{example['question'][:150]}**")
                show_result(example["route"], example["answer"], example["sources"])
