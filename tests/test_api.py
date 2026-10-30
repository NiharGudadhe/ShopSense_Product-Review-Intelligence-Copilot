import json

from fastapi.testclient import TestClient

from shopsense.agent.graph import build_agent
from shopsense.api.app import create_app


def make_client(reviews, products, fake_answer, tmp_path):
    log = tmp_path / "requests.jsonl"
    return TestClient(create_app(build_agent(reviews, products, fake_answer), products, log)), log


def test_health_and_products(reviews, products, fake_answer, tmp_path):
    client, _ = make_client(reviews, products, fake_answer, tmp_path)
    assert client.get("/health").json() == {"status": "ok", "products": 1}      # product B has only 3 reviews
    assert [p["asin"] for p in client.get("/products", params={"query": "alpha"}).json()] == ["A"]


def test_ask_returns_route_and_logs(reviews, products, fake_answer, tmp_path):
    client, log = make_client(reviews, products, fake_answer, tmp_path)
    reply = client.post("/ask", json={"question": "What is the average rating?", "asin": "A"}).json()
    assert reply["route"] == "rating" and reply["value"] == 3.08
    record = json.loads(log.read_text().splitlines()[0])
    assert record["route"] == "rating" and record["refused"] is False


def test_unknown_product_and_short_question_are_rejected(reviews, products, fake_answer, tmp_path):
    client, _ = make_client(reviews, products, fake_answer, tmp_path)
    assert client.post("/ask", json={"question": "average rating?", "asin": "ZZZ"}).status_code == 422
    assert client.post("/ask", json={"question": "x"}).status_code == 422


def test_no_product_is_a_refusal_in_the_log(reviews, products, fake_answer, tmp_path):
    client, log = make_client(reviews, products, fake_answer, tmp_path)
    client.post("/ask", json={"question": "Which stock should I buy?"})
    assert json.loads(log.read_text().splitlines()[0])["refused"] is True
