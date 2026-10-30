import json

from shopsense.monitoring.log_report import load_log, summarise


def test_summary_by_route(tmp_path):
    path = tmp_path / "log.jsonl"
    rows = [{"route": "rating", "seconds": 0.1, "refused": False}, {"route": "reviews", "seconds": 20.0, "refused": False},
            {"route": "reviews", "seconds": 30.0, "refused": True}]
    path.write_text("\n".join(json.dumps(r) for r in rows))
    table = summarise(load_log(path))
    assert table.loc["reviews", "requests"] == 2 and table.loc["reviews", "refused_%"] == 50.0
    assert table.loc["ALL", "requests"] == 3


def test_missing_log_gives_empty_summary(tmp_path):
    assert summarise(load_log(tmp_path / "nothing.jsonl")).empty
