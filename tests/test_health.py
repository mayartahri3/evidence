from fastapi.testclient import TestClient

from backend.app.config import load_project_config
from backend.app.main import app


def test_health_ok():
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_scope_matches_agreed_decisions():
    cfg = load_project_config()
    assert cfg["scope"]["domains"] == ["machine learning", "computer vision"]
    assert cfg["scope"]["content_level"] == "abstracts"


def test_semantic_search_endpoint(monkeypatch):
    monkeypatch.setattr(
        "backend.app.main.search_semantic_papers",
        lambda query, limit, domain: [{"paper_id": "p1", "title": query}],
    )
    response = TestClient(app).get(
        "/search",
        params={"q": "graph learning", "domain": "machine learning"},
    )
    assert response.status_code == 200
    assert response.json()["mode"] == "semantic"
    assert response.json()["results"][0]["paper_id"] == "p1"

    def missing_dependency(query, limit, domain):
        raise ImportError("optional dependency")

    monkeypatch.setattr("backend.app.main.search_semantic_papers", missing_dependency)
    unavailable = TestClient(app).get("/search", params={"q": "graph", "mode": "semantic"})
    assert unavailable.status_code == 503
