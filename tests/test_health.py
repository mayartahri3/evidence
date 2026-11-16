from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.config import load_project_config


def test_health_ok():
    r = TestClient(app).get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_scope_matches_agreed_decisions():
    cfg = load_project_config()
    assert cfg["scope"]["domains"] == ["machine learning", "computer vision"]
    assert cfg["scope"]["content_level"] == "abstracts"
