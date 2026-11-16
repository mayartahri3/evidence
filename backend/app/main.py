from fastapi import FastAPI, Query

from backend.app.config import load_project_config
from pipelines.indexing import search_papers

app = FastAPI(title="EvidenceMap API", version="0.1.0")


@app.get("/health")
def health() -> dict:
    cfg = load_project_config()
    return {
        "status": "ok",
        "project": cfg["project"]["name"],
        "processing_version": cfg["project"]["processing_version"],
    }


@app.get("/search")
def search(
    q: str = Query(default="", description="Free-text paper search"),
    limit: int = Query(default=10, ge=1, le=25),
    domain: str | None = Query(default=None),
) -> dict:
    return {
        "query": q,
        "domain": domain,
        "results": search_papers(q, limit=limit, domain=domain),
    }
