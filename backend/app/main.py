from typing import Literal

from fastapi import FastAPI, HTTPException, Query

from backend.app.config import load_project_config
from pipelines.indexing import search_papers, search_semantic_papers

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
    mode: Literal["lexical", "semantic"] = Query(default="semantic"),
) -> dict:
    if mode == "semantic":
        try:
            results = search_semantic_papers(q, limit=limit, domain=domain)
        except ImportError as exc:
            raise HTTPException(
                status_code=503,
                detail="Semantic search requires the optional dependencies: pip install -e '.[semantic]'",
            ) from exc
    else:
        results = search_papers(q, limit=limit, domain=domain)

    return {
        "query": q,
        "domain": domain,
        "mode": mode,
        "results": results,
    }
