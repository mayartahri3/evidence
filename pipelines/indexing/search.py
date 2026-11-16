from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

from backend.app.config import ROOT

TOKEN_RE = re.compile(r"[a-z0-9]+")
PROCESSED_PAPERS = ROOT / "data" / "processed" / "papers.jsonl"


def _tokenize(value: str | None) -> list[str]:
    if not value:
        return []
    return TOKEN_RE.findall(value.lower())


@lru_cache(maxsize=1)
def load_processed_papers() -> list[dict]:
    if not PROCESSED_PAPERS.exists():
        return []

    papers: list[dict] = []
    with PROCESSED_PAPERS.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            papers.append(json.loads(line))
    return papers


class CorpusIndex:
    def __init__(self, papers: list[dict] | None = None):
        self._papers = list(papers) if papers is not None else load_processed_papers()
        self._by_id = {paper["paper_id"]: paper for paper in self._papers}
        self._term_postings: dict[str, dict[str, int]] = defaultdict(dict)
        self._doc_freq: dict[str, int] = defaultdict(int)
        self._num_docs = len(self._papers)
        self._build()

    def _build(self) -> None:
        for paper in self._papers:
            counts: dict[str, int] = defaultdict(int)
            for token in _tokenize(f"{paper.get('title') or ''} {paper.get('abstract') or ''}"):
                counts[token] += 1
            for term, count in counts.items():
                self._term_postings[term][paper["paper_id"]] = count
                self._doc_freq[term] += 1

    def search(self, query: str, limit: int = 10, domain: str | None = None) -> list[dict]:
        if not query or limit <= 0:
            return []

        terms = _tokenize(query)
        if not terms:
            return []

        scored: dict[str, float] = {}
        for term in terms:
            postings = self._term_postings.get(term)
            if not postings:
                continue
            idf = math.log((1 + self._num_docs) / (1 + self._doc_freq[term])) + 1.0
            for paper_id, tf in postings.items():
                paper = self._by_id.get(paper_id)
                if paper is None:
                    continue
                if domain and domain.lower() not in {d.lower() for d in paper.get("domains", [])}:
                    continue
                scored[paper_id] = scored.get(paper_id, 0.0) + tf * idf

        results = []
        for paper_id, score in scored.items():
            paper = self._by_id[paper_id]
            results.append({
                "paper_id": paper_id,
                "title": paper.get("title"),
                "score": round(score, 4),
                "domains": paper.get("domains", []),
            })

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]


def search_papers(query: str, limit: int = 10, domain: str | None = None) -> list[dict]:
    return CorpusIndex().search(query, limit=limit, domain=domain)
