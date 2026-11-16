from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Protocol

from backend.app.config import ROOT, load_project_config
from pipelines.indexing.search import load_processed_papers

CACHE_DIR = ROOT / ".cache" / "evidencemap"


class TextEncoder(Protocol):
    def encode(
        self,
        sentences: list[str],
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
    ) -> Sequence[Sequence[float]]: ...


class SemanticIndex:
    def __init__(
        self,
        papers: list[dict],
        model_name: str,
        encoder: TextEncoder | None = None,
        cache_dir: Path = CACHE_DIR,
    ):
        import numpy as np

        self._np = np
        self._papers = list(papers)
        self._domains = [
            {domain.lower() for domain in paper.get("domains", [])}
            for paper in self._papers
        ]
        texts = [
            f"{paper.get('title') or ''}. {paper.get('abstract') or ''}"
            for paper in self._papers
        ]
        if not texts:
            self._embeddings = np.empty((0, 0), dtype=np.float32)
            self.model_name = model_name
            self.build_seconds = 0.0
            self._encoder = encoder
            return

        cache_path = cache_dir / f"{self._fingerprint(model_name, texts)}.npy"
        started = time.perf_counter()

        if cache_path.exists():
            embeddings = np.load(cache_path, allow_pickle=False)
        else:
            if encoder is None:
                from sentence_transformers import SentenceTransformer

                encoder = SentenceTransformer(model_name)
            embeddings = np.asarray(
                encoder.encode(
                    texts,
                    batch_size=64,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ),
                dtype=np.float32,
            )
            if embeddings.ndim != 2 or embeddings.shape[0] != len(self._papers):
                raise ValueError("Encoder returned an invalid corpus embedding matrix")
            if not np.isfinite(embeddings).all():
                raise ValueError("Encoder returned non-finite corpus embeddings")
            if len(embeddings):
                norms = np.linalg.norm(embeddings, axis=1)
                if np.any(norms == 0):
                    raise ValueError("Encoder returned a zero-length corpus embedding")
                embeddings /= norms[:, None]
            cache_dir.mkdir(parents=True, exist_ok=True)
            temp_path = cache_path.with_name(f"{cache_path.stem}.{os.getpid()}.tmp")
            with temp_path.open("wb") as file:
                np.save(file, embeddings, allow_pickle=False)
            temp_path.replace(cache_path)

        if embeddings.ndim != 2 or embeddings.shape[0] != len(self._papers):
            raise ValueError(f"Invalid cached embeddings at {cache_path}")
        self._embeddings = embeddings
        self.model_name = model_name
        self.build_seconds = time.perf_counter() - started
        self._encoder = encoder

    @staticmethod
    def _fingerprint(model_name: str, texts: list[str]) -> str:
        digest = hashlib.sha256(model_name.encode("utf-8"))
        for index, text in enumerate(texts):
            digest.update(str(index).encode("ascii"))
            digest.update(text.encode("utf-8"))
            digest.update(b"\0")
        return digest.hexdigest()

    def _ensure_encoder(self) -> None:
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer

            self._encoder = SentenceTransformer(self.model_name)

    def warmup(self) -> float:
        if not self._papers:
            return 0.0
        started = time.perf_counter()
        self._ensure_encoder()
        self._encoder.encode(
            ["EvidenceMap semantic search warm-up"],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return time.perf_counter() - started

    def search(self, query: str, limit: int = 10, domain: str | None = None) -> list[dict]:
        if not query.strip() or limit <= 0 or not self._papers:
            return []

        self._ensure_encoder()
        query_embedding = self._np.asarray(
            self._encoder.encode(
                [query],
                normalize_embeddings=True,
                show_progress_bar=False,
            ),
            dtype=self._np.float32,
        )
        if query_embedding.shape != (1, self._embeddings.shape[1]):
            raise ValueError("Encoder returned an invalid query embedding")

        scores = self._embeddings @ query_embedding[0]
        domain_filter = domain.lower() if domain else None
        candidates = [
            index for index, paper_domains in enumerate(self._domains)
            if not domain_filter or domain_filter in paper_domains
        ]
        candidates.sort(key=lambda index: (-float(scores[index]), index))

        return [
            {
                "paper_id": self._papers[index]["paper_id"],
                "title": self._papers[index].get("title"),
                "score": round(float(scores[index]), 4),
                "domains": self._papers[index].get("domains", []),
            }
            for index in candidates[:limit]
        ]


@lru_cache(maxsize=1)
def get_semantic_index() -> SemanticIndex:
    cfg = load_project_config()
    return SemanticIndex(load_processed_papers(), cfg["models"]["embedding"])


def search_semantic_papers(
    query: str, limit: int = 10, domain: str | None = None
) -> list[dict]:
    return get_semantic_index().search(query, limit=limit, domain=domain)
