"""OpenAlex client + mapping from an OpenAlex 'work' to our Paper schema."""
from __future__ import annotations

import time
from collections.abc import Callable, Iterator

import httpx

from pipelines.ingestion.schema import Paper

SELECT = (
    "id,doi,title,publication_year,type,abstract_inverted_index,"
    "authorships,primary_location,open_access"
)


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str | None:
    """OpenAlex ships abstracts as {word: [positions]}; rebuild the text by sorting positions."""
    if not inverted_index:
        return None
    slots: dict[int, str] = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            slots[pos] = word
    return " ".join(slots[i] for i in sorted(slots)) or None


def normalize_doi(doi: str | None) -> str | None:
    """'https://doi.org/10.1/ABC' -> '10.1/abc' (DOIs are case-insensitive)."""
    if not doi:
        return None
    doi = doi.strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "doi:"):
        if doi.startswith(prefix):
            doi = doi[len(prefix):]
    return doi or None


def work_to_paper(work: dict, domain: str | None = None, processing_version: str = "v1") -> Paper | None:
    title = (work.get("title") or "").strip()
    if not title:
        return None  # unusable record
    short_id = work["id"].rsplit("/", 1)[-1]  # 'https://openalex.org/W123' -> 'W123'
    loc = work.get("primary_location") or {}
    oa = work.get("open_access") or {}
    oa_url = oa.get("oa_url") if oa.get("is_oa") else None
    return Paper(
        paper_id=f"oa_{short_id}",
        source_id=short_id,
        title=title,
        authors=[
            a["author"]["display_name"]
            for a in work.get("authorships") or []
            if a.get("author") and a["author"].get("display_name")
        ],
        publication_year=work.get("publication_year"),
        abstract=reconstruct_abstract(work.get("abstract_inverted_index")),
        doi=normalize_doi(work.get("doi")),
        source_url=loc.get("landing_page_url") or work["id"],
        oa_url=oa_url,
        license=loc.get("license"),
        full_text_status="open_access_link" if oa_url else "metadata_only",
        domains=[domain] if domain else [],
        processing_version=processing_version,
    )


class OpenAlexClient:
    RETRY_STATUS = {429, 500, 502, 503, 504}

    def __init__(
        self,
        api_key: str = "",
        base_url: str = "https://api.openalex.org",
        client: httpx.Client | None = None,
        max_retries: int = 4,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.max_retries = max_retries
        self._sleep = sleep
        self._client = client or httpx.Client(timeout=30)

    def _get(self, path: str, params: dict) -> dict:
        params = dict(params)
        if self.api_key:
            params["api_key"] = self.api_key
        for attempt in range(self.max_retries + 1):
            try:
                r = self._client.get(f"{self.base_url}{path}", params=params)
            except httpx.TransportError:
                if attempt == self.max_retries:
                    raise
                self._sleep(2**attempt)
                continue
            if r.status_code == 200:
                return r.json()
            if r.status_code in self.RETRY_STATUS and attempt < self.max_retries:
                self._sleep(2**attempt)  # exponential backoff: 1s, 2s, 4s, 8s
                continue
            r.raise_for_status()
        raise RuntimeError("unreachable")

    def search_works(
        self,
        query: str,
        year_from: int,
        work_types: list[str],
        max_results: int,
        per_page: int = 200,
    ) -> Iterator[dict]:
        """Yield raw works, following OpenAlex cursor pagination until max_results."""
        filters = (
            f"publication_year:>{year_from - 1},"
            f"type:{'|'.join(work_types)},"
            "has_abstract:true"
        )
        cursor, yielded = "*", 0
        while cursor and yielded < max_results:
            data = self._get(
                "/works",
                {
                    "search": query,
                    "filter": filters,
                    "select": SELECT,
                    "per_page": min(per_page, max_results - yielded),
                    "cursor": cursor,
                },
            )
            results = data.get("results", [])
            if not results:
                break
            for work in results:
                yield work
                yielded += 1
                if yielded >= max_results:
                    return
            cursor = (data.get("meta") or {}).get("next_cursor")
