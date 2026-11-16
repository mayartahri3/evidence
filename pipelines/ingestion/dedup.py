"""Deduplication (FR-08): DOI first, then normalized title; uncertain matches are flagged, not merged."""
from __future__ import annotations

import re
import unicodedata

from pipelines.ingestion.schema import Paper

MIN_TITLE_LEN = 15  # very short titles ("Introduction") are too generic to match on


def normalize_title(title: str) -> str:
    t = unicodedata.normalize("NFKD", title).encode("ascii", "ignore").decode()
    t = re.sub(r"[^a-z0-9]+", " ", t.lower())
    return t.strip()


def _merge(keep: Paper, other: Paper) -> Paper:
    """Keep the richer record, fill its gaps from the other, union the domain tags."""
    a, b = (keep, other) if (keep.abstract and not other.abstract) or (
        bool(keep.abstract) == bool(other.abstract) and len(keep.authors) >= len(other.authors)
    ) else (other, keep)
    data = a.model_dump()
    for field, value in b.model_dump().items():
        if data.get(field) in (None, [], "") and value not in (None, [], ""):
            data[field] = value
    data["domains"] = sorted(set(a.domains) | set(b.domains))
    return Paper(**data)


def deduplicate(papers: list[Paper]) -> tuple[list[Paper], list[dict], list[dict]]:
    """Return (unique_papers, merge_log, possible_duplicates)."""
    by_doi: dict[str, int] = {}
    by_title: dict[str, list[int]] = {}
    unique: list[Paper | None] = []
    merge_log: list[dict] = []
    possible: list[dict] = []

    for p in papers:
        idx = by_doi.get(p.doi) if p.doi else None
        reason = "same_doi"

        if idx is None:
            nt = normalize_title(p.title)
            if len(nt) >= MIN_TITLE_LEN:
                for cand in by_title.get(nt, []):
                    q = unique[cand]
                    if q is None:
                        continue
                    year_gap = abs((p.publication_year or 0) - (q.publication_year or 0))
                    both_years = p.publication_year and q.publication_year
                    if p.doi and q.doi and p.doi != q.doi:
                        # e.g. arXiv preprint vs published version: uncertain -> flag, keep both
                        possible.append({"a": q.paper_id, "b": p.paper_id, "title": p.title,
                                         "reason": "same_title_different_doi"})
                    elif both_years and year_gap > 1:
                        possible.append({"a": q.paper_id, "b": p.paper_id, "title": p.title,
                                         "reason": "same_title_years_differ"})
                    else:
                        idx, reason = cand, "same_title"
                        break

        if idx is None:
            unique.append(p)
            i = len(unique) - 1
            if p.doi:
                by_doi[p.doi] = i
            nt = normalize_title(p.title)
            if len(nt) >= MIN_TITLE_LEN:
                by_title.setdefault(nt, []).append(i)
        else:
            merged = _merge(unique[idx], p)
            merge_log.append({"kept": merged.paper_id, "dropped": p.paper_id
                              if merged.paper_id != p.paper_id else unique[idx].paper_id,
                              "reason": reason})
            unique[idx] = merged
            if merged.doi:
                by_doi[merged.doi] = idx
    return [p for p in unique if p is not None], merge_log, possible
