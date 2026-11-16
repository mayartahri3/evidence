"""Minimal paper schema (spec section 6.5)."""
from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

# metadata_only     : we only know bibliographic data (+ abstract if present)
# open_access_link  : an open-access URL exists, but the text is NOT downloaded yet
FullTextStatus = Literal["metadata_only", "open_access_link"]


class Paper(BaseModel):
    paper_id: str
    source: str = "OpenAlex"
    source_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    publication_year: int | None = None
    abstract: str | None = None
    doi: str | None = None
    source_url: str | None = None
    oa_url: str | None = None
    license: str | None = None
    full_text_status: FullTextStatus = "metadata_only"
    domains: list[str] = Field(default_factory=list)
    ingestion_date: str = Field(default_factory=lambda: date.today().isoformat())
    processing_version: str = "v1"
