"""Data models for arxiv-mcp.

Everything that crosses the MCP boundary is a pydantic model so that clients
receive validated ``structuredContent`` instead of untyped blobs.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class Paper(BaseModel):
    """A single arXiv preprint, flattened out of the Atom entry."""

    arxiv_id: str = Field(description="Bare arXiv identifier without version, e.g. '2401.12345'.")
    version: int = Field(default=1, description="Version of the record that was returned.")
    title: str = Field(description="Paper title, whitespace-normalised.")
    authors: list[str] = Field(description="Author names in the order arXiv lists them.")
    abstract: str = Field(description="Abstract text, whitespace-normalised.")
    published: datetime = Field(description="Date of v1 submission.")
    updated: datetime = Field(description="Date of the latest revision.")
    primary_category: str = Field(description="Primary arXiv category, e.g. 'cs.AI'.")
    categories: list[str] = Field(description="All categories the paper is tagged with.")
    abs_url: str = Field(description="Canonical abstract page.")
    pdf_url: str = Field(description="Canonical PDF URL.")
    doi: str | None = Field(default=None, description="Journal DOI when one is registered.")
    journal_ref: str | None = Field(default=None, description="Journal reference, when present.")
    comment: str | None = Field(
        default=None, description="Author comment, e.g. page count or venue."
    )


class SearchResult(BaseModel):
    """The result of an arXiv API query."""

    query: str = Field(description="The query that produced this result.")
    total_results: int = Field(description="Total matches arXiv reports, ignoring pagination.")
    returned: int = Field(description="Number of papers actually returned in this page.")
    start: int = Field(description="Zero-based offset into the full result set.")
    sort_by: str = Field(description="Sort order arXiv applied.")
    papers: list[Paper] = Field(description="The papers in this page.")


class PdfText(BaseModel):
    """Text extracted from a paper's PDF."""

    arxiv_id: str
    page_count: int = Field(description="Pages in the downloaded PDF.")
    extracted_pages: int = Field(description="Pages that were actually parsed.")
    truncated: bool = Field(description="True when the PDF was longer than max_pages.")
    text: str = Field(description="Extracted text, pages joined by form feeds.")


class CategoryTrend(BaseModel):
    """Submission-volume statistics for one arXiv category."""

    category: str = Field(description="arXiv category, e.g. 'cs.AI'.")
    window_days: int = Field(description="How many days back the sample covers.")
    sampled: int = Field(description="Papers examined (bounded by arXiv's page limit).")
    daily_counts: dict[str, int] = Field(
        description=(
            "ISO date -> submissions among the sampled papers. A sample, not the "
            "category's full volume; days outside the sampled range are absent."
        )
    )
    top_title_terms: list[tuple[str, int]] = Field(
        description="(term, count) pairs from paper titles, most frequent first."
    )
