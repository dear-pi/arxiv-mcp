"""Shared test fixtures: a realistic arXiv Atom feed and a fake client."""

from __future__ import annotations

import pytest

from arxiv_mcp.client import ArxivClient
from arxiv_mcp.models import CategoryTrend, Paper, PdfText, SearchResult

ATOM_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <link
    href="http://arxiv.org/api/query?search_query=all:attention"
    rel="self"
    type="application/atom+xml"
  />
  <title type="html">ArXiv Query</title>
  <id>http://arxiv.org/api/query</id>
  <updated>2024-01-16T00:00:00-05:00</updated>
  <opensearch:totalResults xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">1</opensearch:totalResults>
  <opensearch:startIndex xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">0</opensearch:startIndex>
  <opensearch:itemsPerPage xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">1</opensearch:itemsPerPage>
  <entry>
    <id>http://arxiv.org/abs/2401.12345v2</id>
    <updated>2024-01-16T00:00:00-05:00</updated>
    <published>2024-01-15T00:00:00-05:00</published>
    <title>Attention Is All You Need</title>
    <summary>  The dominant sequence transduction
models are based on complex recurrent or convolutional neural networks.
</summary>
    <author><name>Ashish Vaswani</name></author>
    <author><name>Noam Shazeer</name></author>
    <author><name>Niki Parmar</name></author>
    <arxiv:doi xmlns:arxiv="http://arxiv.org/schemas/atom">10.1234/abcd.2024.01</arxiv:doi>
    <link href="http://arxiv.org/abs/2401.12345v2" rel="alternate" type="text/html"/>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>
"""

EMPTY_FEED = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title type="html">ArXiv Query</title>
  <id>http://arxiv.org/api/query</id>
  <updated>2024-01-16T00:00:00-05:00</updated>
  <opensearch:totalResults xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">0</opensearch:totalResults>
</feed>
"""

PAPER = Paper(
    arxiv_id="2401.12345",
    version=2,
    title="Attention Is All You Need",
    authors=["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
    abstract="The dominant sequence transduction models.",
    published="2024-01-15T00:00:00+00:00",
    updated="2024-01-16T00:00:00+00:00",
    primary_category="cs.CL",
    categories=["cs.CL", "cs.LG"],
    abs_url="https://arxiv.org/abs/2401.12345",
    pdf_url="https://arxiv.org/pdf/2401.12345",
    doi="10.1234/abcd.2024.01",
)


class FakeClient:
    """Records calls and returns canned results - no network in server tests."""

    def __init__(self, *, raises: Exception | None = None) -> None:
        self.calls: list[tuple[str, tuple, dict]] = []
        self.raises = raises

    def _record(self, name: str, *args, **kwargs):
        self.calls.append((name, args, kwargs))
        if self.raises is not None:
            raise self.raises

    async def search(self, query: str, **kwargs) -> SearchResult:
        self._record("search", query, **kwargs)
        return SearchResult(
            query=query,
            total_results=1,
            returned=1,
            start=0,
            sort_by=kwargs.get("sort_by", "relevance"),
            papers=[PAPER],
        )

    async def get_paper(self, arxiv_id: str) -> Paper:
        self._record("get_paper", arxiv_id)
        return PAPER

    async def fetch_pdf_text(self, arxiv_id: str, *, max_pages: int = 5) -> PdfText:
        self._record("fetch_pdf_text", arxiv_id, max_pages=max_pages)
        return PdfText(
            arxiv_id=arxiv_id,
            page_count=8,
            extracted_pages=max_pages,
            truncated=True,
            text="extracted text",
        )

    async def category_trends(self, category: str, **kwargs) -> CategoryTrend:
        self._record("category_trends", category, **kwargs)
        return CategoryTrend(
            category=category,
            window_days=30,
            sampled=3,
            daily_counts={"2024-01-15": 2, "2024-01-16": 1},
            top_title_terms=[["attention", 2], ["models", 1]],
        )


@pytest.fixture
def fake_client() -> FakeClient:
    return FakeClient()


@pytest.fixture
def no_http_client() -> ArxivClient:
    """A real client whose HTTP layer never reaches the network."""
    import httpx

    return ArxivClient(
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: _boom(r)))
    )


async def _boom(request):  # pragma: no cover - safety net
    raise AssertionError(f"unexpected network call to {request.url}")
