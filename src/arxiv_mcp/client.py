"""The arXiv API client.

Responsibilities kept deliberately out of the MCP layer:

* throttle requests to the rate arXiv asks for (>= 3 s apart),
* retry transient failures with exponential backoff + jitter,
* cache raw responses so repeat queries are free,
* parse the Atom feed into :class:`Paper` models.

Everything here is plain async code with no MCP imports, which is what makes
it unit-testable without spinning up a server.
"""

from __future__ import annotations

import asyncio
import io
import random
import re
import time
import xml.etree.ElementTree as ET
from datetime import UTC, datetime, timedelta
from typing import Literal

import httpx
from pydantic import ValidationError

from .cache import TTLCache
from .models import CategoryTrend, Paper, PdfText, SearchResult

ARXIV_API_URL = "https://export.arxiv.org/api/query"
ARXIV_PDF_URL = "https://arxiv.org/pdf/{arxiv_id}"

# arXiv asks for a descriptive User-Agent so they can contact you if you
# misbehave. An anonymous default agent is a good way to get rate-limited.
DEFAULT_USER_AGENT = "arxiv-mcp/0.1.0 (educational project; https://github.com/dear-pi/arxiv-mcp)"

# Accepts the modern form (2401.12345) and the legacy pre-2007 form
# (cs.LO/0112017 or math.GT/0309136), with an optional version suffix.
ARXIV_ID_RE = re.compile(r"^(?:\d{4}\.\d{4,5}|[a-zA-Z][a-zA-Z.\-]*/\d{7})(v\d+)?$")

NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
    "opensearch": "http://a9.com/-/spec/opensearch/1.1/",
}

RETRYABLE_STATUS = frozenset({429, 500, 502, 503, 504})

SortBy = Literal["relevance", "submittedDate", "lastUpdatedDate"]

# English stop words + arXiv boilerplate, used by the trend tool.
_STOP_WORDS = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "of",
        "for",
        "to",
        "in",
        "on",
        "with",
        "without",
        "from",
        "by",
        "via",
        "using",
        "use",
        "based",
        "toward",
        "towards",
        "novel",
        "new",
        "study",
        "studies",
        "paper",
        "approach",
        "method",
        "methods",
        "framework",
        "system",
        "model",
        "models",
        "data",
        "results",
        "result",
        "analysis",
        "is",
        "are",
        "be",
        "we",
        "our",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "at",
        "between",
        "across",
        "more",
        "most",
        "less",
        "least",
        "can",
        "could",
        "may",
        "might",
        "will",
        "would",
        "shall",
        "should",
        "do",
        "does",
        "not",
        "no",
        "non",
        "two",
        "three",
        "one",
        "through",
        "over",
        "under",
        "into",
        "out",
        "up",
        "down",
        "off",
        "again",
        "further",
        "then",
        "once",
        "here",
        "there",
        "when",
        "where",
        "why",
        "how",
        "all",
        "any",
        "both",
        "each",
        "few",
        "other",
        "some",
        "such",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "just",
        "about",
        "above",
        "below",
        "vs",
        "versus",
    ]
)


class ArxivError(RuntimeError):
    """Base class for arXiv failures."""


class PaperNotFoundError(ArxivError):
    """arXiv has no record for the requested identifier."""


class ArxivAPIError(ArxivError):
    """arXiv returned an error status or an unparseable body."""


def is_valid_arxiv_id(value: str) -> bool:
    return bool(ARXIV_ID_RE.match(value.strip()))


def _normalise_ws(text: str) -> str:
    """arXiv wraps titles/abstracts arbitrarily; collapse the whitespace."""
    return re.sub(r"\s+", " ", text or "").strip()


def _strip_version(arxiv_id: str) -> str:
    return arxiv_id.strip().split("v", 1)[0] if "v" in arxiv_id.split("/")[-1] else arxiv_id.strip()


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _entry_to_paper(entry: ET.Element) -> Paper:
    raw_id = entry.findtext("atom:id", default="", namespaces=NS)
    # 'http://arxiv.org/abs/2401.12345v3' -> '2401.12345', version 3
    tail = raw_id.rsplit("/", 1)[-1] if raw_id else ""
    version = 1
    if "v" in tail:
        base, _, ver = tail.rpartition("v")
        if base and ver.isdigit():
            tail, version = base, int(ver)

    categories = [c.get("term", "") for c in entry.findall("atom:category", NS)]
    primary_el = entry.find("arxiv:primary_category", NS)
    primary = (
        primary_el.get("term", "")
        if primary_el is not None
        else (categories[0] if categories else "")
    )

    def text_of(tag: str, namespace: str = "atom") -> str | None:
        value = entry.findtext(f"{namespace}:{tag}", namespaces=NS)
        return value.strip() if value else None

    return Paper(
        arxiv_id=tail,
        version=version,
        title=_normalise_ws(entry.findtext("atom:title", default="", namespaces=NS)),
        authors=[
            _normalise_ws(a.findtext("atom:name", default="", namespaces=NS))
            for a in entry.findall("atom:author", NS)
        ],
        abstract=_normalise_ws(entry.findtext("atom:summary", default="", namespaces=NS)),
        published=_parse_datetime(entry.findtext("atom:published", namespaces=NS)),
        updated=_parse_datetime(entry.findtext("atom:updated", namespaces=NS)),
        primary_category=primary,
        categories=[c for c in categories if c],
        abs_url=f"https://arxiv.org/abs/{tail}",
        pdf_url=f"https://arxiv.org/pdf/{tail}",
        doi=text_of("doi", "arxiv"),
        journal_ref=text_of("journal_ref", "arxiv"),
        comment=text_of("comment", "arxiv"),
    )


def parse_feed(xml: str) -> tuple[list[Paper], int]:
    """Parse an arXiv Atom response into papers and the reported total."""
    try:
        root = ET.fromstring(xml)
    except ET.ParseError as exc:  # pragma: no cover - defensive
        raise ArxivAPIError(f"arXiv returned a malformed Atom feed: {exc}") from exc

    total_el = root.find("opensearch:totalResults", NS)
    try:
        total = int(total_el.text) if total_el is not None and total_el.text else 0
    except ValueError:
        total = 0

    papers: list[Paper] = []
    for entry in root.findall("atom:entry", NS):
        try:
            papers.append(_entry_to_paper(entry))
        except (ValidationError, AttributeError, TypeError):
            # A single malformed entry should not sink the whole query.
            continue
    return papers, total


class ArxivClient:
    """Async arXiv API client with throttling, retries and caching."""

    def __init__(
        self,
        *,
        timeout: float = 20.0,
        max_retries: int = 4,
        min_interval: float = 3.0,
        base_backoff: float = 1.0,
        max_backoff: float = 60.0,
        cache: TTLCache | None = None,
        http_client: httpx.AsyncClient | None = None,
        user_agent: str = DEFAULT_USER_AGENT,
    ) -> None:
        self._min_interval = min_interval
        self._max_retries = max_retries
        self._base_backoff = base_backoff
        self._max_backoff = max_backoff
        self._cache = cache
        self._owns_http = http_client is None
        self._http = http_client or httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            headers={"User-Agent": user_agent},
            follow_redirects=True,
        )
        self._throttle_lock = asyncio.Lock()
        self._last_request_at = 0.0

    async def aclose(self) -> None:
        if self._owns_http:
            await self._http.aclose()

    async def __aenter__(self) -> ArxivClient:
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        await self.aclose()

    async def _throttle(self) -> None:
        """Serialise requests and enforce a minimum gap between them."""
        async with self._throttle_lock:
            wait = self._min_interval - (time.monotonic() - self._last_request_at)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_at = time.monotonic()

    def _retry_delay(self, attempt: int, exc: Exception) -> float:
        """How long to wait before attempt ``attempt + 1``.

        Honours ``Retry-After`` when the server sends one. Otherwise full
        jitter: ``uniform(0, min(base * 2**attempt, cap))``. Capped, because
        arXiv's rate limiter can sit on an IP for minutes and unbounded
        backoff is worse than a clear error.
        """
        retry_after = self._parse_retry_after(exc)
        if retry_after is not None:
            return min(retry_after, self._max_backoff)
        ceiling = min(self._base_backoff * (2**attempt), self._max_backoff)
        return ceiling * random.random()

    @staticmethod
    def _parse_retry_after(exc: Exception) -> float | None:
        response = getattr(exc, "response", None)
        if response is None:
            return None
        raw = response.headers.get("retry-after")
        if not raw:
            return None
        try:
            return max(0.0, float(raw))
        except ValueError:
            return None

    @staticmethod
    def _describe(exc: Exception) -> str:
        """A short, useful description - httpx's own str() is often empty."""
        if isinstance(exc, httpx.HTTPStatusError):
            return f"HTTP {exc.response.status_code}"
        return f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__

    async def _get(self, url: str, params: dict[str, str | int]) -> str:
        """GET with cache lookup, throttling and retry-with-backoff."""
        request = self._http.build_request("GET", url, params=params)
        cache_key = str(request.url)

        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        last_error: Exception | None = None
        for attempt in range(self._max_retries + 1):
            await self._throttle()
            try:
                response = await self._http.send(request, stream=False)
                if response.status_code in RETRYABLE_STATUS:
                    raise httpx.HTTPStatusError(
                        f"arXiv returned {response.status_code}",
                        request=request,
                        response=response,
                    )
                response.raise_for_status()
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                last_error = exc
                if attempt == self._max_retries:
                    break
                await asyncio.sleep(self._retry_delay(attempt, exc))
                continue

            body = response.text
            if self._cache is not None:
                self._cache.set(cache_key, body)
            return body

        raise ArxivAPIError(
            f"arXiv request failed after {self._max_retries + 1} attempts "
            f"({self._describe(last_error) if last_error else 'unknown error'}). "
            "arXiv rate-limits aggressively; wait a minute before retrying."
        )

    async def search(
        self,
        query: str,
        *,
        max_results: int = 10,
        start: int = 0,
        sort_by: SortBy = "relevance",
        category: str | None = None,
    ) -> SearchResult:
        """Run an arXiv query and return a page of results."""
        search_query = query
        if category:
            search_query = f"cat:{category} AND ({query})" if query else f"cat:{category}"

        papers, total = parse_feed(
            await self._get(
                ARXIV_API_URL,
                {
                    "search_query": search_query,
                    "start": start,
                    "max_results": max_results,
                    "sortBy": sort_by,
                    "sortOrder": "descending",
                },
            )
        )
        return SearchResult(
            query=search_query,
            total_results=total,
            returned=len(papers),
            start=start,
            sort_by=sort_by,
            papers=papers,
        )

    async def get_paper(self, arxiv_id: str) -> Paper:
        """Fetch one paper by identifier. Raises PaperNotFoundError on a miss."""
        arxiv_id = _strip_version(arxiv_id)
        if not is_valid_arxiv_id(arxiv_id):
            raise ValueError(f"{arxiv_id!r} is not a valid arXiv identifier")

        papers, _ = parse_feed(
            await self._get(ARXIV_API_URL, {"id_list": arxiv_id, "max_results": 1})
        )
        if not papers:
            raise PaperNotFoundError(f"arXiv has no record for {arxiv_id!r}")
        return papers[0]

    async def fetch_pdf_text(self, arxiv_id: str, *, max_pages: int = 5) -> PdfText:
        """Download a paper's PDF and extract text from its first pages."""
        try:
            from pypdf import PdfReader
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "PDF extraction needs the 'pdf' extra: pip install 'arxiv-mcp[pdf]'"
            ) from exc

        arxiv_id = _strip_version(arxiv_id)
        if not is_valid_arxiv_id(arxiv_id):
            raise ValueError(f"{arxiv_id!r} is not a valid arXiv identifier")

        url = ARXIV_PDF_URL.format(arxiv_id=arxiv_id)
        for attempt in range(self._max_retries + 1):
            await self._throttle()
            try:
                response = await self._http.get(url)
                if response.status_code in RETRYABLE_STATUS:
                    raise httpx.HTTPStatusError(
                        f"arxiv.org returned {response.status_code}",
                        request=response.request,
                        response=response,
                    )
                response.raise_for_status()
                break
            except (httpx.HTTPStatusError, httpx.TransportError) as exc:
                if attempt == self._max_retries:
                    raise ArxivAPIError(
                        f"Could not download the PDF for {arxiv_id} ({self._describe(exc)})."
                    ) from exc
                await asyncio.sleep(self._retry_delay(attempt, exc))
        else:  # pragma: no cover - the loop always breaks or raises
            raise ArxivAPIError("unreachable")

        reader = PdfReader(io.BytesIO(response.content))
        pages: list[str] = []
        for page in reader.pages[:max_pages]:
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # pypdf raises assorted errors on damaged pages
                pages.append("")
        return PdfText(
            arxiv_id=arxiv_id,
            page_count=len(reader.pages),
            extracted_pages=len(pages),
            truncated=len(reader.pages) > max_pages,
            text="\f".join(pages),
        )

    async def category_trends(
        self, category: str, *, window_days: int = 30, sample_size: int = 100
    ) -> CategoryTrend:
        """Summarise recent submission volume and recurring title terms."""
        result = await self.search(
            "",
            max_results=sample_size,
            sort_by="submittedDate",
            category=category,
        )
        cutoff = datetime.now(UTC) - timedelta(days=window_days)
        daily: dict[str, int] = {}
        terms: dict[str, int] = {}

        for paper in result.papers:
            if paper.published < cutoff:
                continue
            day = paper.published.date().isoformat()
            daily[day] = daily.get(day, 0) + 1
            for word in re.findall(r"[A-Za-z][A-Za-z\-]{2,}", paper.title.lower()):
                if word not in _STOP_WORDS:
                    terms[word] = terms.get(word, 0) + 1

        top_terms = sorted(terms.items(), key=lambda kv: (-kv[1], kv[0]))[:20]
        return CategoryTrend(
            category=category,
            window_days=window_days,
            sampled=len(result.papers),
            daily_counts=dict(sorted(daily.items())),
            top_title_terms=[(term, count) for term, count in top_terms],
        )
