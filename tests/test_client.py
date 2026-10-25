"""Tests for the arXiv client: parsing, caching, throttling and retries."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx
from httpx import Response

from arxiv_mcp import client as client_module
from arxiv_mcp.cache import TTLCache
from arxiv_mcp.client import (
    ARXIV_API_URL,
    ArxivAPIError,
    ArxivClient,
    PaperNotFoundError,
    is_valid_arxiv_id,
)

from .conftest import ATOM_FEED, EMPTY_FEED

# ---------------------------------------------------------------------------
# Identifier validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["2401.12345", "2401.12345v3", "cs.LO/0112017", "math.GT/0309136", "hep-th/9901001"],
)
def test_valid_arxiv_ids(value):
    assert is_valid_arxiv_id(value)


@pytest.mark.parametrize(
    "value",
    ["", "attention", "2401.123", "2401.123456", "arxiv.org/abs/2401.12345", "../../etc/passwd"],
)
def test_invalid_arxiv_ids(value):
    assert not is_valid_arxiv_id(value)


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------


@respx.mock
async def test_search_parses_feed():
    route = respx.get(ARXIV_API_URL).mock(return_value=Response(200, text=ATOM_FEED))

    async with ArxivClient(min_interval=0) as client:
        result = await client.search("attention", max_results=5)

    assert route.call_count == 1
    assert result.total_results == 1
    assert result.returned == 1
    assert result.papers[0].arxiv_id == "2401.12345"
    assert "attention" in str(route.calls.last.request.url)


@respx.mock
async def test_search_adds_category_filter():
    route = respx.get(ARXIV_API_URL).mock(return_value=Response(200, text=ATOM_FEED))

    async with ArxivClient(min_interval=0) as client:
        result = await client.search("agents", category="cs.AI")

    assert "cat%3Acs.AI" in str(route.calls.last.request.url) or "cat:cs.AI" in str(
        route.calls.last.request.url
    )
    assert result.query.startswith("cat:cs.AI")


@respx.mock
async def test_search_returns_empty_page():
    respx.get(ARXIV_API_URL).mock(return_value=Response(200, text=EMPTY_FEED))

    async with ArxivClient(min_interval=0) as client:
        result = await client.search("nothing matches this")

    assert result.returned == 0
    assert result.papers == []


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------


@respx.mock
async def test_cache_prevents_second_request(tmp_path):
    route = respx.get(ARXIV_API_URL).mock(return_value=Response(200, text=ATOM_FEED))
    cache = TTLCache(tmp_path / "c.sqlite", ttl_seconds=3600)

    async with ArxivClient(min_interval=0, cache=cache) as client:
        await client.search("attention")
        await client.search("attention")

    assert route.call_count == 1, "the second identical query should be served from cache"
    assert len(cache) == 1


@respx.mock
async def test_cache_miss_on_different_query(tmp_path):
    route = respx.get(ARXIV_API_URL).mock(return_value=Response(200, text=ATOM_FEED))
    cache = TTLCache(tmp_path / "c.sqlite", ttl_seconds=3600)

    async with ArxivClient(min_interval=0, cache=cache) as client:
        await client.search("attention")
        await client.search("transformers")

    assert route.call_count == 2


# ---------------------------------------------------------------------------
# Retries
# ---------------------------------------------------------------------------


@respx.mock
async def test_retries_transient_server_error():
    respx.get(ARXIV_API_URL).mock(side_effect=[Response(503), Response(200, text=ATOM_FEED)])

    async with ArxivClient(min_interval=0, max_retries=2) as client:
        result = await client.search("attention")

    assert result.returned == 1


@respx.mock
async def test_gives_up_after_max_retries():
    respx.get(ARXIV_API_URL).mock(return_value=Response(500))

    async with ArxivClient(min_interval=0, max_retries=1) as client:
        with pytest.raises(ArxivAPIError):
            await client.search("attention")


@respx.mock
async def test_malformed_feed_raises_api_error():
    respx.get(ARXIV_API_URL).mock(return_value=Response(200, text="<broken"))

    async with ArxivClient(min_interval=0) as client:
        with pytest.raises(ArxivAPIError):
            await client.search("attention")


# ---------------------------------------------------------------------------
# get_paper
# ---------------------------------------------------------------------------


@respx.mock
async def test_get_paper_by_id():
    respx.get(ARXIV_API_URL).mock(return_value=Response(200, text=ATOM_FEED))

    async with ArxivClient(min_interval=0) as client:
        paper = await client.get_paper("2401.12345v2")

    assert paper.arxiv_id == "2401.12345"
    assert paper.version == 2


@respx.mock
async def test_get_paper_not_found():
    respx.get(ARXIV_API_URL).mock(return_value=Response(200, text=EMPTY_FEED))

    async with ArxivClient(min_interval=0) as client:
        with pytest.raises(PaperNotFoundError):
            await client.get_paper("2401.99999")


async def test_get_paper_rejects_bad_identifier():
    async with ArxivClient(min_interval=0) as client:
        with pytest.raises(ValueError):
            await client.get_paper("not-an-id")


# ---------------------------------------------------------------------------
# category_trends
# ---------------------------------------------------------------------------


@respx.mock
async def test_category_trends_aggregates():
    feed = ATOM_FEED.replace(
        "<published>2024-01-15T00:00:00-05:00</published>",
        "<published>2024-01-15T00:00:00-05:00</published>",
    )
    respx.get(ARXIV_API_URL).mock(return_value=Response(200, text=feed))

    async with ArxivClient(min_interval=0) as client:
        trend = await client.category_trends("cs.CL", window_days=3650)

    assert trend.category == "cs.CL"
    assert trend.sampled == 1
    assert sum(trend.daily_counts.values()) == 1
    assert isinstance(trend.top_title_terms, list)


# ---------------------------------------------------------------------------
# Backoff policy
# ---------------------------------------------------------------------------


def _status_error(status: int, headers: dict[str, str] | None = None) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", ARXIV_API_URL)
    response = httpx.Response(status, headers=headers or {}, request=request)
    return httpx.HTTPStatusError(f"{status}", request=request, response=response)


def test_retry_delay_honours_retry_after():
    client = ArxivClient(min_interval=0, max_backoff=60)
    assert client._retry_delay(0, _status_error(429, {"Retry-After": "7"})) == 7.0


def test_retry_delay_is_capped():
    client = ArxivClient(min_interval=0, max_backoff=60)
    # A server asking for 10 minutes must not stall the tool for 10 minutes.
    assert client._retry_delay(0, _status_error(429, {"Retry-After": "600"})) == 60.0


def test_retry_delay_ignores_garbage_retry_after():
    client = ArxivClient(min_interval=0, base_backoff=1.0)
    delay = client._retry_delay(0, _status_error(429, {"Retry-After": "soon"}))
    assert 0.0 <= delay <= 1.0


def test_retry_delay_uses_jitter_without_header():
    client = ArxivClient(min_interval=0, base_backoff=1.0, max_backoff=60)
    delays = {client._retry_delay(2, httpx.ConnectError("boom")) for _ in range(30)}
    # Full jitter: uniform in [0, min(base * 2**attempt, cap)] == [0, 4]
    assert delays, "expected some spread"
    assert all(0.0 <= d <= 4.0 for d in delays)


def test_describe_names_the_status():
    client = ArxivClient(min_interval=0)
    assert client._describe(_status_error(503)) == "HTTP 503"
    assert "ConnectError" in client._describe(httpx.ConnectError("refused"))


@respx.mock
async def test_error_message_names_the_status():
    respx.get(ARXIV_API_URL).mock(return_value=Response(500))

    async with ArxivClient(min_interval=0, max_retries=0) as client:
        with pytest.raises(ArxivAPIError, match="HTTP 500"):
            await client.search("attention")


@respx.mock
async def test_retry_after_shortens_the_wait(monkeypatch):
    """A 429 with Retry-After must be respected rather than slept through blindly."""
    respx.get(ARXIV_API_URL).mock(
        side_effect=[
            Response(429, headers={"Retry-After": "2"}),
            Response(200, text=ATOM_FEED),
        ]
    )
    recorded: list[float] = []
    real_sleep = asyncio.sleep

    async def fake_sleep(seconds: float) -> None:
        recorded.append(seconds)
        await real_sleep(0)

    monkeypatch.setattr(client_module.asyncio, "sleep", fake_sleep)

    async with ArxivClient(min_interval=0, max_retries=2) as c:
        result = await c.search("attention")

    assert result.returned == 1
    assert recorded == [2.0], f"expected the Retry-After delay, got {recorded}"
