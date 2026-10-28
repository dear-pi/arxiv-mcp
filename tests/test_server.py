"""End-to-end tests over an in-memory MCP client - no network, no subprocess."""

from __future__ import annotations

import pytest
from mcp import Client, MCPError

import arxiv_mcp.server as server_module
from arxiv_mcp.client import ArxivAPIError, PaperNotFoundError

from .conftest import FakeClient


@pytest.fixture
def server_with_fake(monkeypatch, fake_client):
    """Point the server at a fake client so no HTTP ever happens."""
    monkeypatch.setattr(server_module, "get_client", lambda: fake_client)
    return fake_client


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


async def test_lists_tools_resources_and_prompts(server_with_fake):
    async with Client(server_module.mcp) as client:
        tools = {t.name for t in (await client.list_tools()).tools}
        listed = await client.list_resources()
        templates = await client.list_resource_templates()
        resources = list(listed.resources)
        prompts = {p.name for p in (await client.list_prompts()).prompts}

    assert tools == {
        "search_papers",
        "get_paper",
        "get_pdf_text",
        "category_trends",
    }
    assert {str(r.uri) for r in resources} == {"arxiv://categories"}
    assert {t.uri_template for t in templates.resource_templates} == {"arxiv://paper/{arxiv_id}"}
    assert prompts == {"literature_review"}


async def test_tool_schemas_are_generated_from_type_hints(server_with_fake):
    async with Client(server_module.mcp) as client:
        tools = {t.name: t for t in (await client.list_tools()).tools}

    schema = tools["search_papers"].input_schema
    assert schema["type"] == "object"
    assert "query" in schema["properties"]
    assert "max_results" in schema["properties"]


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


async def test_search_papers_returns_structured_content(server_with_fake):
    async with Client(server_module.mcp) as client:
        result = await client.call_tool("search_papers", {"query": "attention", "max_results": 3})

    assert result.is_error is False
    assert result.structured_content["returned"] == 1
    assert result.structured_content["papers"][0]["arxiv_id"] == "2401.12345"


async def test_search_papers_forwards_arguments(server_with_fake):
    async with Client(server_module.mcp) as client:
        await client.call_tool(
            "search_papers",
            {"query": "agents", "sort_by": "submittedDate", "category": "cs.AI"},
        )

    name, args, kwargs = server_with_fake.calls[-1]
    assert name == "search"
    assert args == ("agents",)
    assert kwargs["sort_by"] == "submittedDate"
    assert kwargs["category"] == "cs.AI"


async def test_search_papers_api_error_becomes_tool_error(monkeypatch):
    monkeypatch.setattr(
        server_module, "get_client", lambda: FakeClient(raises=ArxivAPIError("boom"))
    )
    async with Client(server_module.mcp) as client:
        result = await client.call_tool("search_papers", {"query": "x"})

    # A ToolError is a result the model can read and recover from.
    assert result.is_error is True
    assert "unreachable" in result.content[0].text


async def test_get_paper_rejects_bad_identifier(server_with_fake):
    async with Client(server_module.mcp) as client:
        result = await client.call_tool("get_paper", {"arxiv_id": "not-an-id"})

    assert result.is_error is True
    assert "not a valid arXiv identifier" in result.content[0].text


async def test_get_paper_not_found_is_recoverable(monkeypatch):
    monkeypatch.setattr(
        server_module, "get_client", lambda: FakeClient(raises=PaperNotFoundError("no record"))
    )
    async with Client(server_module.mcp) as client:
        result = await client.call_tool("get_paper", {"arxiv_id": "2401.99999"})

    assert result.is_error is True
    assert "no record" in result.content[0].text


async def test_get_pdf_text_returns_extraction_metadata(server_with_fake):
    async with Client(server_module.mcp) as client:
        result = await client.call_tool("get_pdf_text", {"arxiv_id": "2401.12345", "max_pages": 2})

    assert result.is_error is False
    body = result.structured_content
    assert body["extracted_pages"] == 2
    assert body["truncated"] is True


async def test_category_trends(server_with_fake):
    async with Client(server_module.mcp) as client:
        result = await client.call_tool("category_trends", {"category": "cs.AI"})

    body = result.structured_content
    assert body["category"] == "cs.AI"
    assert sum(body["daily_counts"].values()) == 3
    assert body["top_title_terms"][0] == ["attention", 2]


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


async def test_paper_resource_renders_markdown(server_with_fake):
    async with Client(server_module.mcp) as client:
        result = await client.read_resource("arxiv://paper/2401.12345")

    text = result.contents[0].text
    assert "# Attention Is All You Need" in text
    assert "**arXiv**: [2401.12345]" in text
    assert "10.1234/abcd.2024.01" in text


async def test_categories_resource_lists_taxonomy(server_with_fake):
    async with Client(server_module.mcp) as client:
        result = await client.read_resource("arxiv://categories")

    text = result.contents[0].text
    assert "cs.AI" in text
    assert "stat.ML" in text


async def test_unknown_paper_resource_raises(monkeypatch):
    monkeypatch.setattr(
        server_module, "get_client", lambda: FakeClient(raises=PaperNotFoundError("nope"))
    )
    async with Client(server_module.mcp) as client:
        with pytest.raises(MCPError):
            await client.read_resource("arxiv://paper/9999.99999")


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


async def test_literature_review_prompt_renders(server_with_fake):
    async with Client(server_module.mcp) as client:
        result = await client.get_prompt(
            "literature_review", {"topic": "retrieval augmented generation", "max_papers": "5"}
        )

    assert len(result.messages) == 1
    text = result.messages[0].content.text
    assert "retrieval augmented generation" in text
    assert "category_trends" in text
    assert "search_papers" in text


# ---------------------------------------------------------------------------
# Client construction
# ---------------------------------------------------------------------------


async def test_get_client_wires_the_ttl_cache(monkeypatch, tmp_path):
    """The cache module only earns its keep if the server actually passes it in."""
    monkeypatch.setattr(server_module, "_client", None)
    monkeypatch.setenv("ARXIV_MCP_CACHE_PATH", str(tmp_path / "cache.sqlite"))
    monkeypatch.delenv("ARXIV_MCP_CACHE", raising=False)

    client = server_module.get_client()
    try:
        assert client._cache is not None
        assert (tmp_path / "cache.sqlite").is_file()
    finally:
        server_module._client = None
        await client.aclose()


async def test_get_client_cache_can_be_disabled(monkeypatch, tmp_path):
    monkeypatch.setattr(server_module, "_client", None)
    monkeypatch.setenv("ARXIV_MCP_CACHE", "off")
    monkeypatch.setenv("ARXIV_MCP_CACHE_PATH", str(tmp_path / "cache.sqlite"))

    client = server_module.get_client()
    try:
        assert client._cache is None
    finally:
        server_module._client = None
        await client.aclose()
