# arxiv-mcp

An [MCP](https://modelcontextprotocol.io) server that turns arXiv into a set of tools an
LLM agent can call: search, fetch metadata, extract PDF text, and summarise how active a
research category is.

Built as a learning project while studying for a CS degree, targeting LLM-application /
agent engineering internships.

```
you / an agent
      │  tools/call  "search_papers"
      ▼
┌──────────────┐   throttled + cached HTTP    ┌──────────────┐
│  arxiv-mcp   │ ───────────────────────────▶ │  arXiv API   │
│  (MCP server)│ ◀─────────────────────────── │  Atom feed   │
└──────────────┘        retry w/ backoff      └──────────────┘
      │  validated pydantic models
      ▼
structuredContent the model can actually reason about
```

## What it exposes

### Tools

| Tool | What it does | Why it is not just a proxy |
| --- | --- | --- |
| `search_papers(query, max_results, sort_by, category)` | arXiv search with field syntax (`au:`, `ti:`, `abs:`) | Optional category filter is composed into `cat:X AND (...)` rather than silently ignored |
| `get_paper(arxiv_id)` | Full metadata for one paper | Strips version suffixes, validates both modern (`2401.12345`) and legacy (`cs.LO/0112017`) identifier forms |
| `get_pdf_text(arxiv_id, max_pages)` | Downloads the PDF and extracts text from the first pages | Reports page count, how many were parsed, and whether it truncated — so the model knows what it is missing |
| `category_trends(category, window_days)` | Submission volume per day + most frequent title terms, over a sample of the newest papers | An analysis tool, not a lookup: it aggregates and tokenises so an agent can answer "is this field heating up?" - the counts are a sample estimate, not the category's true volume |

### Resources

- `arxiv://paper/{arxiv_id}` — a Markdown card with the paper's full metadata
- `arxiv://categories` — the arXiv taxonomy the server can filter on

### Prompts

- `literature_review(topic, max_papers)` — a multi-step workflow that instructs an agent
  to check category trends first, try several phrasings, then group findings and cite
  every claim with an `arxiv_id`

## Install and run

```bash
pip install -e ".[pdf]"      # the pdf extra enables get_pdf_text
arxiv-mcp                    # serves over stdio
```

Point any MCP host at it. For Claude Desktop / Claude Code:

```json
{
  "mcpServers": {
    "arxiv": {
      "command": "uv",
      "args": ["run", "--with", "arxiv-mcp[pdf]", "arxiv-mcp"]
    }
  }
}
```

To poke at it interactively:

```bash
uv run mcp dev src/arxiv_mcp/server.py    # opens the MCP Inspector
```

## Design notes

**Rate limiting is a feature, not an afterthought.** arXiv asks clients to keep requests
at least 3 seconds apart and to send a descriptive `User-Agent`. `ArxivClient` enforces
both: an `asyncio.Lock` serialises requests and a monotonic clock enforces the gap, so
concurrent tool calls cannot stampede the API. Getting rate-limited by arXiv would break
every tool at once.

**Retries with full jitter, capped, and `Retry-After`-aware.** Transient
429/5xx responses and transport errors are retried up to `max_retries` times
with a backoff of `uniform(0, min(base * 2**attempt, cap))`. The cap matters:
arXiv's rate limiter can sit on an IP for minutes, and unbounded backoff is
worse than a clear error. When the server sends `Retry-After` it is honoured
(subject to the same cap) instead of guessing.

This is not theoretical. During development the first version retried 3 times
with sub-second backoff and no `Retry-After` handling; testing against the live
API got the sandbox IP rate-limited, and the failure was slow and opaque. Both
the backoff and the error message (`HTTP 429` rather than an empty string) were
rewritten after that.

**The API URL is `https://`.** `http://export.arxiv.org` answers `301` to
`https://`, which silently doubles every request and interacts badly with
retries. Found the same way - by actually running it.

**Caching raw response bodies.** Responses are cached keyed by the fully-built request
URL in SQLite with a TTL. Keying on the URL (rather than a hand-built cache key) means
the key is correct by construction. The server wires this in when it builds its client:
the database lives under the platform cache directory
(`%LOCALAPPDATA%\arxiv-mcp\cache.sqlite` on Windows, `~/.cache/arxiv-mcp/cache.sqlite`
elsewhere), can be moved with `ARXIV_MCP_CACHE_PATH`, and is disabled with
`ARXIV_MCP_CACHE=off`. A read-only home directory degrades to "no cache" instead of
stopping the server - caching is an optimisation, not a requirement.

**Error policy follows the MCP spec.** Every failure is placed in one of three buckets:

| Raised | Model sees | Used here for |
| --- | --- | --- |
| `ToolError` | the message, `is_error=True` | paper not found, arXiv unreachable, empty search result |
| `MCPError` | nothing — the request itself fails | missing optional dependency (`pypdf` not installed) |
| anything else | only `Error executing tool X` | bugs |

The deciding question is *"could a smarter model have avoided this?"*. A misspelled arXiv
id can be corrected by the model, so it gets the message. A missing Python package cannot,
so it is escalated to the host.

**Models everywhere.** Every tool returns a pydantic model, so clients receive validated
`structuredContent` instead of untyped strings. Bad arguments are rejected by the SDK's
schema validation before the handler runs, so the handlers do not re-validate type hints.

**The arXiv layer has no MCP imports.** `arxiv_mcp.client` is plain async code, which is
what makes it unit-testable without starting a server — see `tests/test_client.py`.

## Tests

```bash
pytest                       # 55 tests
ruff check .                # lint
```

- `tests/test_client.py` — feed parsing, cache hit/miss, retry behaviour, identifier
  validation. HTTP is mocked with `respx`, so nothing touches the network. Includes
  the backoff policy: `Retry-After` honoured, capped, jittered.
- `tests/test_server.py` — the whole server exercised through an in-memory MCP `Client`:
  tool/resource/prompt discovery, structured output, and that each error surfaces in the
  right bucket. Also asserts the server actually hands a `TTLCache` to its client
  (`ARXIV_MCP_CACHE_PATH` / `ARXIV_MCP_CACHE=off`), so the cache cannot silently become
  dead code again.
- `tests/test_models.py`, `tests/test_cache.py` — validation and TTL expiry.

## Verification status

| Check | Result |
| --- | --- |
| `pytest` | 55 passed |
| `ruff check` / `ruff format --check` | clean |
| Live `get_paper("1706.03762")` | returns "Attention Is All You Need", 3 authors, `cs.CL`/`cs.LG` |
| Live `category_trends("cs.AI")` | returns daily counts and title terms |
| Live `search_papers("retrieval augmented generation")` | returns 3 results out of 1,029,300 matches |

Free-text search was the last endpoint confirmed live. During development the build
machine's shared egress IP was rate-limited (repeated `429`/`ReadTimeout` on `search`
while `id_list` and `cat:` kept working), so for a while only the `respx`-mocked tests
covered the happy path. Re-run from a different egress and it returns results normally —
the earlier failures were the IP, not the client. The 3-second throttle and the capped
`Retry-After`-aware backoff exist precisely because that endpoint is easy to get wrong.

## Limitations

- `search_papers` returns one page (max 100). arXiv's `start` offset works but is not
  exposed as a tool argument.
- `category_trends` samples the newest N papers once, so `daily_counts` only covers the
  days those papers span. For a busy category like `cs.AI` that is a few days, not the
  requested 30-day window - it estimates activity, it does not count submissions.
- `get_pdf_text` needs the `pdf` extra and is slow — it downloads the whole PDF.
- arXiv has no relevance-ranked "related papers" endpoint, so nothing here does
  citation-graph traversal.
- The server holds one shared `ArxivClient` with a single throttle lock, so tool calls in
  one process are serialised 3 seconds apart. Fine for stdio single-user use, not a
  multi-tenant deployment.

## Attribution

The code in this repository was written for this project. It follows the patterns in the
[MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) documentation
(which is MIT licensed) but contains no copied code.

## License

MIT — see [LICENSE](LICENSE).
