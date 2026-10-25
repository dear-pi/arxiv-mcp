"""The MCP server.

This module is intentionally thin: it validates input, delegates to
:mod:`arxiv_mcp.client`, and decides how each failure should surface to the
model. Three tools, two resources, one prompt.

Error policy (see MCP "Handling errors"):

* ``ToolError``      -> the model reads the message and can retry differently.
* ``MCPError``       -> protocol-level rejection; the model never sees it.
* anything else      -> crash; only "Error executing tool X" reaches the model.

The test in each case is "could a smarter model have avoided this?".
"""

from __future__ import annotations

from typing import Annotated, Literal

from mcp import MCPError
from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ResourceNotFoundError, ToolError
from mcp.types import INVALID_PARAMS
from pydantic import Field

from .client import (
    ArxivAPIError,
    ArxivClient,
    PaperNotFoundError,
    is_valid_arxiv_id,
)
from .models import CategoryTrend, Paper, PdfText, SearchResult

# ---------------------------------------------------------------------------
# arXiv category taxonomy (primary categories only).
# ---------------------------------------------------------------------------
CATEGORY_TAXONOMY: dict[str, list[str]] = {
    "Computer Science": [
        "cs.AI",
        "cs.AR",
        "cs.CC",
        "cs.CE",
        "cs.CG",
        "cs.CL",
        "cs.CR",
        "cs.CV",
        "cs.CY",
        "cs.DB",
        "cs.DC",
        "cs.DL",
        "cs.DM",
        "cs.DS",
        "cs.ET",
        "cs.FL",
        "cs.GL",
        "cs.GR",
        "cs.GT",
        "cs.HC",
        "cs.IR",
        "cs.IT",
        "cs.LG",
        "cs.LO",
        "cs.MA",
        "cs.MM",
        "cs.MS",
        "cs.NA",
        "cs.NE",
        "cs.NI",
        "cs.OH",
        "cs.OS",
        "cs.PF",
        "cs.PL",
        "cs.RO",
        "cs.SC",
        "cs.SD",
        "cs.SE",
        "cs.SI",
        "cs.SY",
    ],
    "Statistics": ["stat.AP", "stat.CO", "stat.ME", "stat.ML", "stat.TH"],
    "Electrical Engineering and Systems Science": [
        "eess.AS",
        "eess.IV",
        "eess.SP",
        "eess.SY",
    ],
    "Mathematics": [
        "math.AC",
        "math.AG",
        "math.AP",
        "math.AT",
        "math.CA",
        "math.CO",
        "math.CT",
        "math.DG",
        "math.DS",
        "math.FA",
        "math.GM",
        "math.GN",
        "math.GR",
        "math.GT",
        "math.HO",
        "math.IT",
        "math.KT",
        "math.LO",
        "math.MG",
        "math.MP",
        "math.NA",
        "math.NT",
        "math.OA",
        "math.OC",
        "math.PR",
        "math.QA",
        "math.RT",
        "math.SG",
        "math.SP",
        "math.ST",
    ],
    "Physics (archives)": [
        "astro-ph.CO",
        "astro-ph.EP",
        "astro-ph.GA",
        "astro-ph.HE",
        "astro-ph.IM",
        "astro-ph.SR",
        "cond-mat.dis-nn",
        "cond-mat.mes-hall",
        "cond-mat.mtrl-sci",
        "cond-mat.other",
        "cond-mat.quant-gas",
        "cond-mat.soft",
        "cond-mat.stat-mech",
        "cond-mat.str-el",
        "cond-mat.supr-con",
        "gr-qc",
        "hep-ex",
        "hep-lat",
        "hep-ph",
        "hep-th",
        "math-ph",
        "nlin.AO",
        "nlin.CD",
        "nlin.CG",
        "nlin.PS",
        "nlin.SI",
        "nucl-ex",
        "nucl-th",
        "physics.acc-ph",
        "physics.ao-ph",
        "physics.app-ph",
        "physics.atm-clus",
        "physics.atom-ph",
        "physics.bio-ph",
        "physics.chem-ph",
        "physics.class-ph",
        "physics.comp-ph",
        "physics.data-an",
        "physics.ed-ph",
        "physics.flu-dyn",
        "physics.gen-ph",
        "physics.geo-ph",
        "physics.hist-ph",
        "physics.ins-det",
        "physics.med-ph",
        "physics.optics",
        "physics.plasm-ph",
        "physics.pop-ph",
        "physics.soc-ph",
        "physics.space-ph",
        "quant-ph",
    ],
    "Quantitative Biology": [
        "q-bio.BM",
        "q-bio.CB",
        "q-bio.GN",
        "q-bio.MN",
        "q-bio.NC",
        "q-bio.OT",
        "q-bio.PE",
        "q-bio.QM",
        "q-bio.SC",
        "q-bio.TO",
    ],
    "Quantitative Finance": [
        "q-fin.CP",
        "q-fin.EC",
        "q-fin.GN",
        "q-fin.MF",
        "q-fin.PM",
        "q-fin.PR",
        "q-fin.RM",
        "q-fin.ST",
        "q-fin.TR",
    ],
    "Economics": ["econ.EM", "econ.GN", "econ.TH"],
}

SORT_OPTIONS = ("relevance", "submittedDate", "lastUpdatedDate")

mcp = MCPServer(
    "arxiv-mcp",
    instructions=(
        "Search arXiv, fetch paper metadata, extract PDF text and summarise "
        "category trends. Prefer search_papers for discovery, get_paper for a "
        "specific identifier, and get_pdf_text only when the abstract is not "
        "enough - it downloads a full PDF and is slow."
    ),
)

_client: ArxivClient | None = None


def get_client() -> ArxivClient:
    """Lazily build the shared client so importing the module has no side effects."""
    global _client
    if _client is None:
        _client = ArxivClient()
    return _client


def _format_paper_card(paper: Paper) -> str:
    """Render a paper as Markdown - the resource representation."""
    lines = [
        f"# {paper.title}",
        "",
        f"**arXiv**: [{paper.arxiv_id}]({paper.abs_url}) (v{paper.version})",
        f"**Authors**: {', '.join(paper.authors) if paper.authors else 'unknown'}",
        f"**Primary category**: {paper.primary_category}",
        f"**Categories**: {', '.join(paper.categories)}",
        f"**Published**: {paper.published.date().isoformat()}  ",
        f"**Updated**: {paper.updated.date().isoformat()}",
    ]
    if paper.doi:
        lines.append(f"**DOI**: {paper.doi}")
    if paper.journal_ref:
        lines.append(f"**Journal**: {paper.journal_ref}")
    if paper.comment:
        lines.append(f"**Comment**: {paper.comment}")
    lines += ["", "## Abstract", "", paper.abstract]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def search_papers(
    query: Annotated[
        str,
        Field(
            description=(
                "Search text. arXiv field syntax is supported, e.g. "
                "'retrieval augmented generation', 'au:hinton AND cat:cs.NE', "
                "'ti:\"chain of thought\"'."
            )
        ),
    ],
    max_results: Annotated[int, Field(ge=1, le=100, description="Page size.")] = 10,
    sort_by: Annotated[
        Literal["relevance", "submittedDate", "lastUpdatedDate"],
        Field(description="Ranking to apply."),
    ] = "relevance",
    category: Annotated[
        str | None,
        Field(description="Optional arXiv category filter, e.g. 'cs.AI'."),
    ] = None,
) -> SearchResult:
    """Search arXiv and return a page of matching papers."""
    try:
        result = await get_client().search(
            query,
            max_results=max_results,
            sort_by=sort_by,
            category=category,
        )
    except ArxivAPIError as exc:
        raise ToolError(f"arXiv is unreachable right now: {exc}") from exc

    if result.returned == 0:
        raise ToolError(
            f"No papers matched {result.query!r}. Try a shorter query, drop the "
            "category filter, or use arXiv field syntax (au:, ti:, abs:)."
        )
    return result


@mcp.tool()
async def get_paper(
    arxiv_id: Annotated[
        str,
        Field(
            description=(
                "arXiv identifier, with or without version. Modern form "
                "'2401.12345', legacy form 'cs.LO/0112017'."
            )
        ),
    ],
) -> Paper:
    """Fetch full metadata for a single arXiv paper by identifier."""
    if not is_valid_arxiv_id(arxiv_id):
        raise ToolError(
            f"{arxiv_id!r} is not a valid arXiv identifier. Expected something "
            "like '2401.12345' or 'cs.LO/0112017'."
        )
    try:
        return await get_client().get_paper(arxiv_id)
    except PaperNotFoundError as exc:
        raise ToolError(str(exc)) from exc
    except ArxivAPIError as exc:
        raise ToolError(f"arXiv is unreachable right now: {exc}") from exc


@mcp.tool()
async def get_pdf_text(
    arxiv_id: Annotated[str, Field(description="arXiv identifier, e.g. '2401.12345'.")],
    max_pages: Annotated[int, Field(ge=1, le=20, description="How many pages to parse.")] = 5,
) -> PdfText:
    """Download a paper's PDF and extract text from its first pages.

    Slow and bandwidth-heavy - use only when the abstract is insufficient.
    """
    if not is_valid_arxiv_id(arxiv_id):
        raise ToolError(f"{arxiv_id!r} is not a valid arXiv identifier.")
    try:
        return await get_client().fetch_pdf_text(arxiv_id, max_pages=max_pages)
    except RuntimeError as exc:
        # Missing optional dependency: the model cannot fix this, the host can.
        raise MCPError(code=INVALID_PARAMS, message=str(exc)) from exc
    except ArxivAPIError as exc:
        raise ToolError(f"Could not read the PDF for {arxiv_id}: {exc}") from exc


@mcp.tool()
async def category_trends(
    category: Annotated[str, Field(description="arXiv category, e.g. 'cs.AI' or 'cs.LG'.")],
    window_days: Annotated[int, Field(ge=1, le=365, description="How far back to look.")] = 30,
    sample_size: Annotated[
        int, Field(ge=10, le=200, description="How many recent papers to sample.")
    ] = 100,
) -> CategoryTrend:
    """Summarise recent submission volume and the most common title terms in a category."""
    try:
        return await get_client().category_trends(
            category, window_days=window_days, sample_size=sample_size
        )
    except ArxivAPIError as exc:
        raise ToolError(f"arXiv is unreachable right now: {exc}") from exc


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@mcp.resource("arxiv://paper/{arxiv_id}", mime_type="text/markdown")
async def paper_resource(arxiv_id: str) -> str:
    """A Markdown card with the full metadata of one paper."""
    if not is_valid_arxiv_id(arxiv_id):
        raise ResourceNotFoundError(f"{arxiv_id!r} is not a valid arXiv identifier")
    try:
        paper = await get_client().get_paper(arxiv_id)
    except PaperNotFoundError as exc:
        raise ResourceNotFoundError(str(exc)) from exc
    return _format_paper_card(paper)


@mcp.resource("arxiv://categories", mime_type="text/markdown")
def categories_resource() -> str:
    """The arXiv category taxonomy this server can filter on."""
    lines = ["# arXiv categories", ""]
    for group, codes in CATEGORY_TAXONOMY.items():
        lines.append(f"## {group}")
        lines.append("")
        lines.append(", ".join(f"`{c}`" for c in codes))
        lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


@mcp.prompt(title="Literature review")
def literature_review(
    topic: Annotated[str, Field(description="The research topic to survey.")],
    max_papers: Annotated[int, Field(ge=3, le=30, description="How many papers to cover.")] = 8,
) -> str:
    """Build a structured literature review from arXiv."""
    return f"""You are writing a literature review on "{topic}".

Work through these steps and show your reasoning as you go:

1. Call `category_trends` on the most relevant category to confirm the topic is
   active and to pick up current terminology.
2. Call `search_papers` with a few different phrasings of the topic. Vary the
   query - do not just repeat the first one.
3. For the {max_papers} most relevant papers, call `get_paper` to get full
   metadata, and `get_pdf_text` only when the abstract leaves a real gap.
4. Group the papers by approach, then write:
   - **What the field agrees on**
   - **Where the approaches diverge**
   - **Open problems** (say so explicitly when the papers do not settle it)
5. Cite every claim as [arxiv_id]. Never state something no retrieved paper
   supports - say "no retrieved paper addresses this" instead.
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Run the server over stdio (what `mcp install` and most hosts expect)."""

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
