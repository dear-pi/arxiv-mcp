"""A minimal tool-using client: search arXiv through the MCP server.

This is deliberately *not* an LLM agent - it needs no API key and its
behaviour is fully deterministic, so it doubles as a smoke test you can run
in CI. The point is to show the other half of MCP: a client driving a server.

    python examples/demo_agent.py "retrieval augmented generation"

Swapping the hard-coded orchestration below for an LLM that decides which
tool to call next is the interesting extension - see README.
"""

from __future__ import annotations

import asyncio
import sys

from mcp import Client

from arxiv_mcp.server import mcp


async def survey(topic: str, per_query: int = 3) -> dict[str, object]:
    """Run a fixed research workflow against the arxiv-mcp server."""
    report: dict[str, object] = {"topic": topic, "steps": []}

    async with Client(mcp) as client:
        # 1. Is the field active, and what vocabulary does it use?
        trends = await client.call_tool(
            "category_trends", {"category": "cs.AI", "window_days": 14, "sample_size": 25}
        )
        if not trends.is_error:
            body = trends.structured_content
            report["steps"].append(
                {
                    "step": "category_trends",
                    "sampled": body["sampled"],
                    "top_terms": body["top_title_terms"][:8],
                }
            )

        # 2. Try several phrasings - a single query misses a lot.
        seen: set[str] = set()
        for phrasing in (topic, topic.split()[0], f"{topic} survey"):
            result = await client.call_tool(
                "search_papers", {"query": phrasing, "max_results": per_query}
            )
            if result.is_error:
                report["steps"].append({"step": "search_papers", "query": phrasing, "error": True})
                continue
            for paper in result.structured_content["papers"]:
                seen.add(paper["arxiv_id"])
                report.setdefault("papers", []).append(
                    {
                        "arxiv_id": paper["arxiv_id"],
                        "title": paper["title"],
                        "categories": paper["categories"],
                    }
                )

        # 3. Pull full metadata for the papers that appeared more than once.
        for arxiv_id in sorted(seen)[:3]:
            detail = await client.call_tool("get_paper", {"arxiv_id": arxiv_id})
            if not detail.is_error:
                report.setdefault("details", []).append(
                    {
                        "arxiv_id": arxiv_id,
                        "authors": detail.structured_content["authors"][:3],
                        "published": detail.structured_content["published"][:10],
                    }
                )

    return report


def render(report: dict[str, object]) -> str:
    lines = [f"Topic: {report['topic']}", ""]
    for step in report.get("steps", []):  # type: ignore[union-attr]
        if step.get("step") == "category_trends":  # type: ignore[union-attr]
            terms = ", ".join(f"{t[0]}({t[1]})" for t in step["top_title_terms"])  # type: ignore[index]
            lines.append(f"cs.AI vocabulary, last 14 days: {terms}")
        elif step.get("error"):  # type: ignore[union-attr]
            lines.append(f"  query {step.get('query')!r} -> no results")
    papers = report.get("papers", [])
    lines += ["", f"Unique papers found: {len(papers)}"]  # type: ignore[arg-type]
    for paper in papers[:10]:  # type: ignore[index]
        lines.append(f"  [{paper['arxiv_id']}] {paper['title'][:70]}")
    return "\n".join(lines)


if __name__ == "__main__":
    topic = sys.argv[1] if len(sys.argv) > 1 else "large language model agents"
    print(render(asyncio.run(survey(topic))))
