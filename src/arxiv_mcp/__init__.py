"""arxiv-mcp - an MCP server for arXiv."""

from __future__ import annotations

from .client import ArxivAPIError, ArxivClient, ArxivError, PaperNotFoundError
from .models import CategoryTrend, Paper, PdfText, SearchResult
from .server import main, mcp

__version__ = "0.1.0"

__all__ = [
    "ArxivAPIError",
    "ArxivClient",
    "ArxivError",
    "CategoryTrend",
    "Paper",
    "PaperNotFoundError",
    "PdfText",
    "SearchResult",
    "main",
    "mcp",
    "__version__",
]
