"""Unit tests for the models and the feed parser."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from arxiv_mcp.client import parse_feed
from arxiv_mcp.models import Paper

from .conftest import ATOM_FEED, EMPTY_FEED


def test_parse_feed_extracts_every_field():
    papers, total = parse_feed(ATOM_FEED)

    assert total == 1
    assert len(papers) == 1

    paper = papers[0]
    assert paper.arxiv_id == "2401.12345"
    assert paper.version == 2
    assert paper.title == "Attention Is All You Need"
    assert paper.authors == ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"]
    # Whitespace and newlines in the summary must be collapsed.
    assert paper.abstract == (
        "The dominant sequence transduction models are based on complex "
        "recurrent or convolutional neural networks."
    )
    assert paper.primary_category == "cs.CL"
    assert paper.categories == ["cs.CL", "cs.LG"]
    assert paper.doi == "10.1234/abcd.2024.01"
    assert paper.abs_url == "https://arxiv.org/abs/2401.12345"


def test_parse_empty_feed():
    papers, total = parse_feed(EMPTY_FEED)
    assert papers == []
    assert total == 0


def test_parse_feed_rejects_garbage():
    from arxiv_mcp.client import ArxivAPIError

    with pytest.raises(ArxivAPIError):
        parse_feed("<not-xml")


def test_paper_requires_core_fields():
    with pytest.raises(ValidationError):
        Paper(
            arxiv_id="2401.12345",
            authors=[],
            abstract="y",
            published="2024-01-15T00:00:00+00:00",
            updated="2024-01-15T00:00:00+00:00",
            primary_category="cs.AI",
            categories=[],
            abs_url="u",
            pdf_url="p",
        )


def test_paper_accepts_iso_strings():
    paper = Paper(
        arxiv_id="2401.12345",
        title="t",
        authors=["a"],
        abstract="s",
        published="2024-01-15T00:00:00+00:00",
        updated="2024-01-15T00:00:00+00:00",
        primary_category="cs.AI",
        categories=["cs.AI"],
        abs_url="u",
        pdf_url="p",
    )
    assert paper.published.year == 2024
