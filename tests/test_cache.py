"""Tests for the TTL cache."""

from __future__ import annotations

import time

from arxiv_mcp.cache import TTLCache


def test_set_and_get(tmp_path):
    cache = TTLCache(tmp_path / "c.sqlite", ttl_seconds=60)
    cache.set("k", "v")
    assert cache.get("k") == "v"
    assert len(cache) == 1


def test_missing_key_returns_none(tmp_path):
    cache = TTLCache(tmp_path / "c.sqlite")
    assert cache.get("absent") is None


def test_expired_entry_is_dropped(tmp_path):
    cache = TTLCache(tmp_path / "c.sqlite", ttl_seconds=0.2)
    cache.set("k", "v")
    assert cache.get("k") == "v"
    time.sleep(0.25)
    assert cache.get("k") is None
    assert len(cache) == 0


def test_overwrite_replaces_value(tmp_path):
    cache = TTLCache(tmp_path / "c.sqlite", ttl_seconds=60)
    cache.set("k", "a")
    cache.set("k", "b")
    assert cache.get("k") == "b"


def test_clear_removes_everything(tmp_path):
    cache = TTLCache(tmp_path / "c.sqlite", ttl_seconds=60)
    cache.set("a", "1")
    cache.set("b", "2")
    cache.clear()
    assert cache.get("a") is None
    assert len(cache) == 0
