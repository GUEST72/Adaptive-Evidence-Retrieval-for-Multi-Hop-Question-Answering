"""Tests for the LLM response cache. No API calls, no network."""

from __future__ import annotations

import pytest

from baseline import llm_cache


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Point each test at its own cache file and reset the module handle."""
    monkeypatch.setenv("LLM_CACHE_DIR", str(tmp_path))
    llm_cache.close()
    llm_cache.reset_stats()
    yield
    llm_cache.close()


BASE = {"model": "m", "prompt": "p", "max_tokens": 64, "temperature": 0.0}


def test_key_is_stable_for_identical_inputs() -> None:
    assert llm_cache.cache_key(**BASE) == llm_cache.cache_key(**BASE)


@pytest.mark.parametrize(
    "override",
    [
        {"model": "other"},
        {"prompt": "different"},
        {"max_tokens": 128},
        {"temperature": 0.7},
    ],
)
def test_key_changes_when_any_input_changes(override) -> None:
    assert llm_cache.cache_key(**{**BASE, **override}) != llm_cache.cache_key(**BASE)


def test_miss_then_store_then_hit() -> None:
    key = llm_cache.cache_key(**BASE)

    assert llm_cache.get(key) is None
    llm_cache.put(key, "the answer", "m")

    assert llm_cache.get(key) == "the answer"
    assert llm_cache.entry_count() == 1


def test_hit_and_miss_counters_track_lookups() -> None:
    key = llm_cache.cache_key(**BASE)

    llm_cache.get(key)
    llm_cache.put(key, "v", "m")
    llm_cache.get(key)
    llm_cache.get(key)

    assert llm_cache.misses == 1
    assert llm_cache.hits == 2


def test_storing_the_same_key_twice_overwrites_rather_than_duplicates() -> None:
    key = llm_cache.cache_key(**BASE)
    llm_cache.put(key, "first", "m")
    llm_cache.put(key, "second", "m")

    assert llm_cache.get(key) == "second"
    assert llm_cache.entry_count() == 1


def test_empty_response_is_cached_and_not_confused_with_a_miss() -> None:
    key = llm_cache.cache_key(**BASE)
    llm_cache.put(key, "", "m")

    assert llm_cache.get(key) == ""
