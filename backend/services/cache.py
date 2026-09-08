"""
SpecSense Persistent Cache Service

Backs product lookups, web search queries, and scraped text using a local
SQLite database (specsense_cache.db).

Why this is critical for enterprise industrial commerce:
1. Zero Quota Waste: Never burn API requests on repeat lookups.
2. Resilience: Survives server restarts and network interruptions.
3. Sub-millisecond Latency: Instant retrieval for previously processed SKUs.
"""
import os
import json
import sqlite3
import time
from typing import Any
from models import StructuredProduct, SourceHit

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "specsense_cache.db")

_stats = {
    "product_hits": 0,
    "product_misses": 0,
    "search_hits": 0,
    "search_misses": 0,
    "scrape_hits": 0,
    "scrape_misses": 0,
}


def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_db():
    conn = _get_db()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS product_cache (
                cache_key TEXT PRIMARY KEY,
                part_number TEXT,
                brand TEXT,
                data_json TEXT,
                updated_at REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS search_cache (
                query_key TEXT PRIMARY KEY,
                results_json TEXT,
                updated_at REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS scrape_cache (
                url TEXT PRIMARY KEY,
                title TEXT,
                raw_text TEXT,
                updated_at REAL
            )
        """)
    conn.close()


_init_db()


def _key(part_number: str, brand: str) -> str:
    b = (brand or "").strip().lower()
    pn = (part_number or "").strip().lower()
    return f"{b}::{pn}"


# ---------------- Product Cache ----------------

def get(part_number: str, brand: str) -> StructuredProduct | None:
    k = _key(part_number, brand)
    conn = _get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT data_json FROM product_cache WHERE cache_key = ?", (k,))
        row = cur.fetchone()
        if row:
            _stats["product_hits"] += 1
            data = json.loads(row["data_json"])
            return StructuredProduct(**data)
        _stats["product_misses"] += 1
        return None
    except Exception as e:
        print(f"[cache] product get error: {e}")
        return None
    finally:
        conn.close()


def set(part_number: str, brand: str, result: StructuredProduct) -> None:
    k = _key(part_number, brand)
    conn = _get_db()
    try:
        data_json = result.model_dump_json() if hasattr(result, "model_dump_json") else result.json()
        with conn:
            conn.execute(
                """
                INSERT INTO product_cache (cache_key, part_number, brand, data_json, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(cache_key) DO UPDATE SET
                    data_json = excluded.data_json,
                    updated_at = excluded.updated_at
                """,
                (k, part_number, brand, data_json, time.time()),
            )
    except Exception as e:
        print(f"[cache] product set error: {e}")
    finally:
        conn.close()


# ---------------- Search Query Cache ----------------

def get_search(query: str) -> list[SourceHit] | None:
    q = query.strip().lower()
    conn = _get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT results_json FROM search_cache WHERE query_key = ?", (q,))
        row = cur.fetchone()
        if row:
            _stats["search_hits"] += 1
            raw_list = json.loads(row["results_json"])
            return [SourceHit(**item) for item in raw_list]
        _stats["search_misses"] += 1
        return None
    except Exception as e:
        print(f"[cache] search get error: {e}")
        return None
    finally:
        conn.close()


def set_search(query: str, hits: list[SourceHit]) -> None:
    q = query.strip().lower()
    conn = _get_db()
    try:
        raw_list = [h.model_dump() if hasattr(h, "model_dump") else h.dict() for h in hits]
        with conn:
            conn.execute(
                """
                INSERT INTO search_cache (query_key, results_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(query_key) DO UPDATE SET
                    results_json = excluded.results_json,
                    updated_at = excluded.updated_at
                """,
                (q, json.dumps(raw_list), time.time()),
            )
    except Exception as e:
        print(f"[cache] search set error: {e}")
    finally:
        conn.close()


# ---------------- Web Scrape Cache ----------------

def get_scrape(url: str) -> tuple[str | None, str | None] | None:
    """Returns (title, raw_text) or None."""
    conn = _get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT title, raw_text FROM scrape_cache WHERE url = ?", (url,))
        row = cur.fetchone()
        if row and row["raw_text"]:
            _stats["scrape_hits"] += 1
            return (row["title"], row["raw_text"])
        _stats["scrape_misses"] += 1
        return None
    except Exception as e:
        print(f"[cache] scrape get error: {e}")
        return None
    finally:
        conn.close()


def set_scrape(url: str, title: str | None, raw_text: str | None) -> None:
    if not url or not raw_text:
        return
    conn = _get_db()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO scrape_cache (url, title, raw_text, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(url) DO UPDATE SET
                    title = excluded.title,
                    raw_text = excluded.raw_text,
                    updated_at = excluded.updated_at
                """,
                (url, title or "", raw_text, time.time()),
            )
    except Exception as e:
        print(f"[cache] scrape set error: {e}")
    finally:
        conn.close()


# ---------------- Analytics & Health ----------------

def get_stats() -> dict[str, Any]:
    conn = _get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM product_cache")
        product_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM search_cache")
        search_count = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM scrape_cache")
        scrape_count = cur.fetchone()[0]
        return {
            "cached_products": product_count,
            "cached_searches": search_count,
            "cached_scraped_pages": scrape_count,
            **_stats,
        }
    except Exception as e:
        return {"error": str(e), **_stats}
    finally:
        conn.close()


def clear() -> None:
    conn = _get_db()
    try:
        with conn:
            conn.execute("DELETE FROM product_cache")
            conn.execute("DELETE FROM search_cache")
            conn.execute("DELETE FROM scrape_cache")
        print("[cache] cleared all SQLite tables")
    finally:
        conn.close()
