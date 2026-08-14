"""
Simple in-memory result cache, keyed by (part_number, brand).

Why this matters more than it sounds: on a free-tier API quota, the
worst thing that can happen is burning your daily allowance on a
product you already looked up five minutes ago -- during testing, or
worse, if a judge asks you to "run that one again." This makes repeat
lookups instant and free.

Day-1 simplicity: dict in memory, cleared on server restart. Swap for
Redis/SQLite if you want it to survive restarts -- not necessary for
a hackathon demo session.
"""
from models import StructuredProduct

_cache: dict[str, StructuredProduct] = {}


def _key(part_number: str, brand: str) -> str:
    return f"{brand.strip().lower()}::{part_number.strip().lower()}"


def get(part_number: str, brand: str) -> StructuredProduct | None:
    return _cache.get(_key(part_number, brand))


def set(part_number: str, brand: str, result: StructuredProduct) -> None:
    _cache[_key(part_number, brand)] = result


def clear() -> None:
    _cache.clear()
