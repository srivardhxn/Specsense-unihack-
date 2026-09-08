"""
Human-in-the-loop review store with SQLite persistence.

Reworked for the flexible attributes model: a "field" needing review is
either category/short_desc/long_desc, OR one of the dynamic attributes
(identified by "attr:<label>").

Persists products and human correction logs into SQLite so decisions
survive server restarts.
"""
import os
import json
import sqlite3
from models import StructuredProduct, ReviewSubmission

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "specsense_cache.db")


def _get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def _init_review_tables():
    conn = _get_db()
    with conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS review_products (
                part_number TEXT PRIMARY KEY,
                product_json TEXT,
                updated_at REAL
            )
        """)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS review_corrections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                part_number TEXT,
                field_name TEXT,
                corrected_value TEXT,
                created_at REAL
            )
        """)
    conn.close()


_init_review_tables()


def save_product(product: StructuredProduct) -> None:
    conn = _get_db()
    try:
        data_json = product.model_dump_json() if hasattr(product, "model_dump_json") else product.json()
        import time
        with conn:
            conn.execute(
                """
                INSERT INTO review_products (part_number, product_json, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(part_number) DO UPDATE SET
                    product_json = excluded.product_json,
                    updated_at = excluded.updated_at
                """,
                (product.part_number, data_json, time.time()),
            )
    except Exception as e:
        print(f"[review_store] save_product error: {e}")
    finally:
        conn.close()


def get_product(part_number: str) -> StructuredProduct | None:
    conn = _get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT product_json FROM review_products WHERE part_number = ?", (part_number,))
        row = cur.fetchone()
        if row:
            return StructuredProduct(**json.loads(row["product_json"]))
        return None
    except Exception as e:
        print(f"[review_store] get_product error: {e}")
        return None
    finally:
        conn.close()


def get_all_products() -> list[StructuredProduct]:
    conn = _get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT product_json FROM review_products ORDER BY updated_at DESC")
        rows = cur.fetchall()
        return [StructuredProduct(**json.loads(r["product_json"])) for r in rows]
    except Exception as e:
        print(f"[review_store] get_all_products error: {e}")
        return []
    finally:
        conn.close()


def get_flagged_fields() -> list[dict]:
    products = get_all_products()
    flagged = []
    for product in products:
        for field_name in ["category", "short_desc", "long_desc"]:
            field = getattr(product, field_name)
            if field.needs_review and field.review_status == "pending":
                flagged.append({
                    "part_number": product.part_number,
                    "brand": product.brand,
                    "field_name": field_name,
                    "current_value": field.value,
                    "confidence": field.confidence,
                    "agreeing_sources": field.agreeing_sources,
                })
        for attr in product.attributes:
            if attr.needs_review and attr.review_status == "pending":
                flagged.append({
                    "part_number": product.part_number,
                    "brand": product.brand,
                    "field_name": f"attr:{attr.label}",
                    "current_value": attr.value,
                    "confidence": attr.confidence,
                    "agreeing_sources": attr.agreeing_sources,
                })
    return flagged


def submit_review(review: ReviewSubmission) -> StructuredProduct | None:
    product = get_product(review.part_number)
    if not product:
        return None

    if review.field_name.startswith("attr:"):
        label = review.field_name[len("attr:"):]
        found = False
        for attr in product.attributes:
            if attr.label.lower() == label.lower():
                attr.value = review.corrected_value
                attr.confidence = 1.0
                attr.needs_review = False
                attr.review_status = "corrected"
                found = True
                break
        if not found:
            return None
    else:
        field = getattr(product, review.field_name, None)
        if field is None:
            return None
        field.value = review.corrected_value
        field.confidence = 1.0
        field.needs_review = False
        field.review_status = "corrected"

    # Log correction in SQLite
    import time
    conn = _get_db()
    try:
        with conn:
            conn.execute(
                """
                INSERT INTO review_corrections (part_number, field_name, corrected_value, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (review.part_number, review.field_name, review.corrected_value, time.time()),
            )
    except Exception as e:
        print(f"[review_store] correction log error: {e}")
    finally:
        conn.close()

    save_product(product)
    return product


def get_correction_log() -> list[dict]:
    conn = _get_db()
    try:
        cur = conn.cursor()
        cur.execute("SELECT part_number, field_name, corrected_value, created_at FROM review_corrections ORDER BY id DESC")
        rows = cur.fetchall()
        return [
            {
                "part_number": r["part_number"],
                "field_name": r["field_name"],
                "corrected_value": r["corrected_value"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]
    except Exception as e:
        print(f"[review_store] get_correction_log error: {e}")
        return []
    finally:
        conn.close()
