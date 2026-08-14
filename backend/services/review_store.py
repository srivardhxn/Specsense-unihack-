"""
Human-in-the-loop review store.

Reworked for the flexible attributes model: a "field" needing review is
now either category/short_desc/long_desc, OR one of the dynamic
attributes (identified by "attr:<label>").
"""
from models import StructuredProduct, ReviewSubmission

_products: dict[str, StructuredProduct] = {}
_correction_log: list[dict] = []


def save_product(product: StructuredProduct) -> None:
    _products[product.part_number] = product


def get_product(part_number: str) -> StructuredProduct | None:
    return _products.get(part_number)


def get_all_products() -> list[StructuredProduct]:
    return list(_products.values())


def get_flagged_fields() -> list[dict]:
    flagged = []
    for product in _products.values():
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
    product = _products.get(review.part_number)
    if not product:
        return None

    if review.field_name.startswith("attr:"):
        label = review.field_name[len("attr:"):]
        for attr in product.attributes:
            if attr.label == label:
                attr.value = review.corrected_value
                attr.confidence = 1.0
                attr.needs_review = False
                attr.review_status = "corrected"
                break
        else:
            return None
    else:
        field = getattr(product, review.field_name, None)
        if field is None:
            return None
        field.value = review.corrected_value
        field.confidence = 1.0
        field.needs_review = False
        field.review_status = "corrected"

    _correction_log.append({
        "part_number": review.part_number,
        "field_name": review.field_name,
        "corrected_value": review.corrected_value,
    })
    save_product(product)
    return product


def get_correction_log() -> list[dict]:
    return _correction_log
