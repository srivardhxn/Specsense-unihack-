"""
Data models shared across the pipeline.

Reworked to align with the hackathon's real expected delivery format
(252-column CSV) rather than a simplified fixed schema. The key change:
instead of hardcoded fields like "material"/"weight", products now carry
a flexible list of (label, value, unit) attributes -- exactly matching
the ATTRIBUTE_LABEL/VALUE/UOM pattern in the real spec, which supports
up to 50 attributes per product.
"""
from pydantic import BaseModel, Field
from typing import Optional, List


class ProductInput(BaseModel):
    part_number: str
    brand: str
    short_description: str


class SourceHit(BaseModel):
    """One discovered source (a webpage, PDF, or ingested dataset chunk) for a product."""
    url: str
    title: str
    snippet: Optional[str] = None
    raw_text: Optional[str] = None  # filled in during extraction
    origin: str = "web"  # "web" (live search) or "rag" (provided dataset)


class FieldValue(BaseModel):
    """A single structured attribute, paired with source + confidence."""
    value: Optional[str] = None
    confidence: float = 0.0
    source_url: Optional[str] = None
    agreeing_sources: int = 0
    needs_review: bool = False
    review_status: str = "pending"  # "pending" | "approved" | "corrected"


class Attribute(BaseModel):
    """
    One (label, value, unit) triple -- e.g. ("Weight", "106", "g") or
    ("Material", "Chrome Steel", None). Maps directly onto the real
    delivery format's ATTRIBUTE_LABEL/VALUE/UOM N columns.
    """
    label: str
    value: str
    uom: Optional[str] = None  # unit of measure, e.g. "mm", "g", "V"
    confidence: float = 0.0
    source_url: Optional[str] = None
    agreeing_sources: int = 0
    needs_review: bool = False
    review_status: str = "pending"
    vocab_validated: bool = False  # True if checked against a real controlled vocabulary and matched


class StructuredProduct(BaseModel):
    part_number: str
    brand: str
    brand_vocab_validated: bool = False  # True if brand matched the approved manufacturer/brand list
    manufacturer: Optional[str] = None  # True manufacturer name extracted from web/RAG sources
    category: FieldValue          # maps to Class / Classpath
    short_desc: FieldValue        # maps to SHORT_DESC / MOBILE_DESC / INVOICE_DESC
    long_desc: FieldValue         # maps to LONG_DESC1 / RETAIL_DESC / MARKETING_DESCRIPTION
    attributes: List[Attribute] = Field(default_factory=list)  # maps to ATTRIBUTE_* columns
    sources_used: List[str] = Field(default_factory=list)      # maps to Ref URL 1-5


class ReviewSubmission(BaseModel):
    part_number: str
    field_name: str  # "category" | "short_desc" | "long_desc" | "attr:<label>"
    corrected_value: str


class BatchRequest(BaseModel):
    products: List[ProductInput]


class BatchResult(BaseModel):
    total: int
    succeeded: int
    failed: int
    elapsed_seconds: float
    results: List[StructuredProduct]
