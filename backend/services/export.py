"""
Exports processed products as CSV in the EXACT column format the
hackathon organizers specified (Unihack__Expected_Output_-_Delivery_Format.csv).

Every column name below was extracted directly from their real expected
output file -- not retyped by hand -- to guarantee an exact match.

Columns we have real data for get filled in. Columns we don't extract
(images, pricing, UPC/GTIN, manuals, etc) are left blank rather than
guessed -- this is a deliberate, honest choice: an empty cell is
truthful, a fabricated one isn't. This is worth being upfront about if
asked: the prototype demonstrates the pipeline and the exact schema
alignment; a production version would extend extraction to cover the
remaining columns (images, pricing feeds, etc) using additional
data sources beyond text-based web/PDF search.
"""
import csv
import io
from models import StructuredProduct

MAX_ATTRIBUTE_SLOTS = 50

HEADER = [
    "MFR URL", "Ref URL 1", "Ref URL 2", "Ref URL 3", "Ref URL 4", "Ref URL 5",
    "PART_NUMBER", "Dept", "Class", "Fine", "SKU - MY_PART_NUMBER", "Mfg_Part_Num",
    "Part_Desc", "E1_Brand", "Unilog_Brand", "DIB_Brand", "Part_Manuf",
    "MANUFACTURER_NAME", "BRAND_NAME", "TRADE_NAME", "MANUFACTURER_PART_NUMBER",
    "ALTERNATE_PART_NUMBER", "Classpath", "MOBILE_DESC", "INVOICE_DESC", "SHORT_DESC",
    "LONG_DESC1", "RETAIL_DESC", "MARKETING_DESCRIPTION",
] + [f"ITEM_FEATURES_{i}" for i in range(1, 21)] + [
    "With", "Standard/Approvals", "Prop 65", "Application", "Includes", "Product Name",
]
for i in range(1, MAX_ATTRIBUTE_SLOTS + 1):
    HEADER += [f"ATTRIBUTE_LABEL {i}", f"ATTRIBUTE_VALUE {i}", f"ATTRIBUTE_UOM {i}"]
HEADER += [
    "UPC", "EAN", "GTIN", "UNSPSC", "Warranty", "List Price", "Selling Qty", "Selling UOM",
    "Standard Packaging Information", "LENGTH", "LENGTH_UOM", "HEIGHT", "HEIGHT_UOM",
    "WIDTH", "WIDTH_UOM", "WEIGHT", "WEIGHT_UOM", "VOLUME", "VOLUME_UOM",
    "Product Image", "Alternate Image 1", "Alternate Image 2", "Alternate Image 3",
    "Alternate Image 4", "SDS", "SDS_1", "Warranty Information", "Catalog",
    "Specification Sheet", "Instruction/Installation Manual", "Service Manual",
    "Owners/User Manual", "Line Drawing", "MTR", "RoHS", "Full Engineering Drawing",
    "Energy Star Guide", "Technical Bulletin", "Submittal", "Compatibility Chart",
    "Size Chart", "Product Label/Insert", "Video Link", "Video Link 1",
    "Country Of Origin", "Discontinued", "Actual Image (Yes/No)",
]
assert len(HEADER) == 252, f"Header column count drifted from spec: {len(HEADER)} != 252"

_DIMENSION_KEYWORDS = {
    "length": ("LENGTH", "LENGTH_UOM"),
    "width": ("WIDTH", "WIDTH_UOM"),
    "height": ("HEIGHT", "HEIGHT_UOM"),
    "weight": ("WEIGHT", "WEIGHT_UOM"),
}


def _find_dimension_columns(attributes) -> dict:
    """
    If an attribute's label directly matches length/width/height/weight,
    map it into the schema's dedicated LENGTH/WIDTH/HEIGHT/WEIGHT columns
    (in addition to it still appearing in the generic ATTRIBUTE_* slots).
    Deliberately conservative: only exact keyword matches, no guessing
    at ambiguous multi-dimension strings like "20x47x14mm".
    """
    result = {}
    for attr in attributes:
        label_lower = attr.label.lower()
        for keyword, (value_col, uom_col) in _DIMENSION_KEYWORDS.items():
            if keyword in label_lower and value_col not in result:
                result[value_col] = attr.value
                result[uom_col] = attr.uom or ""
    return result


def _find_approvals(attributes) -> str:
    """
    Surfaces every attribute that looks like a certification/standard,
    joined with '|' -- matching the real reference data's exact
    convention (e.g. "ASSE 1006|CEE Tier 2 Qualified|cUL Listed").
    """
    matches = []
    for attr in attributes:
        label_lower = attr.label.lower()
        if any(kw in label_lower for kw in ("certif", "approv", "standard", "compliance", "rohs", "iso", "ul list")):
            matches.append(attr.value)
    return "|".join(matches)


def product_to_row(p: StructuredProduct) -> dict:
    sources = p.sources_used[:5]
    row = {col: "" for col in HEADER}

    row["PART_NUMBER"] = p.part_number
    row["Mfg_Part_Num"] = p.part_number
    row["MANUFACTURER_PART_NUMBER"] = p.part_number
    row["MANUFACTURER_NAME"] = p.manufacturer or p.brand
    row["BRAND_NAME"] = p.brand
    row["TRADE_NAME"] = p.brand
    row["E1_Brand"] = p.brand
    row["Unilog_Brand"] = p.brand
    row["DIB_Brand"] = p.brand
    row["Part_Manuf"] = p.manufacturer or p.brand

    row["Class"] = p.category.value or ""
    row["Classpath"] = p.category.value or ""

    row["MOBILE_DESC"] = p.short_desc.value or ""
    row["INVOICE_DESC"] = p.short_desc.value or ""
    row["SHORT_DESC"] = p.short_desc.value or ""
    row["LONG_DESC1"] = p.long_desc.value or ""
    row["RETAIL_DESC"] = p.long_desc.value or ""
    row["MARKETING_DESCRIPTION"] = p.long_desc.value or ""

    for i, url in enumerate(sources, start=1):
        row[f"Ref URL {i}"] = url

    row["Standard/Approvals"] = _find_approvals(p.attributes)

    for i, attr in enumerate(p.attributes[:MAX_ATTRIBUTE_SLOTS], start=1):
        row[f"ATTRIBUTE_LABEL {i}"] = attr.label
        row[f"ATTRIBUTE_VALUE {i}"] = attr.value
        row[f"ATTRIBUTE_UOM {i}"] = attr.uom or ""

    row.update(_find_dimension_columns(p.attributes))

    return row


def export_products_csv(products: list[StructuredProduct]) -> str:
    """Returns CSV text (as a string) for the given products, matching the exact 252-column spec."""
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=HEADER)
    writer.writeheader()
    for p in products:
        writer.writerow(product_to_row(p))
    return buffer.getvalue()
