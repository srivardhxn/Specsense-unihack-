"""
Controlled vocabulary validation layer.

Per Unilog's Solution Guide: "Attribute values must come from the LOV
files; manufacturer and brand names must match the approved list
exactly, symbols and all; units must use the approved abbreviation."

This module is the extension point for that requirement. It does NOT
contain any fabricated vocabulary -- inventing a fake "approved list"
and pretending it's real would be worse than having no compliance
layer at all, since it could produce confidently wrong output that
looks validated but isn't.

Instead: at startup, this module looks for real reference files in
./reference_data/. If found, it loads them and validation is ACTIVE.
If not found, validation is INACTIVE and every field passes through
unchanged, clearly flagged as "not vocabulary-checked" rather than
silently claiming compliance it doesn't have.

Expected files (drop into backend/reference_data/, then restart the
server -- see reference_data/README.md for exact format):

  manufacturer_brand.csv   -- columns: MANUFACTURER_NAME, BRAND_NAME
                               (exported from UniCat_Manufacturer_and_Brand_List.xlsx)
  uom_standards.csv        -- columns: Measurement_Type, Approved_Abbreviation
                               (exported from Unilog_Master_UOM_Standards...xlsx)
  fittings_lov.csv         -- columns: Attribute_Label, Attribute_Value, Normalized_Value
                               (exported from the relevant sheet of Fittings_LOV.xlsx
                               or Unicat_Lov_v1_0...xlsx filtered to the Fittings classpath)

Any of the three can be present independently -- e.g. having just the
UOM file activates unit normalization even without the others.
"""
import os
import csv
import difflib
import re

REFERENCE_DIR = os.getenv("REFERENCE_DATA_DIR", os.path.join(os.path.dirname(__file__), "..", "reference_data"))

_manufacturer_brands: list[tuple[str, str]] = []  # (MANUFACTURER_NAME, BRAND_NAME)
_uom_map: dict[str, str] = {}                      # lowercase variant -> approved abbreviation
_lov_map: dict[str, str] = {}                      # "label::lowercase_value" -> normalized value

status = {
    "manufacturer_brand_active": False,
    "uom_active": False,
    "lov_active": False,
}


def _load_csv_rows(filename: str) -> list[dict]:
    path = os.path.join(REFERENCE_DIR, filename)
    if not os.path.isfile(path):
        return []
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            return list(csv.DictReader(f))
    except Exception as e:
        print(f"[vocabulary] failed to read {filename}: {e}")
        return []


def load_all() -> None:
    """Called once at server startup. Populates whatever reference files are actually present."""
    global _manufacturer_brands, _uom_map, _lov_map

    mb_rows = _load_csv_rows("manufacturer_brand.csv")
    if mb_rows:
        _manufacturer_brands = [
            (r.get("MANUFACTURER_NAME", "").strip(), r.get("BRAND_NAME", "").strip())
            for r in mb_rows if r.get("MANUFACTURER_NAME") or r.get("BRAND_NAME")
        ]
        status["manufacturer_brand_active"] = len(_manufacturer_brands) > 0
        print(f"[vocabulary] manufacturer/brand list ACTIVE: {len(_manufacturer_brands)} entries loaded")
    else:
        print("[vocabulary] manufacturer/brand list INACTIVE -- reference_data/manufacturer_brand.csv not found, names pass through unvalidated")

    uom_rows = _load_csv_rows("uom_standards.csv")
    if uom_rows:
        for r in uom_rows:
            variant = (r.get("Measurement_Type") or r.get("Variant") or "").strip().lower()
            approved = (r.get("Approved_Abbreviation") or r.get("Approved") or "").strip()
            if variant and approved:
                _uom_map[variant] = approved
        status["uom_active"] = len(_uom_map) > 0
        print(f"[vocabulary] UOM standards ACTIVE: {len(_uom_map)} mappings loaded")
    else:
        print("[vocabulary] UOM standards INACTIVE -- reference_data/uom_standards.csv not found, units pass through unvalidated")

    lov_rows = _load_csv_rows("fittings_lov.csv")
    if lov_rows:
        for r in lov_rows:
            label = (r.get("Attribute_Label") or "").strip()
            value = (r.get("Attribute_Value") or "").strip()
            normalized = (r.get("Normalized_Value") or value).strip()
            if label and value:
                _lov_map[f"{label.lower()}::{value.lower()}"] = normalized
        status["lov_active"] = len(_lov_map) > 0
        print(f"[vocabulary] Fittings LOV ACTIVE: {len(_lov_map)} value mappings loaded")
    else:
        print("[vocabulary] Fittings LOV INACTIVE -- reference_data/fittings_lov.csv not found, attribute values pass through unvalidated")


def normalize_brand(brand: str) -> tuple[str, bool]:
    """
    Returns (normalized_brand, was_validated). If the vocabulary is
    inactive, returns the input unchanged with was_validated=False --
    never silently invents a "corrected" spelling.
    """
    if not brand:
        return brand, False

    clean_input = re.sub(r'\s*\([^)]*\)\s*$', '', str(brand).strip())
    # Clean common placeholder prefixes
    clean_input = re.sub(r'^--\s*|\s*--$', '', clean_input).strip()

    if not status["manufacturer_brand_active"]:
        return clean_input, False

    # Check direct or case-insensitive match first
    for mfr, b in _manufacturer_brands:
        if b and (b.lower() == clean_input.lower() or b.lower() == brand.lower()):
            return b, True
        if mfr and (mfr.lower() == clean_input.lower() or mfr.lower() == brand.lower()):
            return b if b else mfr, True

    # Fuzzy match against approved brands
    candidates = [b for _, b in _manufacturer_brands if b]
    matches = difflib.get_close_matches(clean_input, candidates, n=1, cutoff=0.75)
    if matches:
        return matches[0], True

    # Fuzzy match against manufacturers to find paired brand
    mfr_candidates = [m for m, _ in _manufacturer_brands if m]
    mfr_matches = difflib.get_close_matches(clean_input, mfr_candidates, n=1, cutoff=0.75)
    if mfr_matches:
        for m, b in _manufacturer_brands:
            if m == mfr_matches[0] and b:
                return b, True

    return clean_input, False


def normalize_uom(uom: str | None) -> tuple[str | None, bool]:
    """Returns (approved_abbreviation, was_validated) adhering to Unilog UOM standards."""
    if not uom:
        return None, False
    clean_u = uom.strip()
    if not status["uom_active"]:
        return clean_u, False
    match = _uom_map.get(clean_u.lower())
    if match:
        return match, True
    return clean_u, False


def normalize_attribute_value(label: str, value: str, uom: str | None = None) -> tuple[str, str | None, bool]:
    """
    Applies Decimal-to-Fraction conversion and LOV canonicalization.
    Returns (normalized_value, normalized_uom, was_validated).
    """
    from services.decimal_fraction import convert_measurement_string

    if not value:
        return value, uom, False

    clean_val = str(value).strip()
    norm_uom, uom_valid = normalize_uom(uom)

    # Convert imperial decimal dimensions to trade fractions
    is_dimension = any(kw in label.lower() for kw in ["bore", "outside", "outer", "inside", "width", "height", "length", "depth", "thickness", "dimension", "size"])
    if is_dimension or (norm_uom and norm_uom.lower() in ["in", "inch"]):
        converted, did_conv = convert_measurement_string(clean_val, norm_uom)
        if did_conv:
            clean_val = converted

    # Apply LOV normalization
    lov_valid = False
    if status["lov_active"] and label:
        match = _lov_map.get(f"{label.lower()}::{clean_val.lower()}")
        if match:
            clean_val = match
            lov_valid = True

    return clean_val, norm_uom, (lov_valid or uom_valid)


# Canonical Unilog Classpath Taxonomy mapping
CLASSPATH_TAXONOMY = {
    "dishwasher": "Appliances & Consumer Electronics > Kitchen Appliances > Built-In Dishwashers",
    "refrigerator": "Appliances & Consumer Electronics > Kitchen Appliances > Refrigerators",
    "bearing": "Bearings & Power Transmission > Bearings > Deep Groove Ball Bearings",
    "ball bearing": "Bearings & Power Transmission > Bearings > Deep Groove Ball Bearings",
    "roller bearing": "Bearings & Power Transmission > Bearings > Roller Bearings",
    "sanding belt": "Abrasives & Cutting Tools > Abrasives > Sanding Belts",
    "abrasive": "Abrasives & Cutting Tools > Abrasives > Discs & Sheets",
    "disc": "Abrasives & Cutting Tools > Abrasives > Sanding Discs",
    "coupling": "Pipe, Tubing & Hose > Pipe & Tube Fittings > Couplings",
    "fitting": "Pipe, Tubing & Hose > Pipe & Tube Fittings > Industrial Fittings",
    "valve": "Plumbing & Pumps > Valves > Control & Ball Valves",
    "faucet": "Plumbing & Pumps > Faucets > Kitchen & Bath Sink Faucets",
    "plc": "Electrical & Industrial Automation > Controls > PLCs & Automation",
    "controller": "Electrical & Industrial Automation > Controls > Industrial Controllers",
    "motor": "Motors & Power Equipment > Electric Motors > AC/DC Motors",
    "pump": "Plumbing & Pumps > Pumps > Industrial Pumps",
    "sensor": "Electrical & Industrial Automation > Sensors > Proximity & Photoelectric",
    "circuit breaker": "Electrical & Industrial Automation > Distribution > Circuit Breakers",
}


def get_canonical_classpath(category: str | None, desc: str | None = None) -> str:
    combined = f"{category or ''} {desc or ''}".lower()
    for kw, cp in CLASSPATH_TAXONOMY.items():
        if kw in combined:
            return cp
    if category and category.strip():
        return f"Industrial Supplies > Equipment & Hardware > {category.strip()}"
    return "Industrial Supplies > General Industrial Equipment"


def is_any_active() -> bool:
    return any(status.values())

