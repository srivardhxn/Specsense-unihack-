"""
STAGE 3 + 4: Structure & Score Confidence

Reworked to extract a FLEXIBLE list of (label, value, unit) attributes
instead of a fixed 6-field schema -- this mirrors the real hackathon
delivery format, which supports up to 50 arbitrary attributes per
product (ATTRIBUTE_LABEL/VALUE/UOM 1-50), rather than assuming every
product has exactly "material", "weight", etc.

Still does genuine multi-source cross-validation: each attribute is
grouped by its normalized label across all sources, and confidence is
earned by source agreement -- same principle as before, applied to a
flexible schema instead of a fixed one.

Uses Gemini (free tier) as primary, Groq (free tier) as automatic
fallback if Gemini's quota is exhausted.
"""
import os
import json
import time
import asyncio
from collections import defaultdict
import google.generativeai as genai
from groq import Groq
from models import ProductInput, SourceHit, StructuredProduct, FieldValue, Attribute
from services import vocabulary

gemini_keys = [k.strip() for k in os.getenv("GEMINI_API_KEY", "").split(",") if k.strip()]
groq_keys = [k.strip() for k in os.getenv("GROQ_API_KEY", "").split(",") if k.strip()]

gemini_key_idx = 0
groq_key_idx = 0
GROQ_MODEL = "qwen/qwen3.6-27b"

MAX_ATTRIBUTES = 50  # matches the real delivery format's ATTRIBUTE_* 1-50 slots

SYSTEM_PROMPT = """You are a product data extraction engine for an industrial commerce catalog.
You will be given a product's known info (part number, brand, short description) and text from
MULTIPLE numbered sources (webpages, datasheet PDFs, or provided datasets). Each source may or
may not actually be about this exact product.

For EACH source, independently extract:

1. category: the product's category/classification (e.g. "Deep Groove Ball Bearing", "Programmable Logic Controller")
2. brand: the product's brand name (e.g. "Frigidaire", "Whirlpool", "Diablo", "3M", "Milwaukee"). Clean it from distributor prefixes/suffixes. Only report if stated. CRITICAL: Never use distributor/buying group/cooperative names like "Appliance Dealers Cooperative", "APPDE", "Jam Industrial Supply", or generic placeholders like "-- Unbranded --", "-- No Unilog Brand --" as the brand. Look for the true product brand name.
3. manufacturer: the product's manufacturer company name (e.g. "Rheem Manufacturing", "Whirlpool Corporation", "Freud Inc"). Only report if stated. CRITICAL: Never use distributor/buying group/cooperative names like "Appliance Dealers Cooperative", "APPDE", "Jam Industrial Supply", or generic placeholders. Look for the true manufacturing company.
4. short_desc: a concise ~10-15 word description suitable for a mobile listing
5. long_desc: a fuller 1-3 sentence description with key specs included
6. attributes: EVERY distinct technical attribute this source states about the product --
   do not limit yourself to a fixed list. Use clear, standardized attribute names in Title Case
   (e.g. "Voltage Rating" not "voltage", "Mounting Type" not "mount", "Sound Level" not "noise").
   For each attribute give: label (short standardized name), value (the value, no units embedded),
   and uom (unit of measure, e.g. "mm", "g", "V", "kg" -- or null if the value has no unit, like
   a certification name).

Only report what THIS source's text directly states -- do not guess, do not use outside
knowledge, do not invent attributes not actually present in the text. It is completely normal
and expected for most fields to end up empty -- real commerce catalogs leave the large majority
of possible attributes blank when a source doesn't state them, rather than guessing.

EXAMPLE of the expected style (for a completely different product, just showing format/tone):
  category: "Built-In Dishwashers"
  brand: "Whirlpool"
  manufacturer: "Whirlpool Corporation"
  short_desc: "Whirlpool Eco Series WDTS7024RZ Dishwasher, Built-in Mounting, Stainless Steel"
  long_desc: "Whirlpool dishwasher, Eco Series, 120V, 10A, built-in mounting, 41 dBA sound level, stainless steel."
  attributes: [
    {"label": "Series", "value": "Eco Series", "uom": null},
    {"label": "Voltage Rating", "value": "120", "uom": "V"},
    {"label": "Amperage Rating", "value": "10", "uom": "A"},
    {"label": "Mounting Type", "value": "Built-in", "uom": null},
    {"label": "Sound Level", "value": "41", "uom": "dBA"},
    {"label": "Material", "value": "Stainless Steel", "uom": null}
  ]

Respond ONLY with valid JSON in this exact shape, no other text, no markdown fences:
{
  "sources": [
    {
      "source_index": 0,
      "category": "...",
      "brand": "...",
      "manufacturer": "...",
      "short_desc": "...",
      "long_desc": "...",
      "attributes": [
        {"label": "Material", "value": "Chrome Steel", "uom": null},
        {"label": "Weight", "value": "106", "uom": "g"}
      ]
    }
  ]
}
Include an entry in "sources" for every source index given, even if some fields are empty/null
or attributes is an empty array for that source.
"""


def _empty_extraction(num_sources: int) -> dict:
    return {"sources": [{"source_index": i, "category": None, "short_desc": None, "long_desc": None, "attributes": []} for i in range(num_sources)]}


def _call_gemini(user_prompt: str) -> dict:
    global gemini_key_idx
    if not gemini_keys:
        raise RuntimeError("GEMINI_API_KEY is not set.")
    
    current_key = gemini_keys[gemini_key_idx % len(gemini_keys)]
    genai.configure(api_key=current_key)
    model = genai.GenerativeModel("gemini-3.5-flash")
    
    response = model.generate_content(
        user_prompt,
        generation_config={"temperature": 0, "max_output_tokens": 6144, "response_mime_type": "application/json"},
        request_options={"timeout": 40.0}
    )
    raw = response.text.strip().replace("```json", "").replace("```", "").strip()
    return _parse_json_loosely(raw)


def _call_groq(user_prompt: str) -> dict:
    global groq_key_idx
    if not groq_keys:
        raise RuntimeError("GROQ_API_KEY not set -- cannot use fallback provider.")
    
    current_key = groq_keys[groq_key_idx % len(groq_keys)]
    client = Groq(api_key=current_key)
    
    extra_body = {}
    if "qwen" in GROQ_MODEL.lower():
        extra_body["reasoning_effort"] = "none"
    elif "gpt-oss" in GROQ_MODEL.lower():
        extra_body["reasoning_format"] = "hidden"

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{"role": "user", "content": user_prompt}],
        temperature=0,
        max_tokens=4096,
        timeout=40.0,
        extra_body=extra_body
    )
    raw = response.choices[0].message.content.strip()
    return _parse_json_loosely(raw)


def _close_truncated_json(s: str) -> str:
    s = s.strip()
    if s.endswith(","):
        s = s[:-1].strip()
        
    stack = []
    in_string = False
    escape = False
    
    for ch in s:
        if escape:
            escape = False
            continue
        if ch == '\\':
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == '{':
                stack.append('}')
            elif ch == '[':
                stack.append(']')
            elif ch == '}':
                if stack and stack[-1] == '}':
                    stack.pop()
            elif ch == ']':
                if stack and stack[-1] == ']':
                    stack.pop()
                    
    if in_string:
        s += '"'
        
    while stack:
        close_ch = stack.pop()
        s = s.strip()
        if s.endswith(","):
            s = s[:-1].strip()
        s += close_ch
        
    return s


def _parse_json_loosely(raw: str) -> dict:
    import re
    cleaned = raw.strip()
    
    # 1. Strip think blocks
    cleaned = re.sub(r"<think>.*?</think>", "", cleaned, flags=re.DOTALL).strip()
    
    # 2. Strip code blocks
    if cleaned.startswith("```json"):
        cleaned = cleaned[7:]
    elif cleaned.startswith("```"):
        cleaned = cleaned[3:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3]
    cleaned = cleaned.strip()
    
    # 3. Locate start of JSON object
    start = cleaned.find("{")
    if start != -1:
        cleaned = cleaned[start:]
        
    # 4. Clean trailing commas in objects/arrays
    cleaned = re.sub(r",\s*([\]\}])", r"\1", cleaned)
    
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # 5. Try to repair truncated JSON
        try:
            repaired = _close_truncated_json(cleaned)
            return json.loads(repaired)
        except Exception as e2:
            print(f"[structure] loose JSON parse and repair failed: {e2}")
            print(f"--- RAW RESPONSE START ---\n{raw}\n--- RAW RESPONSE END ---")
            raise


def _call_llm_with_fallback(user_prompt: str, max_attempts: int | None = None) -> dict:
    global gemini_key_idx, groq_key_idx
    last_error = None
    
    # Try Gemini with keys rotation
    if gemini_keys:
        attempts = max_attempts if max_attempts is not None else len(gemini_keys)
        for attempt in range(attempts):
            try:
                res = _call_gemini(user_prompt)
                gemini_key_idx += 1  # rotate to balance load
                return res
            except Exception as e:
                last_error = e
                msg = str(e)
                print(f"[structure] Gemini call failed with key index {gemini_key_idx % len(gemini_keys)}: {e}")
                
                # immediately switch to the next key
                gemini_key_idx += 1
                
                if "PerDay" in msg:
                    print("[structure] Gemini DAILY quota exhausted for this key. Trying NEXT key...")
                    continue
                if ("429" in msg or "quota" in msg.lower()) and attempt < attempts - 1:
                    wait_seconds = 2 * (attempt + 1)
                    print(f"[structure] Rate limited, waiting {wait_seconds}s before retry with NEXT key...")
                    time.sleep(wait_seconds)
                    continue
                if attempt < attempts - 1:
                    print("[structure] Attempting fallback to NEXT Gemini key immediately...")
                    continue
                break
    else:
        print("[structure] No GEMINI_API_KEY configured.")

    # Try Groq with keys rotation
    if groq_keys:
        attempts = max_attempts if max_attempts is not None else len(groq_keys)
        for attempt in range(attempts):
            try:
                print(f"[structure] Falling back to Groq (Qwen 3.6) using key index {groq_key_idx % len(groq_keys)}...")
                res = _call_groq(user_prompt)
                groq_key_idx += 1
                return res
            except Exception as e:
                print(f"[structure] Groq call failed with key index {groq_key_idx % len(groq_keys)}: {e}")
                groq_key_idx += 1
                last_error = e
    else:
        print("[structure] No GROQ_API_KEY configured -- no fallback available.")

    raise last_error


def _extract_all_sources(product: ProductInput, sources: list[SourceHit]) -> dict:
    if not sources:
        return _empty_extraction(0)

    source_blocks = []
    for i, s in enumerate(sources):
        text = s.raw_text
        if not text and s.snippet:
            # Fall back to the search snippet if the web scraper was blocked or failed
            text = f"[Web scrape failed - falling back to search snippet]: {s.snippet}"
        text = (text or "")[:5000]
        source_blocks.append(f"--- SOURCE {i} ({s.origin}): {s.url} ---\n{text}")
    combined = "\n\n".join(source_blocks)

    user_prompt = f"""{SYSTEM_PROMPT}

KNOWN PRODUCT INFO:
Part Number: {product.part_number}
Brand: {product.brand}
Short Description: {product.short_description}

SOURCES:
{combined}
"""
    try:
        return _call_llm_with_fallback(user_prompt)
    except Exception as e:
        print(f"[structure] extraction failed on ALL providers: {e}")
        return _empty_extraction(len(sources))


def _normalize_label(label: str) -> str:
    return "".join(ch for ch in label.lower() if ch.isalnum())


async def structure_product(product: ProductInput, sources: list[SourceHit]) -> StructuredProduct:
    usable_sources = [s for s in sources if s.raw_text or s.snippet]
    parsed = await asyncio.to_thread(_extract_all_sources, product, usable_sources)
    per_source = parsed.get("sources", [])

    # ---- Category, short_desc, long_desc: same cross-validation pattern as before ----
    def resolve_text_field(field_name: str, freeform: bool) -> FieldValue:
        supporting = []
        for entry in per_source:
            idx = entry.get("source_index")
            value = entry.get(field_name)
            if value and idx is not None and 0 <= idx < len(usable_sources):
                supporting.append((value, usable_sources[idx]))

        if not supporting:
            return FieldValue(value=None, confidence=0.15, source_url=None, agreeing_sources=0, needs_review=True)

        if freeform:
            supporting.sort(key=lambda pair: len(pair[0]), reverse=True)
            chosen_value, chosen_source = supporting[0]
            return FieldValue(
                value=chosen_value,
                confidence=0.75 if len(supporting) >= 2 else 0.6,
                source_url=(chosen_source.url or None),
                agreeing_sources=len(supporting),
                needs_review=False,
            )

        groups = defaultdict(list)
        for value, src in supporting:
            groups[value.strip().lower()].append((value, src))
        best_key = max(groups, key=lambda k: len(groups[k]))
        best_group = groups[best_key]
        agreeing_count = len(best_group)
        chosen_value, chosen_source = best_group[0]
        conflicting = len(groups) > 1

        if agreeing_count >= 2 and not conflicting:
            confidence, needs_review = 0.95, False
        elif agreeing_count >= 2 and conflicting:
            confidence, needs_review = 0.7, True
        elif conflicting:
            confidence, needs_review = 0.35, True
        else:
            confidence, needs_review = 0.6, False

        return FieldValue(
            value=chosen_value, confidence=confidence, source_url=(chosen_source.url or None),
            agreeing_sources=agreeing_count, needs_review=needs_review,
        )

    category = resolve_text_field("category", freeform=False)
    extracted_brand_fv = resolve_text_field("brand", freeform=False)
    extracted_mfr_fv = resolve_text_field("manufacturer", freeform=False)
    
    # Fallback to product.brand if LLM could not resolve it
    resolved_brand = extracted_brand_fv.value if extracted_brand_fv.value else product.brand
    resolved_mfr = extracted_mfr_fv.value if extracted_mfr_fv.value else resolved_brand

    # Clean distributor names if resolved_brand or resolved_mfr is still bad
    bad_brands = ["appliance dealers cooperative", "appde", "-- unbranded --", "-- no unilog brand --", "unknown", "-- no dib brand --"]
    if not resolved_brand or resolved_brand.lower().strip() in bad_brands:
        desc_lower = (product.short_description or "").lower()
        if "frigidaire" in desc_lower:
            resolved_brand = "FRIGIDAIRE®"
        elif "whirlpool" in desc_lower:
            resolved_brand = "Whirlpool®"
        elif "kitchenaid" in desc_lower or "kitchen aid" in desc_lower:
            resolved_brand = "KitchenAid®"
        elif "lg" in desc_lower:
            resolved_brand = "LG®"
        elif "speed queen" in desc_lower:
            resolved_brand = "Speed Queen®"
        elif "beko" in desc_lower:
            resolved_brand = "Beko"

    if not resolved_mfr or resolved_mfr.lower().strip() in bad_brands or resolved_mfr == resolved_brand:
        clean_brand_name = (resolved_brand or "").lower().strip()
        if "frigidaire" in clean_brand_name:
            resolved_mfr = "Rheem Manufacturing"
        elif "whirlpool" in clean_brand_name:
            resolved_mfr = "Whirlpool Corporation"
        else:
            resolved_mfr = resolved_brand

    short_desc = resolve_text_field("short_desc", freeform=True)
    long_desc = resolve_text_field("long_desc", freeform=True)

    # ---- Attributes: group by normalized label across all sources ----
    attr_groups = defaultdict(list)  # normalized_label -> list of (label, value, uom, source)
    for entry in per_source:
        idx = entry.get("source_index")
        if idx is None or not (0 <= idx < len(usable_sources)):
            continue
        src = usable_sources[idx]
        for attr in entry.get("attributes", []) or []:
            label = (attr.get("label") or "").strip()
            value = (attr.get("value") or "").strip()
            uom = attr.get("uom")
            if not label or not value:
                continue
            attr_groups[_normalize_label(label)].append((label, value, uom, src))

    final_attributes = []
    for norm_label, entries in attr_groups.items():
        value_groups = defaultdict(list)
        for label, value, uom, src in entries:
            value_groups[value.strip().lower()].append((label, value, uom, src))
        best_key = max(value_groups, key=lambda k: len(value_groups[k]))
        best_group = value_groups[best_key]
        agreeing_count = len(best_group)
        label, value, uom, src = best_group[0]
        conflicting = len(value_groups) > 1

        if agreeing_count >= 2 and not conflicting:
            confidence, needs_review = 0.95, False
        elif agreeing_count >= 2 and conflicting:
            confidence, needs_review = 0.7, True
        elif conflicting:
            confidence, needs_review = 0.35, True
        else:
            confidence, needs_review = 0.6, False

        final_attributes.append(Attribute(
            label=label, value=value, uom=uom, confidence=confidence,
            source_url=(src.url or None), agreeing_sources=agreeing_count, needs_review=needs_review,
        ))

    # Highest-confidence attributes first, capped to the schema's 50 slots.
    final_attributes.sort(key=lambda a: a.confidence, reverse=True)
    final_attributes = final_attributes[:MAX_ATTRIBUTES]

    # ---- Controlled vocabulary validation (active only if real reference
    # files are present in backend/reference_data/ -- see vocabulary.py) ----
    normalized_brand, brand_validated = vocabulary.normalize_brand(resolved_brand)
    for attr in final_attributes:
        normalized_value, value_validated = vocabulary.normalize_attribute_value(attr.label, attr.value)
        normalized_uom, uom_validated = vocabulary.normalize_uom(attr.uom)
        attr.value = normalized_value
        attr.uom = normalized_uom
        attr.vocab_validated = value_validated or uom_validated

    return StructuredProduct(
        part_number=product.part_number,
        brand=normalized_brand,
        brand_vocab_validated=brand_validated,
        manufacturer=resolved_mfr,
        category=category,
        short_desc=short_desc,
        long_desc=long_desc,
        attributes=final_attributes,
        sources_used=[s.url for s in usable_sources if s.url],
    )
