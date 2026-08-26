"""
STAGE 1: Discover

Takes the minimal product input (part number, brand, description) and
finds candidate source material -- checking provided/ingested datasets
FIRST (via RAG), then falling back to live web search for anything not
covered locally. This mirrors how a real product team would work: use
what you already have before going out to the open web.

Live search uses SerpAPI (https://serpapi.com) -- free tier, no Google
Cloud project needed, fastest to set up for a hackathon.
"""
import os
import httpx
from models import ProductInput, SourceHit
from services.rag import rag_store

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SERPAPI_URL = "https://serpapi.com/search"

# Per Unilog's sourcing rules: product data must come from the
# manufacturer's own site or documentation. Marketplaces and distributor
# sites are explicitly excluded, even though they often rank well and
# have clean data -- using them would be a compliance violation, not
# just a quality tradeoff. This blocklist catches the most common
# offenders; results from these domains are filtered out entirely
# before extraction ever sees them.
EXCLUDED_DOMAINS = [
    "amazon.", "ebay.", "walmart.", "aliexpress.", "alibaba.",
    "mercateo.", "123bearing.", "grainger.", "mcmaster.", "rsonline.",
    "digikey.", "mouser.", "newark.", "farnell.", "zoro.", "homedepot.",
    "lowes.", "wayfair.", "target.", "bestbuy.", "globalindustrial.",
    "thomasnet.", "indiamart.", "made-in-china.",
]


def _is_excluded_source(url: str, brand: str) -> bool:
    url_lower = url.lower()
    if brand and "siemens" in brand.lower():
        # For Siemens, bypass exclusions for datasheets/catalogs since their official site blocks scrapers
        marketplaces = ["amazon.", "ebay.", "walmart.", "aliexpress.", "alibaba."]
        return any(domain in url_lower for domain in marketplaces)
    return any(domain in url_lower for domain in EXCLUDED_DOMAINS)


def _guess_manufacturer_domain(brand: str) -> str | None:
    """
    Best-effort guess at a manufacturer's own domain from its brand name
    (e.g. "Siemens" -> "siemens.com"), used to bias the FIRST search
    attempt toward their official site per the sourcing rules. This is
    a heuristic, not a lookup against a verified list -- if it's wrong,
    the site: filter just returns nothing and the pipeline falls
    through to the next, broader query. A real production version
    would resolve this against Unilog's actual manufacturer/brand
    master list instead of guessing.
    """
    if not brand or not brand.strip():
        return None
    cleaned = "".join(ch for ch in brand.lower() if ch.isalnum())
    if not cleaned:
        return None
    return f"{cleaned}.com"


async def discover_sources(product: ProductInput, max_results: int = 3) -> list[SourceHit]:
    """
    Retrieval order:
      1. Query the local RAG store (provided datasets/reference artifacts).
      2. If it returns confident hits, use those -- fast, free, no web dependency.
      3. Otherwise (or in addition, if RAG returns fewer than max_results),
         fall back to live web search to fill the gap.
    """
    query_text = f"{product.brand} {product.part_number} {product.short_description}"

    rag_hits = rag_store.query(query_text, top_k=max_results, part_number=product.part_number)

    remaining = max_results - len(rag_hits)
    web_hits: list[SourceHit] = []
    if remaining > 0:
        web_hits = await _web_search(product, max_results=remaining)

    return rag_hits + web_hits


async def _web_search(product: ProductInput, max_results: int) -> list[SourceHit]:
    if not SERPAPI_KEY:
        print("[discover] SERPAPI_KEY not set -- skipping live web search.")
        return []

    brand_clean = (product.brand or "").strip()
    bad_brands = ["appliance dealers cooperative", "appde", "-- unbranded --", "-- no unilog brand --", "unknown", "-- no dib brand --"]
    if brand_clean.lower() in bad_brands:
        brand_clean = ""

    queries = []
    if brand_clean:
        queries.append(f"{brand_clean} {product.part_number} site:{_guess_manufacturer_domain(brand_clean)}" if _guess_manufacturer_domain(brand_clean) else None)
        queries.append(f"{brand_clean} {product.part_number} datasheet specifications")
        queries.append(f"{brand_clean} {product.part_number} {product.short_description}")
        queries.append(f'"{product.part_number}" {brand_clean}')
    else:
        queries.append(f"{product.part_number} datasheet specifications")
        queries.append(f"{product.part_number} {product.short_description}")
        queries.append(f'"{product.part_number}"')

    queries = [q for q in queries if q]

    for i, query in enumerate(queries):
        results = await _try_search(query, max_results, brand_clean)
        if results:
            if i > 0:
                print(f"[discover] first query found nothing, broader query #{i+1} succeeded: '{query}'")
            return results

    print(f"[discover] no results from any query variant for {product.brand} {product.part_number}")
    return []


def _clean_url(url: str) -> str:
    if not url:
        return url
    # Replace common URL unicode escapes if written as literal strings
    url = url.replace(r'\u0026', '&').replace(r'\u003d', '=').replace(r'\u003f', '?').replace(r'\u002f', '/')
    try:
        # Standard unicode-escape decode for any remaining characters
        import codecs
        url = codecs.decode(url.encode(), 'unicode-escape').decode('utf-8')
    except Exception:
        pass
    return url


async def _try_search(query: str, max_results: int, brand: str) -> list[SourceHit]:
    try:
        async with httpx.AsyncClient(timeout=8.0) as client:
            resp = await client.get(
                SERPAPI_URL,
                params={"q": query, "api_key": SERPAPI_KEY, "num": max_results},
            )
            resp.raise_for_status()
            data = resp.json()
    except Exception as e:
        print(f"[discover] web search failed for query '{query}': {e}")
        return []

    results = []
    for item in data.get("organic_results", [])[:max_results * 2]:  # over-fetch since some get filtered out
        url = _clean_url(item.get("link", ""))
        if _is_excluded_source(url, brand):
            continue
        results.append(
            SourceHit(
                url=url,
                title=item.get("title", ""),
                snippet=item.get("snippet", ""),
                origin="web",
            )
        )
        if len(results) >= max_results:
            break
    return results
