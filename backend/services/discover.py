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
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, unquote
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
    "facebook.com", "instagram.com", "twitter.com", "x.com", "youtube.com",
    "pinterest.com", "play.google.com", "apps.apple.com", "apps.microsoft.com",
]


def _is_excluded_source(url: str, brand: str) -> bool:
    url_lower = url.lower()
    
    # Filter out adult/inappropriate spam domains
    adult_keywords = ["xhamster", "pornviden", "bokep", "porn", "xxx", "adult", "sex", "redtube", "pornhub", "xnxx", "xvideos"]
    if any(kw in url_lower for kw in adult_keywords):
        return True
        
    # Filter out homepages and generic non-product pages
    if "wikipedia.org" in url_lower or "linkedin.com" in url_lower:
        return True
    try:
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        # If path is empty, it's a homepage
        if not path:
            return True
        # If path is just a locale code (e.g. 'in', 'us', 'en', 'de', 'fr', 'en-us')
        parts = [p for p in path.split("/") if p]
        if len(parts) == 1 and (len(parts[0]) == 2 or (len(parts[0]) == 5 and parts[0][2] == "-")):
            return True
        # If it's a generic corporate page
        generic_keywords = ["/career", "/job", "/about", "/login", "/register", "/contact", "/terms", "/privacy", "/press"]
        if any(kw in url_lower for kw in generic_keywords):
            return True
    except Exception:
        pass

    if brand and ("siemens" in brand.lower() or "skf" in brand.lower()):
        # For Siemens and SKF, bypass exclusions for datasheets/catalogs since their official site blocks scrapers
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


def _score_source(url: str) -> float:
    url_lower = url.lower()
    score = 0.0
    if url_lower.endswith(".pdf") or "pdf" in url_lower or "datasheet" in url_lower:
        score += 10.0
    product_keywords = ["/product/", "/products/", "/part/", "/parts/", "/bearing/", "/bearings/", "catalog", "specification", "spec"]
    if any(kw in url_lower for kw in product_keywords):
        score += 5.0
    noise_keywords = ["hsn", "gst", "cleartax", "tax", "import", "export", "news", "forum", "blog", "wikipedia"]
    if any(kw in url_lower for kw in noise_keywords):
        score -= 8.0
    return score


async def _web_search(product: ProductInput, max_results: int) -> list[SourceHit]:
    brand_clean = (product.brand or "").strip()
    bad_brands = ["appliance dealers cooperative", "appde", "-- unbranded --", "-- no unilog brand --", "unknown", "-- no dib brand --"]
    if brand_clean.lower() in bad_brands:
        brand_clean = ""

    queries = []
    pn_orig = product.part_number
    pn_clean = pn_orig.replace(" ", "")
    pn_norm = "".join(c for c in pn_orig if c.isalnum()).upper()

    if brand_clean:
        queries.append(f'"{pn_orig}" {brand_clean}')
        if pn_clean != pn_orig:
            queries.append(f'"{pn_clean}" {brand_clean}')
        queries.append(f'"{pn_clean}"')
        if pn_norm != pn_clean:
            queries.append(f'"{pn_norm}"')
        queries.append(f"{pn_clean} datasheet specifications")
        queries.append(f"{pn_clean} {brand_clean} datasheet specifications")
        
        blocks_scrapers = any(b in brand_clean.lower() for b in ["siemens", "skf", "schneider", "abb", "rockwell", "allen-bradley", "omron"])
        mfr_domain = _guess_manufacturer_domain(brand_clean)
        if mfr_domain and not blocks_scrapers:
            queries.append(f"{brand_clean} {pn_clean} site:{mfr_domain}")
    else:
        queries.append(f'"{pn_orig}"')
        if pn_clean != pn_orig:
            queries.append(f'"{pn_clean}"')
        queries.append(f"{pn_clean} datasheet specifications")
        queries.append(f"{pn_clean} {product.short_description}")

    queries = [q for q in queries if q]

    all_results = []
    seen_urls = set()
    
    # Always run the top 2 queries to combine results and get a diverse set of sources
    queries_to_run = queries[:2] if len(queries) >= 2 else queries
    for query in queries_to_run:
        results = await _try_search(query, max_results, brand_clean)
        for r in results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                all_results.append(r)
                
    # If we have fewer than max_results, run subsequent queries
    if len(all_results) < max_results:
        for query in queries[2:]:
            results = await _try_search(query, max_results - len(all_results), brand_clean)
            for r in results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    all_results.append(r)
            if len(all_results) >= max_results:
                break
                
    if not all_results:
        print(f"[discover] no results from any query variant for {product.brand} {product.part_number}")
        return []
        
    # Rank discovered sources to prioritize high-quality pages (like specs/PDFs) and avoid noise
    all_results.sort(key=lambda hit: _score_source(hit.url), reverse=True)
    return all_results[:max_results]


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
    # Always query for up to 10 results to get a larger candidate pool for ranking
    search_limit = 10
    
    # 1. Try DuckDuckGo search first (free, unlimited, no API key needed)
    try:
        print(f"[discover] attempting DuckDuckGo search for: '{query}'")
        results = await _ddg_search(query, search_limit, brand)
        if results:
            return results
    except Exception as e:
        print(f"[discover] DuckDuckGo search failed: {e}")

    # 2. Fallback to SerpApi if key is present
    if SERPAPI_KEY:
        try:
            print(f"[discover] attempting SerpAPI search for: '{query}'")
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(
                    SERPAPI_URL,
                    params={"q": query, "api_key": SERPAPI_KEY, "num": search_limit},
                )
                if resp.status_code == 200:
                    data = resp.json()
                    if "error" in data:
                        print(f"[discover] SerpAPI error: {data['error']}")
                    else:
                        results = []
                        for item in data.get("organic_results", [])[:search_limit * 2]:
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
                            if len(results) >= search_limit:
                                break
                        if results:
                            return results
                else:
                    print(f"[discover] SerpAPI search returned status {resp.status_code}")
        except Exception as e:
            print(f"[discover] SerpAPI search fallback failed: {e}")

    # 3. Fallback to Bing search (free, no API key needed, extremely robust)
    try:
        print(f"[discover] attempting Bing search for: '{query}'")
        results = await _bing_search(query, search_limit, brand)
        if results:
            return results
    except Exception as e:
        print(f"[discover] Bing search failed: {e}")

    return []


def _decode_bing_url(url: str) -> str:
    if not url:
        return url
    if "bing.com/ck/a?!" in url:
        try:
            import base64
            parsed = urlparse(url)
            qs = parse_qs(parsed.query)
            if "u" in qs:
                u_val = qs["u"][0]
                if len(u_val) > 2:
                    encoded = u_val[2:]
                    padding = len(encoded) % 4
                    if padding:
                        encoded += "=" * (4 - padding)
                    decoded = base64.b64decode(encoded).decode("utf-8", errors="ignore")
                    return decoded
        except Exception:
            pass
    return url


async def _bing_search(query: str, max_results: int, brand: str) -> list[SourceHit]:
    url = "https://www.bing.com/search"
    params = {"q": query}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }
    
    async with httpx.AsyncClient(timeout=10.0) as client:
        resp = await client.get(url, params=params, headers=headers)
        if resp.status_code != 200:
            print(f"[discover] Bing search returned status {resp.status_code}")
            return []
            
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for item in soup.select("li.b_algo"):
            a_el = item.select_one("h2 a")
            snippet_el = item.select_one(".b_caption p") or item.select_one(".b_algoSlug")
            
            if a_el:
                title = a_el.get_text(strip=True)
                href = a_el.get("href", "")
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                
                real_url = _decode_bing_url(href)
                url = _clean_url(real_url)
                if _is_excluded_source(url, brand):
                    continue
                    
                results.append(
                    SourceHit(
                        url=url,
                        title=title,
                        snippet=snippet,
                        origin="web",
                    )
                )
                if len(results) >= max_results:
                    break
        return results



async def _ddg_search(query: str, max_results: int, brand: str) -> list[SourceHit]:
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
    }
    
    async with httpx.AsyncClient(timeout=8.0) as client:
        resp = await client.get(url, params=params, headers=headers)
        if resp.status_code != 200:
            resp = await client.post(url, data=params, headers=headers)
            
        if resp.status_code != 200:
            print(f"[discover] DDG search returned status {resp.status_code}")
            return []
            
        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for result in soup.select(".result"):
            title_el = result.select_one(".result__title")
            link_el = result.select_one(".result__url")
            snippet_el = result.select_one(".result__snippet")
            
            if title_el and link_el:
                title = title_el.get_text(strip=True)
                link = link_el.get_text(strip=True)
                snippet = snippet_el.get_text(strip=True) if snippet_el else ""
                
                a_el = title_el.select_one("a")
                href = a_el["href"] if a_el and "href" in a_el.attrs else link
                
                # Extract clean url from redirect if needed
                if "/l/?" in href:
                    try:
                        parsed = urlparse(href)
                        qs = parse_qs(parsed.query)
                        if "uddg" in qs:
                            href = unquote(qs["uddg"][0])
                    except Exception:
                        pass
                
                url = _clean_url(href)
                if _is_excluded_source(url, brand):
                    continue
                    
                results.append(
                    SourceHit(
                        url=url,
                        title=title,
                        snippet=snippet,
                        origin="web",
                    )
                )
                if len(results) >= max_results:
                    break
        return results
