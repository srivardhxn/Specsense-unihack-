"""
STAGE 1: Discover

Finds candidate source material for minimal product inputs.
Retrieval Hierarchy:
  1. Local RAG store (provided reference datasets in backend/datasets/)
  2. Persistent SQLite Search Cache (0ms latency, zero API calls)
  3. Free, Zero-Quota Web Discovery (DuckDuckGo Lite + Bing en-US)
  4. SerpAPI (optional, gracefully bypassed if searches are depleted)
  5. Industrial Manufacturer Spec Direct Resolver (for known industrial lines)

Strict Sourcing Compliance:
  Excludes consumer marketplaces (Amazon, eBay, Alibaba, Walmart) and
  distributor storefronts per Unilog industrial procurement compliance rules.
"""
import os
import re
import httpx
from bs4 import BeautifulSoup
from urllib.parse import urlparse, parse_qs, unquote
from models import ProductInput, SourceHit
from services.rag import rag_store
from services import cache

SERPAPI_KEY = os.getenv("SERPAPI_KEY")
SERPAPI_URL = "https://serpapi.com/search"

# Track SerpApi status to prevent spamming exhausted accounts
_serpapi_exhausted = False

# Marketplace and non-compliant source exclusions
EXCLUDED_DOMAINS = [
    "amazon.", "ebay.", "walmart.", "aliexpress.", "alibaba.",
    "mercateo.", "123bearing.", "grainger.", "mcmaster.", "rsonline.",
    "digikey.", "mouser.", "newark.", "farnell.", "zoro.", "homedepot.",
    "lowes.", "wayfair.", "target.", "bestbuy.", "globalindustrial.",
    "thomasnet.", "indiamart.", "made-in-china.",
    "facebook.com", "instagram.com", "twitter.com", "x.com", "youtube.com",
    "pinterest.com", "play.google.com", "apps.apple.com", "apps.microsoft.com",
    "zhihu.com", "baidu.com", "weibo.com", "reddit.com", "quora.com",
    "google.", "bing.com", "microsoft.com", "yahoo.com", "duckduckgo.com",
    "gsmarena.com", "phonearena.com", "t-mobile.com", "verizon.com", "att.com",
    "flipkart.com", "psychforums.com", "resumeviking.com", "requestletters.com",
    "manaui.com", "forum.", "forums.",
]


def _is_excluded_source(url: str, brand: str) -> bool:
    if not url:
        return True
    url_lower = url.lower()
    
    # Filter spam, adult, and off-topic non-industrial content
    spam_keywords = [
        "xhamster", "pornviden", "bokep", "porn", "xxx", "adult", "sex", "redtube",
        "pornhub", "xnxx", "xvideos", "cell-phone", "smartphone", "/recipe/",
        "/resignation-letter", "psychology", "paraphilias", "topic300", "topic450",
    ]
    if any(kw in url_lower for kw in spam_keywords):
        return True
        
    # Filter generic non-product corporate pages
    if "wikipedia.org" in url_lower or "linkedin.com" in url_lower:
        return True
        
    try:
        parsed = urlparse(url)
        path = parsed.path.strip("/")
        if not path:
            return True
        parts = [p for p in path.split("/") if p]
        if len(parts) == 1 and (len(parts[0]) <= 3 or (len(parts[0]) == 5 and parts[0][2] == "-")):
            return True
        generic_keywords = ["/career", "/job", "/about", "/login", "/register", "/contact", "/terms", "/privacy", "/press", "/threads/", "/thread/"]
        if any(kw in url_lower for kw in generic_keywords):
            return True
    except Exception:
        pass

    # Exception for direct datasheets / PDFs of major industrial brands
    if brand and any(b in brand.lower() for b in ["siemens", "skf", "schneider", "abb"]):
        if url_lower.endswith(".pdf") or "datasheet" in url_lower or "catalog" in url_lower:
            marketplaces = ["amazon.", "ebay.", "walmart.", "aliexpress.", "alibaba."]
            return any(domain in url_lower for domain in marketplaces)

    return any(domain in url_lower for domain in EXCLUDED_DOMAINS)


def _guess_manufacturer_domain(brand: str) -> str | None:
    if not brand or not brand.strip():
        return None
    cleaned = "".join(ch for ch in brand.lower() if ch.isalnum())
    if not cleaned:
        return None
    return f"{cleaned}.com"


def _score_source(url: str) -> float:
    url_lower = url.lower()
    score = 0.0
    if url_lower.endswith(".pdf") or "pdf" in url_lower or "datasheet" in url_lower:
        score += 12.0
    product_keywords = ["/product/", "/products/", "/part/", "/parts/", "/bearing/", "/bearings/", "catalog", "specification", "spec", "datasheets.com", "alldatasheet.com"]
    if any(kw in url_lower for kw in product_keywords):
        score += 6.0
    noise_keywords = ["hsn", "gst", "cleartax", "tax", "import", "export", "news", "forum", "blog", "wikipedia"]
    if any(kw in url_lower for kw in noise_keywords):
        score -= 10.0
    return score


def _clean_url(url: str) -> str:
    if not url:
        return ""
    url = url.replace(r'\u0026', '&').replace(r'\u003d', '=').replace(r'\u003f', '?').replace(r'\u002f', '/')
    try:
        import codecs
        url = codecs.decode(url.encode(), 'unicode-escape').decode('utf-8')
    except Exception:
        pass
    # Clean DDG redirect links
    if "/l/?" in url or "duckduckgo.com/l/?" in url:
        try:
            parsed = urlparse(url)
            qs = parse_qs(parsed.query)
            if "uddg" in qs:
                url = unquote(qs["uddg"][0])
        except Exception:
            pass
    return url


async def discover_sources(product: ProductInput, max_results: int = 4) -> list[SourceHit]:
    """
    Zero-Quota Discovery Engine:
      1. Local RAG store (pre-ingested datasets)
      2. Multi-tier web search with persistent caching (DDG Lite -> Bing -> SerpApi)
    """
    query_text = f"{product.brand} {product.part_number} {product.short_description}".strip()

    # 1. Check local RAG store
    rag_hits = rag_store.query(query_text, top_k=max_results, part_number=product.part_number)

    remaining = max_results - len(rag_hits)
    web_hits: list[SourceHit] = []
    if remaining > 0:
        web_hits = await _web_search(product, max_results=remaining)

    combined = rag_hits + web_hits
    return combined[:max_results]


async def _web_search(product: ProductInput, max_results: int) -> list[SourceHit]:
    brand_clean = (product.brand or "").strip()
    bad_brands = ["appliance dealers cooperative", "appde", "-- unbranded --", "-- no unilog brand --", "unknown", "-- no dib brand --"]
    if brand_clean.lower() in bad_brands:
        brand_clean = ""

    pn_orig = product.part_number.strip()
    pn_clean = pn_orig.replace(" ", "")

    desc_words = [w for w in re.sub(r'[^a-zA-Z0-9]', ' ', product.short_description or '').split() if len(w) >= 3 and w.lower() not in pn_orig.lower()]
    desc_snippet = " ".join(desc_words[:2])

    queries = []
    if brand_clean and desc_snippet:
        queries.append(f'"{pn_clean}" {brand_clean} {desc_snippet}')
        queries.append(f'"{pn_clean}" {brand_clean} datasheet specifications')
    elif brand_clean:
        queries.append(f'"{pn_clean}" {brand_clean} datasheet specifications')
        queries.append(f'"{pn_orig}" {brand_clean}')
    else:
        queries.append(f'"{pn_clean}" {desc_snippet}'.strip())
        queries.append(f'"{pn_clean}" datasheet specifications')

    all_results = []
    seen_urls = set()

    for query in queries[:2]:
        results = await _try_search_cached(query, max_results, brand_clean)
        for r in results:
            if r.url and r.url not in seen_urls:
                seen_urls.add(r.url)
                all_results.append(r)
        if len(all_results) >= max_results:
            break

    # If still no results, try manufacturer domain heuristic
    if not all_results and brand_clean:
        mfr_domain = _guess_manufacturer_domain(brand_clean)
        if mfr_domain:
            heuristic_query = f"{pn_clean} site:{mfr_domain}"
            results = await _try_search_cached(heuristic_query, max_results, brand_clean)
            for r in results:
                if r.url and r.url not in seen_urls:
                    seen_urls.add(r.url)
                    all_results.append(r)

    # Rank discovered sources to prioritize genuine datasheets/specs
    all_results.sort(key=lambda hit: _score_source(hit.url), reverse=True)
    return all_results[:max_results]


async def _try_search_cached(query: str, max_results: int, brand: str) -> list[SourceHit]:
    # 1. Check persistent SQLite cache
    cached_hits = cache.get_search(query)
    if cached_hits is not None:
        print(f"[discover] cache hit for query: '{query}' ({len(cached_hits)} hits)")
        return cached_hits[:max_results]

    # 2. Execute live search across resilient engines
    results = await _execute_live_search(query, brand)

    # 3. Cache discovered results permanently
    if results:
        cache.set_search(query, results)

    return results[:max_results]


async def _execute_live_search(query: str, brand: str) -> list[SourceHit]:
    search_limit = 8

    # Tier 1: DuckDuckGo Lite (Free, zero-quota, highly resilient, returns official manufacturer links)
    try:
        results = await _ddg_lite_search(query, search_limit, brand)
        if results:
            print(f"[discover] DDG Lite found {len(results)} candidate sources for '{query}'")
            return results
    except Exception as e:
        print(f"[discover] DDG Lite search exception: {e}")

    # Tier 2: Bing Search (Free, zero-quota, en-US filtered)
    try:
        results = await _bing_search(query, search_limit, brand)
        if results:
            print(f"[discover] Bing found {len(results)} candidate sources for '{query}'")
            return results
    except Exception as e:
        print(f"[discover] Bing search exception: {e}")

    # Tier 3: SerpAPI (Optional fallback, checked only if not previously exhausted)
    global _serpapi_exhausted, SERPAPI_KEY
    if SERPAPI_KEY and not _serpapi_exhausted:
        try:
            results = await _serpapi_search(query, search_limit, brand)
            if results:
                return results
        except Exception as e:
            print(f"[discover] SerpAPI search exception: {e}")

    return []


async def _ddg_lite_search(query: str, max_results: int, brand: str) -> list[SourceHit]:
    url = "https://lite.duckduckgo.com/lite/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"q": query}

    async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
        resp = await client.post(url, data=data, headers=headers)
        if resp.status_code != 200:
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        
        # In DDG lite, results are structured in tables with class .result-link
        rows = soup.select("table tbody tr")
        current_title = ""
        current_url = ""
        
        for row in rows:
            link_el = row.select_one("a.result-link")
            snippet_el = row.select_one("td.result-snippet")
            
            if link_el:
                current_title = link_el.get_text(strip=True)
                raw_href = link_el.get("href", "")
                current_url = _clean_url(raw_href)
            elif snippet_el and current_url:
                snippet = snippet_el.get_text(strip=True)
                if not _is_excluded_source(current_url, brand):
                    results.append(
                        SourceHit(
                            url=current_url,
                            title=current_title,
                            snippet=snippet,
                            origin="web",
                        )
                    )
                current_url = ""
                current_title = ""
                if len(results) >= max_results:
                    break

        # Fallback to standard a.result-link elements if snippet rows differed
        if not results:
            for link_el in soup.select("a.result-link"):
                raw_href = link_el.get("href", "")
                clean = _clean_url(raw_href)
                title = link_el.get_text(strip=True)
                if clean and not _is_excluded_source(clean, brand):
                    results.append(SourceHit(url=clean, title=title, snippet="", origin="web"))
                if len(results) >= max_results:
                    break

        return results


def _decode_bing_url(url: str) -> str:
    if not url:
        return ""
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
                    return base64.b64decode(encoded).decode("utf-8", errors="ignore")
        except Exception:
            pass
    return url


async def _bing_search(query: str, max_results: int, brand: str) -> list[SourceHit]:
    url = "https://www.bing.com/search"
    params = {"q": query, "setlang": "en-US", "cc": "US"}
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    }

    async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
        resp = await client.get(url, params=params, headers=headers)
        if resp.status_code != 200:
            return []

        soup = BeautifulSoup(resp.text, "html.parser")
        results = []
        for item in soup.select("li.b_algo"):
            a_el = item.select_one("h2 a")
            snippet_el = item.select_one(".b_caption p") or item.select_one(".b_algoSlug")
            if not a_el:
                continue

            raw_href = a_el.get("href", "")
            clean = _clean_url(_decode_bing_url(raw_href))
            if not clean or _is_excluded_source(clean, brand):
                continue

            # Ensure ASCII/English clean title to filter out foreign forum scrapes
            title = a_el.get_text(strip=True).encode("ascii", "ignore").decode("ascii")
            snippet = snippet_el.get_text(strip=True).encode("ascii", "ignore").decode("ascii") if snippet_el else ""

            results.append(
                SourceHit(
                    url=clean,
                    title=title or a_el.get_text(strip=True),
                    snippet=snippet,
                    origin="web",
                )
            )
            if len(results) >= max_results:
                break
        return results


async def _serpapi_search(query: str, max_results: int, brand: str) -> list[SourceHit]:
    global _serpapi_exhausted, SERPAPI_KEY
    if not SERPAPI_KEY or _serpapi_exhausted:
        return []

    async with httpx.AsyncClient(timeout=6.0) as client:
        resp = await client.get(
            SERPAPI_URL,
            params={"q": query, "api_key": SERPAPI_KEY, "num": max_results},
        )
        if resp.status_code != 200:
            return []

        data = resp.json()
        if "error" in data:
            err_msg = str(data["error"])
            print(f"[discover] SerpAPI response notice: {err_msg}")
            if "run out of searches" in err_msg.lower() or "limit" in err_msg.lower() or "quota" in err_msg.lower():
                _serpapi_exhausted = True
                print("[discover] SerpAPI searches exhausted. Bypassing SerpAPI for subsequent lookups.")
            return []

        results = []
        for item in data.get("organic_results", []):
            url = _clean_url(item.get("link", ""))
            if not url or _is_excluded_source(url, brand):
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
