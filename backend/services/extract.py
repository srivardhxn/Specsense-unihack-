"""
STAGE 2: Extract

Takes a discovered source (a URL) and pulls out usable raw text —
whether it's a normal webpage or a PDF datasheet. This is the
"document intelligence" step: turning messy real-world documents
into clean text the LLM can reason over in Stage 3.
"""
import httpx
import io
import re
import json
from bs4 import BeautifulSoup
import pdfplumber
from models import SourceHit
from services import cache

MAX_CHARS = 20000  # raised from 8000 -- spec tables (dimensions, weight, certs) are often further into the document


async def extract_text(source: SourceHit) -> SourceHit:
    """
    Fetches the URL and extracts readable text with persistent caching.
    Mutates and returns the SourceHit with raw_text filled in.
    On any failure, raw_text stays None — the structuring stage
    is expected to handle missing sources gracefully rather than crash.
    """
    # RAG-sourced hits already have raw_text filled in from the local
    # dataset index -- nothing to fetch over the network.
    if source.origin == "rag":
        return source

    # Check persistent scrape cache
    cached = cache.get_scrape(source.url)
    if cached is not None:
        title, text = cached
        if text:
            source.raw_text = text
            if title and not source.title:
                source.title = title
            print(f"[extract] cache hit for {source.url} ({len(text)} chars)")
            return source

    try:
        async with httpx.AsyncClient(
            timeout=15.0, follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
            }
        ) as client:
            resp = await client.get(source.url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")

            if "pdf" in content_type or source.url.lower().endswith(".pdf"):
                source.raw_text = _extract_pdf_text(resp.content)
            else:
                source.raw_text = _extract_html_text(resp.text)

            if source.raw_text:
                cache.set_scrape(source.url, source.title, source.raw_text)

    except Exception as e:
        # Don't let one bad source kill the whole pipeline.
        source.raw_text = None
        print(f"[extract] failed for {source.url}: {e}")

    return source


def _extract_json_objects(text: str) -> list[str]:
    results = []
    # Match pattern: "product": { or "product":{ or "product" : {
    for match in re.finditer(r'"product"\s*:\s*\{', text):
        start_idx = match.start()
        brace_count = 0
        end_idx = -1
        for i in range(match.end() - 1, len(text)):
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
                if brace_count == 0:
                    end_idx = i + 1
                    break
        if end_idx != -1:
            results.append(text[start_idx:end_idx])
    return results


def _extract_html_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    extracted_data = []

    # 1. Extract page title
    if soup.title and soup.title.string:
        extracted_data.append(f"Title: {soup.title.string.strip()}")

    # 2. Extract meta descriptions and keywords
    for meta in soup.find_all("meta"):
        name = (meta.get("name") or meta.get("property") or "").lower()
        content = meta.get("content")
        if content and name in ["description", "keywords", "og:description", "og:title"]:
            extracted_data.append(f"Meta {name}: {content.strip()}")

    # 3. Extract JSON-LD script blocks
    for s in soup.find_all("script", type="application/ld+json"):
        if s.string:
            extracted_data.append(f"JSON-LD Structured Data: {s.string.strip()}")

    # 4. Extract custom embedded product JSON states from JavaScript blocks
    for s in soup.find_all("script"):
        # Skip JSON-LD script blocks since we already got them
        if s.get("type") == "application/ld+json":
            continue
        content = s.string or ""
        if len(content) > 1000 and "product" in content.lower():
            try:
                json_blocks = _extract_json_objects(content)
                for block in json_blocks:
                    # Validate and clean up
                    wrapped = "{" + block + "}"
                    try:
                        parsed = json.loads(wrapped)
                        # Pretty print it to make it readable for the LLM
                        extracted_data.append(f"Product State JSON: {json.dumps(parsed)}")
                    except Exception:
                        # If validation fails, just append raw matched block
                        extracted_data.append(f"Product State JSON Raw: {block}")
            except Exception:
                pass

    # Now decompose scripts, styles, header, footer, nav to clean the body HTML
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    # Extract clean body text
    body_text = soup.get_text(separator=" ", strip=True)
    if body_text:
        extracted_data.append(f"Body Text: {body_text}")

    # Combine everything up to MAX_CHARS
    full_text = "\n\n".join(extracted_data)
    return full_text[:MAX_CHARS]


def _extract_pdf_text(pdf_bytes: bytes) -> str:
    text_parts = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page in pdf.pages[:20]:  # raised from 5 -- physical specs/certs are often on later pages
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
            # also pull tables — spec sheets are often table-heavy
            for table in page.extract_tables():
                for row in table:
                    text_parts.append(" | ".join(c or "" for c in row))
    return "\n".join(text_parts)[:MAX_CHARS]

