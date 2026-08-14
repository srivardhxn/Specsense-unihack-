"""
STAGE 2: Extract

Takes a discovered source (a URL) and pulls out usable raw text —
whether it's a normal webpage or a PDF datasheet. This is the
"document intelligence" step: turning messy real-world documents
into clean text the LLM can reason over in Stage 3.
"""
import httpx
import io
from bs4 import BeautifulSoup
import pdfplumber
from models import SourceHit

MAX_CHARS = 20000  # raised from 8000 -- spec tables (dimensions, weight, certs) are often further into the document


async def extract_text(source: SourceHit) -> SourceHit:
    """
    Fetches the URL and extracts readable text.
    Mutates and returns the SourceHit with raw_text filled in.
    On any failure, raw_text stays None — the structuring stage
    is expected to handle missing sources gracefully rather than crash.
    """
    # RAG-sourced hits already have raw_text filled in from the local
    # dataset index -- nothing to fetch over the network.
    if source.origin == "rag":
        return source

    try:
        async with httpx.AsyncClient(
            timeout=8.0, follow_redirects=True,
            headers={"User-Agent": "Mozilla/5.0 (SpecSense hackathon bot)"}
        ) as client:
            resp = await client.get(source.url)
            resp.raise_for_status()
            content_type = resp.headers.get("content-type", "")

            if "pdf" in content_type or source.url.lower().endswith(".pdf"):
                source.raw_text = _extract_pdf_text(resp.content)
            else:
                source.raw_text = _extract_html_text(resp.text)

    except Exception as e:
        # Don't let one bad source kill the whole pipeline.
        source.raw_text = None
        print(f"[extract] failed for {source.url}: {e}")

    return source


def _extract_html_text(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()
    text = soup.get_text(separator=" ", strip=True)
    return text[:MAX_CHARS]


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
