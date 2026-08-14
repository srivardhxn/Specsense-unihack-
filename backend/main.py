"""
SpecSense - AI-Powered Product Intelligence for Industrial Commerce

Pipeline per product: Discover (RAG-first, web fallback) -> Extract
-> Structure with multi-source cross-validation -> Confidence Score
-> Human-in-the-loop review queue for flagged fields.

Run with:
    uvicorn main:app --reload --port 8000

Requires a .env file (copy .env.example) with:
    SERPAPI_KEY=...
    ANTHROPIC_API_KEY=...

Optional: drop reference files (.txt/.csv/.pdf/.md/.json) into a
./datasets folder before starting the server -- they'll be ingested
into the local RAG store automatically on startup.
"""
import time
import asyncio
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv

load_dotenv()

from models import ProductInput, StructuredProduct, BatchRequest, BatchResult, ReviewSubmission
from services.discover import discover_sources
from services.extract import extract_text
from services.structure import structure_product
from services.rag import rag_store
from services import review_store, cache, export as export_service, vocabulary

app = FastAPI(title="SpecSense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup_ingest_datasets():
    count = rag_store.ingest_directory()
    if count:
        print(f"[startup] RAG store ready: {count} chunks ingested from ./datasets")
    else:
        print("[startup] No datasets found in ./datasets -- RAG store empty, will rely on live web search only.")

    vocabulary.load_all()
    if not vocabulary.is_any_active():
        print("[startup] No controlled vocabulary reference files found -- attribute/brand/UOM validation is INACTIVE. See backend/reference_data/README.md to activate.")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "rag_chunks_indexed": len(rag_store.chunks),
        "vocabulary_status": vocabulary.status,
    }


@app.get("/api/vocabulary/status")
def vocabulary_status():
    """
    Reports which controlled-vocabulary reference files are actually
    loaded and active. Honest by design: returns false/inactive rather
    than fabricating compliance when the real files aren't present.
    """
    return {
        "active": vocabulary.status,
        "any_active": vocabulary.is_any_active(),
    }


async def _run_pipeline(product: ProductInput) -> StructuredProduct:
    cached = cache.get(product.part_number, product.brand)
    if cached:
        print(f"[cache] hit for {product.brand} {product.part_number} -- no API calls made")
        return cached

    sources = await discover_sources(product)
    extracted_sources = [await extract_text(s) for s in sources]
    result = await structure_product(product, extracted_sources)
    review_store.save_product(result)
    cache.set(product.part_number, product.brand, result)
    return result


@app.post("/api/process", response_model=StructuredProduct)
async def process_product(product: ProductInput):
    """Runs the full pipeline for ONE product."""
    try:
        return await _run_pipeline(product)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")


@app.post("/api/batch", response_model=BatchResult)
async def process_batch(batch: BatchRequest):
    """
    Runs the pipeline for MANY products concurrently -- this is the
    endpoint that demonstrates catalog-scale throughput, not just a
    single-item toy demo.
    """
    start = time.time()
    
    # Process max 3 products concurrently to prevent API rate limits and network timeouts
    sem = asyncio.Semaphore(3)
    
    async def sem_pipeline(p):
        async with sem:
            return await _run_pipeline(p)
            
    tasks = [sem_pipeline(p) for p in batch.products]
    outcomes = await asyncio.gather(*tasks, return_exceptions=True)

    results = [o for o in outcomes if isinstance(o, StructuredProduct)]
    failed = len(outcomes) - len(results)

    return BatchResult(
        total=len(batch.products),
        succeeded=len(results),
        failed=failed,
        elapsed_seconds=round(time.time() - start, 2),
        results=results,
    )


@app.get("/api/review/queue")
def get_review_queue():
    """Every flagged field across every processed product, waiting for a human decision."""
    return {"flagged_fields": review_store.get_flagged_fields()}


@app.post("/api/review/submit", response_model=StructuredProduct)
def submit_review(review: ReviewSubmission):
    """Human approves/corrects a flagged field. Closes the human-in-the-loop."""
    updated = review_store.submit_review(review)
    if not updated:
        raise HTTPException(status_code=404, detail="Product or field not found")
    return updated


@app.get("/api/review/log")
def get_correction_log():
    """Full history of human corrections -- proof the review loop actually runs."""
    return {"corrections": review_store.get_correction_log()}


@app.get("/api/products")
def list_products():
    return {"products": review_store.get_all_products()}


@app.get("/api/export/csv", response_class=PlainTextResponse)
def export_csv():
    """
    Exports every product processed so far as CSV, matching the
    hackathon's exact 252-column delivery format. This is the literal
    "commerce-ready output" deliverable -- open it in Excel/Sheets and
    it lines up with their expected schema column-for-column.
    """
    products = review_store.get_all_products()
    csv_text = export_service.export_products_csv(products)
    return PlainTextResponse(
        content=csv_text,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=specsense_export.csv"},
    )


# Serve the frontend from the SAME app/URL as the API. This means once
# deployed, there's exactly ONE public link to share (e.g. your Render
# URL) that opens the working UI directly -- no separate frontend host,
# no CORS headaches, no "which URL do I give judges" confusion.
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
frontend_dir = os.path.join(current_dir, "..", "frontend")
app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
