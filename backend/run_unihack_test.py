import os
import csv
import asyncio
import argparse
import time
from dotenv import load_dotenv

# Load env variables before importing services
load_dotenv()

from models import ProductInput, StructuredProduct
from services.discover import discover_sources
from services.extract import extract_text
from services.structure import structure_product
from services.rag import rag_store
from services import vocabulary
from services.export import export_products_csv, HEADER

async def run_pipeline_for_product(part_number: str, brand: str, short_desc: str) -> StructuredProduct:
    product = ProductInput(
        part_number=part_number,
        brand=brand,
        short_description=short_desc
    )
    print(f"\nProcessing: PN={part_number}, Brand={brand}...")
    
    # 1. Discover
    sources = await discover_sources(product)
    print(f"  - Discovered {len(sources)} sources")
    for s in sources:
        print(f"    * {s.origin}: {s.url}")
        
    # 2. Extract
    extracted_sources = []
    for s in sources:
        ext = await extract_text(s)
        if ext.raw_text:
            print(f"    * Extracted {len(ext.raw_text)} chars from {ext.url}")
            extracted_sources.append(ext)
        else:
            print(f"    * Extraction failed/empty for {ext.url}")
            
    # 3. Structure & Score & Vocabulary Validation
    result = await structure_product(product, extracted_sources)
    print(f"  - Category: {result.category.value} (Confidence: {result.category.confidence})")
    print(f"  - Brand matched approved list: {result.brand_vocab_validated} ({result.brand})")
    print(f"  - Attributes Extracted: {len(result.attributes)}")
    for attr in result.attributes[:5]:
        print(f"    * {attr.label} = {attr.value} {attr.uom or ''} (vocab-validated: {attr.vocab_validated})")
        
    return result

def clean_brand(e1_brand: str, part_manuf: str) -> str:
    if e1_brand and e1_brand.strip() and e1_brand.strip() != "-- Unbranded --":
        return e1_brand.strip()
    if part_manuf and part_manuf.strip():
        # Remove trailing parenthetical identifiers like " (2435)"
        import re
        brand = part_manuf.strip()
        brand = re.sub(r'\s*\([^)]*\)\s*$', '', brand)
        return brand
    return "Unknown"

async def main():
    parser = argparse.ArgumentParser(description="SpecSense UniHack test runner")
    parser.add_argument("--parts", type=str, help="Comma-separated part numbers to process (runs only these)")
    parser.add_argument("--limit", type=int, default=5, help="Limit total number of products to process if no specific parts given")
    args = parser.parse_args()
    
    # Ingest directory (RAG)
    rag_store.ingest_directory()
    # Load controlled vocabulary reference datasets
    vocabulary.load_all()
    
    input_file = os.path.join(os.path.dirname(__file__), "datasets", "unihack_sample_input.csv")
    output_file = os.path.join(os.path.dirname(__file__), "datasets", "unihack_output.csv")
    
    if not os.path.exists(input_file):
        print(f"Error: Input file {input_file} not found!")
        return
        
    target_parts = []
    if args.parts:
        target_parts = [p.strip() for p in args.parts.split(",")]
        
    products_to_process = []
    with open(input_file, newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pn = row.get("Mfg_Part_Num", "").strip()
            desc = row.get("Part_Desc", "").strip()
            brand = clean_brand(row.get("E1_Brand", ""), row.get("Part_Manuf", ""))
            
            if target_parts:
                if pn in target_parts:
                    products_to_process.append((pn, brand, desc))
            else:
                products_to_process.append((pn, brand, desc))
                if len(products_to_process) >= args.limit:
                    break
                    
    print(f"Starting pipeline run for {len(products_to_process)} products...")
    start_time = time.time()
    
    results = []
    for pn, brand, desc in products_to_process:
        try:
            res = await run_pipeline_for_product(pn, brand, desc)
            results.append(res)
        except Exception as e:
            print(f"Error processing {pn}: {e}")
            
    elapsed = time.time() - start_time
    print(f"\nPipeline finished in {elapsed:.2f} seconds.")
    
    if results:
        # Export
        csv_text = export_products_csv(results)
        with open(output_file, "w", newline="", encoding="utf-8") as out:
            out.write(csv_text)
            
        print(f"Successfully generated {output_file}")
        # Verify columns
        with open(output_file, newline="", encoding="utf-8") as check:
            reader = csv.reader(check)
            headers = next(reader)
            print(f"Verification: Output contains {len(headers)} columns (Expected: 252)")
            assert len(headers) == 252, "Column count mismatch!"
            
            row_count = sum(1 for _ in reader)
            print(f"Verification: Output contains {row_count} data rows.")
    else:
        print("No results to export.")

if __name__ == "__main__":
    asyncio.run(main())
