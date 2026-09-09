# SpecSense
**AI-Powered Product Catalog Intelligence for Industrial Commerce | Unilog UniHack Edition**

SpecSense transforms messy, cryptic industrial distributor inputs (`Part Number`, `Brand`, `Short Description`) into complete, standardized, search-ready product records adhering strictly to the **Unilog Rule Book** and **252-Column Schema**.

---

## 🚀 Key Differentiators & Unilog Rule-Book Compliance

1. **Strict Controlled Vocabularies**
   - Integrates `manufacturer_brand.csv`, `uom_standards.csv`, and `fittings_lov.csv`.
   - Brand names match approved casing and registered symbols (`Diablo®`, `3M®`, `Mirka®`).
   - Zero invented attribute values or hallucinated fields.

2. **Decimal-to-Trade-Fraction Engine**
   - Automatically converts imperial decimal dimensions into industry-standard trade fractions (e.g., `0.5 in` -> `1/2 in`, `2.75 in` -> `2-3/4 in`).

3. **Multi-Length Description Builders**
   - **Invoice Description**: Strictly <= 40 characters, ALL CAPS, standard abbreviations (`SND BLT`, `DG BB`, `CPLG`, `BRS`).
   - **Mobile Description**: Strictly 60–80 character window (`[Brand], [Category], [Part Number], [Key Specs]`).
   - **Short Description & Long Description**

4. **Full 252-Column Unilog Ground Truth Export**
   - Maps normalized specs across `ATTRIBUTE_LABEL 1-50`, `VALUE 1-50`, `UOM 1-50`, canonical classpath taxonomy breadcrumbs, and manufacturer reference URLs.

5. **Zero-Quota Resilient Discovery**
   - Hierarchical retrieval: Local RAG -> Persistent SQLite Cache (0ms) -> DuckDuckGo Lite & Bing -> SerpAPI fallback.
   - 100% manufacturer sourcing compliance (consumer marketplaces like Amazon, eBay, and Alibaba are strictly excluded).

6. **Dual LLM Failover**
   - Primary: Google Gemini 2.5 Flash (free tier).
   - Instant automated fallback: Groq LLaMA 3.3 70B for 100% demo uptime.

---

## 🏗️ Architecture

```
Input (Part Number, Brand, Short Description)
        │
        ▼
 ┌─────────────┐   Checks Local RAG Store (Reference CSVs)
 │  DISCOVER   │──▶ Multi-Tier Zero-Quota Web Search (DDG Lite + Bing)
 └─────────────┘   Strictly filters out non-manufacturer consumer domains
        │
        ▼
 ┌─────────────┐   Extracts raw technical specs from datasheets & PDFs
 │  EXTRACT    │   Per-source LLM extraction (Gemini / Groq failover)
 └─────────────┘
        │
        ▼
 ┌─────────────┐   Cross-source consensus validation & conflict scoring
 │  STRUCTURE  │   Builds Invoice (<40 chars) & Mobile (60-80 chars) descriptions
 └─────────────┘
        │
        ▼
 ┌─────────────┐   Validates against Unilog LOV & UOM controlled vocabularies
 │  NORMALIZE  │   Converts imperial decimals to fractions (0.5 in -> 1/2 in)
 └─────────────┘
        │
        ▼
 ┌─────────────┐   Generates exact 252-column ground truth CSV & UI SpecSheet
 │   EXPORT    │   Live Unilog Compliance Scorecard (100% Pass)
 └─────────────┘
```

---

## ⚡ Quick Start (3 Minutes)

### 1. Install Dependencies
```bash
cd backend
python -m venv venv
venv\Scripts\activate          # Linux/macOS: source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure Environment Variables
```bash
cp .env.example .env
# Fill in free API keys in .env:
# GEMINI_API_KEY (from https://aistudio.google.com)
# GROQ_API_KEY   (optional failover from https://console.groq.com)
```

### 3. Run the Backend Dev Server
```bash
uvicorn main:app --reload --port 8000
```
Open `http://localhost:8000` to view the live dashboard and interactive stepper.

### 4. Run the UniHack Batch Benchmark
```bash
python run_unihack_test.py --limit 10
```
This executes the pipeline against `datasets/unihack_sample_input.csv` and outputs `datasets/unihack_output.csv` with all 252 columns populated.

---

## 📊 Deliverables & Submission Files

- **Pitch Presentation**: `[EXT] UniHack-Protoype Template_Populated.pptx` (Complete with 4-panel architecture diagram and working MVP snapshot).
- **Verified Benchmark Output**: `backend/datasets/unihack_output.csv` (252 columns, 10 sample SKUs).
- **Controlled Vocabularies**: `backend/reference_data/` (`manufacturer_brand.csv`, `uom_standards.csv`, `fittings_lov.csv`).
- **Live UI**: `frontend/index.html` (Accessible at `http://localhost:8000`).

---

## 👥 Team SpecSense
- **Srivardhan Kosuru**
- **T Aysha Jannah**
- **Ron Ittyavirah**
