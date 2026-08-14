# SpecSense
**AI-Powered Product Intelligence for Industrial Commerce**

Turns minimal product input (Part Number, Brand, Short Description) into rich, structured,
commerce-ready product data — with explainable, source-cited, confidence-scored output and
a human-in-the-loop review queue for anything uncertain.

## Architecture

```
Input (Part Number, Brand, Short Description)
        │
        ▼
 ┌─────────────┐   checks local RAG store first (provided datasets)
 │  DISCOVER   │──▶ falls back to live web search (SerpAPI) if nothing found
 └─────────────┘
        │
        ▼
 ┌─────────────┐   pulls text/tables from webpages & PDF datasheets
 │  EXTRACT    │
 └─────────────┘
        │
        ▼
 ┌─────────────┐   LLM extraction run PER SOURCE independently
 │  STRUCTURE  │
 └─────────────┘
        │
        ▼
 ┌─────────────┐   cross-checks sources: 2+ agree = high confidence,
 │  VALIDATE   │   conflict/single-source = flagged for review
 └─────────────┘
        │
        ▼
 ┌─────────────┐   every field carries its source URL
 │  EXPLAIN    │
 └─────────────┘
        │
        ▼
 ┌─────────────┐   flagged fields queue up for a human to approve/correct
 │ HUMAN REVIEW│
 └─────────────┘
        │
        ▼
   Structured, commerce-ready product record
```

## Setup (5 minutes)

1. **Install dependencies**
   ```bash
   cd backend
   python3 -m venv venv
   source venv/bin/activate        # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Add your API keys (all free, no credit card)**
   ```bash
   cp .env.example .env
   # edit .env and fill in:
   #   SERPAPI_KEY   -> free at https://serpapi.com (250 free searches/month, no card)
   #   GEMINI_API_KEY -> free at https://aistudio.google.com (sign in with Google,
   #                     click "Get API key" -> "Create API key")
   #   GROQ_API_KEY  -> free at https://console.groq.com -- this is a BACKUP LLM
   #                     provider. If Gemini's daily quota runs out, the pipeline
   #                     automatically switches to Groq instead of failing. Strongly
   #                     recommended before a live demo -- takes 2 minutes to set up.
   ```

3. **(Optional) Add reference data for RAG**
   Drop any `.txt` / `.csv` / `.pdf` / `.md` / `.json` files into `backend/datasets/`.
   A sample file is already there to demo the RAG path — replace it with whatever
   your hackathon organizers provide.

4. **Run the backend**
   ```bash
   uvicorn main:app --reload --port 8000
   ```
   Visit `http://localhost:8000/health` — you should see `{"status": "ok", ...}`.

5. **Open the frontend**
   Just open `frontend/index.html` directly in a browser (no build step, no server needed).
   It talks to `http://localhost:8000` by default.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| POST | `/api/process` | Run the full pipeline for one product |
| POST | `/api/batch` | Run the pipeline concurrently for many products — proves catalog-scale throughput |
| GET | `/api/review/queue` | List every field currently flagged for human review |
| POST | `/api/review/submit` | Submit a human correction, closes the loop |
| GET | `/api/review/log` | Full history of human corrections |
| GET | `/api/products` | All products processed so far |

## Why it's built this way (for judge Q&A)

- **RAG-first, web-fallback discovery** — provided datasets are checked before live search,
  so it works even offline for known parts, and scales to the long tail via search.
- **Per-source LLM extraction + cross-validation** — the pipeline never trusts a single
  unverified source for a high-confidence field. Confidence is *earned* by source agreement,
  not guessed.
- **Every field cites its source** — nothing is a black box; a judge can click any value
  and see exactly where it came from.
- **Human-in-the-loop is a real workflow**, not just a flag — flagged fields go to a queue,
  get corrected, and the correction is logged.
- **Batch endpoint proves scale** — run 20-50 products concurrently live in the demo, not
  just one at a time.

## Known limitations (Day 1 — be upfront about these if asked)

- RAG retrieval uses TF-IDF keyword similarity, not semantic embeddings — fast and
  dependency-light for a hackathon, upgradeable to a real embedding model later.
- Review store is in-memory — restarting the server clears it. Fine for a demo session;
  swap in SQLite for persistence if you have time.
- Confidence scoring is a transparent heuristic (source agreement count), not a trained
  model — this is a feature for explainability, not a shortcut.
