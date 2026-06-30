# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Purpose

A RAG (Retrieval-Augmented Generation) pipeline that helps UC Davis students choose and plan courses. Students ask questions in Chinese or English; the system retrieves semantically relevant courses and generates answers via Claude. A separate DAG module encodes prerequisite logic to support tiered recommendations (available now / need 1–2 prereqs / long-term path).

## Setup

```bash
pip install -r requirements.txt
# Also needed for scraping (not in requirements.txt):
pip install requests beautifulsoup4
```

API key can be set via environment variable or `.env` file (loaded automatically via `python-dotenv`):
```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# or add ANTHROPIC_API_KEY=sk-ant-... to .env
```

## Pipeline: Run Steps in Order

### 1. Scrape the catalog (one-time)
```bash
python scrape-catalog.py
```
Outputs `courses_raw.json` and `courses.csv` in the project root. Set `SUBJECTS_LIMIT = 3` in the CONFIG block at the top to scrape only a few subjects for debugging.

### 2. Build the vector + BM25 index (one-time or when data changes)
```bash
python build_index.py
```
Reads `courses_raw.json` from the project root. Downloads `BAAI/bge-m3` (~500MB–1GB) on first run. Creates `chroma_db/` and `bm25_index.pkl`.

### 3. Build the prerequisite DAG (one-time or when data changes)
```bash
python build_dag.py              # builds DAG, visualizes EAE subject, opens browser
python build_dag.py STA          # visualize a different subject
python build_dag.py STA --recommend "STA 013, MAT 021C" --level undergraduate
python build_dag.py --rebuild    # force rebuild even if course_dag.pkl exists
```
Reads `courses_raw.json`, creates `course_dag.pkl`. The `--recommend` flag prints courses the student can take now (all prereqs satisfied). Outputs an interactive HTML visualization (`dag_<SUBJECT>.html`).

### 4. Query (interactive or single-shot)
```bash
python query.py                        # interactive REPL
python query.py "有哪些数据科学相关的课？"  # single query
```

Interactive mode (when `course_dag.pkl` exists) first collects a user profile — major and completed courses. Completed courses can be entered as comma-separated codes (`ECS 20, MAT 021A`) or as a file path to a transcript PDF/PNG/JPG, which is parsed via Claude's vision API. The profile affects query embedding augmentation, completed-course filtering, and DAG tier annotation.

Single-shot mode skips profile collection and returns flat (un-tiered) results.

### 5. Run the HTTP API server
```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```
`api.py` is a FastAPI wrapper around `RAGPipeline`. It loads all models once at startup (~30 s on CPU) and serialises concurrent inference with an `asyncio.Lock`. Endpoints:
- `GET /` — serves `index.html` (the chat UI frontend).
- `GET /health` — liveness check; returns index size and whether the DAG is loaded.
- `POST /query` — accepts `{"query": "...", "profile": {"major": "", "completed": [], "level": ""}}`, returns `answer`, `sources` (with DAG tier/missing/path fields), and `elapsed_seconds`.
- `POST /extract-courses` — accepts a multipart file upload (PDF/PNG/JPG/WEBP transcript), extracts course codes via Claude vision, and returns `{"courses": [...]}`. Does **not** acquire `_lock` since it bypasses the local ML models entirely.

The `profile` field is optional; omitting it returns flat results (no tier annotation, no completed-course filtering). `RAGPipeline.query` accepts `return_sources=True` to return `(answer, courses)` instead of just `answer` — the API relies on this signature.

### 6. Build the textbook database (optional, ECS courses only)
```bash
python build_textbook_db.py              # process all ECS courses (skips already done)
python build_textbook_db.py --limit 5   # process only first 5 courses (debug)
python build_textbook_db.py --rebuild   # ignore cache, regenerate all
```
Reads `courses.csv`, calls Claude API to generate textbook recommendations per ECS course, and writes `textbook_db.json` (keyed by `course_code`). Saves incrementally after each course so progress survives interruption. Configure `SUBJECT` at the top to target a different department.

### 7. Evaluate retrieval quality
```bash
python eval.py
```
Runs the full pipeline (BM25 + vector → RRF → dedup → reranker) on `TEST_CASES` in `eval.py` and prints ranked results with rerank scores. If `textbook_db.json` exists, also shows the primary textbook and first three chapters for each hit. Add test cases there to cover new query types. Note: `eval.py` runs without a profile, so DAG tier annotation and level boost are not exercised here.

### 8. Docker / Fly.io deployment (production)
```bash
# Local Docker
docker build -t ucd-advisor .
docker run -e ANTHROPIC_API_KEY=sk-ant-... -p 8000:8000 ucd-advisor

# Fly.io (configured in fly.toml — app: ucd-course-advisor, region: sjc)
fly deploy
fly secrets set ANTHROPIC_API_KEY=sk-ant-...
```
The `Dockerfile` bakes `BAAI/bge-m3` and `BAAI/bge-reranker-v2-m3` into the image (~1.1 GB added) so the first request is fast. It copies only runtime files (`api.py`, `query.py`, `build_dag.py`) plus pre-built artifacts (`chroma_db/`, `bm25_index.pkl`, `course_dag.pkl`). Scraping and index-building happen outside the container.

The container runs a single Uvicorn worker intentionally — `api.py` uses an `asyncio.Lock` to serialize concurrent model inference, and that lock only works within one process. The Fly.io config matches this: `soft_limit = 1`. The VM is `performance-2x` with 4 GB RAM, which is the minimum needed to fit both models + ChromaDB in memory.

## Architecture

```
scrape-catalog.py  →  courses_raw.json
                              ↓                          ↓
                       build_index.py             build_dag.py
                    (bge-m3 embeddings          (networkx DiGraph,
                     → chroma_db/ +              prereq logic →
                       bm25_index.pkl)            course_dag.pkl)
                              ↓
query.py (RAGPipeline):
  user query
    → embed (bge-m3)
    → vector retrieve (Chroma, top 20) + BM25 retrieve (top 20)
    → weighted RRF fusion (vector 0.7 / BM25 0.3)
    → deduplicate by course_code
    → rerank (bge-reranker-v2-m3, top 20 → top 5)
    → LEVEL_BOOST multiplier (if profile.level set)
    → filter completed courses (if profile)
    → DAG tier annotation (if profile + DAG loaded)
    → Claude API → answer

api.py wraps RAGPipeline in FastAPI (GET /, GET /health, POST /query, POST /extract-courses). `index.html` is the chat UI served at the root — it calls `/query` and `/extract-courses` directly.
```

`query.py` imports `load_dag` and `compute_distance` from `build_dag` at runtime (inside `RAGPipeline.__init__` and `_annotate_dag`). Changing those function signatures in `build_dag.py` breaks `query.py`.

`eval.py` calls private RAGPipeline methods directly (`_retrieve_vector`, `_retrieve_bm25`, `_deduplicate`, `_rerank`). Renaming or refactoring those methods in `query.py` also breaks `eval.py`.

`build_index.py:build_document` structures course text with labeled fields (`"Description: ..."`, `"Prerequisites: ..."`). `query.py:_format_single_course` parses it back with `doc.split("Description:", 1)`. Changing the document format in either place requires updating the other.

**Debugging retrieval:** Pass `verbose=True` to `pipeline.query()` (or run interactively — interactive mode sets `verbose=True` automatically) to print the ranked candidates with rerank scores and tier labels, plus the full context string sent to Claude.

**Key config values** (each file has a `CONFIG` block at the top):

| File | Key config |
|------|-----------|
| `scrape-catalog.py` | `SUBJECTS_LIMIT`, `DELAY_SEC` |
| `build_index.py` | `DATA_PATH`, `EMBED_MODEL`, `CHROMA_DIR`, `BM25_PATH` |
| `build_dag.py` | `DATA_PATH`, `DAG_PATH` |
| `build_textbook_db.py` | `DATA_PATH` (courses.csv), `OUTPUT_PATH`, `SUBJECT`, `CLAUDE_MODEL`, `DELAY_SEC` |
| `query.py` | `RETRIEVAL_K`, `RERANK_TOP_N`, `CONTEXT_K`, `VECTOR_WEIGHT`, `BM25_WEIGHT`, `CLAUDE_MODEL`, `EMBED_MODEL`, `RERANKER_MODEL`, `LEVEL_BOOST` |

`EMBED_MODEL` must be identical in `build_index.py` and `query.py` — a mismatch silently produces wrong retrieval results. Same applies to `COLLECTION` (the Chroma collection name, currently `"ucdavis_courses"`).

`LEVEL_BOOST` in `query.py` is a 2×3 multiplier table applied to rerank scores after the reranker runs. Rows are the student's level (`undergraduate`/`graduate`), columns are the course level. Example: an undergrad querying for courses gets graduate courses boosted down to ×0.6 and undergrad courses up to ×1.3.

## Data Schema

Each record in `courses_raw.json` has these fields (all strings):

| Field | Notes |
|-------|-------|
| `course_code` | e.g. `"ECS 020"` |
| `title` | Full course title |
| `subject_code` | e.g. `"ECS"` |
| `subject_name` | e.g. `"Computer Science"` |
| `college` | May be empty |
| `level` | `"undergraduate"`, `"graduate"`, or `"unknown"` |
| `units` | e.g. `"4"` |
| `description` | Course catalog description; courses without one are filtered by `build_index.py` |
| `prerequisites` | Raw text from catalog; parsed into CNF by `build_dag.py` |
| `source_url` | URL scraped from |

Chroma IDs use `{course_code}_{enumerate_index}` to avoid collisions from duplicate entries in the raw data. Deduplication in the query pipeline uses the `course_code` metadata field, not the Chroma ID.

## DAG Prerequisite Logic

`build_dag.py:parse_prereq_logic` converts raw prerequisite text to CNF (AND of OR groups):
- `;` separates AND groups
- `or` within a group means any one suffices
- Result: `[["STA 013", "STA 032"], ["MAT 021C"]]` = (STA 013 OR STA 032) AND MAT 021C

OR groups get intermediate "junction" nodes in the graph (yellow diamonds in the visualization). The `recommend()` function checks CNF satisfaction against a completed-courses set.

`compute_distance` returns a dict with `tier` (0/1/2), `missing` (list of best-choice missing prereqs), and `path` (topologically-sorted course sequence for tier 2). Tier 1 means ≤2 missing prereqs that are themselves immediately available.

After `_annotate_dag` runs, `query.py` bumps the tier by 1 (capped at 2) for any course whose `prerequisites` field contains the word `"consent"` — even if all formal prereqs are met, instructor-consent courses are never surfaced as immediately available.

Generated artifacts: `course_dag.pkl` (networkx DiGraph), `dag_<SUBJECT>.html` (pyvis visualization).

## Known Issues (from note.md)

- Some queries retrieve internship courses — treated as acceptable behavior
- Duplicate courses from different subject codes (e.g., MGB vs MGP) are partially addressed by deduplication; full resolution planned via DAG integration
- BM25 tokenization is whitespace-only (`lower().split()`), which hurts Chinese queries — translation pre-processing is a future improvement
