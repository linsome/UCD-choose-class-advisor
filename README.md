# UC Davis Course Advisor

A RAG-powered course advisor for UC Davis students. Ask questions in English or Chinese — the system retrieves semantically relevant courses and generates answers via Claude.

**Live demo:** https://ucd-course-advisor.fly.dev

---

## Architecture

```
scrape-catalog.py  →  courses_raw.json
                              ↓                     ↓
                       build_index.py          build_dag.py
                    (bge-m3 embeddings       (prerequisite DAG
                     → chroma_db/ +           → course_dag.pkl)
                       bm25_index.pkl)
                              ↓
                          query.py (RAGPipeline)
                    vector + BM25 → RRF fusion
                    → reranker → level boost
                    → DAG tier annotation
                    → Claude API → answer
                              ↓
                           api.py (FastAPI)
                       GET  /          → index.html
                       GET  /health    → liveness check
                       POST /query     → main RAG query
                       POST /extract-courses → transcript OCR
```

**Retrieval pipeline:** dual-path retrieval (bge-m3 vector + BM25) → weighted RRF fusion (0.7/0.3) → deduplication → bge-reranker-v2-m3 → level boost → DAG tier annotation → Claude generation.

**DAG tiers** (requires user profile):
- `[Available Now]` — all prerequisites met
- `[Coming Soon]` — 1–2 prerequisites missing
- `[Long-term Plan]` — multi-course path needed

Courses requiring *consent of instructor* are automatically bumped one tier higher.

---

## Setup

### Requirements

```bash
pip install -r requirements.txt

# Also needed for scraping (not in requirements.txt):
pip install requests beautifulsoup4
```

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
# or create a .env file with ANTHROPIC_API_KEY=sk-ant-...
```

### Build the data pipeline (one-time)

**1. Scrape the catalog**

```bash
python scrape-catalog.py
```

Outputs `courses_raw.json` and `courses.csv`. Set `SUBJECTS_LIMIT = 3` in the CONFIG block to scrape only a few subjects for testing.

**2. Build the vector + BM25 index**

```bash
python build_index.py
```

Downloads `BAAI/bge-m3` (~500MB–1GB) on first run. Creates `chroma_db/` and `bm25_index.pkl`.

**3. Build the prerequisite DAG**

```bash
python build_dag.py              # builds DAG, visualizes EAE subject
python build_dag.py STA          # visualize a different subject
python build_dag.py --rebuild    # force rebuild
```

Creates `course_dag.pkl`.

---

## Running Locally

```bash
uvicorn api:app --host 0.0.0.0 --port 8000 --reload
```

Then open http://localhost:8000 — the chat UI is served from the same server.

For a single query without the server:

```bash
python query.py "What data science courses are available?"
python query.py   # interactive mode with profile collection
```

---

## Deployment (Fly.io)

The app is deployed on Fly.io as a single `performance-2x` machine (4GB RAM) with a 10GB volume for HuggingFace model cache.

```bash
fly deploy
fly secrets set ANTHROPIC_API_KEY=sk-ant-...
```

Models are downloaded to the Fly volume on first startup (~60 seconds). Subsequent starts are faster (~30 seconds from volume).

The server uses a single Uvicorn worker and serializes concurrent ML inference with an `asyncio.Lock` — concurrent `/query` requests are queued, not parallelized.

---

## Key Config

Each file has a `CONFIG` block at the top. Important values:

| File | Key settings |
|------|-------------|
| `scrape-catalog.py` | `SUBJECTS_LIMIT`, `DELAY_SEC` |
| `build_index.py` | `EMBED_MODEL`, `CHROMA_DIR`, `BM25_PATH` |
| `build_dag.py` | `DAG_PATH` |
| `query.py` | `RETRIEVAL_K`, `RERANK_TOP_N`, `CONTEXT_K`, `VECTOR_WEIGHT`, `BM25_WEIGHT`, `LEVEL_BOOST` |

`EMBED_MODEL` must be identical in `build_index.py` and `query.py`.

---

## Evaluating Retrieval

```bash
python eval.py
```

Runs the full pipeline on test cases in `eval.py` and prints ranked results with rerank scores.
