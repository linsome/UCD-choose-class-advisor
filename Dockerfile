FROM python:3.13-slim

# build-essential is needed by some chromadb/hnswlib native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps as a separate layer so they are cached on code-only rebuilds
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# HuggingFace models are downloaded at first startup into a mounted Fly volume
# (/app/.hf_cache). Baking them into the image exceeded Fly's 8 GB uncompressed limit.
ENV HF_HOME=/app/.hf_cache

# Runtime source files only (scrape-catalog, build_index, eval not needed)
COPY api.py query.py build_dag.py index.html ./

# Pre-built retrieval artifacts
COPY chroma_db/    ./chroma_db/
COPY bm25_index.pkl course_dag.pkl ./

EXPOSE 8000

# Single worker: each worker would load both models (~2 GB each),
# and the asyncio.Lock in api.py only serialises within one process.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
