"""
api.py
======
FastAPI wrapper around RAGPipeline.

Run:
    uvicorn api:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health   — liveness check, reports index size
    POST /query    — main RAG query, returns answer + source courses
"""

import asyncio
import base64
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from query import RAGPipeline

# ── Startup ──────────────────────────────────────────────

_pipeline: RAGPipeline | None = None
_lock = asyncio.Lock()          # serialise concurrent model inference


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = RAGPipeline()   # loads models + indexes once (~30s on CPU)
    yield


app = FastAPI(title="UC Davis Course Advisor API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
)


# ── Frontend ─────────────────────────────────────────────

@app.get("/", include_in_schema=False)
def root():
    return FileResponse(Path(__file__).parent / "index.html", media_type="text/html")


# ── Schemas ───────────────────────────────────────────────

class Profile(BaseModel):
    major: str = ""
    completed: list[str] = []           # e.g. ["ECS 020", "MAT 021A"]
    level: str = ""                     # "undergraduate" | "graduate" | ""


class QueryRequest(BaseModel):
    query: str
    profile: Profile | None = None


class CourseInfo(BaseModel):
    course_code: str
    title: str
    units: str
    level: str
    subject_name: str
    prerequisites: str
    rerank_score: float
    dag_tier: int | None = None
    dag_missing: list[str] = []
    dag_path: list[str] = []


class QueryResponse(BaseModel):
    answer: str
    sources: list[CourseInfo]
    elapsed_seconds: float


# ── Endpoints ─────────────────────────────────────────────

@app.get("/health")
def health():
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")
    return {
        "status": "ok",
        "courses_indexed": _pipeline.collection.count(),
        "dag_loaded": _pipeline.dag is not None,
    }


@app.post("/query", response_model=QueryResponse)
async def query(req: QueryRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    profile = None
    if req.profile:
        profile = {
            "major": req.profile.major,
            "completed": set(req.profile.completed),
            "level": req.profile.level,
        }

    t0 = time.time()
    async with _lock:
        answer, courses = await asyncio.to_thread(
            _pipeline.query, req.query, profile=profile, return_sources=True
        )

    sources = [
        CourseInfo(
            course_code=h["meta"]["course_code"],
            title=h["meta"]["title"],
            units=h["meta"]["units"],
            level=h["meta"].get("level", ""),
            subject_name=h["meta"]["subject_name"],
            prerequisites=h["meta"].get("prerequisites", ""),
            rerank_score=round(h.get("rerank_score", 0.0), 4),
            dag_tier=h["dag_info"]["tier"] if "dag_info" in h else None,
            dag_missing=h["dag_info"].get("missing", []) if "dag_info" in h else [],
            dag_path=h["dag_info"].get("path", []) if "dag_info" in h else [],
        )
        for h in courses
    ]

    return QueryResponse(
        answer=answer,
        sources=sources,
        elapsed_seconds=round(time.time() - t0, 2),
    )


@app.post("/extract-courses")
async def extract_courses(file: UploadFile = File(...)):
    """Accept a transcript PDF or image, return extracted course codes via Claude vision."""
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    suffix = Path(file.filename or "").suffix.lower()
    if suffix == ".pdf":
        content_block = {
            "type": "document",
            "source": {"type": "base64", "media_type": "application/pdf",
                       "data": base64.standard_b64encode(await file.read()).decode()},
        }
    elif suffix in (".png", ".jpg", ".jpeg", ".webp"):
        media_type = "image/png" if suffix == ".png" else "image/jpeg"
        content_block = {
            "type": "image",
            "source": {"type": "base64", "media_type": media_type,
                       "data": base64.standard_b64encode(await file.read()).decode()},
        }
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type. Accepted formats: PDF, PNG, JPG, WEBP")

    # No _lock here — Claude Vision doesn't use the ML models (embedder/reranker)
    response = await asyncio.to_thread(
        _pipeline.claude.messages.create,
        model="claude-sonnet-4-6",
        max_tokens=512,
        messages=[{
            "role": "user",
            "content": [
                content_block,
                {"type": "text", "text": (
                    "Extract all completed course codes from this transcript or academic record. "
                    "Return only a comma-separated list of course codes, e.g.: ECS 20, MAT 021A, STA 013. "
                    "No other text."
                )},
            ],
        }],
        )

    courses = [c.strip() for c in response.content[0].text.split(",") if c.strip()]
    return {"courses": courses}
