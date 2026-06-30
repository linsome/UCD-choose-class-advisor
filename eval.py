"""
eval.py
=======
对每个测试问题，走完整 pipeline（BM25 + 向量检索 → 加权 RRF → 去重 → reranker）
打印前 5 条结果，评估真实检索效果。结果附带推荐教材和章节（来自 textbook_db.json）。

运行方式：
    python eval.py
"""

import json
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from query import RAGPipeline, weighted_rrf, CONTEXT_K, RETRIEVAL_K, RERANK_TOP_N

# ─────────────────────── 测试问题 ───────────────────────
TEST_CASES = [
    "Introduction to aerospace engineering",
    "课程涉及无人机和飞行器历史",
    "child development and family",
    "machine learning or artificial intelligence",
    "What courses cover data analysis?",
    "我想学 AI 和机器学习，应该选什么课？",
]
# ────────────────────────────────────────────────────────

TEXTBOOK_DB_PATH = "textbook_db.json"


def load_textbook_db(path: str = TEXTBOOK_DB_PATH) -> dict:
    p = Path(path)
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


def eval_query(pipeline: RAGPipeline, query: str) -> list[dict]:
    """走完整 pipeline，返回最终 CONTEXT_K 条结果"""
    vector_hits = pipeline._retrieve_vector(query, top_k=RETRIEVAL_K)
    bm25_hits   = pipeline._retrieve_bm25(query, top_k=RETRIEVAL_K)
    merged      = weighted_rrf(vector_hits, bm25_hits)
    unique      = pipeline._deduplicate(merged)
    reranked    = pipeline._rerank(query, unique[:RERANK_TOP_N])
    return reranked[:CONTEXT_K]


def print_results(query: str, hits: list[dict], textbook_db: dict):
    print("\n" + "=" * 60)
    print(f"Query: {query}")
    print("=" * 60)
    for rank, hit in enumerate(hits, 1):
        meta = hit["meta"]
        doc  = hit["doc"]
        desc = ""
        for line in doc.splitlines():
            if line.startswith("Description:"):
                desc = line[len("Description:"):].strip()
                break
        print(f"\n  [{rank}] {meta['course_code']} — {meta['title']}")
        print(f"       Level   : {meta.get('level', '')}")
        print(f"       Subject : {meta['subject_name']}")
        print(f"       Units   : {meta['units']}")
        if meta.get("prerequisites"):
            print(f"       Prereqs : {meta['prerequisites']}")
        if desc:
            print(f"       Desc    : {desc[:180]}{'...' if len(desc) > 180 else ''}")
        print(f"       Rerank  : {hit['rerank_score']:.4f}")

        # ── 教材信息 ──────────────────────────────────────
        tb = textbook_db.get(meta["course_code"])
        if tb and tb.get("textbooks"):
            primary = next((b for b in tb["textbooks"] if b["relevance"] == "primary"),
                           tb["textbooks"][0])
            edition = f" ({primary['edition']})" if primary.get("edition") else ""
            authors = ", ".join(primary["authors"][:2])
            print(f"       Textbook: {primary['title']}{edition} — {authors}")
            for ch in primary.get("chapters", [])[:3]:
                topics = ", ".join(ch["topics"][:4])
                print(f"         ch{ch['number']:>2}: {ch['title']}  [{topics}]")
            if len(primary.get("chapters", [])) > 3:
                print(f"               ... {len(primary['chapters']) - 3} more chapters")
            if tb.get("notes"):
                print(f"       Note    : {tb['notes'][:120]}")


def main():
    pipeline = RAGPipeline()
    textbook_db = load_textbook_db()
    print(f"Textbook DB loaded: {len(textbook_db)} courses")

    for query in TEST_CASES:
        hits = eval_query(pipeline, query)
        print_results(query, hits, textbook_db)


if __name__ == "__main__":
    main()
