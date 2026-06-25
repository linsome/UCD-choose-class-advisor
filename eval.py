"""
eval.py
=======
对每个测试问题，走完整 pipeline（BM25 + 向量检索 → 加权 RRF → 去重 → reranker）
打印前 5 条结果，评估真实检索效果。

运行方式：
    python eval.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

from query import RAGPipeline, weighted_rrf, CONTEXT_K, RETRIEVAL_K, RERANK_TOP_N

# ─────────────────────── 测试问题 ───────────────────────
TEST_CASES = [
    "Introduction to aerospace engineering",
    "课程涉及无人机和飞行器历史",
    "child development and family",
    "machine learning or artificial intelligence",
    "What courses cover data analysis?",
]
# ────────────────────────────────────────────────────────


def eval_query(pipeline: RAGPipeline, query: str) -> list[dict]:
    """走完整 pipeline，返回最终 CONTEXT_K 条结果"""
    vector_hits = pipeline._retrieve_vector(query, top_k=RETRIEVAL_K)
    bm25_hits   = pipeline._retrieve_bm25(query, top_k=RETRIEVAL_K)
    merged      = weighted_rrf(vector_hits, bm25_hits)
    unique      = pipeline._deduplicate(merged)
    reranked    = pipeline._rerank(query, unique[:RERANK_TOP_N])
    return reranked[:CONTEXT_K]


def print_results(query: str, hits: list[dict]):
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
            print(f"       Desc    : {desc[:200]}{'...' if len(desc) > 200 else ''}")
        print(f"       Rerank  : {hit['rerank_score']:.4f}")


def main():
    pipeline = RAGPipeline()
    for query in TEST_CASES:
        hits = eval_query(pipeline, query)
        print_results(query, hits)


if __name__ == "__main__":
    main()
