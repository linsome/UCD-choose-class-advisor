"""
query.py
========
RAG pipeline:
  1. 双路检索：向量检索（Chroma + bge-m3）+ 关键词检索（BM25）
  2. 加权 RRF 融合（向量 0.7 / BM25 0.3）
  3. 按 course_code 去重
  4. Reranker 精排（bge-reranker-v2-m3，多语言）
  5. 取前 CONTEXT_K 条送给 Claude

运行方式：
    python query.py                        # 交互模式
    python query.py "有哪些数据科学相关的课？"  # 单次查询

需要设置环境变量：
    export ANTHROPIC_API_KEY="sk-ant-..."
"""

import os
import sys
sys.stdout.reconfigure(encoding="utf-8")
import pickle
import time
from dotenv import load_dotenv
load_dotenv()

import anthropic
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

# ─────────────────────── CONFIG ───────────────────────
CHROMA_DIR     = "chroma_db"
COLLECTION     = "ucdavis_courses"
BM25_PATH      = "bm25_index.pkl"
DAG_PATH       = "course_dag.pkl"
EMBED_MODEL    = "BAAI/bge-m3"          # 必须和 build_index.py 里一致
RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"  # 多语言 reranker，支持中文提问

RETRIEVAL_K    = 20   # 每路检索召回候选数
RERANK_TOP_N   = 20   # 去重后送入 reranker 的最大数量
CONTEXT_K      = 5    # 最终送给 Claude 的条数

VECTOR_WEIGHT  = 0.7  # 加权 RRF：向量检索权重
BM25_WEIGHT    = 0.3  # 加权 RRF：BM25 权重
RRF_K          = 60   # RRF 平滑系数（标准值）

CLAUDE_MODEL   = "claude-sonnet-4-6"

# rerank_score 乘数：行=学生层次，列=课程层次
# 本科生：本科课 ×1.3，研究生课 ×0.6；研究生：研究生课 ×1.2，本科课 ×0.85
LEVEL_BOOST: dict[str, dict[str, float]] = {
    "undergraduate": {"undergraduate": 1.3, "graduate": 0.6,  "unknown": 1.0},
    "graduate":      {"undergraduate": 0.85, "graduate": 1.2, "unknown": 1.0},
}
# ──────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a helpful course advisor for UC Davis students.
You help students explore courses and find ones that match their interests, goals, or graduation requirements.

When answering:
- Base your answer ONLY on the course information provided below
- If the provided courses don't fully answer the question, say so honestly
- When recommending courses, briefly explain WHY each course is relevant
- If the student asks in Chinese, answer in Chinese; otherwise answer in English
- Keep answers concise but informative
- When listing multiple courses, order them by difficulty: foundational first (course numbers below 100), then advanced undergraduate (100–199), then graduate level (200+). Exception: if the student is a graduate student, list graduate-level courses (200+) first

When course information includes tier labels, preserve the tiered structure in your answer:
- [Available Now] Student meets all prerequisites — recommend these directly
- [Coming Soon] Student is 1–2 prerequisites away — note the specific missing prerequisite(s)
- [Long-term Plan] Student needs a multi-course path — present as future planning, show the suggested path

If you cannot find relevant courses in the provided context, say:
"I couldn't find courses matching your query in the current catalog. Try rephrasing or asking about a different topic."
"""


def weighted_rrf(
    vector_hits: list[dict],
    bm25_hits: list[dict],
    vector_weight: float = VECTOR_WEIGHT,
    bm25_weight: float = BM25_WEIGHT,
    k: int = RRF_K,
) -> list[dict]:
    """加权 RRF：两路结果按排名融合，向量检索享有更高权重"""
    all_hits: dict[str, dict] = {}
    scores: dict[str, float] = {}

    for rank, hit in enumerate(vector_hits, 1):
        doc_id = hit["id"]
        all_hits[doc_id] = hit
        scores[doc_id] = scores.get(doc_id, 0.0) + vector_weight / (k + rank)

    for rank, hit in enumerate(bm25_hits, 1):
        doc_id = hit["id"]
        all_hits.setdefault(doc_id, hit)
        scores[doc_id] = scores.get(doc_id, 0.0) + bm25_weight / (k + rank)

    sorted_ids = sorted(scores, key=lambda x: scores[x], reverse=True)
    return [all_hits[i] for i in sorted_ids]


def _format_single_course(i: int, hit: dict) -> str:
    meta = hit["meta"]
    doc  = hit["doc"]
    desc = doc.split("Description:", 1)[-1].strip()[:400]
    parts = [
        f"[Course {i}]",
        f"Code: {meta['course_code']} | Units: {meta['units']} | Level: {meta.get('level', '')}",
        f"Title: {meta['title']}",
        f"Subject: {meta['subject_name']}" + (f" | College: {meta['college']}" if meta.get("college") else ""),
        f"Description: {desc}",
    ]
    if meta.get("prerequisites"):
        parts.append(f"Prerequisites: {meta['prerequisites']}")
    return "\n".join(parts)


def format_courses_for_prompt(candidates: list[dict], profile: dict | None = None) -> str:
    # No profile or no DAG info on hits → flat format (backward compatible)
    has_dag = profile is not None and any("dag_info" in h for h in candidates)
    if not has_dag:
        lines = [_format_single_course(i, hit) for i, hit in enumerate(candidates, 1)]
        return "\n\n---\n\n".join(lines)

    # Group by tier
    tiers: dict[int, list[dict]] = {0: [], 1: [], 2: []}
    for hit in candidates:
        tier = hit.get("dag_info", {}).get("tier", 0)
        tiers[tier].append(hit)

    sections = []
    counter = 1

    if tiers[0]:
        block = ["[Available Now — Prerequisites all met]"]
        for hit in tiers[0]:
            block.append(_format_single_course(counter, hit))
            counter += 1
        sections.append("\n\n".join(block))

    if tiers[1]:
        block = ["[Coming Soon — Missing 1–2 prerequisites]"]
        for hit in tiers[1]:
            block.append(_format_single_course(counter, hit))
            missing = hit["dag_info"].get("missing", [])
            if missing:
                block.append(f"Missing prerequisites: {', '.join(missing)}")
            counter += 1
        sections.append("\n\n".join(block))

    if tiers[2]:
        block = ["[Long-term Plan — Multiple prerequisites needed]"]
        for hit in tiers[2]:
            block.append(_format_single_course(counter, hit))
            path = hit["dag_info"].get("path", [])
            if path:
                block.append(f"Suggested path: {' → '.join(path)}")
            counter += 1
        sections.append("\n\n".join(block))

    return "\n\n===\n\n".join(sections)


class RAGPipeline:
    def __init__(self):
        print("Loading embedding model...")
        self.embedder = SentenceTransformer(EMBED_MODEL)

        print("Loading reranker...")
        self.reranker = CrossEncoder(RERANKER_MODEL)

        print("Connecting to Chroma...")
        client = chromadb.PersistentClient(path=CHROMA_DIR)
        self.collection = client.get_collection(COLLECTION)
        print(f"  {self.collection.count()} courses in index")

        print("Loading BM25 index...")
        with open(BM25_PATH, "rb") as f:
            bm25_data = pickle.load(f)
        self.bm25: BM25Okapi       = bm25_data["bm25"]
        self.bm25_ids: list[str]   = bm25_data["ids"]
        self.bm25_docs: list[str]  = bm25_data["documents"]
        self.bm25_metas: list[dict] = bm25_data["metadatas"]

        print("Loading DAG...")
        self.dag = None
        if os.path.exists(DAG_PATH):
            from build_dag import load_dag
            self.dag = load_dag(DAG_PATH)
            print(f"  DAG loaded ({self.dag.number_of_nodes()} nodes)")
        else:
            print("  course_dag.pkl not found — DAG annotation disabled")
        print()

        self.claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    # ── 用户画像 ──────────────────────────────────────────────────────

    def _extract_courses_from_file(self, file_path: str) -> set[str]:
        """用 Claude vision 从成绩单 PDF 或截图中提取已修课程编号"""
        import base64
        from pathlib import Path

        path = Path(file_path)
        if not path.exists():
            print(f"  文件不存在：{file_path}")
            return set()

        suffix = path.suffix.lower()
        with open(path, "rb") as f:
            data = base64.standard_b64encode(f.read()).decode("utf-8")

        if suffix == ".pdf":
            source = {"type": "base64", "media_type": "application/pdf", "data": data}
            file_content = {"type": "document", "source": source}
        elif suffix in (".png", ".jpg", ".jpeg", ".webp"):
            media_type = "image/png" if suffix == ".png" else "image/jpeg"
            source = {"type": "base64", "media_type": media_type, "data": data}
            file_content = {"type": "image", "source": source}
        else:
            print(f"  不支持的文件格式：{suffix}（支持 PDF / PNG / JPG）")
            return set()

        try:
            response = self.claude.messages.create(
                model=CLAUDE_MODEL,
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": [
                        file_content,
                        {
                            "type": "text",
                            "text": (
                                "Extract all completed course codes from this transcript or academic record. "
                                "Return only a comma-separated list of course codes, e.g.: ECS 20, MAT 021A, STA 013. "
                                "No other text."
                            ),
                        },
                    ],
                }],
            )
        except Exception as e:
            print(f"  API 调用失败：{e}")
            return set()

        claude_response = response.content[0].text.strip()
        print(f"\n  Claude 识别结果：\n  {claude_response}\n")
        courses = {c.strip() for c in claude_response.split(",") if c.strip()}
        return courses

    def _collect_user_profile(self) -> dict:
        print("在开始之前，请告诉我一些信息：")
        major = input("1. 你的专业方向是？(如 Computer Science, Statistics): ").strip()

        print("2. 你已完成的课程？")
        print("   · 直接输入课程代码（逗号分隔，如 ECS 20, MAT 021A）")
        print("   · 或输入成绩单文件路径（支持 PDF / PNG / JPG）")
        raw = input("   > ").strip().strip('"').strip("'")

        completed: set[str] = set()
        if raw.lower().endswith((".pdf", ".png", ".jpg", ".jpeg", ".webp")):
            print("  正在识别文件...")
            completed = self._extract_courses_from_file(raw)
            if completed:
                print(f"  共识别到 {len(completed)} 门课程：")
                for i, c in enumerate(sorted(completed), 1):
                    print(f"    {i}. {c}")
                confirm = input("\n  识别结果是否正确？直接回车确认，或重新输入课程列表：").strip()
                if confirm:
                    completed = {c.strip() for c in confirm.split(",") if c.strip()}
            else:
                print("  未识别到课程，请手动输入：")
                fallback = input("  > ").strip()
                completed = {c.strip() for c in fallback.split(",") if c.strip()}
        elif raw:
            completed = {c.strip() for c in raw.split(",") if c.strip()}

        print("3. 你的学籍状态？")
        print("   1. undergraduate（本科生）")
        print("   2. graduate（研究生）")
        raw_level = input("   > ").strip().lower()
        if raw_level in ("2", "graduate", "研究生"):
            level = "graduate"
        elif raw_level in ("1", "undergraduate", "本科生"):
            level = "undergraduate"
        else:
            level = ""

        print()
        return {"major": major, "completed": completed, "level": level}

    # ── 检索 ─────────────────────────────────────────────────────────

    def _retrieve_vector(self, query: str, top_k: int) -> list[dict]:
        embedding = self.embedder.encode(query, normalize_embeddings=True).tolist()
        res = self.collection.query(
            query_embeddings=[embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"id": id_, "doc": doc, "meta": meta, "distance": dist}
            for id_, doc, meta, dist in zip(
                res["ids"][0], res["documents"][0],
                res["metadatas"][0], res["distances"][0],
            )
        ]

    def _retrieve_bm25(self, query: str, top_k: int) -> list[dict]:
        tokens = query.lower().split()
        scores = self.bm25.get_scores(tokens)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [
            {"id": self.bm25_ids[i], "doc": self.bm25_docs[i], "meta": self.bm25_metas[i]}
            for i in top_indices
        ]

    # ── DAG 标注 ─────────────────────────────────────────────────────

    def _annotate_dag(self, hits: list[dict], completed: set[str]) -> list[dict]:
        from build_dag import compute_distance
        for hit in hits:
            code = hit["meta"].get("course_code", "")
            info = compute_distance(self.dag, code, completed)
            # Courses requiring instructor consent are not freely available even if prereqs are met
            if "consent" in hit["meta"].get("prerequisites", "").lower() and info["tier"] < 2:
                info["tier"] += 1
            hit["dag_info"] = info
        return hits

    # ── 层次权重调整 ──────────────────────────────────────────────────

    def _apply_level_boost(self, hits: list[dict], student_level: str) -> list[dict]:
        """按学生层次调整 rerank_score，然后重排。"""
        multipliers = LEVEL_BOOST.get(student_level, {})
        if not multipliers:
            return hits
        for hit in hits:
            course_level = hit["meta"].get("level", "unknown")
            hit["rerank_score"] *= multipliers.get(course_level, 1.0)
        return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)

    # ── 去重 & 精排 ──────────────────────────────────────────────────

    def _deduplicate(self, hits: list[dict]) -> list[dict]:
        """按 course_code 去重，保留 RRF 分数最高的那条"""
        seen: set[str] = set()
        unique = []
        for hit in hits:
            code = hit["meta"].get("course_code", hit["id"])
            if code not in seen:
                seen.add(code)
                unique.append(hit)
        return unique

    def _rerank(self, query: str, hits: list[dict]) -> list[dict]:
        pairs = [(query, hit["doc"]) for hit in hits]
        scores = self.reranker.predict(pairs)
        for hit, score in zip(hits, scores):
            hit["rerank_score"] = float(score)
        return sorted(hits, key=lambda h: h["rerank_score"], reverse=True)

    # ── 生成 ─────────────────────────────────────────────────────────

    def generate(self, query: str, context: str, profile: dict | None = None) -> str:
        profile_info = ""
        if profile and profile.get("major"):
            profile_info = f"Student profile: {profile['major']} major.\n\n"
        user_message = (
            f"{profile_info}"
            f"Here are the relevant UC Davis courses I found:\n\n"
            f"{context}\n\n---\n\nStudent's question: {query}"
        )
        response = self.claude.messages.create(
            model=CLAUDE_MODEL,
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_message}],
        )
        return response.content[0].text

    # ── 完整 pipeline ────────────────────────────────────────────────

    def query(self, user_query: str, profile: dict | None = None, verbose: bool = False, return_sources: bool = False) -> str | tuple:
        # 1. Query 增强：拼接学籍层次 + 专业信息以优化 embedding 召回
        search_query = user_query
        if profile:
            prefix_parts = []
            if profile.get("level"):
                prefix_parts.append(profile["level"])
            if profile.get("major"):
                prefix_parts.append(f"{profile['major']} major")
            if prefix_parts:
                search_query = f"{' '.join(prefix_parts)} student. {user_query}"

        # 2. 双路检索
        vector_hits = self._retrieve_vector(search_query, top_k=RETRIEVAL_K)
        bm25_hits   = self._retrieve_bm25(search_query, top_k=RETRIEVAL_K)

        # 3. 加权 RRF 融合
        merged = weighted_rrf(vector_hits, bm25_hits)

        # 4. 按 course_code 去重
        unique = self._deduplicate(merged)

        # 5. Reranker 精排
        reranked = self._rerank(search_query, unique[:RERANK_TOP_N])

        # 6. 按学籍层次调整 rerank_score 并重排
        if profile and profile.get("level"):
            reranked = self._apply_level_boost(reranked, profile["level"])

        # 7. 过滤掉已修课程
        if profile and profile.get("completed"):
            reranked = [
                hit for hit in reranked
                if hit["meta"].get("course_code", "") not in profile["completed"]
            ]

        # 8. 取前 CONTEXT_K 送给 Claude
        final = reranked[:CONTEXT_K]

        # 9. DAG 标注：给每条结果打距离标签
        if profile and self.dag:
            final = self._annotate_dag(final, profile.get("completed", set()))

        if verbose:
            print("\n── Retrieved & Reranked ──")
            for i, hit in enumerate(final, 1):
                tier_label = ""
                if "dag_info" in hit:
                    tier_label = f"  tier={hit['dag_info']['tier']}"
                print(
                    f"  {i}. {hit['meta']['course_code']} — {hit['meta']['title']}"
                    f"  (rerank={hit['rerank_score']:.3f}){tier_label}"
                )
            print()

        context = format_courses_for_prompt(final, profile=profile)
        if verbose:
            print("\n── Full context sent to Claude ──")
            print(context)
            print("─" * 40 + "\n")
        answer = self.generate(user_query, context, profile=profile)
        if return_sources:
            return answer, final
        return answer


def interactive_mode(pipeline: RAGPipeline):
    profile = pipeline._collect_user_profile() if pipeline.dag else None

    print("UC Davis Course Advisor (type 'quit' to exit)\n")
    if profile:
        completed_display = ", ".join(sorted(profile["completed"])) or "（无）"
        print(f"专业：{profile['major'] or '（未填写）'}  |  已修课程：{completed_display}")
    print("=" * 50)

    while True:
        try:
            query = input("\n你的问题 / Your question: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not query:
            continue
        if query.lower() in ("quit", "exit", "q"):
            print("Bye!")
            break

        print("\nSearching...\n")
        t0 = time.time()
        answer = pipeline.query(query, profile=profile, verbose=True)
        elapsed = time.time() - t0

        print("── Answer ──")
        print(answer)
        print(f"\n(took {elapsed:.1f}s)")


def main():
    pipeline = RAGPipeline()

    if len(sys.argv) > 1:
        query = " ".join(sys.argv[1:])
        print(f"Query: {query}\n")
        answer = pipeline.query(query, verbose=True)
        print("Answer:")
        print(answer)
    else:
        interactive_mode(pipeline)


if __name__ == "__main__":
    main()
