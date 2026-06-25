"""
build_index.py
==============
一次性运行：读取 courses_raw.json → 生成 embedding → 存入 Chroma

运行方式：
    python build_index.py

完成后会在当前目录生成 chroma_db/ 文件夹（持久化的向量数据库）
"""

import json
import pickle
import time
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

# ─────────────────────── CONFIG ───────────────────────
DATA_PATH   = "courses_raw.json"
CHROMA_DIR  = "chroma_db"
COLLECTION  = "ucdavis_courses"
BM25_PATH   = "bm25_index.pkl"

# 推荐模型：多语言支持，中文提问也能匹配英文课程描述
# 如果网络不好，可改为 "paraphrase-multilingual-MiniLM-L12-v2"（更小更快）
EMBED_MODEL = "BAAI/bge-m3"
# ──────────────────────────────────────────────────────


def build_document(course: dict) -> str:
    """
    把一门课的各字段拼成一段自然语言文本，作为 embedding 的输入

    为什么不直接用 description？
    因为用户可能用课程编号、课程名称、学分来搜索，
    把所有字段拼在一起能让 embedding 捕捉到更多匹配角度
    """
    parts = [
        f"Course: {course.get('course_code', '')}",
        f"Title: {course.get('title', '')}",
        f"Subject: {course.get('subject_name', '')}",
        f"College: {course.get('college', '')}",
        f"Level: {course.get('level', '')}",
        f"Units: {course.get('units', '')}",
        f"Description: {course.get('description', '')}",
    ]
    if course.get("prerequisites"):
        parts.append(f"Prerequisites: {course['prerequisites']}")
    return "\n".join(p for p in parts if p.split(": ", 1)[1])  # 跳过空字段


def main():
    # 1. 读数据
    print(f"Loading courses from {DATA_PATH}...")
    with open(DATA_PATH, encoding="utf-8") as f:
        courses = json.load(f)
    print(f"  {len(courses)} courses loaded")

    # 过滤掉没有描述的课（噪音）
    courses = [c for c in courses if c.get("description", "").strip()]
    print(f"  {len(courses)} courses after filtering empty descriptions")

    # 2. 准备文档、ID、metadata
    documents = [build_document(c) for c in courses]

    # Chroma 要求 ID 是字符串，用 course_code 做 ID
    # 同一门课可能因为爬虫重复出现，用 enumerate 加序号避免冲突
    ids = [f"{c.get('course_code', 'UNKNOWN')}_{i}" for i, c in enumerate(courses)]

    # metadata 存结构化字段，检索后直接用，不需要重新解析文本
    metadatas = [
        {
            "course_code":   c.get("course_code", ""),
            "title":         c.get("title", ""),
            "subject_code":  c.get("subject_code", ""),
            "subject_name":  c.get("subject_name", ""),
            "college":       c.get("college", ""),
            "level":         c.get("level", ""),
            "units":         c.get("units", ""),
            "prerequisites": c.get("prerequisites", ""),
            "source_url":    c.get("source_url", ""),
        }
        for c in courses
    ]

    # 3. 加载 embedding 模型
    print(f"\nLoading embedding model: {EMBED_MODEL}")
    print("  (首次运行会下载模型，大约 500MB-1GB，需要几分钟)")
    model = SentenceTransformer(EMBED_MODEL)

    # 4. 生成 embedding（分批处理，避免内存溢出）
    print("\nGenerating embeddings...")
    BATCH_SIZE = 64
    all_embeddings = []
    for i in range(0, len(documents), BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        embeddings = model.encode(
            batch,
            show_progress_bar=False,
            normalize_embeddings=True,  # 归一化后余弦相似度 = 点积，更快
        )
        all_embeddings.extend(embeddings.tolist())
        print(f"  [{i + len(batch)}/{len(documents)}] done")

    # 5. 存入 Chroma
    print(f"\nSaving to Chroma at {CHROMA_DIR}/...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # 如果已存在同名 collection，先删除（重新构建索引时用）
    existing = [c.name for c in client.list_collections()]
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
        print(f"  Deleted existing collection '{COLLECTION}'")

    collection = client.create_collection(
        name=COLLECTION,
        # 用余弦相似度（配合 normalize_embeddings=True）
        metadata={"hnsw:space": "cosine"},
    )

    # Chroma 的 add() 也有批量限制，分批写入
    CHROMA_BATCH = 500
    for i in range(0, len(ids), CHROMA_BATCH):
        collection.add(
            ids=ids[i : i + CHROMA_BATCH],
            embeddings=all_embeddings[i : i + CHROMA_BATCH],
            documents=documents[i : i + CHROMA_BATCH],
            metadatas=metadatas[i : i + CHROMA_BATCH],
        )
    
    print(f"\nIndex built successfully!")
    print(f"   Collection '{COLLECTION}' contains {collection.count()} documents")
    print(f"   Saved to: {Path(CHROMA_DIR).resolve()}")

    # 6. 构建 BM25 索引（用于关键词检索，与向量检索互补）
    print(f"\nBuilding BM25 index...")
    tokenized = [doc.lower().split() for doc in documents]
    bm25 = BM25Okapi(tokenized)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({
            "bm25":      bm25,
            "ids":       ids,
            "documents": documents,
            "metadatas": metadatas,
        }, f)
    print(f"   BM25 index saved to {BM25_PATH}")


if __name__ == "__main__":
    start = time.time()
    main()
    print(f"\nTotal time: {time.time() - start:.1f}s")
