"""
build_textbook_db.py
====================
为 ECS 课程生成教材数据库。
使用 Claude API 根据课程信息（标题、描述、层次）生成推荐教材及相关章节和主题词。

运行方式：
    python build_textbook_db.py              # 处理所有 ECS 课程（跳过已有记录）
    python build_textbook_db.py --limit 5   # 只处理前 5 门课（调试用）
    python build_textbook_db.py --rebuild   # 忽略已有记录，全部重新生成

输出：textbook_db.json
"""

import csv
import json
import os
import time
import argparse
from pathlib import Path
from dotenv import load_dotenv
import anthropic

load_dotenv()

# ─────────────────────── CONFIG ───────────────────────
DATA_PATH    = "courses.csv"
OUTPUT_PATH  = "textbook_db.json"
SUBJECT      = "ECS"
CLAUDE_MODEL = "claude-sonnet-4-6"
DELAY_SEC    = 0.5   # API 调用之间的等待时间（秒）
# ──────────────────────────────────────────────────────

PROMPT_TEMPLATE = """\
You are an expert computer science educator with deep knowledge of standard university textbooks.

Given the following UC Davis {level} course, suggest the most commonly used textbooks. \
For each textbook, list only the chapters that would be covered in this specific course.

Course Information:
- Code: {course_code}
- Title: {title}
- Level: {level}
- Units: {units}
- Description: {description}
- Prerequisites: {prerequisites}

Respond with a JSON object in exactly this format (no other text):
{{
  "textbooks": [
    {{
      "title": "Full Book Title",
      "authors": ["Author Last, First", "..."],
      "edition": "8th" or null,
      "relevance": "primary" or "supplementary",
      "chapters": [
        {{
          "number": "1",
          "title": "Chapter Title",
          "topics": ["topic 1", "topic 2", "topic 3"]
        }}
      ]
    }}
  ],
  "notes": "e.g. 'Course relies primarily on lecture notes' or null"
}}

Guidelines:
- Suggest 1–3 textbooks. Only suggest textbooks that genuinely exist and are widely used.
- List only chapters relevant to this specific course (not the entire book).
- For graduate courses, prefer research monographs or classic graduate-level texts.
- If no standard textbook exists (e.g., seminar courses), set "textbooks" to [] and explain in "notes".
- Each chapter should have 3–6 concise topic keywords useful for search."""


def load_db(path: str) -> dict:
    if Path(path).exists():
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_db(db: dict, path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(db, f, ensure_ascii=False, indent=2)


def clean_description(raw: str) -> str:
    """Strip the 'Course Description:' prefix that the scraper adds."""
    stripped = raw.strip()
    if stripped.startswith("Course Description:"):
        stripped = stripped[len("Course Description:"):].strip()
    return stripped


def generate_textbooks(client: anthropic.Anthropic, course: dict) -> dict | None:
    prompt = PROMPT_TEMPLATE.format(
        course_code  = course["course_code"],
        title        = course["title"],
        level        = course["level"],
        units        = course["units"],
        description  = clean_description(course.get("description", "")),
        prerequisites= course.get("prerequisites", "") or "None",
    )

    try:
        response = client.messages.create(
            model      = CLAUDE_MODEL,
            max_tokens = 2048,
            messages   = [{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text.strip()

        # Strip markdown code fences if present
        if raw.startswith("```"):
            raw = raw.split("```", 2)[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.rsplit("```", 1)[0].strip()

        return json.loads(raw)

    except json.JSONDecodeError as e:
        print(f"    [WARN] JSON parse error for {course['course_code']}: {e}")
        return None
    except anthropic.APIError as e:
        print(f"    [ERROR] API error for {course['course_code']}: {e}")
        return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",   type=int, default=None, help="处理前 N 门课（调试用）")
    parser.add_argument("--rebuild", action="store_true",    help="忽略缓存，重新生成所有课程")
    args = parser.parse_args()

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    db = {} if args.rebuild else load_db(OUTPUT_PATH)

    # Load ECS courses from CSV
    courses = []
    with open(DATA_PATH, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["subject_code"] == SUBJECT:
                courses.append(row)

    if args.limit:
        courses = courses[: args.limit]

    to_process = [c for c in courses if c["course_code"] not in db]
    print(f"{SUBJECT} courses total : {len(courses)}")
    print(f"Already in DB      : {len(courses) - len(to_process)}")
    print(f"To generate        : {len(to_process)}")
    print()

    for i, course in enumerate(to_process, 1):
        code = course["course_code"]
        print(f"[{i}/{len(to_process)}] {code} — {course['title']} ({course['level']})")

        result = generate_textbooks(client, course)
        if result is None:
            print("    Skipped (generation failed)")
            time.sleep(DELAY_SEC)
            continue

        n_books    = len(result.get("textbooks", []))
        n_chapters = sum(len(b.get("chapters", [])) for b in result.get("textbooks", []))
        print(f"    {n_books} textbook(s), {n_chapters} chapter(s)")

        db[code] = {
            "course_code": code,
            "title"      : course["title"],
            "level"      : course["level"],
            "units"      : course["units"],
            **result,
            "generated_at": time.strftime("%Y-%m-%d"),
        }

        save_db(db, OUTPUT_PATH)   # incremental save
        time.sleep(DELAY_SEC)

    print(f"\nDone. {len(db)} courses in {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
