"""
UC Davis Course Catalog Scraper
================================
抓取 https://catalog.ucdavis.edu/courses-subject-code/ 下所有学科的课程信息

输出文件：
  - courses_raw.json   每门课的原始字段
  - courses.csv        方便在 Excel / Sheets 里浏览

使用方法：
  pip install requests beautifulsoup4
  python scrape_ucdavis.py

可选参数（直接修改下面的 CONFIG）：
  SUBJECTS_LIMIT  只抓前 N 个学科，调试时设成 2-3 即可；None = 全量
  DELAY_SEC       每次请求之间的间隔，建议 >=1，避免对服务器造成压力
"""

import json
import csv
import time
import logging
import re
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional

import requests
from bs4 import BeautifulSoup

# ─────────────────────────── CONFIG ───────────────────────────
BASE_URL      = "https://catalog.ucdavis.edu"
INDEX_URL     = f"{BASE_URL}/courses-subject-code/"
DELAY_SEC     = 1.2          # 请求间隔（秒），别改太小
SUBJECTS_LIMIT = None       # 调试时改成 3；正式抓改成 None
OUTPUT_JSON   = "courses_raw.json"
OUTPUT_CSV    = "courses.csv"
# ──────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

session = requests.Session()
session.headers.update(HEADERS)


@dataclass
class Course:
    subject_code: str        # e.g. "EAE"
    subject_name: str        # e.g. "Aerospace Science & Engineering"
    college: str             # e.g. "College of Engineering"
    course_code: str         # e.g. "EAE 001"
    title: str               # e.g. "Introduction to Aerospace Science Engineering"
    units: str               # e.g. "1" or "3-4"
    level: str               # "undergraduate" or "graduate"
    description: str         # 课程描述全文
    prerequisites: str = ""       # 从 description 里提取的先修要求全文
    prerequisite_codes: str = ""  # 先修要求中的课程编号，逗号分隔，如 "ENG 104, ENG 104V"
    source_url: str = ""          # 来源页面 URL，方便溯源


def fetch(url: str) -> Optional[BeautifulSoup]:
    """发起 GET 请求，失败时最多重试 3 次"""
    for attempt in range(1, 4):
        try:
            resp = session.get(url, timeout=15)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.RequestException as e:
            log.warning(f"Attempt {attempt}/3 failed for {url}: {e}")
            if attempt < 3:
                time.sleep(3)
    log.error(f"Giving up on {url}")
    return None


def get_subject_links(soup: BeautifulSoup) -> list[tuple[str, str]]:
    """
    从总览页解析所有学科链接
    结构：div.az_sitemap > ul > li > a[href]
    返回：[(href, display_text), ...]
    """
    sitemap = soup.find("div", class_="az_sitemap")
    if not sitemap:
        log.error("Cannot find div.az_sitemap — page structure may have changed")
        return []

    links = []
    for a in sitemap.find_all("a", href=True):
        href = a["href"]
        # 只保留指向学科代码页面的链接（如 /courses-subject-code/eae/）
        if "/courses-subject-code/" in href and href != "/courses-subject-code/":
            full_url = BASE_URL + href if href.startswith("/") else href
            links.append((full_url, a.get_text(strip=True)))

    log.info(f"Found {len(links)} subject links")
    return links


def clean(text: str) -> str:
    """统一把 \xa0（HTML 非断行空格）替换为普通空格，防止课程编号匹配失败"""
    return text.replace("\xa0", " ").strip()


def parse_units(units_tag) -> str:
    """从 detail-hours_html span 提取学分字符串，例如 '3 Units' -> '3'"""
    if not units_tag:
        return ""
    text = units_tag.get_text(separator=" ", strip=True)
    # 匹配 "1 Unit", "3 Units", "1-4 Units" 等
    m = re.search(r"([\d.]+(?:[-–][\d.]+)?)\s*units?", text, re.IGNORECASE)
    return m.group(1) if m else text


def get_level(course_code: str) -> str:
    """根据课号里的数字判断层级，字母后缀忽略（如 200A → 200）"""
    # 取最后一个空格分隔的部分（即课号本身），再提取开头数字
    number_part = course_code.split()[-1] if " " in course_code else course_code
    m = re.match(r"(\d+)", number_part)
    if m:
        return "graduate" if int(m.group(1)) >= 200 else "undergraduate"
    return "unknown"


def extract_prerequisites(description: str) -> str:
    """从描述文本里提取完整先修要求，正确处理 Prerequisite(s): / Prerequisites: 两种写法"""
    m = re.search(
        r"prerequisite(?:s|\(s\))?\s*:\s*(.+?)(?=\s{2,}|\.\s+[A-Z]|$)",
        description,
        re.IGNORECASE,
    )
    return m.group(1).strip().rstrip(".") if m else ""


def extract_prerequisite_codes(prerequisites: str) -> str:
    """从先修要求文本中提取课程编号列表，如 'EAE 001, ECS 036B'"""
    codes = re.findall(r"\b([A-Z]{2,4}\s+\d{1,3}[A-Z]{0,2})\b", prerequisites)
    # 去重并保持顺序
    seen: set[str] = set()
    unique = [c for c in codes if not (c in seen or seen.add(c))]  # type: ignore[func-returns-value]
    return ", ".join(unique)


def parse_courses(soup: BeautifulSoup, subject_url: str) -> list[Course]:
    """
    解析单个学科页面，提取所有 courseblock

    HTML 结构（根据截图）：
      div.sc_sccoursedescs
        div.courseblock
          h3.cols.noindent
            span.detail-code   → 课程编号
            span.detail-title  → 课程名称
            span.detail-hours_html → 学分
          div.noindent (多个)  → 描述等
    """
    courses = []

    # 学科代码和名称从 URL 和页面 h1.page-title 提取
    slug = subject_url.rstrip("/").split("/")[-1].upper()  # "eae" -> "EAE"

    h1 = soup.find("h1", class_="page-title")
    if h1:
        college_span = h1.find("span", class_="title-college")
        college = college_span.get_text(strip=True) if college_span else ""
        if college_span:
            college_span.extract()  # 先摘掉，避免污染 subject_name
        subject_name = h1.get_text(strip=True)
        # 部分页面 college 直接拼在 h1 文本里（无 span），尝试从末尾切分
        # 常见格式："Subject Name (CODE) College of XXX"
        if not college:
            m = re.search(r"\s+((?:College|School|Division)\s+of\s+\S.+)$", subject_name)
            if m:
                college = m.group(1).strip()
                subject_name = subject_name[: m.start()].strip()
    else:
        subject_name = slug
        college = ""

    # 去掉 subject_name 里夹带的学科代码，如 "Aerospace Science & Engineering (EAE)"
    subject_name = re.sub(r"\s*\([A-Z0-9]+\)\s*$", "", subject_name).strip()

    container = soup.find("div", class_="sc_sccoursedescs")
    if not container:
        log.warning(f"No sc_sccoursedescs found at {subject_url}")
        return courses

    blocks = container.find_all("div", class_="courseblock")
    log.info(f"  {slug}: {len(blocks)} courses found")

    for block in blocks:
        # 跳过已过期的历史版本（页面上会同时显示旧版和新版）
        if block.find(string=re.compile(r"this version has ended", re.IGNORECASE)):
            continue

        # ── 课程编号 ──────────────────────────────────────────────
        code_span = block.find("span", class_="detail-code")
        course_code = clean(code_span.get_text(strip=True)) if code_span else ""

        # ── 课程名称 ──────────────────────────────────────────────
        title_span = block.find("span", class_="detail-title")
        title = re.sub(r"^[\s\-–—]+", "", clean(title_span.get_text(strip=True))) if title_span else ""

        # ── 学分 ─────────────────────────────────────────────────
        units_span = block.find("span", class_="detail-hours_html")
        units = parse_units(units_span)

        # ── 课程描述 ──────────────────────────────────────────────
        # 描述通常在 h3 之后的 div.noindent 里，取所有文本拼接
        desc_parts = []
        for div in block.find_all("div", class_="noindent"):
            # 跳过空的或仅有空白的 div
            text = div.get_text(separator=" ", strip=True)
            if text:
                desc_parts.append(text)
        description = " ".join(desc_parts)

        # 如果上面方式没拿到描述，fallback 到整个 block 文本（去掉标题部分）
        if not description:
            h3 = block.find("h3")
            if h3:
                h3.decompose()
            description = block.get_text(separator=" ", strip=True)

        prereqs = extract_prerequisites(description)

        courses.append(Course(
            subject_code=slug,
            subject_name=subject_name,
            college=college,
            course_code=course_code,
            title=title,
            units=units,
            level=get_level(course_code),
            description=description,
            prerequisites=prereqs,
            prerequisite_codes=extract_prerequisite_codes(prereqs),
            source_url=subject_url,
        ))

    return courses


def save_json(courses: list[Course], path: str):
    data = [asdict(c) for c in courses]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    log.info(f"Saved {len(data)} courses → {path}")


def save_csv(courses: list[Course], path: str):
    if not courses:
        return
    fields = list(asdict(courses[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(asdict(c) for c in courses)
    log.info(f"Saved CSV → {path}")


def main():
    log.info("Step 1: Fetching subject index...")
    index_soup = fetch(INDEX_URL)
    if not index_soup:
        log.error("Failed to fetch index page. Exiting.")
        return

    subject_links = get_subject_links(index_soup)
    if SUBJECTS_LIMIT:
        subject_links = subject_links[:SUBJECTS_LIMIT]
        log.info(f"DEBUG MODE: limited to {SUBJECTS_LIMIT} subjects")

    all_courses: list[Course] = []

    log.info(f"Step 2: Scraping {len(subject_links)} subject pages...")
    for i, (url, _label) in enumerate(subject_links, 1):
        log.info(f"[{i}/{len(subject_links)}] {url}")
        soup = fetch(url)
        if soup:
            courses = parse_courses(soup, url)
            all_courses.extend(courses)
        time.sleep(DELAY_SEC)

    log.info(f"\nTotal courses scraped: {len(all_courses)}")

    save_json(all_courses, OUTPUT_JSON)
    save_csv(all_courses, OUTPUT_CSV)

    # 简单统计
    subjects = {c.subject_code for c in all_courses}
    log.info(f"Subjects covered: {len(subjects)}")
    log.info("Done!")


if __name__ == "__main__":
    main()