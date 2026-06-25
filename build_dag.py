"""
build_dag.py
============
根据 prerequisites 原文解析 AND/OR 先修逻辑，构建课程 DAG 并生成交互式可视化。

先修逻辑解析规则：
  ;   →  AND（所有组都必须满足）
  or  →  OR（组内满足一门即可）
  无 or 的组内用逗号隔开的多门课 → 每门都是独立 AND 要求

不在 catalog 里的课程编号直接忽略。
在 catalog 里但属于其他学科的课程（如 MAT、BST）保留为外部灰色节点。

可视化约定：
  蓝色圆节点   → 本科课程
  橙色圆节点   → 研究生课程
  灰色圆节点   → 外学科先修课（catalog 内，其他学科）
  绿色圆节点   → 已修完成
  黄色菱形节点 → OR 选择节点（满足其中一门即可）
  蓝色实线     → AND 边（必须满足）
  橙色虚线     → OR 边（任选之一）

运行方式：
    python build_dag.py                     # 可视化 EAE 学科（默认）
    python build_dag.py STA                 # 可视化 STA 学科
    python build_dag.py STA --recommend "STA 013, MAT 021C" undergraduate
"""

import json
import pickle
import argparse
import re
import networkx as nx
from pyvis.network import Network

# ─────────────────────── CONFIG ───────────────────────
DATA_PATH = "courses_raw.json"
DAG_PATH  = "course_dag.pkl"
# ──────────────────────────────────────────────────────

COLOR_UNDERGRAD   = "#4A90D9"   # 蓝
COLOR_GRAD        = "#E87722"   # 橙
COLOR_EXTERNAL    = "#AAAAAA"   # 灰（catalog 内、其他学科）
COLOR_OR_JUNCTION = "#F5C518"   # 黄（OR 选择节点）

COURSE_CODE_RE = re.compile(r'\b([A-Z]{2,5})\s+(\d+[A-Z0-9]*)\b')


# ── 解析先修逻辑 ──────────────────────────────────────

def _norm(s: str) -> str:
    return s.replace("\xa0", " ").strip()


def parse_prereq_logic(prereq_text: str, code_set: set[str]) -> list[list[str]]:
    """
    将先修文本解析为 CNF（合取范式）：AND of OR groups。
    只保留在 code_set（catalog）中存在的课程编号，其余忽略。

    返回值为列表的列表，外层 AND，内层 OR。

    例：
      "STA 013 or STA 032; MAT 021C"
      → [["STA 013", "STA 032"], ["MAT 021C"]]
      意为：(STA 013 OR STA 032) AND MAT 021C
    """
    if not prereq_text:
        return []

    text = _norm(prereq_text)
    result: list[list[str]] = []

    for group_text in text.split(";"):
        group_text = group_text.strip()
        if not group_text:
            continue
        # 以 "or ..." 开头的分组通常是 "or consent of instructor" / "or equivalent"，跳过
        if re.match(r'or\b', group_text, re.IGNORECASE):
            continue

        if re.search(r'\bor\b', group_text, re.IGNORECASE):
            # OR 组：收集所有 catalog 内的备选课程
            or_courses: list[str] = []
            for part in re.split(r'\bor\b', group_text, flags=re.IGNORECASE):
                for subj, num in COURSE_CODE_RE.findall(part):
                    code = f"{subj} {num}"
                    if code in code_set and code not in or_courses:
                        or_courses.append(code)
            if or_courses:
                result.append(or_courses)
        else:
            # 无 or：组内每门课都是独立的 AND 要求
            for subj, num in COURSE_CODE_RE.findall(group_text):
                code = f"{subj} {num}"
                if code in code_set:
                    result.append([code])

    return result


# ── 构建图 ────────────────────────────────────────────

def build_dag(courses: list[dict]) -> nx.DiGraph:
    G = nx.DiGraph()

    for c in courses:
        G.add_node(
            _norm(c["course_code"]),
            title=c.get("title", ""),
            subject_code=c.get("subject_code", ""),
            subject_name=c.get("subject_name", ""),
            college=c.get("college", ""),
            level=c.get("level", "unknown"),
            units=c.get("units", ""),
            prerequisites=c.get("prerequisites", ""),
            node_type="course",
            prereq_logic=[],
        )

    code_set = {_norm(c["course_code"]) for c in courses}

    edge_count = 0
    for c in courses:
        target = _norm(c["course_code"])
        logic = parse_prereq_logic(c.get("prerequisites", ""), code_set)
        G.nodes[target]["prereq_logic"] = logic

        for gi, or_group in enumerate(logic):
            if len(or_group) == 1:
                # 单课 AND 要求：直接连边
                prereq = or_group[0]
                if not G.has_edge(prereq, target):
                    G.add_edge(prereq, target, edge_type="and")
                    edge_count += 1
            else:
                # OR 组：创建中间菱形节点
                junction_id = f"_OR_{target}_{gi}"
                G.add_node(junction_id, node_type="or_junction", target_course=target)
                G.add_edge(junction_id, target, edge_type="and")
                for prereq in or_group:
                    if not G.has_edge(prereq, junction_id):
                        G.add_edge(prereq, junction_id, edge_type="or")
                        edge_count += 1

    print(f"DAG built: {G.number_of_nodes()} nodes, {edge_count} edges")
    return G


def save_dag(G: nx.DiGraph, path: str = DAG_PATH):
    with open(path, "wb") as f:
        pickle.dump(G, f)
    print(f"DAG saved to {path}")


def load_dag(path: str = DAG_PATH) -> nx.DiGraph:
    with open(path, "rb") as f:
        return pickle.load(f)


# ── 可视化 ────────────────────────────────────────────

def _compute_positions(sub: nx.DiGraph, x_gap: int = 200, y_gap: int = 150) -> dict[str, tuple[int, int]]:
    try:
        generations = list(nx.topological_generations(sub))
    except nx.NetworkXUnfeasible:
        return {}

    pos = {}
    for level, nodes in enumerate(generations):
        nodes = sorted(nodes)
        n = len(nodes)
        for i, node in enumerate(nodes):
            x = int((i - (n - 1) / 2) * x_gap)
            y = level * y_gap
            pos[node] = (x, y)
    return pos


def visualize(
    G: nx.DiGraph,
    subject: str,
    completed: set[str] | None = None,
    output: str | None = None,
):
    """生成以 subject 为核心的交互式 HTML 图，含 AND/OR 可视化。"""
    core_nodes = {n for n, d in G.nodes(data=True) if d.get("subject_code") == subject}

    # 收集外部先修节点（catalog 内、其他学科）和 OR junction 节点
    external_prereqs: set[str] = set()
    or_junctions: set[str] = set()
    for node in core_nodes:
        for pred in G.predecessors(node):
            ndata = G.nodes[pred]
            if ndata.get("node_type") == "or_junction":
                or_junctions.add(pred)
                for pred2 in G.predecessors(pred):
                    if pred2 not in core_nodes:
                        external_prereqs.add(pred2)
            elif pred not in core_nodes:
                external_prereqs.add(pred)

    all_nodes = core_nodes | external_prereqs | or_junctions
    sub = G.subgraph(all_nodes).copy()

    if output is None:
        output = f"dag_{subject}.html"

    pos = _compute_positions(sub)
    net = Network(height="900px", width="100%", directed=True, bgcolor="#1a1a2e")
    completed = completed or set()

    for nid in sub.nodes():
        data = G.nodes[nid]
        x, y = pos.get(nid, (0, 0))

        if data.get("node_type") == "or_junction":
            alts = list(G.predecessors(nid))
            net.add_node(
                nid,
                label="OR",
                title="<b>OR 选择节点</b><br>满足以下任意一门即可:<br>" + "<br>".join(alts),
                color=COLOR_OR_JUNCTION,
                x=x, y=y,
                physics=False,
                font={"color": "#000000", "size": 11, "bold": True},
                size=14,
                shape="diamond",
            )
            continue

        level = data.get("level", "unknown")
        prereq_logic = data.get("prereq_logic", [])
        logic_str = ""
        if prereq_logic:
            parts = []
            for group in prereq_logic:
                if len(group) == 1:
                    parts.append(group[0])
                else:
                    parts.append("(" + " OR ".join(group) + ")")
            logic_str = "<br>先修逻辑: " + " AND ".join(parts)

        tooltip = (
            f"<b>{nid}</b><br>"
            f"{data.get('title', '')}<br>"
            f"Units: {data.get('units', '')} | Level: {level}<br>"
            f"Subject: {data.get('subject_name', '')}"
            f"{logic_str}"
        )

        if nid in completed:
            color = "#2ECC71"
        elif nid in external_prereqs:
            color = COLOR_EXTERNAL
        elif level == "graduate":
            color = COLOR_GRAD
        else:
            color = COLOR_UNDERGRAD

        net.add_node(
            nid,
            label=nid,
            title=tooltip,
            color=color,
            x=x, y=y,
            physics=False,
            font={"color": "#FFFFFF", "size": 13},
            size=22,
        )

    for src, dst in sub.edges():
        edge_type = G.edges[src, dst].get("edge_type", "and")
        if edge_type == "or":
            net.add_edge(src, dst,
                         color={"color": COLOR_OR_JUNCTION, "highlight": "#FFFFFF"},
                         dashes=True, arrows="to", width=1.5)
        else:
            net.add_edge(src, dst,
                         color={"color": "#6666AA", "highlight": "#FFFFFF"},
                         arrows="to", width=2)

    net.set_options("""{
      "physics": { "enabled": false },
      "interaction": { "hover": true, "tooltipDelay": 80, "navigationButtons": true }
    }""")

    net.save_graph(output)
    print(
        f"Visualization saved → {output}  "
        f"({len(core_nodes)} core, {len(external_prereqs)} external, {len(or_junctions)} OR junctions)"
    )
    return output


# ── 推荐 ─────────────────────────────────────────────

def recommend(
    G: nx.DiGraph,
    completed: set[str],
    student_level: str,
    subjects: list[str] | None = None,
    top_n: int = 10,
) -> list[dict]:
    """
    推荐当前可修的课程。
    使用 prereq_logic（CNF）检查：每个 OR 组中至少有一门已修完。
    不在 catalog 里的先修要求已在构建阶段忽略。
    """
    candidates = []
    for node, data in G.nodes(data=True):
        if data.get("node_type") == "or_junction":
            continue
        if node in completed:
            continue
        if data.get("level") not in (student_level, "unknown"):
            continue
        if subjects and data.get("subject_code") not in subjects:
            continue

        prereq_logic: list[list[str]] = data.get("prereq_logic", [])
        if prereq_logic:
            satisfied = all(
                any(c in completed for c in or_group)
                for or_group in prereq_logic
            )
            if not satisfied:
                continue

        candidates.append({
            "course_code":  node,
            "title":        data.get("title", ""),
            "subject_name": data.get("subject_name", ""),
            "units":        data.get("units", ""),
            "level":        data.get("level", ""),
        })

    return candidates[:top_n]


# ── DAG 距离计算 ──────────────────────────────────────

def _count_unmet(G: nx.DiGraph, course_code: str, completed: set[str]) -> int:
    """返回 course_code 还有多少个 AND 组尚未满足（用于选最优备选项）"""
    data = G.nodes.get(course_code, {})
    prereq_logic: list[list[str]] = data.get("prereq_logic", [])
    return sum(
        1 for grp in prereq_logic
        if not any(c in completed for c in grp)
    )


def _build_path(G: nx.DiGraph, target: str, completed: set[str]) -> list[str]:
    """
    从 target 向前 BFS，收集所有需要先修但尚未完成的课程，
    返回拓扑排序后的建议路径（含 target 本身）。
    遇到 OR junction 节点时选择 unmet 数量最少的前驱。
    """
    needed: set[str] = set()
    visited: set[str] = set()
    queue: list[str] = [target]

    while queue:
        node = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)

        ndata = G.nodes.get(node, {})
        if ndata.get("node_type") == "or_junction":
            preds = list(G.predecessors(node))
            if preds:
                # 选 unmet 最少的前驱；已完成的优先（unmet=0）
                best = min(preds, key=lambda p: _count_unmet(G, p, completed))
                if best not in completed:
                    queue.append(best)
            continue

        if node not in completed:
            if node != target:
                needed.add(node)
            for pred in G.predecessors(node):
                queue.append(pred)

    if not needed:
        return [target]

    sub = G.subgraph(needed | {target})
    try:
        order = list(nx.topological_sort(sub))
    except nx.NetworkXUnfeasible:
        order = list(needed) + [target]

    # 只返回 needed 中的节点（course 节点，跳过 OR junction）加 target
    result = [
        n for n in order
        if n in needed and G.nodes.get(n, {}).get("node_type") != "or_junction"
    ]
    result.append(target)
    return result


def compute_distance(G: nx.DiGraph, course_code: str, completed: set[str]) -> dict:
    """
    计算某门课对当前学生的"距离"：

    tier 0 — 现在可选（先修全部满足）
      {"tier": 0, "missing": [], "path": []}

    tier 1 — 即将可选（只缺 1-2 门先修，且那些先修自身也可立即选）
      {"tier": 1, "missing": ["STA 013"], "path": []}

    tier 2 — 长期规划（缺较多先修，或先修本身也有未满足的先修）
      {"tier": 2, "missing": [...], "path": ["A", "B", "target"]}
    """
    data = G.nodes.get(course_code)
    if data is None or data.get("node_type") == "or_junction":
        return {"tier": 0, "missing": [], "path": []}

    prereq_logic: list[list[str]] = data.get("prereq_logic", [])
    if not prereq_logic:
        return {"tier": 0, "missing": [], "path": []}

    # 对每个未满足的 OR 组，选最优备选课
    missing: list[str] = []
    for or_group in prereq_logic:
        if not any(c in completed for c in or_group):
            best = min(or_group, key=lambda c: _count_unmet(G, c, completed))
            missing.append(best)

    if not missing:
        return {"tier": 0, "missing": [], "path": []}

    # tier 1：缺 ≤2 门，且每门缺少的课自身先修也都满足
    is_shallow = len(missing) <= 2 and all(
        _count_unmet(G, m, completed) == 0 for m in missing
    )
    if is_shallow:
        return {"tier": 1, "missing": missing, "path": []}

    # tier 2：需要多步规划
    path = _build_path(G, course_code, completed)
    return {"tier": 2, "missing": missing, "path": path}


# ── 入口 ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("subject", nargs="?", default="EAE", help="学科代码，如 STA / ECS / HDE")
    parser.add_argument("--recommend", default="", help="已修课程，逗号分隔，如 'STA 013, MAT 021C'")
    parser.add_argument("--level", default="undergraduate", choices=["undergraduate", "graduate"])
    parser.add_argument("--rebuild", action="store_true", help="强制重建 DAG（忽略缓存）")
    args = parser.parse_args()

    import os
    if not args.rebuild and os.path.exists(DAG_PATH):
        print(f"Loading cached DAG from {DAG_PATH}  (use --rebuild to force rebuild)")
        G = load_dag(DAG_PATH)
    else:
        print(f"Loading {DATA_PATH}...")
        with open(DATA_PATH, encoding="utf-8") as f:
            courses = json.load(f)
        G = build_dag(courses)
        save_dag(G)

    completed = {c.strip() for c in args.recommend.split(",") if c.strip()} if args.recommend else set()

    output = visualize(G, subject=args.subject, completed=completed)

    if completed:
        print(f"\n── Recommendations for {args.level} (completed: {completed}) ──")
        recs = recommend(G, completed, student_level=args.level, subjects=[args.subject])
        if recs:
            for r in recs:
                print(f"  {r['course_code']}  {r['title']}  ({r['units']} units)")
        else:
            print("  No unlocked courses found for this subject.")

    import webbrowser, os
    webbrowser.open(f"file://{os.path.abspath(output)}")


if __name__ == "__main__":
    main()
