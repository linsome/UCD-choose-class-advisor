# Benchmark
Advanced RAG systems dynamically decide what to retrieve, when to retrieve, and how to verify.

# Purpose
1. 帮助学生选择出心仪的课程
2. 帮助学生规划课程
3. 当有课程更新的时候，可以及时更新到模型里

# Data

数据来源：
1. 学院提供的syllabus
2. course catalog 上的简单介绍
3. canvas上syllabus大纲
4. 有部分课程没有syllabus （例如NSC 287）
5. 教授信息（Rate my professor）-- 可选
6. 教授研究方向 -- 可选
7. degree requirement
8. 课程所用的教材，对应的章节

数据内容：
1. 长篇的syllabus
2. 简短的课程介绍
3. degree requirement 表格
4. department course arrangement

# Methodology
1. 可以先以简短的课程介绍为主

----------
爬取Syllabus:
1. 只针对开放课程来进行爬取

爬取课程频率：
1. 只爬取学院的课程安排

爬取学院课程：
1. 爬取学院课程要求

----------
# MVP-1:
1. 可能的问题 -- 有些query会采样到例如internship course
2. 会混合研究生和本科生的课程
3. hallucination -- "无人机和飞行器历史" 会 检索出Avian Reproduction
4. data - analysis 会检索到相同的课程两次
    “  [4] MGB 403AY — — Data Analysis for Managers
        Subject : 2026-2027 General Catalog
        Units   : 4
        Desc    : Course Description: Introduction to statistics and data analysis for managerial decision making. Descriptive statistics, principles of data collection, sampling, quality control, statistical inference...
        Score   : 0.6570

    [5] MGP 403AY — — Data Analysis for Managers
        Subject : 2026-2027 General Catalog
        Units   : 4
        Desc    : Course Description: Introduction to statistics and data analysis for managerial decision making. Descriptive statistics, principles of data collection, sampling, quality control, statistical inference...
        Score   : 0.6543”
5. data - analysis会检索到MGP的课程和研究生的课程
6. subject 都是：2026-2027 - 只保留subject code
------------------------------
# MVP-1.5:
1. internship 的问题我们展示先不理会，这确实是个选项
2. 研究生/本科生的问题尚未解决
3. 针对hallucination，我加入BM25来提高关键词的提取 + Reranker -- 模型对中文的友好程度会差一点
    a. 考虑针对不同语言提高关键词的检索能力（提前翻译然后输入）
4. 课程去重 -- 对于少部分课程课程内容高度相关，但是属于不同专业的课程，我们期望可以在DAG的过程解决
5. 原始数据中，学院，prerequisite的部分已经提炼的更加精准
想要进一步优化：
a. 判断是否跨学院推荐/ 跨level推荐 -- 收集跨年级，跨专业的课程，结合专业内的课程做判断

--了解当前dag是如何工作的