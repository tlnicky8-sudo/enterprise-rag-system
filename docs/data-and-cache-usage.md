# 数据处理、入库与缓存策略

本文说明企业知识库语料、FAQ 数据与语义缓存的职责边界、入库流程和运行时命中策略。README 保留快速启动路径，本文用于日常数据维护和调参。

---

## 一、数据分层

系统将运行数据划分为三层，各层职责不同，应分别维护：

| 层级 | 存什么 | 怎么写入 | 运行时干什么 |
|------|--------|----------|--------------|
| **RAG 语料** | 企业文档（制度/流程/手册等）切块 | `python setup_data.py` | Milvus 检索制度内容，作为 LLM 生成上下文 |
| **FAQ 问答对** | 人工维护的高频标准问答 | `python setup_faq_data.py` | 对稳定高频问题提供低延迟响应 |
| **语义缓存** | 问题向量 + 答案（FAQ 预热 + RAG 回写） | 入库预热；运行时按质量策略写入 | 相似问法命中后跳过检索与生成 |

```text
User Query
  │
  ├─► [1] Semantic Cache Hit? ──yes──► Return Answer
  │
  ├─► [2] FAQ BM25 Hit? ──yes──► Return Answer
  │
  └─► [3] Miss ──► RAG (Milvus Retrieval + LLM Generation)
                          │
                          └─► Cacheable? ──yes──► Write Semantic Cache
```

---

## 二、RAG 语料入库

### 2.1 适用场景

- 你有企业规章制度、流程手册、产品文档等**原始文档**
- 希望系统能「查制度再回答」，而不是只靠预设问答

### 2.2 操作步骤

**第 1 步：放文件**

```bash
# 默认目录（与 config.ini 中 valid_sources = ["enterprise"] 对应）
data/enterprise_data/
├── employee_handbook.md
├── it_security_policy.md
└── ...
```

支持：`.pdf` `.docx` `.ppt` `.pptx` `.md` `.txt` `.jpg` `.jpeg` `.png`

**第 2 步：确认 Milvus 和模型**

- Milvus 已启动（默认 `localhost:19530`）
- `config.ini` / `.env` 中 `BGE_M3_PATH` 指向本地 BGE-M3 模型

**第 3 步：执行入库**

```bash
# 首次入库
python setup_data.py

# 建议：先 dry-run 看流程是否正常
python setup_data.py --dry-run

# 可选：开启切块增强（关键词 + 假设问题，检索效果更好，但更慢）
python setup_data.py --enhance

# 可选：集合里已有数据就跳过（快速启动用）
python setup_data.py --skip-if-exists
```

**第 4 步：看结果**

- 终端会打印每个 source 写入的 chunk 数量
- 中间 Markdown：`data/processed/enterprise/markdown/`
- 血缘报告：`data/ingest_reports/ingest_report_*.json`

### 2.3 切块规则（`core/ingest/`）

入库时文档会按以下优先级切成父块，再细分为子块写入 Milvus：

1. **Markdown 标题** — `##` / `###`（员工手册、制度 MD 优先走这条）
2. **「第X条」编号** — 兼容 PDF/Word 转出的条文式文档
3. **固定长度回退** — 由 `config.ini` 中 `parent_chunk_size` / `child_chunk_size` 控制

清洗阶段（`clean.py`）会统一 `第X章/第X节/第X条` 为 Markdown 标题，并脱敏手机号、身份证号。

可选 `--enhance` 会为每个子块生成关键词和「可能问题」，提升检索召回。

### 2.4 日常维护

| 需求 | 做法 |
|------|------|
| 新增几份文档 | 放进 `data/enterprise_data/`，再跑 `python setup_data.py`（相同内容不会重复插入） |
| 文档大改版 | 替换文件后重新 `setup_data.py`；或改 `--doc-version 2` 区分版本 |
| 只想验证解析 | `python setup_data.py --dry-run` |

### 2.5 未入库时的行为

- Web / CLI 仍能启动
- 企业知识类问题会检索不到上下文，触发 **grounding 拒答** 或「信息不足」

---

## 三、FAQ 问答对入库

### 3.1 适用场景

- 业务或运营人员已整理「标准问 + 标准答」
- 希望稳定高频问题不经过 LLM 生成，直接返回经过审核的标准答案

### 3.2 数据格式

默认文件：`data/faq_data/faq_pairs.jsonl`（**每行一条 JSON**）。

可从仓库示例复制起步：

```bash
mkdir -p data/faq_data
cp examples/faq_pairs.example.jsonl data/faq_data/faq_pairs.jsonl
# 然后编辑 faq_pairs.jsonl，追加你的问答
```

单条示例：

```json
{"question": "公司年假有多少天？", "answer": "根据员工手册，累计工作满 1 年不满 10 年享 5 天/年……"}
```

| 字段 | 必填 | 说明 |
|------|------|------|
| `question` | 是 | 用户可能问法（也可用别名 `q`） |
| `answer` | 是 | 标准答案，建议引用制度依据（也可用别名 `a`） |
| `subject_name` | 否 | 领域标签，默认用 `config.ini` 的 `domain.name` |

也支持 `.json` 数组文件：可以是数组，或 `{ "items": [...] }` / `{ "data": [...] }` / `{ "faqs": [...] }`。

### 3.3 操作步骤

**第 1 步：确认 Redis**

- Redis 已启动
- `config.ini` / `.env` 中 `REDIS_*` 正确

**第 2 步：校验数据（推荐）**

```bash
python setup_faq_data.py --dry-run
```

会打印有效条数、跳过条数，**不写 Redis**。

**第 3 步：入库**

```bash
# 标准入库：写 Redis + 预热语义缓存
python setup_faq_data.py

# 全量替换（清空旧 FAQ 后重导）
python setup_faq_data.py --replace

# 指定文件
python setup_faq_data.py --source data/faq_data/my_faq.jsonl
```

**第 4 步：验证**

启动 Web 后，用 FAQ 里有的问题提问；日志应出现 `Pipeline answered from FAQ layer`。

### 3.4 日常维护

| 需求 | 做法 |
|------|------|
| 追加几条 FAQ | 编辑 `faq_pairs.jsonl`，再 `python setup_faq_data.py`（会去重合并） |
| 整表替换 | `python setup_faq_data.py --replace` |
| Redis 有数据但缓存丢了 | `python scripts/preheat_faq_cache.py`（从 Redis 或 JSONL 重新预热向量缓存） |

### 3.5 未入库时的行为

- Redis 不可用时：FAQ 层自动关闭，**全部走 RAG**（不影响基本可用）
- Redis 可用但没 FAQ 数据：同上，只是少了一层加速

---

## 四、问答缓存策略

缓存分 **读路径（命中）** 和 **写路径（回写）** 两部分。

### 4.1 读路径：用户提问时怎么命中

顺序固定（见 `core/faq/search.py`）：

1. **语义缓存** — 用 BGE-M3 算问题向量，在 Redis 向量索引里找最相似的  
   - 相似度 ≥ `faq.semantic_threshold`（默认 `0.92`）→ 命中  
2. **BM25** — 在 `faq:qa_pairs` 里做关键词匹配  
   - softmax 分数 ≥ `faq.bm25_threshold` 且原始分 ≥ `faq.bm25_min_score` → 命中  
3. **都没命中** → 进入 RAG

**注意**：Web 端如果选了「文档来源」过滤（`source_filter`），默认会跳过 FAQ 层，直接 RAG。开关：`features.skip_faq_when_source_filter`。

### 4.2 写路径：RAG 回答后什么时候写回缓存

RAG 生成答案后，`core/cache_policy.py` 判断是否写入语义缓存。需**同时满足**：

- `features.enable_cache_write = true`
- 答案来自 RAG 且检索到了上下文
- 通过了 grounding（有引用、相关性够）
- 答案长度 ≥ `faq.min_answer_length`
- 不含时效性关键词（如「今天」「实时」，见 `cache.time_sensitive_pattern`）
- 加权分 ≥ `faq.cache_write_threshold`：

```text
write_score = rerank_weight × rerank_score + llm_weight × llm_confidence
            ─────────────────────────────────────────────────────────
                    rerank_weight + llm_weight
```

默认：`rerank_weight=0.6`，`llm_weight=0.4`，`cache_write_threshold=0.8`。

写入后，相似问题下次可走语义缓存，不必再调 LLM。

### 4.3 缓存相关配置（`config/runtime.yaml`）

```yaml
features:
  enable_faq: true              # 总开关：是否走 FAQ 快路径
  enable_cache_write: true      # 是否允许 RAG 答案写回缓存

faq:
  semantic_threshold: 0.92    # 语义命中阈值（越高越严）
  bm25_threshold: 0.85        # BM25 命中阈值
  bm25_min_score: 0.1         # BM25 原始分下限
  cache_ttl: 86400            # 语义缓存过期时间（秒）
  cache_write_threshold: 0.8  # 写回缓存的最低加权分
  rerank_weight: 0.6          # 写回分数中 rerank 权重
  llm_weight: 0.4             # 写回分数中 LLM 置信度权重
  min_answer_length: 20       # 写回答案最短字数

cache:
  time_sensitive_pattern: "今天|本周|本月|今年|现在|实时|当前|几点|天气"
```

调参建议：

| 现象 | 可尝试 |
|------|--------|
| FAQ 误命中太多 | 提高 `semantic_threshold` / `bm25_threshold` |
| FAQ 命中率太低 | 略降低阈值，或补充 `faq_pairs.jsonl` |
| 不希望自动写缓存 | `enable_cache_write: false` |
| 缓存答案质量不稳 | 提高 `cache_write_threshold` |

### 4.4 手动预热缓存

入库时 `setup_faq_data.py` 会自动预热。若 Redis 里已有问答对、但向量索引需要重建：

```bash
python scripts/preheat_faq_cache.py

# 指定 JSONL 作为 Redis 为空时的回退数据源
python scripts/preheat_faq_cache.py --source data/faq_data/faq_pairs.jsonl
```

---

## 五、端到端初始化流程

首次部署或重建本地环境时，建议按以下顺序执行：

```bash
# 0. 配置
cp config.example.ini config.ini
cp config/runtime.example.yaml config/runtime.yaml
cp .env.example .env
# 编辑 .env 填入 API Key

# 1. RAG 语料（把 Markdown 等放进 data/enterprise_data/）
python setup_data.py

# 2. FAQ 问答对
mkdir -p data/faq_data
cp examples/faq_pairs.example.jsonl data/faq_data/faq_pairs.jsonl
# 编辑 faq_pairs.jsonl
python setup_faq_data.py

# 3. 启动
python web_app.py
```

验证清单：

- [ ] 问一个 FAQ 里已有的问题 → 应命中 FAQ 层，`source=faq`
- [ ] 问一个需要查知识库的新问题 → 走 RAG，返回答案 + 引用 `[1][2]`
- [ ] 重复问类似的新问题（若上次 RAG 质量够高）→ 可能命中语义缓存

---

## 六、命令速查

| 命令 | 作用 |
|------|------|
| `python setup_data.py` | RAG 语料入库 Milvus |
| `python setup_data.py --dry-run` | 只跑处理流程，不写库 |
| `python setup_data.py --enhance` | 开启切块增强 |
| `python setup_faq_data.py` | FAQ 入库 Redis + 预热缓存 |
| `python setup_faq_data.py --dry-run` | 只校验 JSONL |
| `python setup_faq_data.py --replace` | 清空后全量重导 FAQ |
| `python scripts/preheat_faq_cache.py` | 重新预热语义向量缓存 |

---

## 七、进一步阅读

- [../config/runtime.example.yaml](../config/runtime.example.yaml) — 全部运行时开关
- [../models/README.md](../models/README.md) — 本地模型路径
