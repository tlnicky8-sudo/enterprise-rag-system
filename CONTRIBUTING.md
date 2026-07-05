# Contributing

感谢你对本项目的关注。以下是参与贡献的简要说明。

## 开发环境

```bash
git clone <your-fork-url>
cd rag_projector
uv venv && uv sync --extra retrieval --extra documents --extra faq --extra dev
cp config.example.ini config.ini
cp config/runtime.example.yaml config/runtime.yaml
cp .env.example .env
```

## 提交前检查

```bash
pytest
```

CI 会在 push / pull request 时自动运行轻量 smoke tests，不依赖 Milvus、Redis 或大模型。

## Pull Request 规范

1. 一个 PR 聚焦一个改动（功能、修复或文档）
2. 如涉及配置变更，同步更新 `config.example.ini`、`config/runtime.example.yaml` 或文档
3. 不要提交 `.env`、`config.ini`、`config/runtime.yaml`、模型权重或完整语料
4. 在 PR 描述中说明测试方式

## 报告问题

请使用 [Bug Report](.github/ISSUE_TEMPLATE/bug_report.yml) 模板，附上复现步骤、期望行为与实际行为。
