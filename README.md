# AI_CRM — RAG 知识问答客服机器人（Flask + Chroma + LangChain/LangGraph）

![tests](https://github.com/lgx-666/ai-crm-rag-chatbot/actions/workflows/ci.yml/badge.svg)

面向大模型与 Java 后端课程资料的问答机器人：意图分类 + 主题路由检索 + SSE 流式 +
多轮记忆（滑窗 + 滚动摘要）+ JWT 鉴权 + RAGAS 量化评估 + Langfuse 自托管埋点 +
Redis 问答缓存 + Docker 化。

## 目录结构约定（重要）

本仓库根是 `AI_CRM/`，但它依赖三个**同级目录**，都不在仓库内（体积 / 版权原因）：

```
A3实战_deepseek/
  AI_CRM/            ← 本仓库
  chinese_model/     ← 中文嵌入模型 BAAI/bge-base-zh-v1.5（约 400MB，从 HuggingFace 下载）
  knowledge/         ← 课程 PDF 语料（供 import_knowledge.py 灌库）
  docker_wheels/     ← torch CPU wheel（仅 Docker 构建需要，见下）
```

## 快速开始（本地）

```
python -m venv venv && venv\Scripts\activate        :: Windows；Linux/macOS 用 source venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
copy .env.example .env                              :: 填入自己的 key
python app.py                                       :: http://localhost:5000
```

灌知识库（**集合名必须传全名**，传 `market` / `java` 会静默不导入）：

```
python import_knowledge.py ../knowledge/llm market_knowledge
python import_knowledge.py ../knowledge/java java_knowledge
```

## 测试与评估

```
python -m pytest              :: 76 用例；覆盖率 91%，分母故意只圈三个纯逻辑模块，范围见 pytest.ini
python eval_rag.py            :: RAGAS 四指标，会调用付费 LLM，结果落 data/ragas_report_<时间戳>.csv
```

检索与评估**走同一条线上链路**：`eval_rag.py` 与 `response_generator.py` 都调用
`kb_router.retrieval_plan`，否则测的就是已经不存在的旧管线。

## Docker

先准备 torch CPU wheel：国内没有任何可用的新版 CPU 镜像源，需手动下载
`torch-2.8.0+cpu-cp39-cp39-manylinux_2_28_x86_64.whl`（cp39 要对应基础镜像的 Python 版本），
放到**与 AI_CRM 同级的 `docker_wheels/`** 目录。

之所以不能放项目内：`.dockerignore` 排除了 `*.whl`，而 BuildKit 的
`RUN --mount=type=bind` 同样受 `.dockerignore` 约束，文件会被挡在构建上下文之外；
放项目外 + `build.additional_contexts: wheels` 才能既被排除又可挂载，
并且不给镜像多留一层 184MB（`COPY` 出来的层事后 `RUN rm` 也回收不掉）。

```
docker compose up -d --build        :: 数据在 ./data，模型只读挂载 ../chinese_model:/app/chinese_model
```

## 安全

`.env` 已被忽略，切勿提交；`data/crm.db` 含真实用户数据，同样不入库
（`.gitignore` 用 `data/*` 把整个数据目录挡在仓库外）。评估报告
`data/ragas_report_*.csv` **同样不公开**——它的 `retrieved_contexts` / `reference` 两列是
课程 PDF 的原文块，放进公开仓库等于把课程材料发出去；对外只给聚合分数，
见 `项目报告/项目报告.md` 的 3.3 与 3.3.1 两张表。

**图片同样是审查对象**：`项目报告/images/` 里的 PNG 会随仓库公开，而截图最容易漏的恰好是
手机号、密钥这类账号信息——`git check-ignore` 只看文件名，对像素是盲的。拍图前先切到测试账号；
提交前用 `git log --oneline -- 项目报告/images` 复核哪些图已经在远端。**改图再提交只修 HEAD**：旧提交里的原图按 SHA 照样能下载（实测强推之后仍返回 200，要等 GitHub 后台回收不可达对象）。彻底处置 = `git filter-repo --invert-paths --path-glob "*.png"` 把路径从全部历史剔除 + `--force-with-lease` 强推 + 本地 `git gc --prune=now`（重加 `origin` 后一次 `git fetch` 会把旧历史连敏感 blob 一起拉回来，这步不能省）。
