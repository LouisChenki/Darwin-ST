# 机制知识库数据 (P3-b 建图产物)

本目录存放 `scripts/build_kg.py` 的产物 (运行后生成, 已被 .gitignore 排除大文件):

- `mechanism_cards.json`  最终机制卡库 (人类可读, 供用户复核)。
- `coverage.json`         每轮迭代的覆盖统计 + LLM 自评诊断 (审计建库过程)。
- `graph_store.json`      InMemory 后端持久化 (无 Neo4j 时的灌库结果, 含 embedding)。

Neo4j 可用时直接灌入 Neo4j (bolt://localhost:7687), 不写 graph_store.json。

复跑:
    export DEEPSEEK_API_KEY=...
    export DEEPSEEK_MODEL=deepseek-v4-flash
    python scripts/build_kg.py            # 全量 500 篇
    python scripts/build_kg.py --limit 40 # 小批量试跑
    python scripts/build_kg.py --dry-run  # 无 LLM, 演示后处理/灌库链路
