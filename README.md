# Darwin-ST

🌐 **Darwin-ST** 是一套**自主时空预测研究系统**:针对交通流预测(PeMS04/08、METR-LA、PEMS-BAY),通过进化式 NAS + 超参优化(HPO)自动逼近 SOTA;并且——这是它的核心研究赌注——用 LLM **通过跨域类比发明新算子**(例如把 CV 的掩码自编码迁移进时空预测)。

它是一个**自包含的 Python 系统**:确定性的 Python 自治循环是主体,LLM 只在停滞时作为 Tier-2 工具被调用来合成新机制。系统全部位于 `src/darwin_st/` 包中。

---

## 双层自治架构

系统的脊梁是**确定性 Python 主导(Tier-1,高频)+ 把"创造力"外包给 LLM(Tier-2,低频,仅停滞触发)**:

- **Tier-1 优化引擎**:aging 进化 NAS + Optuna TPE/ASHA + MAP-Elites 档案 + 多卡并行调度。`while True` 自治循环,程序化停止(超越 SOTA 才停),永不暂停问人。
- **Tier-2 创造层**:停滞时诊断瓶颈 → 跨域机制知识库检索互补机制 → DeepSeek 合成 N 个融合假设 → Aider 在 git 沙箱写算子 → 验证门把关 → 注入算子库参与进化与真实评测。

底层是可信评测地基(masked MAE/RMSE/MAPE、按数据集分流的协议、邻接矩阵)和 SQLite 结构化记忆(成功/失败墓地)。

---

## 安装

需要 [uv](https://github.com/astral-sh/uv)。

```bash
uv sync          # 创建 .venv 并装依赖 (macOS 用 CPU/MPS torch, Linux 用 cu128)
```

`pythonpath=["src"]` 已在 pyproject 配好,导入直接 `from darwin_st...`,无需 `pip install`。

跨域知识库与 Tier-2 创造层另需:Neo4j(可选,不可用自动回退内存图)、`sentence-transformers`(真实语义嵌入)、DeepSeek API key(`DEEPSEEK_API_KEY` 环境变量)。

---

## 运行

所有入口在 `scripts/`,均由环境变量配置。

| 脚本 | 用途 |
|---|---|
| `run_autoresearch.py` | Tier-1 自治优化(进化 NAS + HPO + MAP-Elites + 多卡) |
| `run_creation_loop.py` | Tier-2 创造闭环端到端(真实 DeepSeek 合成 + 真实 PeMS04 训练评测) |
| `build_kg.py` | 从文献语料构建跨域机制知识库(机制卡) |
| `qc_mechanisms.py` | 机制库质检报告(DeepSeek 当质检员,只读) |
| `apply_qc.py` | 按质检结果整理机制库(默认 dry-run,`--apply` 改库) |
| `verify_retrieval.py` | 把机制库灌进检索栈跑跨域类比检索验收 |
| `baseline_smoke.py` | 编译一个 genotype 在真实数据上训练的冒烟测试 |

例:

```bash
# Tier-1 优化
DATASET=PeMS04 N_GPUS=4 python scripts/run_autoresearch.py

# Tier-2 创造闭环 (需 DeepSeek key)
export DEEPSEEK_API_KEY=...
KB_SOURCE=cards python scripts/run_creation_loop.py
```

---

## 测试

测试是正确性契约——每个模块都有 `tests/test_<module>.py`,设备无关,CPU 即可全绿。

```bash
uv run pytest tests/ -q                          # 全套
uv run pytest tests/test_metrics.py -v           # 单个文件
uv run pytest tests/test_metrics.py::test_zeros_are_masked_out   # 单个测试
```

---

## 设计文档

治理性设计在 `docs/`,做实质性改动前应先读:

- `ANALYSIS.md` — 原始诊断(项目为何这样重构)
- `BLUEPRINT.md` — P0–P4 路线图
- `P2_ALGORITHM_DESIGN.md` — NAS/HPO/MAP-Elites 决策 + 架构铁律(节点维约束)
- `TIER2_DESIGN_DECISIONS.md` + `TIER2_RESEARCH_FINDINGS.md` — LLM 创造层
- `SERVER_VALIDATION.md` — 真实硬件上实际跑过什么

面向开发者的总览见 `CLAUDE.md`。
