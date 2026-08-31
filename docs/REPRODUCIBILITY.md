# 可复现性清单 (Reproducibility Checklist) —— E14

> 目标：第三方按本清单能在同硬件上复跑主结果的关键环节；并防御"LLM 方法不可复现"的天然质疑。
> 状态：随实验并行维护，每个投稿用实验收关时当次冻结。

## 代码

- [ ] 定稿 commit + git tag（沿用 `b7-v1-routing-failure-18.476` 的 tag 规程：`版本-定性-best`）
- [x] 依赖锁定：`uv.lock` 在库；Python 版本见 `.python-version`
- [x] 测试门槛：`uv run pytest tests/ -q` 全绿是任何 commit 的收门条件
- [ ] 运行约定入文档：`python -u`、环境变量全集（见下）

## 数据

- [ ] 数据集版本与来源：PeMS04/07/08、METR-LA（PEMS-BAY 可选）的原始文件哈希
- [ ] 协议 profile 版本：`src/darwin_st/data/protocol.py` 的 split/horizon/null_val（6/2/2 + 12 步平均；METR-LA 7/1/2 + @3/6/12）
- [ ] 邻接矩阵生成参数：PeMS 高斯核 σ 与阈值 κ、METR-LA 的 DCRNN 镜像来源
- [x] 评测口径：masked MAE/RMSE/MAPE，inverse-transform 回真实尺度（`data/prepare.py: evaluate`）

## 配置归档

- [ ] 每次长跑的完整环境变量集（`DATASET/N_GPUS/POP_SIZE/HPO_TRIALS/MAX_EPOCHS/STAGNATION/KB_SOURCE/USE_PROXY/AUX_CREATION/AUX_FORCE_EVERY/REFLECT/REFLECT_EVERY/REFINE_BREAKER_*/DEEPSEEK_MODEL/...`）随收关冻结（脱敏）
- [x] 冻结规程样例：`frozen-runs/b7v1/RUN_MANIFEST.md`（含 SHA-256 manifest）

## LLM

- [x] 模型版本披露：deepseek-v4-flash（调用日期随实验记录）
- [x] prompt/response 全量留档：`LLM_ARCHIVE=<path>.jsonl` 环境变量开启（`creation/llm.py`，追加写含时间戳/模型/温度/消息/响应）
- [ ] 关键创造步骤的留档重放演示（用留档 response 离线重放，证明结论不依赖模型未来版本）

## 资产冻结（每实验收关一次）

- [ ] memory db（sqlite `.backup` 一致副本）+ 创造履历 jsonl + 动作账本 + direction_state + 日志 + 动态算子目录整包 + SHA-256 manifest
- [ ] checkpoint 目录（大体积留数据盘，哈希入 manifest）
- [x] 规程：`frozen-runs/<run>/RUN_MANIFEST.md` + `*_manifest_server.sha256` + `*_manifest_local.sha256`

## 结果公开

- [ ] top-K checkpoint + 模型卡（genotype/hparams/operators/复现说明）
- [ ] 多 seed 原始数值（每 seed 的 val/test MAE/RMSE/MAPE）随结果目录公开
- [ ] 榜单双口径（val 历史 + test 复评）见 `leaderboard/README.md`

## 投稿前最后闸门（E16）

- [ ] masked 指标与 BasicTS/DCRNN 官方实现对拍（含 test 集路径）
- [ ] scaler 仅训练集拟合 + inverse-transform 后评测的断言复核
- [ ] 邻接矩阵重建与文献参数核对
- [ ] `uv run pytest tests/ -q` 全绿
