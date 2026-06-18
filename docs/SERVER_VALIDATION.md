# 服务器实跑验证报告 (2026-06-17)

> 分支 feat/p0-evaluation-foundation @ e6d3670 · 服务器 4×RTX5090 / Python3.12 / torch2.8.0+cu128

## ✅ 验证通过项

| 项 | 结果 |
|---|---|
| pytest 全套 (Linux 真环境) | **112/112 全绿** (与 macOS 一致, 证明设备无关逻辑正确) |
| 4× RTX 5090 可见 | ✓ `torch.cuda.device_count()==4` |
| 真实数据集准备 PeMS04 | ✓ npz+distance.csv 下载、邻接矩阵(307,307)、6/2/2 切分(train10181/val3393/test3395) |
| GPU 端到端冒烟 | ✓ load_split→GPU训练→evaluate→masked指标 闭环正常, 指标有限合理 |
| 镜像加速 (新修) | ✓ GITHUB_MIRROR=gh-proxy.com 实测 4MB/s (直连仅 8KB/s) |

## 🐛 发现的潜在问题

### P-1 [已修复] 服务器拉 GitHub 数据集极慢
- 现象: 直连 GitHub raw ~8KB/s, PeMS04(14MB)下了 13 分钟未完。
- 修复: 新增 `GITHUB_MIRROR` 环境变量 (commit e6d3670), 设 `https://gh-proxy.com/` 后 4MB/s。
- 状态: ✅ 已修复并验证。

### P-2 [已修复·高优先级] 缓存默认落在系统盘 (30GB), 非数据盘 (50GB)
- 现象: `~/.cache/darwin-st` → `/root/.cache` 在 30GB 系统盘; PeMS04 单数据集处理后占 **986MB**。
- 风险: 4 数据集 + P2 大量实验检查点会撑爆系统盘 (仅剩 29GB)。
- 修复: CACHE_DIR 支持 `DARWIN_ST_CACHE` 环境变量覆盖 (默认仍 ~/.cache)。
  服务器用法: `export DARWIN_ST_CACHE=/root/autodl-tmp/darwin-st-cache`。
- 状态: ✅ 已修复 (+1 测试)。

### P-3 [低优先级·体验] `python -m darwin_st.data.prepare` 的 RuntimeWarning
- 现象: `'darwin_st.data.prepare' found in sys.modules ...` runpy 警告。
- 原因: 包 __init__ 导入了 prepare, 再用 -m 执行触发 runpy 重复导入告警。无害。
- 方案: 提供独立 CLI 入口 (如 scripts 或 console_scripts), 或文档改用函数调用。
- 状态: ⬜ 可延后 (不影响功能)。

### P-4 [观察项] 处理后 npy 体积偏大
- 现象: PeMS04 的 train_x `[10181,12,307,3]` float32 ≈ 450MB 单文件。
- 影响: 磁盘 + load 内存。P2 多数据集时需留意。
- 方案: 暂不优化 (M4 Max/服务器内存充足); 若紧张可改 float16 存储或按需窗口化。
- 状态: ⬜ 观察, 暂不动。

## 结论
P0/P1 在服务器真实环境验证**通过**。P-1 已修。P-2 需在进入 P2 前修掉(避免撑爆盘)。P-3/P-4 可延后。

## P2-c 基线验证 (2026-06-18, HEAD 5e8c92f)

genotype→builder→真实 PeMS04→GPU 训练→masked 评测 **整条链贯通**:
- 2-block (adaptive 图 + tcn + 节点嵌入, hidden=64), 15 epoch ~40s
- 训练 loss 持续下降 (0.19→0.17), test MAE=26.6 (符合预期: 未调优冒烟基线, 验证管道非精度)

观察到的**待 P2 后续解决**的现象 (非 bug, 记录):
- val_MAE 在 27-31 震荡未稳定下降 (train_loss 在降) → 训练不稳, LR 可能偏高、缺早停取最优。
  这正是 HPO(Optuna 调 LR)+ 多 seed + best-checkpoint 要解决的。当前不修。

服务器实跑补充经验:
- `nohup bash -c` **不加 -l 则 conda PATH 不载入** → python not found。
  服务器后台跑必须 `nohup bash -lc "..."` 或用 `/root/miniconda3/bin/python` 全路径。
- 长 git fetch 在后台 SSH 会被截断 → 服务器 git 操作也要 detached + 轮询。

## P2 点火验证 (2026-06-18, HEAD 609bd94)

scripts/run_autoresearch.py 在真实 PeMS04 + 4×RTX5090 跑通完整自治循环:
- 3 轮 × 4 架构(=12 评估), 907s, **0 崩溃**, 程序化 max_rounds 正常停止。
- 4 卡全程并行工作; 试验逐条落 memory.db; 进化+HPO+scheduler+archive+graveyard 全链协作。
- **机器全部验证通过**: 自治循环确实自己转、永不暂停、程序化停止。

**发现的问题 (待修, 不是 bug 是标定)**:
- 短 epoch(12)冒烟下 12 个架构 MAE 23-37, 全部劣于 PeMS04 最差基线(ASTGCN 22.93)→
  **全判 DISCARD, archive 空, best=inf**。后果: 进化拿不到 KEEP 信号、archive 不积累。
- 根因: DISCARD 阈值用"劣于已发表最差基线"假设了充分训练; 短训练下不成立。
- 修法: DISCARD 阈值改为**相对/自适应**(如相对当前 best 的退化幅度, 或冷启动期一律 KEEP
  让 archive 先积累), 而非绝对基线。正式跑(长 epoch)时绝对阈值才合理。

**额外教训**: git fetch 也被 GitHub 限流 → `git remote set-url origin https://gh-proxy.com/https://github.com/...` 走镜像; python 输出 piped 全缓冲 → sys.stdout.reconfigure(line_buffering)(已修), 或改看 memory.db 实时进度。

## P2 点火复跑成功 (2026-06-18, HEAD 25eb4e4, 修复后)

warmup+相对退化标定修复后, 同配置(3轮×4架构, 12 epoch)复跑:
- **结束: max_rounds(3) 980s, 12 KEEP / 0 DISCARD / 0 CRASH, 程序化停止。**
- **archive 填充 5/36 (coverage 0.14, QD 0.180)** —— 修复生效, 进化拿到 KEEP 信号。
- 进化逐轮改进: best MAE 25.59 → 24.51 → 24.51(3轮太短)。
- 最优架构: depth=1 hidden=64 adj=sym **emb_node=True** [(adaptive,tcn,residual)] ——
  带节点嵌入, 符合研究指明方向。
- gap vs SOTA(STD-MAE 17.8) = +6.7(短训练冒烟, 正式跑需 max_epochs↑ + 轮数↑ + 多seed)。

**整套 P2 自治系统在真实数据+4卡上验证通过**: 进化NAS+HPO+并行调度+MAP-Elites档案+
结构化记忆+graveyard+程序化自治 全链路真实跑通、永不暂停、自己收敛。

**正式冲SOTA建议**(后续): max_epochs 80-100、max_rounds 数十、POP_SIZE 20、多seed平均、
开 time_budget 熔断; 由 orchestrator 长跑(可 /loop 或 nohup 24/7)。
