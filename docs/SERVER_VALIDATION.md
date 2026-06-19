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

## P2.5-e Tier-2 创造闭环端到端实跑 (2026-06-19)

**配置**: PeMS04, 4×GPU, 弱基线gcn+tcn, pop=6, max_rounds=6, HPO=2, epochs=8, stagnation=2。

**验证通过(闭环鲁棒性)**:
- ✅ 优化引擎真实训练: round1 best MAE=33.4 → round3=30.1(进化在降误差)。
- ✅ 停滞**正确触发创造**: 诊断瓶颈"长程依赖不足"→跨域检索拉回dilated_causal_convolution(SSM域)→DeepSeek融合计划(膨胀卷积并行分支+零初始化残差, 理由专业)→Aider写代码。
- ✅ **验证守卫工作**: 第一版算子输出恒等输入被门trivial_identity拒, 触发重试。
- ✅ **自主性铁律**: 创造失败(3次重试未过验证)**不中止循环**, 继续进化round3降了MAE。4卡全程可用, memory落库正常。

**发现的真问题(待优化, 非bug)**:
- ⚠️ **创造延迟过高**: 单次创造~10分钟(每次重试=完整Aider+DeepSeek往返~1-2min × max_retries=3 + 验证)。多次停滞触发多次慢创造, 整体太慢。
- ⚠️ **合成成功率不稳**: 观察到的创造尝试多次未过验证(trivial_identity等)。deepseek-v4-pro写的算子质量波动。
- 根因: ①Aider每次重试都全量重跑(慢)②验证反馈循环长③弱基线下DeepSeek倾向产平凡解。

**优化方向(后续)**:
- 降 max_retries(如2)或加快验证反馈; 创造异步化(不阻塞进化主循环, 后台合成完再入队);
- prompt 强化"非平凡"约束 + few-shot 好例子; 或换更快模型做初版、pro做精修(AlphaEvolve式model ensemble);
- 限制创造频率(如每N轮最多一次), 避免慢创造拖垮节奏。

**结论**: 创造闭环在真实环境**机制全通、鲁棒(失败不崩)**, 但**创造的速度与成功率需调优**才实用。这是真实实跑才能发现的工程问题, 已记录待优化。

## P2.5-g 多假设创造闭环实跑 (2026-06-19) —— 决定性成功

**配置**: 同P2.5-e但用多假设(N_HYPOTHESES=3, max_retries=2)。

**决定性成功(对比上次单假设完全失败)**:
- ✅ **创造成功**: round3触发创造, 3个假设【全部】过验证并注入(上次单假设0注入)。
- ✅ **真实评测+竞争**: 7+个含synth算子的试验真实训练, MAE 29.46~35.85, 差的被淘汰好的留下(评测器当裁判)。最优合成算子MAE=29.46 = 当前全局最优(一个跨域融合算子打平进化最好结果)。
- ✅ **多假设多样性**: 共注入9个跨域融合算子(残差融合/门控调制/门控路由/并行多尺度/序列预处理...组合方式各异), 正是"单Generator生N个"的多样性体现。
- ✅ 进化降MAE 31.4→29.46, 4卡正常, 持续触发创造(创造次数=2+)。

**完整闭环铁证**: 跨域检索(静态机制库)→Generator生N假设→Aider写码→验证harness→注入算子库→进化+真实MAE评测→合成算子参与竞争。整条链在真实PeMS04+4GPU+DeepSeek跑通。

**残留工程问题(已知, 待优化)**: 创造延迟仍高(单次创造N假设×Aider往返~8min)。但多假设把"成功率"问题解决了(总有假设成功)。延迟优化(异步创造/model ensemble)留后续。

**结论**: 简化的多假设流程**完全验证** —— 解决了上次单假设的成功率问题, 创造的算子真正进入进化、真实评测、参与竞争, 最优合成算子打平全局最优。这是论文核心创新(LLM跨域涌现追平进化)的端到端铁证。
