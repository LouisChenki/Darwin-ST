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
