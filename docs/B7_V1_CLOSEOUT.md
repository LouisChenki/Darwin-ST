# B7 v1 收关说明（路由失效定性）

日期：2026-08-20 ｜ RUN_COMMIT：`a7cda24` ｜ tag：`b7-v1-routing-failure-18.476`

## 一句话结论

B7 v1 期间长跑基建经受了多次断电/关机恢复的考验（恢复后均正常续跑），
但 **aux（训练方式创造）通道路由失效：activation = 0**。本轮 best 18.476 不能作为
辅助任务机制有效或无效的证据——该机制从未真正上场。

## 关键数字

- best val MAE = **18.47564697265625**（PeMS04，总评估 550，masked metric，12 horizons 平均，split 6/2/2）
- 创造履历记录 38 条（creation_archive_b7v1.jsonl）。
  **注意：一次创造动作可写入多条履历记录（多假设成败各一条），38 条记录 ≠ 38 次创造动作；
  动作粒度的精确统计自 v2 动作账本（tier2_actions.jsonl）起可用，本轮不做追溯换算。**
- aux_attempts = 0；insights = 0（反思环节因游标未持久化 + watchdog 重启而从未触发）
- refine:best_shot 集中于 ScaleGatedGCNExpertMixer 一族（精炼至 v20）

## 代码版本披露

tag 指向本地 commit `a7cda24`。**服务器 repo 的 git HEAD 实际停在 `4290db0`**，
运行代码是经 rsync 覆盖工作树部署的；抽样 5 个改动文件的 SHA-256 与本地 `a7cda24`
一致（唯一例外 `leaderboard/README.md`，为榜单导出物，不影响运行逻辑）。
即：`a7cda24` 是运行内容的来源 commit，服务器 git 元数据未同步推进。

## 三个已定位的失效点（v2 修复对象）

1. **反思环节（F1）**：游标是进程内存快照，watchdog 重启即重置，未满 20 条创造履历的增量被永久跳过；
   叠加计数口径是创造履历（约占评估 5.7%），单次进程存活期内数学上不可能攒够。
2. **aux 路由（F2）**：检索打分天然压死掩码族卡（前提词供给断裂 + 子模选择 tie-break），
   且 refine-first 调度使配额放在 create 内部也拿不到执行机会。
3. **精炼垄断（F3）**：`_pick_refine_candidate` 纯贪心，最强家族恒满足候选条件且排第一。

## 资产冻结

原始证据（db / 创造履历 / direction_state / 日志 / 动态算子目录 / SHA-256 manifest）冻结于
`frozen-runs/b7v1/`（不入 git）；checkpoint（1.0G）留服务器数据盘，哈希已入服务器侧 manifest。
服务器侧原始文件在 v2 上线前保持不动。
