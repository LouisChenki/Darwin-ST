#!/usr/bin/env bash
# 24h 冲 SOTA 看门狗 (Watchdog) —— 进程崩了自动 RESUME 续跑, 直到达 TARGET_MAE 或超时。
#
# 续跑铁律: synth 算子由 registry.load_persisted 从 dynamic_ops 续上; RESUME=1 让 base
# 从 memory 最优 KEEP 暖启动 → 崩溃重启不丢算子库也不丢进化起点。靠独立 MEMORY_DB 累积。
#
# 用法 (服务器, 在 Darwin-ST 根目录):
#   source /root/autodl-tmp/darwin-secrets.env   # 先 source key (本脚本不碰 key)
#   DEADLINE_HOURS=24 TARGET_MAE=17.80 bash scripts/watchdog_sota.sh
# 其余配置走 run_creation_loop.py 的 env (POP_SIZE/MAX_ROUNDS/.../ADAPTIVE_PATIENCE 等), 由调用方 export。
set -u

DEADLINE_HOURS="${DEADLINE_HOURS:-24}"
LOG="${WATCHDOG_LOG:-logs/sota_24h.log}"
RUN_LOG="${RUN_LOG:-logs/sota_24h_run.log}"
MEMORY_DB="${MEMORY_DB:?必须设 MEMORY_DB (独立库累积续跑)}"
TARGET_MAE="${TARGET_MAE:?必须设 TARGET_MAE (如 17.80, 程序化停)}"
mkdir -p logs

# 截止时间戳 (秒)。注: 服务器允许 date, 仅本地 Claude 环境禁 date。
START_TS=$(date +%s)
DEADLINE_TS=$(( START_TS + DEADLINE_HOURS * 3600 ))
ATTEMPT=0

log() { echo "[watchdog $(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

log "启动: deadline=${DEADLINE_HOURS}h target_mae=${TARGET_MAE} db=${MEMORY_DB}"

while :; do
  NOW=$(date +%s)
  if [ "$NOW" -ge "$DEADLINE_TS" ]; then
    log "到达 ${DEADLINE_HOURS}h 截止, 停止看门狗。"
    break
  fi

  # 程序化达标? memory 最优已破 target → 收工
  BEST=$(python -c "
import sqlite3,sys
try:
    c=sqlite3.connect('$MEMORY_DB')
    r=c.execute(\"select min(val_mae) from experiments where status='KEEP' and val_mae is not null\").fetchone()[0]
    print(r if r is not None else 999)
except Exception:
    print(999)
" 2>/dev/null)
  REACHED=$(python -c "print(1 if float('$BEST') < float('$TARGET_MAE') else 0)" 2>/dev/null)
  if [ "$REACHED" = "1" ]; then
    log "🎯 已达标! best=$BEST < target=$TARGET_MAE, 收工。"
    break
  fi

  ATTEMPT=$(( ATTEMPT + 1 ))
  # 第 2 次起 RESUME=1 (从 memory 最优暖启动); 首次也可 RESUME (库空则用弱基线, from_dict 失败兜底)
  export RESUME=1
  log "第 ${ATTEMPT} 次启动 run (best 至今=$BEST, RESUME=1)..."
  python -u scripts/run_creation_loop.py >> "$RUN_LOG" 2>&1
  RC=$?
  log "run 退出 rc=$RC (创造闭环正常结束=程序化停/max_rounds; 非0=崩溃将续跑)。"

  # 正常结束 (rc=0 且达 target) 上面已 break; 否则短歇后续跑 (防紧密崩溃刷屏)
  sleep 15
done
log "看门狗结束。最终 best=$(python -c "
import sqlite3
try:
    c=sqlite3.connect('$MEMORY_DB')
    print(c.execute(\"select min(val_mae) from experiments where status='KEEP' and val_mae is not null\").fetchone()[0])
except Exception: print('?')
" 2>/dev/null)"
