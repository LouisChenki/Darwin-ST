"""离线反思巩固 (Reflection Consolidation bootstrap) —— 对历史 db 直接跑反思。

把积压的创造履历 (creation_archive.jsonl) 一次性反思蒸馏进 memory insights 表:
LLM 产 ADD/EDIT/UPVOTE/DOWNVOTE 操作 → 校验 → 落库 → Hermes 容量淘汰 → 打印结果。
反思产出的条件式教训随后会被 run_creation_loop 的合成/诊断 prompt 回注 (经验块)。

用法 (服务器, 需先 source secrets 拿 DEEPSEEK_API_KEY):
    source /root/autodl-tmp/darwin-secrets.env
    MEMORY_DB=/root/autodl-tmp/cache/memory_creation.db \
        python scripts/reflect_consolidate.py
    # 连续跑 3 批 (断点重试/增量消化):
    REFLECT_ROUNDS=3 MEMORY_DB=... python scripts/reflect_consolidate.py

环境变量 (命令行同名参数优先):
    MEMORY_DB          必需, memory SQLite 库路径
    CREATION_ARCHIVE   履历本 jsonl, 默认 db 同目录 creation_archive.jsonl
    REFLECT_ROUNDS     连续 force 批次数, 默认 1 (游标推进后无增量自动提前退出)
    REFLECT_EVERY      仅影响日志语义, force 模式必触发 (默认 20, 透传 ReflectionConfig)
    INSIGHTS_CAPACITY  insights 表容量上限 (默认 50)
    DEEPSEEK_API_KEY   必需 (OpenAICompatLLM 从 env 读, 同 run_creation_loop)
    DEEPSEEK_MODEL     反思模型 (默认 deepseek-v4-pro, 可切 flash 省成本)

退出码: 0 正常; 2 参数/环境缺失; 1 反思批次全部失败 (LLM/解析)。
"""

from __future__ import annotations

import argparse
import os
import sys

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass


def _env_int(k: str, d: int) -> int:
    v = os.environ.get(k)
    return int(v) if v else d


def main() -> int:
    ap = argparse.ArgumentParser(description="离线反思巩固: 历史履历 → insights 语义记忆层")
    ap.add_argument("--memory-db", default=os.environ.get("MEMORY_DB", ""),
                    help="memory SQLite 库路径 (env MEMORY_DB)")
    ap.add_argument("--creation-archive", default=os.environ.get("CREATION_ARCHIVE", ""),
                    help="履历本 jsonl (默认 db 同目录 creation_archive.jsonl; env CREATION_ARCHIVE)")
    ap.add_argument("--rounds", type=int, default=_env_int("REFLECT_ROUNDS", 1),
                    help="连续 force 批次数 (env REFLECT_ROUNDS, 默认 1)")
    args = ap.parse_args()

    if not args.memory_db:
        print("错误: 必须给 MEMORY_DB (--memory-db 或环境变量)", file=sys.stderr)
        return 2
    if not os.path.exists(args.memory_db):
        print(f"错误: MEMORY_DB 不存在: {args.memory_db}", file=sys.stderr)
        return 2
    archive_path = args.creation_archive or os.path.join(
        os.path.dirname(os.path.abspath(args.memory_db)), "creation_archive.jsonl")
    if not os.path.exists(archive_path):
        print(f"错误: 履历本不存在: {archive_path} (CREATION_ARCHIVE 可覆盖)", file=sys.stderr)
        return 2

    from darwin_st.creation.creation_archive import CreationArchive
    from darwin_st.creation.llm import OpenAICompatLLM
    from darwin_st.creation.reflection import ReflectionConfig, ReflectionLoop
    from darwin_st.memory.store import MemoryStore

    cfg = ReflectionConfig(reflect_every_records=_env_int("REFLECT_EVERY", 20),
                           insights_capacity=_env_int("INSIGHTS_CAPACITY", 50))
    mem = MemoryStore(args.memory_db)
    archive = CreationArchive(archive_path)
    n_records = len(archive._load())
    print(f"=== 离线反思巩固 ===")
    print(f"db={args.memory_db}\n履历本={archive_path} (共 {n_records} 条, "
          f"游标从 0 起全量反思)")
    print(f"模型={os.environ.get('DEEPSEEK_MODEL', 'deepseek-v4-pro')} "
          f"容量={cfg.insights_capacity} 每批ADD上限={cfg.max_new_lessons}")
    print(f"反思前 insights: {len(mem.list_insights_by_confidence())} 条")

    try:
        llm = OpenAICompatLLM()          # key 走 DEEPSEEK_API_KEY; DEEPSEEK_MODEL 可切
    except RuntimeError as e:
        print(f"错误: {e}", file=sys.stderr)
        return 2

    loop = ReflectionLoop(mem, archive, llm, cfg, cursor=0)  # 离线 bootstrap: 从头全量反思

    n_ok = n_fail = 0
    for i in range(max(args.rounds, 1)):
        rep = loop.maybe_reflect(force=True)
        if rep is None:
            print(f"[批 {i + 1}] 无新增履历, 提前结束")
            break
        if "error" in rep:
            n_fail += 1
            print(f"[批 {i + 1}] 失败: {rep['error']} (游标不动, 可重跑重试)")
            continue
        n_ok += 1
        print(f"[批 {i + 1}] 新履历 {rep['n_new_records']} 条 → 操作 {rep['n_ops']} 个, "
              f"接受 {rep['accepted']}, 拒绝 {len(rep['rejected'])}")
        for r in rep["rejected"]:
            print(f"    拒: {r.get('op')} —— {r.get('reason')}")
        st = rep["apply"]
        print(f"    落库: 新增{st['added']} 改{st['edited']} 赞{st['upvoted']} "
              f"踩{st['downvoted']} 删{st['deleted']}; 容量淘汰 {rep['evicted']}")

    insights = mem.list_insights_by_confidence()
    print(f"\n=== 最终 insights 表 ({len(insights)} 条, 按置信度降序) ===")
    for ins in insights:
        ev = ",".join(f"#{e}" for e in ins["evidence_ids"])
        print(f"  [{ins['confidence']:.2f}] (id={ins['id']}) {ins.get('condition') or '通用'}"
              f" → {ins['insight_text'][:110]}" + (f"  (依据 {ev})" if ev else ""))
    mem.close()
    print(f"\n完成: 成功批 {n_ok}, 失败批 {n_fail}")
    return 0 if n_ok > 0 or n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
