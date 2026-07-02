-- Darwin-ST 结构化试验记忆 (Structured Experiment Memory) —— SQLite schema
--
-- 把过往优化的成功与失败经验结构化记录, 供演化系统:
--   1. 程序化判定「是否超越 SOTA」(best_so_far)
--   2. 硬阻断已知失败配置, 防止无效重复优化 (query_graveyard by signature)
--   3. 最近邻经验检索, 让新变异借鉴相似历史 (nearest_experiments)
--   4. 谱系追踪, 支撑进化式 NAS 的父子变异关系 (lineage)
--   5. ExpeL 式经验蒸馏 (insights)
--
-- 设计要点 (见 docs/BLUEPRINT.md §3):
--   - genotype/hp/behavior_descriptor 用 JSON 文本存 (P2 中 genotype schema 会演化, 保持灵活)
--   - signature = genotype+hp 的稳定哈希, 用于 Graveyard 秒级查重
--   - 指标多 seed: 存 mean+std (SOTA 提升 <1%, 必须多 seed 抗噪声)

PRAGMA journal_mode = WAL;       -- 并发读写更稳 (集群多 worker 写同一库)
PRAGMA foreign_keys = ON;

-- 实验主表: 每行 = 一次完整 trial
CREATE TABLE IF NOT EXISTS experiments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_tag         TEXT    NOT NULL,            -- 实验批次/分支标识
    dataset         TEXT    NOT NULL,            -- 目标数据集 (PeMS04 等)
    space_version   TEXT,                        -- 搜索空间代际 (作用域隔离键; NULL=旧库/未知代)
    signature       TEXT    NOT NULL,            -- genotype+hp 稳定哈希 (Graveyard 查重键)
    genotype_json   TEXT    NOT NULL,            -- 架构基因型 (JSON)
    hp_json         TEXT    NOT NULL DEFAULT '{}', -- 超参 (JSON)

    -- 指标 (masked, 已 inverse-transform 真实尺度; 多 seed)
    val_mae         REAL,
    val_mae_std     REAL,
    val_rmse        REAL,
    val_mape        REAL,
    test_mae        REAL,                        -- 最终 held-out 测试(仅里程碑时算)
    num_seeds       INTEGER NOT NULL DEFAULT 1,

    -- 结局: KEEP(强化) / DISCARD(劣化) / CRASH(数值崩溃) / PRUNED(多保真早停)
    status          TEXT    NOT NULL,
    fail_reason     TEXT,                        -- Graveyard 死因: nan/oom/timeout/regression/...

    -- MAP-Elites 行为描述子 (JSON: {graph_conv, temporal, depth, params_bucket})
    behavior_descriptor TEXT NOT NULL DEFAULT '{}',

    -- 资源画像
    num_params      INTEGER,
    wall_seconds    REAL,
    peak_mem_gb     REAL,

    parent_id       INTEGER,                     -- 父代 experiment.id (谱系冗余, 便于回溯)
    commit_hash     TEXT,
    created_at      TEXT    NOT NULL,            -- ISO8601 (由调用方传入, 不用 SQLite 时间函数)

    FOREIGN KEY (parent_id) REFERENCES experiments(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_exp_signature ON experiments(signature);
CREATE INDEX IF NOT EXISTS idx_exp_dataset_status ON experiments(dataset, status);
CREATE INDEX IF NOT EXISTS idx_exp_val_mae ON experiments(dataset, val_mae);
CREATE INDEX IF NOT EXISTS idx_exp_run_tag ON experiments(run_tag);
-- 注: idx_exp_scope (含 space_version 列) 由 store._migrate() 创建, 不在此处 —— 旧库经
-- CREATE TABLE IF NOT EXISTS 是 no-op 不会加列, 若在此引用 space_version 会在迁移前崩。

-- 谱系表: 显式记录变异关系 (一条变异边 = 父 → 子, 附变异算子)
CREATE TABLE IF NOT EXISTS lineage (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    parent_id       INTEGER NOT NULL,
    child_id        INTEGER NOT NULL,
    mutation_op     TEXT,                        -- op-swap / edge-rewire / depth / hp / agent-semantic
    mutation_detail TEXT,                        -- 变异细节描述
    FOREIGN KEY (parent_id) REFERENCES experiments(id) ON DELETE CASCADE,
    FOREIGN KEY (child_id)  REFERENCES experiments(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_lineage_parent ON lineage(parent_id);
CREATE INDEX IF NOT EXISTS idx_lineage_child ON lineage(child_id);

-- 洞察表: ExpeL 式从试验历史蒸馏的条件式自然语言经验
CREATE TABLE IF NOT EXISTS insights (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset         TEXT,                        -- 适用数据集 (NULL=通用)
    condition       TEXT,                        -- 适用条件 (何种 regime 下成立)
    insight_text    TEXT    NOT NULL,            -- 洞察内容
    evidence_ids    TEXT    NOT NULL DEFAULT '[]', -- 支撑该洞察的 experiment.id 列表 (JSON)
    confidence      REAL    NOT NULL DEFAULT 0.5,
    created_at      TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_insights_dataset ON insights(dataset);
