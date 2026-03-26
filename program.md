# AutoResearch: Spatio-Temporal Edition

This is an experiment to have the AI agent autonomously conduct Deep Learning research on Spatio-Temporal Neural Networks (like Traffic Flow Prediction).
The agent will iteratively modify `train.py`, run experiments for a specified time budget, keep improvements, and discard failures.

## 全局规范 (Global Constraints - MUST READ)

1. **中文双语注释 (Bilingual Comments)**: The agent must write ALL code comments primarily in Simplified Chinese. Professional deep learning terms should use Chinese accompanied by their English names (e.g. 损失函数 (Loss Function), 时空图卷积 (Spatio-Temporal Graph Convolution)).
2. **张量维度追踪 (Dimension Tracking)**: Every tensor manipulation (like `view`, `permute`, `reshape`, `einsum`) must be strictly annotated with a Chinese comment showing the exact dimension changes dynamically, e.g. `# [B, C, H, W] -> [B, 512]` to avoid shape hallucinations.

## Phase 1: Setup & Initialization

To set up a new experiment, you must interact with the user to do the following:

1. **理解创新点建议 (Understand Innovation Prompt)**: Ask the user about the core methodological innovation they want to explore (e.g. Liquid Neural Networks, Time-Space Attention, State Space Models, etc.).
2. **构建基座代码 (Construct Initial `train.py`)**: Based on the user's requirement, write the initial `train.py` from scratch.
   - You MUST wrap the user's core innovation in a clearly marked block, class, or function with a loud comment marking it as the "【创新点保护区 (Innovation Protected Zone)】".
   - **Constraint**: During the later autonomous experiment loop, you are FORBIDDEN from deleting this protected module or fundamentally changing its core mechanism. You can only tune its hyperparameters, adjust the surrounding feature-extraction layers, or change the optimizer/training loop.
3. **验证数据 (Verify Data)**: Run `python3 prepare.py` if not already run. This will auto-download the PeMS data and build cached matrices.
4. **初始化结果日志 (Initialize `results.tsv`)**: Create a `results.tsv` file with the header: `commit\tval_mae\tval_rmse\tpeak_vram_mb\tstatus\tdescription`.
5. **创建 Git 分支 (Create Branch)**: Propose a tag, create the branch from master, and ask the user if you should kick off the autonomous experimentation loop.

## Phase 2: The Autonomous Experimentation Loop

Each experiment runs on a single device (CUDA, MPS, or CPU). The training script MUST run and cleanly terminate within a **fixed time budget of 15 minutes** (wall clock training time). 

Launch it simply as: `python3 train.py`.

**What you CAN do:**
- Modify `train.py` — this is the only file you edit. You can change architectures outside the protected zone, adjust hyperparameters, optimizers, learning schedules, etc.
- Adjust the number of epochs dynamically such that the script finishes in roughly 15 minutes safely.

**What you CANNOT do:**
- Modify `prepare.py`. It is read-only and handles the sliding window DataLoaders and the fixed objective metric evaluation (MAE evaluator).
- Delete or replace the user's Core Innovation Module defined in Phase 1.

**The Goal: Get the lowest `val_mae`.** 
Since the time budget is 15 minutes, you must structure `train.py` to train as efficiently as possible within this limit and achieve the best predictive generalization.

## Output Format

Once the `train.py` script finishes, it must print a summary:

```
---
val_mae:          15.4200
val_rmse:         22.1400
training_seconds: 900.1
total_seconds:    905.9
peak_vram_mb:     4506.2
num_steps:        1500
num_params_M:     5.3
```

You can extract the metrics during the loop using `grep "^val_mae:" run.log`.

## Logging and Plotting

After every single run, log it to `results.tsv` (tab-separated):
```
commit	val_mae	val_rmse	peak_vram_mb	status	description
```

**MANDATORY OUTPUT**: You MUST write code in `train.py` or a helper script that, at the end of every successful loop, generates a data visualization plot `progress.png`. This plot should visually trace the `val_mae` descending trend against commits based on `results.tsv`. **This is unequivocally required.**

## The Step-by-Step Flow 

LOOP FOREVER:
1. Tune `train.py` with a new experimental architectural idea or hyperparameter sweep.
2. `git commit -am "experiment: <short description>"`
3. Run: `python3 train.py > run.log 2>&1`
4. Read results: `grep "^val_mae:" run.log`
5. If crash/OOM: Try fixing it once. If fundamentally broken, log status `crash` in tsv and move on (Reset changes).
6. If `val_mae` is LOWER than baseline: Record improvements, KEEP changes, advance the branch.
7. If `val_mae` is HIGHER or equal (or `val_mae` is inf/nan): `git reset --hard HEAD^`, Discard changes.

**NEVER STOP**: Once the loop starts, execute it indefinitely while the human sleeps. Never ask "should I continue?". If you run out of obvious ideas, try radical structural sweeps outside the protected zone.
