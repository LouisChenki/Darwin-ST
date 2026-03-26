"""
Spatio-Temporal AutoResearch Base Scaffold.
Usage: python3 train.py
"""

import os
import time
import math
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

from prepare import TIME_BUDGET, load_data, evaluate_mae

# ---------------------------------------------------------------------------
# 【创新点保护区 (Innovation Protected Zone)】
# ---------------------------------------------------------------------------
# 智能体注意 (Agent Note):
# 这个模块包含用户的核心创新点（Phase 1 自动生成）。
# 在 Phase 2 的自动研究循环中，您【不可】删除或本末倒置此处的前向核心推理逻辑与理念。
# 但您可以任意向外扩展层级、更改通道数量、或者在外部空间包裹新的特征提取架构。

class SpatioTemporalModule(nn.Module):
    """
    一个用于初始化的结构占位符 (An initialization placeholder).
    此结构由智能体在 Phase 1 阶段基于用户的需求（如加入液态神经网络 LNN，或图注意力 GAT）直接重写生成。
    """
    def __init__(self, in_channels, out_channels, num_nodes, seq_len_in, seq_len_out):
        super().__init__()
        self.seq_len_out = seq_len_out
        self.num_nodes = num_nodes
        
        # 极简的全连接作为平替 (Minimal Flatten & Linear Baseline)
        hidden_dim = 64
        self.fc1 = nn.Linear(seq_len_in * num_nodes * in_channels, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, seq_len_out * num_nodes)
        
    def forward(self, x):
        # x 形状追踪 (Dimension Tracking): [B, T_in, N, C]
        B = x.shape[0]
        # [B, T_in, N, C] -> [B, T_in * N * C]
        x_flat = x.view(B, -1) 
        
        h = torch.relu(self.fc1(x_flat))
        
        # [B, Hidden] -> [B, T_out * N]
        out = self.fc2(h)
        
        # [B, T_out * N] -> [B, T_out, N]
        out = out.view(B, self.seq_len_out, self.num_nodes)
        return out

# ---------------------------------------------------------------------------
# 周边自由探索网络 (Free Exploration Zone)
# ---------------------------------------------------------------------------

class AutoResearchModel(nn.Module):
    def __init__(self, in_channels=3, num_nodes=307, seq_in=12, seq_out=12):
        super().__init__()
        # 模型主体被嵌套，智能体可以在外部随心所欲增加各种归一化、残差或者新型网络结构
        self.core = SpatioTemporalModule(in_channels, 1, num_nodes, seq_in, seq_out)
        
    def forward(self, x):
        # x: [B, T_in, N, C_in]
        return self.core(x)

# ---------------------------------------------------------------------------
# 工具与日志生成 (Utilities & Logging)
# ---------------------------------------------------------------------------

def generate_progress_plot(tsv_path="results.tsv", out_path="progress.png"):
    """基于 results.tsv 绘制优化曲线 (Plot MAE descent curve)"""
    if not os.path.exists(tsv_path):
        return
    try:
        df = pd.read_csv(tsv_path, sep='\t')
        valid_df = df[df['status'] == 'keep']
        if len(valid_df) == 0:
            return
            
        plt.figure(figsize=(10, 5))
        plt.plot(range(len(valid_df)), valid_df['val_mae'], marker='o', linestyle='-', color='indigo', linewidth=2)
        plt.title('Validation MAE Progress Over Auto-Research Generational Steps', fontsize=14)
        plt.xlabel('Successful Generational Commits (Kept)', fontsize=12)
        plt.ylabel('Validation MAE', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"生成的进度曲线图保存至 (Progress plot saved to) {out_path}")
    except Exception as e:
        print(f"生成进度图失败 (Failed to generate progress plot): {e}")

# ---------------------------------------------------------------------------
# 主训练脚本 (Main Training Loop)
# ---------------------------------------------------------------------------

def main():
    # 多端设备自适应 (Hardware Agnostic Device Selection)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')
    print(f"使用计算设备 (Using Device): {device}")

    # 超参数设置 (Hyperparameters) - 智能体可以任意修改此区域
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    WEIGHT_DECAY = 1e-4
    MAX_EPOCHS = 500  # 提供远超能力的 Epoch 以确保纯靠 Time Budget 被中断
    
    model = AutoResearchModel(in_channels=3, num_nodes=307, seq_in=12, seq_out=12)
    model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"初始化模型参数总数 (Model Parameters): {num_params:,}")

    # 支持探索优化器 (Optimizers exploration)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 损失函数 (Loss Function) 默认采用 MAE (L1 Loss) 以对齐评估指标
    criterion = nn.L1Loss() 
    
    # 启用混合精度加速 (AMP) 支持
    use_amp_cuda = ('cuda' in str(device))
    use_amp_mps = False # 关闭 MPS mixed precision 以避免特定版本的不支持警告
    scaler = torch.amp.GradScaler(device.type) if use_amp_cuda else None

    # 调用提前编译的数据模块 (Invoke dataloader)
    train_loader = load_data("train", BATCH_SIZE, device)
    
    total_training_time = 0.0
    global_step = 0
    t_start = time.time()
    
    print("开始时空自动化研究实验流水线 (Commencing Spatio-Temporal Experiment Pipeline)...")
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for x, y in train_loader:
            t_batch_start = time.time()
            
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            
            # 引入通用防御性编程混合精度架构 (Defensive Mix Precision)
            if use_amp_cuda:
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
                    preds = model(x)
                    loss = criterion(preds.float(), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            elif use_amp_mps:
                with torch.amp.autocast(device_type='mps', dtype=torch.bfloat16):
                    preds = model(x)
                    loss = criterion(preds.float(), y)
                loss.backward()
                optimizer.step()
            else:
                preds = model(x)
                loss = criterion(preds, y)
                loss.backward()
                optimizer.step()
                
            loss_val = loss.item()
            if math.isnan(loss_val) or math.isinf(loss_val):
                print("FAIL: Loss Exploded to NaN or Inf.")
                exit(1)
                
            batch_time = time.time() - t_batch_start
            total_training_time += batch_time
            global_step += 1
            
            if global_step % 50 == 0:
                print(f"Epoch {epoch} | Step {global_step} | Loss: {loss_val:.4f} | Time Used: {total_training_time:.1f}s / {TIME_BUDGET}s")
                
            if total_training_time >= TIME_BUDGET:
                break
                
        if total_training_time >= TIME_BUDGET:
            print("达到强制时间预算上限 (Forced time budget reached). 停止训练序列流.")
            break

    total_seconds = time.time() - t_start
    print("\n--- 训练进程关闭 (Training Closed), 启动验证 (Commencing Verification) ---")
    
    # 评测验证集表现获取绝对客观的指标反馈
    val_mae, val_rmse = evaluate_mae(model, BATCH_SIZE, device)
    
    peak_vram_mb = 0.0
    if device.type == 'cuda':
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    elif device.type == 'mps':
        peak_vram_mb = torch.mps.driver_allocated_memory() / 1024 / 1024

    print("---")
    print(f"val_mae:          {val_mae:.6f}")
    print(f"val_rmse:         {val_rmse:.6f}")
    print(f"training_seconds: {total_training_time:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
    print(f"num_steps:        {global_step}")
    print(f"num_params_M:     {num_params / 1e6:.6f}")

    # 调用内置日志画图生成
    generate_progress_plot()

if __name__ == "__main__":
    main()
