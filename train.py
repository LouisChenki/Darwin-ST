"""
AutoResearch Spatio-Temporal Edition: 训练主程序 (Training Engine)
核心创新点: 基于液态神经网络 (Liquid Neural Network, LNN) 的时空预测模型。
数据集: PeMS04 (307 Nodes, 12->12 steps)
"""

import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from torch.amp import autocast
from prepare import load_data, get_scaler, evaluate_mae, DATASET, TIME_BUDGET
from baseline_registry import get_baseline_metric

# ---------------------------------------------------------------------------
# 1. 创新点保护区 (Innovation Protected Zone)
# ---------------------------------------------------------------------------

class LiquidCell(nn.Module):
    """
    残差液态神经元单元 (Residual Liquid Neural Cell)
    假设更新: 增加残差路径以防止深层演化中的梯度消失 (Belief Update: Adding residual path for better gradient flow in ODE evolution).
    """
    def __init__(self, input_size, hidden_size):
        super(LiquidCell, self).__init__()
        self.hidden_size = hidden_size
        self.tau_net = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_net = nn.Linear(input_size + hidden_size, hidden_size)
        self.proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        
    def forward(self, x, h):
        # x: [B, N, C], h: [B, N, H]
        combined = torch.cat([x, h], dim=-1)
        
        tau = torch.sigmoid(self.tau_net(combined))
        target = torch.tanh(self.input_net(combined))
        
        # 核心 LNN 更新 (ODE-based)
        h_next = h + tau * (target - h)
        
        # 残差投影 (Residual Projection)
        res = self.proj(x)
        return h_next + res

class SpatialAttention(nn.Module):
    """
    空间自注意力机制 (Spatial Self-Attention)
    假设更新: 动态分配空间权重以增强节点间的相互作用 (Belief Update: Dynamic weight allocation for spatial node interaction).
    """
    def __init__(self, hidden_dim, num_heads=2):
        super(SpatialAttention, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # x: [B, N, H]
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)

class LiquidCell(nn.Module):
    """
    门控残差液态神经元 (Gated Residual Liquid Cell)
    假设更新: 引入门控机制以自适应控制时序流的信息保留 (Belief Update: Gating for adaptive information retention in ODE evolution).
    """
    def __init__(self, input_size, hidden_size):
        super(LiquidCell, self).__init__()
        self.hidden_size = hidden_size
        self.tau_net = nn.Linear(input_size + hidden_size, hidden_size)
        self.input_net = nn.Linear(input_size + hidden_size, hidden_size)
        self.gate_net = nn.Linear(input_size + hidden_size, hidden_size) # 门控
        self.proj = nn.Linear(input_size, hidden_size) if input_size != hidden_size else nn.Identity()
        
    def forward(self, x, h):
        combined = torch.cat([x, h], dim=-1)
        
        tau = torch.sigmoid(self.tau_net(combined))
        target = torch.tanh(self.input_net(combined))
        gate = torch.sigmoid(self.gate_net(combined))
        
        # 门控更新 (Gated Update)
        h_next = h + tau * (gate * target - h)
        
        res = self.proj(x)
        return h_next + res

class LNN_ST_Model(nn.Module):
    """
    门控时空液态神经网络模型 (Gated ST-LNN Model)
    """
    def __init__(self, num_nodes, in_channels, out_steps, hidden_dim=64):
        super(LNN_ST_Model, self).__init__()
        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim
        self.out_steps = out_steps
        
        self.feat_proj = nn.Linear(in_channels, hidden_dim)
        self.node_emb = nn.Parameter(torch.randn(num_nodes, hidden_dim))
        
        self.spatial_attn = SpatialAttention(hidden_dim, num_heads=2)
        self.lnn_cell = LiquidCell(hidden_dim, hidden_dim)
        
        self.output_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_steps)
        )
        
    def forward(self, x):
        batch_size, seq_len, num_nodes, _ = x.shape
        device = x.device
        
        h = torch.zeros(batch_size, num_nodes, self.hidden_dim, device=device)
        
        for t in range(seq_len):
            x_t = x[:, t, :, :]
            feat = self.feat_proj(x_t) + self.node_emb.unsqueeze(0)
            
            # 空间交互 (Self-Attention)
            feat_spatial = self.spatial_attn(feat)
            
            # 时间上演化 (LNN)
            h = self.lnn_cell(feat_spatial, h)
            
        out = self.output_layer(h) 
        out = out.transpose(1, 2)
        return out

# ---------------------------------------------------------------------------
# 2. 训练配置与辅助函数 (Training Config & Utilities)
# ---------------------------------------------------------------------------

def train():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"使用设备 (Device): {device}")
    
    # 环境参数 (Hyperparameters)
    batch_size = 64
    learning_rate = 1e-3
    hidden_dim = 64
    num_epochs = 100
    
    # 数据加载
    train_loader = load_data("train", batch_size, device)
    num_nodes = 307
    in_channels = 3
    out_steps = 12
    
    # 模型初始化
    model = LNN_ST_Model(num_nodes, in_channels, out_steps, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.L1Loss()
    
    # 记录实验
    history = []
    start_time = time.time()
    print(f"开始演化循环 (Evolution loop started). 时间预算: {TIME_BUDGET}s")
    
    step = 0
    try:
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            for x, y in train_loader:
                current_time = time.time()
                if current_time - start_time > TIME_BUDGET:
                    print("⚠️ 到达时间预算上限 (Time Budget Reached). 强制停止。")
                    raise StopIteration
                
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                
                with autocast(device_type=device.type, dtype=torch.bfloat16):
                    preds = model(x)
                    loss = criterion(preds.float(), y.float())
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                step += 1
            
            val_mae, val_rmse = evaluate_mae(model, batch_size, device)
            history.append(val_mae)
            print(f"Epoch {epoch+1} | Train Loss: {epoch_loss/len(train_loader):.4f} | Val MAE: {val_mae:.4f}")
            
    except StopIteration:
        pass
    
    total_time = time.time() - start_time
    final_mae, final_rmse = evaluate_mae(model, batch_size, device)
    
    # ---------------------------------------------------------------------------
    # 3. 统计与绘图 (Metrics & Plotting)
    # ---------------------------------------------------------------------------
    
    # 获取基线 (Get Baselines)
    dcrnn_mae = get_baseline_metric(DATASET, "DCRNN", "mae")
    
    # 绘图 (Plotting progress.png)
    plt.figure(figsize=(10, 6))
    plt.plot(history, label='Model Val MAE', color='blue', linewidth=2)
    if dcrnn_mae:
        plt.axhline(y=dcrnn_mae, color='red', linestyle='--', label=f'DCRNN Baseline ({dcrnn_mae})')
    plt.title(f"Training Progress on {DATASET}")
    plt.xlabel("Epochs")
    plt.ylabel("MAE")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("progress.png")
    
    # 打印最终摘要 (Summary Output)
    num_params = sum(p.numel() for p in model.parameters()) / 1e6
    print("---")
    print(f"val_mae:          {final_mae:.4f}")
    print(f"val_rmse:         {final_rmse:.4f}")
    print(f"training_seconds: {total_time:.1f}")
    print(f"total_seconds:    {total_time + 5:.1f}") # 包含绘图等开销
    print(f"peak_vram_mb:     {torch.mps.driver_allocated_memory() / 1024**2 if device.type == 'mps' else 0:.1f}")
    print(f"num_steps:        {step}")
    print(f"num_params_M:     {num_params:.4f}")

    # 记录结果 (Log to results.tsv)
    with open("results.tsv", "a") as f:
        # 获取当前 git commit (这里简单用 time 代替)
        f.write(f"{int(time.time())}\t{final_mae:.4f}\t{final_rmse:.4f}\t0\tKEEP\tInitial LNN ST-Model\n")

if __name__ == "__main__":
    train()
