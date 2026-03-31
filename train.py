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
from baseline_registry import get_baseline_metric

# ---------------------------------------------------------------------------
# 用户任务指令承接 (User Prompt Fulfillment)
# ---------------------------------------------------------------------------
# [AGENT CONFIG] 智能体在接收阶段 1 模板后，需将用户选定的 Baseline 和 Dataset 写入于此：
DATASET = os.environ.get("DATASET", "PeMS04")
SELECTED_BASELINES = ["DCRNN"] # 基于用户给定的对比基线
# ===========================================================================

# ---------------------------------------------------------------------------
# 【创新点保护区 (Innovation Protected Zone)】
# ---------------------------------------------------------------------------
# 智能体注意 (Agent Note):
# 这个模块包含用户的核心创新点（Phase 1 自动生成）。
# 在 Phase 2 的自动研究循环中，您【不可】删除或本末倒置此处的前向核心推理逻辑与理念。
# 但您可以任意向外扩展层级、更改通道数量、或者在外部空间包裹新的特征提取架构。

class LiquidTimeConstantNode(nn.Module):
    def __init__(self, in_features, state_size):
        super().__init__()
        self.state_size = state_size
        self.W_in = nn.Linear(in_features, state_size)
        self.W_rec = nn.Linear(state_size, state_size)
        # 初始化流体时间常数 (Liquid Time Constant)
        self.tau_inv = nn.Parameter(torch.ones(state_size) * 0.1)
        self.A = nn.Parameter(torch.ones(state_size))
        
    def forward(self, x_t, h_t, dt=1.0):
        # x_t 维度追踪 (Dimension Tracking): [B, in_features]
        # h_t 维度追踪 (Dimension Tracking): [B, state_size]
        f_val = torch.sigmoid(self.W_in(x_t) + self.W_rec(h_t))
        dh = - (torch.abs(self.tau_inv) + f_val) * h_t + f_val * self.A
        return h_t + dh * dt

class SpatioTemporalModule(nn.Module):
    """
    基于液态神经网络 (Liquid Neural Network, LNN) 的时空架构.
    利用连续时间 ODE 推演时序动态演化，适合非均匀序列建模。
    """
    def __init__(self, in_channels, out_channels, num_nodes, seq_len_in, seq_len_out):
        super().__init__()
        self.seq_len_out = seq_len_out
        self.num_nodes = num_nodes
        
        self.hidden_dim = 128
        # 输入特征数为每个节点通道叠加: in_channels * num_nodes
        self.lnn_cell = LiquidTimeConstantNode(in_channels * num_nodes, self.hidden_dim)
        self.fc_out = nn.Linear(self.hidden_dim, seq_len_out * num_nodes)
        
    def forward(self, x):
        # x 维度追踪 (Dimension Tracking): [B, T_in, N, C]
        B, T_in, N, C = x.shape
        x_flat = x.view(B, T_in, N * C)
        
        # 统一设备与精度 (Initialize hidden state)
        h_t = torch.zeros(B, self.hidden_dim, device=x.device, dtype=x.dtype)
        
        # 按时间步模拟液体系统计算流
        for t in range(T_in):
            h_t = self.lnn_cell(x_flat[:, t, :], h_t, dt=1.0)
            
        out_flat = self.fc_out(h_t)
        return out_flat.view(B, self.seq_len_out, self.num_nodes)

# ---------------------------------------------------------------------------
# 周边自由探索网络 (Free Exploration Zone)
# ---------------------------------------------------------------------------

class GraphConvolution(nn.Module):
    """
    物理图先验卷积 (Physical Prior Graph Convolution)
    能利用真实的物理邻接矩阵 A 拓宽感知域。
    """
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)
        
    def forward(self, x, adj):
        # x 形状: [B, T, N, C]
        # adj 形状: [N, N]
        B, T, N, C = x.shape
        assert adj.shape == (N, N), f"Adjacency matrix shape mismatch! Expected ({N}, {N}) but got {adj.shape}"
        
        # 张量变换: 1. 节点特征映射 (Feature Projection)
        # [B, T, N, C] @ [C, F] -> [B, T, N, F]
        support = torch.matmul(x, self.weight)
        
        # 张量变换: 2. 图物理结构信息传播 (Graph Message Passing)
        # adj: [N, N], support: [B, T, N, F] -> out: [B, T, N, F]
        out = torch.einsum('nm,btmf->btnf', adj, support)
        
        return out + self.bias

class TemporalEmbedding(nn.Module):
    """
    时间周期特征编码器 (Temporal Periodicity Encoder)
    """
    def __init__(self, d_model):
        super().__init__()
        # 取 时间戳(TOD) 和 星期(DOW) 2个维度
        self.time_proj = nn.Linear(2, d_model)
        
    def forward(self, time_features):
        # 张量变换: [B, T, N, 2] -> [B, T, N, d_model]
        return self.time_proj(time_features)

class AutoResearchModel(nn.Module):
    def __init__(self, in_channels=3, num_nodes=307, seq_in=12, seq_out=12):
        super().__init__()
        self.d_model = 16 # 使用较小维度以保证内存与计算效率 (Occam's Razor)
        
        # 将原始交通流与时间分布特征分流提取
        self.flow_proj = nn.Linear(1, self.d_model)
        self.time_emb = TemporalEmbedding(self.d_model)
        
        # 引入物理空间先验，废除无规律的全局互相关
        self.gcn = GraphConvolution(self.d_model, self.d_model)
        self.norm = nn.LayerNorm(self.d_model)
        
        # 维持底层液态神经网络 (LNN) 接稳信号
        self.core = SpatioTemporalModule(self.d_model, 1, num_nodes, seq_in, seq_out)
        
    def forward(self, x, adj):
        # 强制性防御形状验证
        B, T_in, N, C = x.shape
        assert C >= 3, f"Shape Mismatch: Models expects Flow, TOD, DOW, got {C} features!"
        
        # 张量切片分离属性 (Feature Defusing)
        flow_feat = x[..., 0:1] # [B, T_in, N, 1]
        time_feat = x[..., 1:3] # [B, T_in, N, 2]
        
        # 张量映射与融合 (Embedding & Add Fusion)
        h_flow = self.flow_proj(flow_feat) # [B, T, N, d_model]
        h_time = self.time_emb(time_feat)  # [B, T, N, d_model]
        
        # 融合周期相位特征
        h_fuse = h_flow + h_time 
        
        # 利用 Graph 物理邻接矩阵 A 扩散拓扑属性
        h_graph = torch.relu(self.gcn(h_fuse, adj))
        h_norm = self.norm(h_graph)
        
        # 移交具有时空完备洞察的信号矩阵给下游动态 LNN 层
        return self.core(h_norm)

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
            
        plt.figure(figsize=(12, 6))
        y_vals = valid_df['val_mae'].values
        x_vals = range(len(valid_df))
        plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='indigo', linewidth=2, label='AutoResearch Model')
        
        # 读取并在显著提升点绘制架构进化反思气泡 (Evolution Roadmap Annotations)
        log_path = "evolution_log.jsonl"
        evolution_data = {}
        if os.path.exists(log_path):
            import json
            with open(log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            record = json.loads(line)
                            evolution_data[record.get("commit", "")] = record
                        except:
                            pass
                            
        for i in range(1, len(valid_df)):
            prev_mae = y_vals[i-1]
            curr_mae = y_vals[i]
            commit_id = str(valid_df.iloc[i]['commit'])
            
            # 如果此轮下降具有显著意义 (例如 MAE 下滑 > 1%)，提取反思绘制科技树气泡
            if prev_mae > 0 and (prev_mae - curr_mae) / prev_mae > 0.01:
                bubble_text = f"MAE: {curr_mae:.2f}"
                if commit_id in evolution_data:
                    mut_info = evolution_data[commit_id]
                    op = mut_info.get("操作类型", "")
                    if op:
                        bubble_text += f"\n[{op}]"
                        
                plt.annotate(
                    bubble_text,
                    xy=(i, curr_mae),
                    xytext=(i, curr_mae + (prev_mae - curr_mae) * 0.5 + 0.5), # 悬浮在点上方
                    arrowprops=dict(facecolor='gray', shrink=0.05, width=1, headwidth=5, alpha=0.6),
                    bbox=dict(boxstyle="round,pad=0.4", fc="#FFF9C4", ec="#FBC02D", lw=1.5, alpha=0.9),
                    fontsize=8, zorder=10, ha='center'
                )
        
        # 将用户选取的 Baselines 以虚线水平绘制
        colors = ['r', 'g', 'c', 'orange', 'm']
        for i, baseline in enumerate(SELECTED_BASELINES):
            b_mae = get_baseline_metric(DATASET, baseline, metric='mae')
            if b_mae is not None:
                plt.axhline(y=b_mae, color=colors[i % len(colors)], linestyle='--', alpha=0.8, linewidth=1.5, label=f'{baseline} (MAE: {b_mae})')
                
        plt.title(f'Validation MAE Progress on {DATASET}', fontsize=14)
        plt.xlabel('Successful Generational Commits (Kept)', fontsize=12)
        plt.ylabel('Validation MAE', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend(loc='upper right')
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
    
    # 获取节点的动态数目 (由于不同 Dataset 节点数不同)
    try:
        import numpy as np
        # 探测随便一个 x 文件以读取维度
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", DATASET.lower())
        sample_x = np.load(os.path.join(cache_dir, "train_x.npy"), mmap_mode='r')
        num_nodes = sample_x.shape[2]
    except Exception:
        num_nodes = 307 # Default PeMS04
        
    model = AutoResearchModel(in_channels=3, num_nodes=num_nodes, seq_in=12, seq_out=12)
    model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    print(f"初始化模型参数总数 (Model Parameters): {num_params:,}")

    # 支持探索优化器 (Optimizers exploration)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 损失函数 (Loss Function) 默认采用 MAE (L1 Loss) 以对齐评估指标
    criterion = nn.L1Loss() 
    
    # 启用混合精度加速 (AMP) 支持
    use_amp_cuda = ('cuda' in str(device))
    use_amp_mps = ('mps' in str(device)) # 必须集成 AMP M4 Max 支持
    scaler = torch.amp.GradScaler(device.type) if use_amp_cuda else None

    # 调用提前编译的数据模块 (Invoke dataloader)
    train_loader = load_data("train", BATCH_SIZE, device)
    
    # 挂载预计算的物理图空间先验邻接矩阵 (Mount Spatial Graph Prior Adjacency)
    adj_path = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch", DATASET.lower(), "adj.npy")
    if os.path.exists(adj_path):
        import numpy as np
        adj_matrix = torch.from_numpy(np.load(adj_path)).to(device)
    else:
        print("警告: 缺失物理图缓存！使用单位阵占位验证防御性代码。")
        adj_matrix = torch.eye(num_nodes).to(device)
        
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
                    preds = model(x, adj_matrix)
                    loss = criterion(preds.float(), y)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            elif use_amp_mps:
                with torch.amp.autocast(device_type='mps', dtype=torch.bfloat16):
                    preds = model(x, adj_matrix)
                    loss = criterion(preds.float(), y)
                loss.backward()
                optimizer.step()
            else:
                preds = model(x, adj_matrix)
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
    val_mae, val_rmse = evaluate_mae(model, BATCH_SIZE, device, adj_matrix=adj_matrix)
    
    peak_vram_mb = 0.0
    if device.type == 'cuda':
        peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    elif device.type == 'mps':
        peak_vram_mb = torch.mps.driver_allocated_memory() / 1024 / 1024

    # 报告最终评价 (Report Evaluation and Benchmarks)
    baseline_mae = get_baseline_metric(DATASET, SELECTED_BASELINES[0], metric='mae') if len(SELECTED_BASELINES) > 0 else None
    
    print("---")
    print(f"val_mae:          {val_mae:.6f}")
    if baseline_mae is not None:
        relative_improvement = (baseline_mae - val_mae) / baseline_mae * 100
        print(f"vs_{SELECTED_BASELINES[0]}_%:  {relative_improvement:+.2f}%")
        
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
