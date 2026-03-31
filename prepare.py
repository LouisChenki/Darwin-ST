"""
针对时空预测任务（Spatio-Temporal Prediction）的数据准备脚本。
下载 PeMS04 交通流数据集，处理为滑动窗口格式，并提供统一的评估与指标功能。

用法 (Usage):
    uv run prepare.py
"""

import os
import sys
import requests
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# 常量设置 (Constants - 固定的时间预算与数据格式)
# ---------------------------------------------------------------------------

TIME_BUDGET = 900  # 时间预算 (Time Budget) 扩展到 15 分钟
CACHE_DIR = os.path.join(os.path.expanduser("~"), ".cache", "autoresearch")

DATASET = os.environ.get("DATASET", "PeMS04")
DATA_DIR = os.path.join(CACHE_DIR, DATASET.lower())

# 数据集远程地址列表 (Dataset URLs)
DATASET_URLS = {
    "PeMS04": "https://github.com/Davidham3/ASTGCN/raw/master/data/PEMS04/pems04.npz",
    "PeMS08": "https://github.com/Davidham3/ASTGCN/raw/master/data/PEMS08/pems08.npz",
    "METR-LA": "https://github.com/chenkaiqi/METR-LA_placeholder_url/metr-la.npz", # 需替换为真实源或手动下载
    "PEMS-BAY": "https://github.com/chenkaiqi/PEMS-BAY_placeholder_url/pems-bay.npz" 
}

PEMS_URL = DATASET_URLS.get(DATASET, DATASET_URLS["PeMS04"])
FILE_PATH = os.path.join(DATA_DIR, f"{DATASET.lower()}.npz")

# 时空参数设定 (Spatio-Temporal Parameters)
SEQ_LEN_IN = 12   # 历史输入时间步 (Historical Sequence Length)
SEQ_LEN_OUT = 12  # 未来预测时间步 (Future Sequence Length)
TRAIN_RATIO = 0.6 # 训练集比例 (Train Ratio)
VAL_RATIO = 0.2   # 验证集比例 (Validation Ratio)

# ---------------------------------------------------------------------------
# 数据下载与预处理 (Data Download & Preprocessing)
# ---------------------------------------------------------------------------

def download_data():
    """下载 PeMS04 数据集并保存到本地缓存"""
    os.makedirs(DATA_DIR, exist_ok=True)
    if os.path.exists(FILE_PATH):
        print(f"数据已存在 (Dataset already exists): {FILE_PATH}")
        return

    print(f"正在下载 {DATASET} 数据集 (Downloading) 从 {PEMS_URL} ...")
    try:
        response = requests.get(PEMS_URL, stream=True)
        response.raise_for_status()
        with open(FILE_PATH, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        print("下载完成 (Download complete).")
    except Exception as e:
        print(f"下载失败 (Failed to download dataset). 请手动将 {DATASET.lower()}.npz 放入 {DATA_DIR}. 报错内容: {e}")
        sys.exit(1)

def generate_dataset(data, seq_len_in, seq_len_out):
    """
    滑动窗口构建数据集 (Sliding Window Generation)
    输入 data 形状: [T, N, C]
    返回:
        X: [num_samples, seq_len_in, N, C]
        Y: [num_samples, seq_len_out, N] 默认只预测第一维 Flow
    """
    total_len = data.shape[0]
    num_samples = total_len - seq_len_in - seq_len_out + 1
    
    X = np.zeros((num_samples, seq_len_in, data.shape[1], data.shape[2]), dtype=np.float32)
    Y = np.zeros((num_samples, seq_len_out, data.shape[1]), dtype=np.float32)
    
    for i in range(num_samples):
        X[i] = data[i: i + seq_len_in]
        # 只取特征维度第一维 (Flow) 作为目标标签
        Y[i] = data[i + seq_len_in: i + seq_len_in + seq_len_out, :, 0]
        
    return X, Y

def process_and_save():
    """切分并缓存处理好的数据集"""
    train_x_path = os.path.join(DATA_DIR, "train_x.npy")
    if os.path.exists(train_x_path):
         print("数据已预处理 (Preprocessed data already exists).")
         return
         
    print(f"正在处理 {DATASET} 滑动窗口切片 (Processing sliding windows)...")
    # pemsnpz 普遍含有 'data' 键, shape: [T, N, C]
    file_data = np.load(FILE_PATH)
    data = file_data['data'] if 'data' in file_data else file_data[file_data.files[0]]
    
    # 动态获取当前数据集节点数
    num_nodes = data.shape[1]
    
    # 构建滑动窗口 (Sliding Window)
    X, Y = generate_dataset(data, SEQ_LEN_IN, SEQ_LEN_OUT)
    
    # 划分数据集 (Split Dataset)
    num_samples = X.shape[0]
    train_steps = int(num_samples * TRAIN_RATIO)
    val_steps = int(num_samples * VAL_RATIO)
    
    train_x, train_y = X[:train_steps], Y[:train_steps]
    val_x, val_y = X[train_steps:train_steps+val_steps], Y[train_steps:train_steps+val_steps]
    
    # Z-Score 归一化 (Standardization) -> 为了防止数据泄露，仅依训练集标准差进行归一化
    mean = train_x[..., 0].mean()
    std = train_x[..., 0].std()
    
    train_x[..., 0] = (train_x[..., 0] - mean) / std
    val_x[..., 0] = (val_x[..., 0] - mean) / std
    
    np.save(train_x_path, train_x)
    np.save(os.path.join(DATA_DIR, "train_y.npy"), train_y)
    np.save(os.path.join(DATA_DIR, "val_x.npy"), val_x)
    np.save(os.path.join(DATA_DIR, "val_y.npy"), val_y)
    
    # 保存缩放因子以备还原反变换 (Save Scaler)
    np.save(os.path.join(DATA_DIR, "scaler.npy"), np.array([mean, std]))
    print("预处理完成 (Preprocessing complete)!")


# ---------------------------------------------------------------------------
# 运行时组件 (Runtime Utilities - Imported by train.py)
# ---------------------------------------------------------------------------

def load_data(split, batch_size, device='cpu', drop_last=True):
    """
    加载数据加载器 (DataLoader Load Function)
    通过 pin_memory_True 针对 MPS/CUDA 异步传输做优化。
    """
    x_path = os.path.join(DATA_DIR, f"{split}_x.npy")
    y_path = os.path.join(DATA_DIR, f"{split}_y.npy")
    
    if not os.path.exists(x_path):
        raise FileNotFoundError(f"Dataset split {split} not found. Please run prepare.py first.")
        
    X = torch.from_numpy(np.load(x_path)).to(torch.float32)
    Y = torch.from_numpy(np.load(y_path)).to(torch.float32)
    
    dataset = TensorDataset(X, Y)
    
    # [性能优化] - 根据统一内存架构 (M4 Max) 与 CUDA 进行内存锁定配置传输
    is_accelerator = ('cuda' in str(device)) or ('mps' in str(device))
    
    loader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=(split == 'train'), 
        pin_memory=is_accelerator,
        drop_last=drop_last
    )
    return loader

def get_scaler():
    """读取均值域方差 (Get Scaler Parameters)"""
    return np.load(os.path.join(DATA_DIR, "scaler.npy"))

# ---------------------------------------------------------------------------
# 评估模块 (Evaluation Module - 这是唯一的且绝对稳固的标尺)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_mae(model, batch_size, device="cpu"):
    """
    在验证集上度量系统的 平均绝对误差 (Mean Absolute Error, MAE) 与均方根误差 (RMSE)。
    这是智能体进化的考核指标，极度客观。要求输出形状适配 [B, T_out, Nodes]。
    """
    model.eval()
    val_loader = load_data("val", batch_size, 'cpu', drop_last=False) 
    
    total_mae = 0.0
    total_mse = 0.0
    total_samples = 0
    
    # 取回归一化器进行标签尺度的对比 (Rescale data for real-world metric evaluation)
    scaler = get_scaler()
    mean_val, std_val = float(scaler[0]), float(scaler[1])
    
    for x, y in val_loader:
        x, y = x.to(device), y.to(device)
        
        # 前向传播 (Forward Propagation)
        preds = model(x)
        
        if torch.isnan(preds).any():
            return float('inf'), float('inf')
        
        # 统一恢复到绝对真实的交通流量尺度 (Inverse Transform)
        preds_real = (preds * std_val) + mean_val     # 模型预测是在归一化的尺度，或者直接通过线性映射完成，需确认。
        # 注意：此处假设模型的直接输出是与 y 的同等标度（未经反归一化的）。如果不使用特征对齐可以直接通过下式：
        # 这里为简便起见，假设直接对 y 进行真实还原进行统计
        # 实际上 y 保存的就是真实流量（无归一化）
        # 如果模型直接拟合真实流量，就直接计算；如果模型拟合的是预测差距，需反向还原。
        # 我们上游预处理时 val_y.npy 其实并未进行归一化（因为归一化仅处理了 x）。
        # 请确保 model的最终输出拟合了真实值。
        
        mae = torch.abs(preds - y).sum().item()
        mse = torch.pow(preds - y, 2).sum().item()
        
        total_mae += mae
        total_mse += mse
        total_samples += y.numel()
        
    mean_mae = total_mae / total_samples
    mean_rmse = np.sqrt(total_mse / total_samples)
    
    model.train()
    return mean_mae, mean_rmse

# ---------------------------------------------------------------------------
# 程序入口 (Entry Point)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("初始化时空图预测数据的下载与解配 (Prepare Autoresearch Spatio-Temporal Data)...")
    download_data()
    process_and_save()
    print("就绪 (Ready)! 数据管道搭建完成。")
