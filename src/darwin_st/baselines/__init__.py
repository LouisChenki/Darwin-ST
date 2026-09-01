"""经典时空预测基线 (Literature Baselines) —— IJGIS E5 实验的同炉复现。

四个公认基线的 PyTorch 实现, 与演化模型共用同一条数据/训练/评测管线
(prepare.py 切分与 scaler + masked_mae 训练损失 + evaluate 真实尺度 masked
指标), 保证「同炉可比」:

  - DCRNN  (Li et al., ICLR'18):     扩散卷积 + DCGRU seq2seq encoder-decoder
  - GWNet  (Wu et al., IJCAI'19):    自适应邻接 + dilated causal conv + 图卷积
  - AGCRN  (Bai et al., NeurIPS'20): NAPL 节点自适应参数 + AGCRU encoder + 线性头
  - STID   (Shao et al., CIKM'22):   纯 MLP + 节点/tod/dow 身份嵌入

统一契约 (与 search/builder.py STModel 对齐):
  构造: Model(num_nodes, in_channels, seq_len_in, seq_len_out, adj=None, **kw)
  前向: forward(x [B,T_in,N,C], tod_idx=None, dow_idx=None) -> [B,T_out,N]
        (x 为归一化尺度, 返回目标通道预测; 非 STID 模型忽略 tod/dow)
"""

from __future__ import annotations

import torch.nn as nn

from darwin_st.baselines.agcrn import AGCRN
from darwin_st.baselines.dcrnn import DCRNN
from darwin_st.baselines.gwnet import GWNet
from darwin_st.baselines.statistical import AutoregressiveBaseline, FCLSTM, HistoricalAverage
from darwin_st.baselines.stid import STID

__all__ = ["AGCRN", "DCRNN", "GWNet", "STID",
           "HistoricalAverage", "AutoregressiveBaseline", "FCLSTM",
           "BASELINES", "build_baseline"]

# 跑批入口 (scripts/run_baseline.py 的 BASELINE 环境变量) 的名字注册表
BASELINES: dict[str, type[nn.Module]] = {
    "dcrnn": DCRNN,
    "gwnet": GWNet,
    "agcrn": AGCRN,
    "stid": STID,
    "ha": HistoricalAverage,
    "ar": AutoregressiveBaseline,
    "fc_lstm": FCLSTM,
}


def build_baseline(name: str, num_nodes: int, in_channels: int, seq_len_in: int,
                   seq_len_out: int, adj=None, **kw) -> nn.Module:
    """按名构造基线 (大小写容错)。未知名抛 ValueError 并列出可选项。"""
    cls = BASELINES.get(name.lower())
    if cls is None:
        raise ValueError(f"未知基线 '{name}'. 可选: {sorted(BASELINES)}")
    return cls(num_nodes=num_nodes, in_channels=in_channels,
               seq_len_in=seq_len_in, seq_len_out=seq_len_out, adj=adj, **kw)
