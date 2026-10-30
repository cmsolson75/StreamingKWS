# 学习目标：让多种库的随机操作从指定起点开始，便于比较两次实验。
# 输入是一个整数 seed；函数修改各库的随机状态，没有需要接收的返回结果。
import random
import torch
import numpy as np


# ======================================================================
# 1. 设置随机起点：Python、NumPy、PyTorch 和 CUDA 各有自己的随机数状态
# ======================================================================
def seed_everything(seed: int = 42):
    # 为什么要分别设置：只设 PyTorch，不会同时固定 Python random 或 NumPy 的抽样。
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)  # 为所有 CUDA 设备设置种子。


# 预测 / 验证：seed_everything(42) 后调用 torch.rand(3)，再重新设 42 并调用，比较结果。
# 修改：换成 seed=43；同一序列中的连续抽样仍不同，重新设种子才会回到序列起点。
# 自己写：先只固定 random，再加入 NumPy、PyTorch，观察各库的输出是否随之重复。
# 固定种子帮助复现，但不保证不同设备、库版本及所有并行算法都得到完全相同的结果。
