"""M4（收尾）/ M5 · 把数据集、采样器、增强组装成 DataLoader

这是“胶水代码”，但很多 bug 藏在这里：增强用错了切分、验证集被随机采样、worker 随机数重复……

先读文档
--------
- DataLoader 的参数：batch_size、sampler、shuffle、num_workers、collate_fn、pin_memory、
  persistent_workers、drop_last、generator（全部查一遍，说清每个参数在本项目里该怎么设）

思考题
------
1. 训练集用 GroupWeightedSampler 无限地按权重抽样。验证集和测试集还应该加权抽样吗？
   为什么用有放回抽样评估，结果的可信度会变差？
2. 数据增强应该作用在哪些切分上？（StreamingKWS 原代码在这里有一个值得讨论的选择。）
3. 验证集里的 SILENCE 样本从哪来？每次评估都应该是同一批吗？

写完的标准：`uv run pytest tests/test_m04_data.py -k loader` 全绿（M5 写完后再回来接上增强）。
"""

from __future__ import annotations

from torch.utils.data import DataLoader

from kws.config import Config
from kws.dataset import Split


def build_dataloader(cfg: Config, split: Split, labels: list[str]) -> DataLoader:
    """返回某个切分的 DataLoader，每个 batch 是 (waveforms (B, T), labels (B,))。

    - train：四个数据集 → ConcatDataset → GroupWeightedSampler（num_samples 取 batch_size * max_steps）
             → 只在 cfg.augment.enabled 时接上增强（M5 写完再接）
    - val / test：完整、确定地遍历一遍，不打乱，不增强
    - 所有 DataLoader 参数都从 cfg 里读，不要写死
    """
    raise NotImplementedError("M4: build_dataloader")
