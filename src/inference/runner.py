# 学习目标：把一段 NumPy 波形变成“每个类别的概率”，暂时不处理麦克风或唤醒状态。
# 输入通常为 [N]，默认模型与配置匹配时，输出 [1, 类别数]，每行概率之和约为 1。
# 阅读时沿 forward 追踪：数组 → Tensor → 统一波形 → 频谱 → 类别分数 → 概率。
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from ..configs import Config
from ..transforms import AudioTransform, DbMelSpec


# ======================================================================
# 1. 准备工具：复用训练时的波形、频谱格式，关闭随机频谱增强
# ======================================================================
class InferenceRunner(nn.Module):
    def __init__(self, model: nn.Module, cfg: Config):
        super().__init__()
        self.model = model
        model.eval()  # 调整 Dropout / BatchNorm 的行为；它不等同于关闭梯度记录。

        self.cfg = cfg
        self.transform = AudioTransform(cfg)
        self.db_mel_spec = DbMelSpec(cfg, augment=False)

    # ------------------------------------------------------------------
    # 2. 单窗口推理：只预测，不准备训练梯度
    # ------------------------------------------------------------------
    # inference_mode 装饰器减少用于求导的记录，适用于这里只读取输出的场景。
    @torch.inference_mode()
    def forward(self, x: np.ndarray) -> torch.Tensor:
        device = next(self.model.parameters()).device  # 查看第一个模型参数，确定数据应送到哪里。
        # from_numpy 创建 Tensor；.to 同时转换设备和数值类型，float32 是常用浮点格式。
        x = torch.from_numpy(x).to(device=device, dtype=torch.float32)
        # 此处假定传入波形已使用训练配置的采样率；调用方应保证录音设置与它一致。
        x = self.transform(x, self.cfg.preprocess.sample_rate)
        x = self.db_mel_spec(x)
        logits = self.model(x)
        # softmax 将每行原始分数变成非负、总和为 1 的类别概率；dim=1 是类别方向。
        return F.softmax(logits, dim=1)


# 预测 / 验证（已有 runner 后，在单独脚本中进行，不必打开麦克风）：
# x = np.zeros(cfg.preprocess.sample_rate, dtype=np.float32)，调用 probs = runner(x)。
# 打印 probs.shape 和 probs.sum(dim=1)；概率总和为 1 不代表模型一定识别正确。
# 修改：输入改为半秒，追踪 AudioTransform 如何补成配置要求的长度。
# 自己写：先用 Tensor 直接预测，再补 NumPy 转换和统一波形，最后加 softmax。
