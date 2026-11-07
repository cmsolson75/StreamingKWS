"""M12（选做）· 从零写 Transformer 分类器

规则：读论文和 nn.MultiheadAttention 的文档，**不调用** nn.MultiheadAttention、
nn.TransformerEncoderLayer、F.scaled_dot_product_attention（写完后可以用它们对照）。

论文：Attention Is All You Need https://arxiv.org/abs/1706.03762（3.2 节和 3.5 节）
参考（写完再看）：nanoGPT https://github.com/karpathy/nanoGPT 里的 CausalSelfAttention

输入输出约定和 model.py 一致：(B, n_mels, n_frames) → (B, num_classes)。

思考题
------
1. 为什么要除以 sqrt(d_head)？不除会怎样？（想想 d_head 很大时点积的方差。）
2. 多头注意力里，(B, T, D) 怎么变成 (B, H, T, D/H)？用 view + transpose 时，为什么之后有时需要 contiguous()？
3. mask 在 softmax 之前还是之后加？被 mask 的位置填什么值？整行都被 mask 掉会怎样？
4. Pre-norm 和 Post-norm（LayerNorm 放在残差前还是后）训练稳定性有什么区别？
5. 分类时怎么把 (B, T, D) 变成 (B, D)？平均池化时，补齐的帧该不该算进去？
6. 和 TC-ResNet8 比，参数量、准确率、单次推理延迟各是多少？（做成表格写进学习日志。）
"""

from __future__ import annotations

import torch
from torch import nn


class MultiHeadSelfAttention(nn.Module):
    """x: (B, T, D) → (B, T, D)。

    - causal=True 时，位置 t 只能看到 <= t 的位置
    - key_padding_mask: (B, T) bool，True 表示该位置是补齐的、不能被看到
    """

    def __init__(
        self, d_model: int, n_heads: int, dropout: float = 0.0, causal: bool = False
    ) -> None:
        super().__init__()
        raise NotImplementedError("M12: MultiHeadSelfAttention.__init__")

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        raise NotImplementedError("M12: MultiHeadSelfAttention.forward")


class TransformerBlock(nn.Module):
    """Pre-norm：x = x + attn(LN(x))；x = x + mlp(LN(x))。MLP 隐层宽度为 mlp_ratio * d_model。"""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        mlp_ratio: int = 4,
        dropout: float = 0.0,
        causal: bool = False,
    ) -> None:
        super().__init__()
        raise NotImplementedError("M12: TransformerBlock.__init__")

    def forward(
        self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        raise NotImplementedError("M12: TransformerBlock.forward")


class KWSTransformer(nn.Module):
    """Mel 帧 → 线性投影到 d_model → 加可学习的位置编码 → n_layers 个 block → LN → 池化 → 分类头。

    forward 的 frame_mask: (B, n_frames) bool，True = 真实帧。为 None 表示全部有效。
    要求：补齐帧的内容不能影响输出。
    """

    def __init__(
        self,
        n_mels: int,
        num_classes: int,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_frames: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        raise NotImplementedError("M12: KWSTransformer.__init__")

    def forward(self, x: torch.Tensor, frame_mask: torch.Tensor | None = None) -> torch.Tensor:
        raise NotImplementedError("M12: KWSTransformer.forward")
