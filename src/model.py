# 学习目标：沿着输入的形状变化，理解“频谱怎样变成每个类别的分数”。
# 阅读顺序：先看 AudioClassifier 的 forward，再看 encoder / head，最后读 TCResNet8。
# Tensor 是多维数组，shape 描述各维大小；B=一批样本数，F=频率格数，T=时间格数。
# C=特征通道数，可以理解成模型从输入中提取了多少种特征。
# 输入频谱 [B, 1, F, T] → 模型 → 类别分数 [B, num_classes]。
# 分数叫 logits，还不是概率；本项目把它交给损失函数衡量预测误差。
# 每个“预测 / 验证”关注形状，“自己写”练习放在底部，不用先启动完整训练。
import torch
from torch import nn
import torch.nn.functional as F
from typing import Literal, Callable


# ======================================================================
# 1. CNN 的基础积木：卷积 → 调整数值分布 → 加入非线性
# ======================================================================
# nn.Module 是 PyTorch 模型的基类；继承它后，框架能找到并管理模型参数。
class ConvBlock(nn.Module):
    def __init__(self, n_in, n_out):
        # 子目标：改变特征通道数，但保持这一步输入的频谱高宽。
        # 输入 [B, n_in, F, T] → 输出 [B, n_out, F, T]。
        # __init__ 在创建对象时执行；super() 先完成 nn.Module 的初始化。
        super().__init__()
        # Sequential(a, b, c)：让输入依次经过 a、b、c。
        # n_in / n_out 是输入 / 输出通道数；可把输出通道看成不同的特征图。
        self.block = nn.Sequential(
            # Conv2d：用 3×3 小窗口扫描频谱；padding=1 在边缘补值，保持高宽。
            # bias=False 不额外学习偏置值；这里后面接了 BatchNorm。
            nn.Conv2d(n_in, n_out, kernel_size=3, padding=1, bias=False),
            # BatchNorm2d：调整各通道数值的分布，帮助后续层处理；训练和评估方式不同。
            nn.BatchNorm2d(n_out),
            # ReLU：负数变为 0，正数保留，让模型能表达更复杂的关系。
            # inplace=True 尽量复用原来的存储空间。
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward 定义数据怎么走；平时调用 block(x)，PyTorch 会进入这里。
        return self.block(x)


# ======================================================================
# 2. 二维 CNN：把频谱当成一张单通道图片来识别
# ======================================================================
class AudioClassifier(nn.Module):
    def __init__(self, num_classes: int, p: float = 0.5):
        super().__init__()
        # 输入参数：num_classes 决定输出几个分数，p 决定训练时随机置零的比例。
        # 子目标 A：encoder 找特征；通道数增加，频谱高宽通过池化逐渐减小。
        # 为什么缩小高宽：在后续层用较少的位置汇总前面找到的局部信息。
        self.encoder = nn.Sequential(
            ConvBlock(1, 32),
            # AvgPool2d(2)：每个 2×2 区域取平均，高宽大致各缩小一半。
            nn.AvgPool2d(2),
            ConvBlock(32, 64),
            nn.AvgPool2d(2),
            ConvBlock(64, 128),
            nn.AvgPool2d(2),
            ConvBlock(128, 256),
        )

        # 子目标 B：head 汇总特征，再给每个类别打分。
        # 为什么先平均：把大小不固定的特征图变成固定的 256 个数，再接分类层。
        self.head = nn.Sequential(
            # 每个特征图取整体平均：[B, 256, 高, 宽] → [B, 256, 1, 1]。
            nn.AdaptiveAvgPool2d(1),
            # Flatten 默认保留第 0 维（样本数），其余铺平，得到 [B, 256]。
            nn.Flatten(),
            # Dropout(p)：训练时随机把部分数值置零，减少对固定特征的依赖。
            nn.Dropout(p),
            # Linear(输入数, 输出数)：学习如何将 256 个特征组合成各类别的分数。
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 预测：num_classes=4，输入是 [2, 1, 64, 101]，输出应该有几行、几列？
        # 验证（已创建 model 和 x 后）：先看 model.encoder(x).shape，再看 model(x).shape。
        # 为什么分两段看：先确认特征形状，再确认最后输出的类别数是否正确。
        return self.head(self.encoder(x))


# ======================================================================
# 3. 一维残差积木：主路处理数据，旁路保留另一份信息，最后相加
# ======================================================================
class TCBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 2):
        super().__init__()
        # 子目标：将 [B, 输入通道, 时间长度] 变成 [B, 输出通道, 更短的时间长度]。
        # Conv1d 沿时间方向扫描，输入形状为 [B, C, T]。
        # kernel_size=9 每次看 9 个时间位置；stride=2 每次移动 2 格，缩短序列。
        # padding=4 在两端补值；stride=1 时配合大小为 9 的窗口保持长度。
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=9,
            stride=stride,
            padding=4,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        # 第二层步长为 1，继续处理特征，不再缩短时间长度。
        self.conv2 = nn.Conv1d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=9,
            stride=1,
            padding=4,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        # 为什么旁路也要卷积：主路已改变通道数和长度，原输入无法直接和它相加。
        # kernel_size=1、相同步长的卷积把两路的形状对齐。
        self.skip_conv = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            bias=False,
        )

        self.skip_bn = nn.BatchNorm1d(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先计算旁路；主路是下面连续两次卷积。
        skip = self.skip_bn(self.skip_conv(x))

        # F.relu(x) 是直接调用函数；上面的 nn.ReLU() 是创建一个可放进 Sequential 的层。
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        # 预测：输入 [2, 16, 101]、out_channels=24、stride=2，两路相加前的形状是什么？
        # 验证：在此处设断点查看 x.shape、skip.shape，确认两路对应位置能一一相加。
        # 相同位置的数相加形成残差连接，给训练中的信息传递提供另一条路径。
        return F.relu(x + skip)


# ======================================================================
# 4. TCResNet8：把频率格当成输入通道，沿时间方向找特征
# ======================================================================
class TCResNet8(nn.Module):
    def __init__(self, num_classes: int, n_mels: int = 64, k: int = 1, p: float = 0.5):
        super().__init__()

        # 与 CNN 的区别：这里把 F 个频率格作为通道，Conv1d 只沿 T（时间）移动。
        # n_mels 必须等于输入的 F；k 将内部通道数放大，但不改变类别数。
        def scale_width(n):
            return int(n * k)

        # 先做一次卷积，再经过三个残差积木；每个积木默认缩短时间长度。
        self.encoder = nn.Sequential(
            nn.Conv1d(
                in_channels=n_mels,
                out_channels=scale_width(16),
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),  # 此层后接 BatchNorm，这里不使用卷积偏置。
            nn.BatchNorm1d(scale_width(16)),
            nn.ReLU(),
            TCBlock(scale_width(16), scale_width(24)),
            TCBlock(scale_width(24), scale_width(32)),
            TCBlock(scale_width(32), scale_width(48)),
        )
        # AdaptiveAvgPool1d(1) 对时间取平均，每个通道最后只留下一个数。
        # [B, C, T] → [B, C, 1] → [B, C] → [B, num_classes]。
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p),
            nn.Linear(scale_width(48), num_classes, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 子目标：把频谱形状适配到 Conv1d 需要的 [B, 通道, 时间]。
        # API：dim() 查看维度个数；squeeze(1) 去掉第 1 维上大小为 1 的维度。
        # [B, 1, F, T] → [B, F, T]；本来就是三维时则保持不变。
        # 预测：squeeze 后 F 和 T 会交换吗？验证：比较处理前后的 x.shape。
        x = x.squeeze(1) if x.dim() == 4 else x
        return self.head(self.encoder(x))


# ======================================================================
# 5. 按名字创建模型：让训练代码只需要提供模型名称和参数
# ======================================================================
# Literal 是类型提示，列出支持的名字；Callable 描述“接收什么、返回什么”的函数。
ModelName = Literal["cnn", "tc_resnet8"]
ModelCtor = Callable[[int, float], nn.Module]


def tc_resnet8_ctor(num_classes: int, p: float = 0.5):
    # ctor 是 constructor（创建对象的函数）的缩写；这里固定使用两倍通道数。
    return TCResNet8(num_classes, k=2, p=p)


def cnn_ctor(num_classes: int, p: float = 0.5):
    return AudioClassifier(num_classes, p=p)


# 字典把名字对应到创建函数；这里保存函数本身，还没有调用它们。
_MODEL_CTORS: dict[ModelName, ModelCtor] = {
    "cnn": cnn_ctor,
    "tc_resnet8": tc_resnet8_ctor,
}


def model_factory(name: str, num_classes: int, p: float) -> nn.Module:
    # 输入：模型名字、类别数、Dropout 比例；输出：一个可以 model(x) 的模型对象。
    # 为什么用统一入口：train.py 可以通过配置换模型，而不用改创建模型的代码。
    # dict.get(name) 按名字查找；找不到会返回 None。
    ctor = _MODEL_CTORS.get(name)
    if ctor is None:
        # raise ValueError 主动报错；sorted(...) 将支持的名字排序后显示。
        raise ValueError(
            f"model name does not exist, got {name}. want {sorted(_MODEL_CTORS)}"
        )
    return ctor(num_classes, p)  # 调用查到的函数，返回创建好的模型。


# ======================================================================
# 6. 从读懂到自己写：先验证形状，再改变一个参数
# ======================================================================
# 在单独脚本中 import torch，并从 src.model 导入 AudioClassifier 后：
# model = AudioClassifier(num_classes=4).eval()  # eval() 切到评估方式，关闭 Dropout 的随机置零。
# x = torch.randn(2, 1, 64, 101)                # 创建一批随机频谱，用来检查数据能否通过模型。
# with torch.no_grad():                       # 只观察输出，不记录用于训练求导的计算过程。
#     print(model.encoder(x).shape, model(x).shape)
# 修改练习：先预测，再把 num_classes 改为 6；观察特征形状和输出形状各有什么变化。
# 自己写：遮住 AudioClassifier 的实现，按“提取特征 → 整体平均 → 铺平 → 类别分数”复写。
# 完成后再加 Dropout，并说明它为何只在训练时随机置零。
#
# 以下保留原有调试入口，作为入口运行时执行，被其他文件导入时不执行。
# 注意：原有示例的 in_channels 参数与上面的构造函数不匹配。
# 实际输入应为 [B, n_mels, T] 或 [B, 1, n_mels, T]；这里只保留示例原代码。
if __name__ == "__main__":
    model = TCResNet8(in_channels=80, num_classes=12)
    x = torch.randn(1, 100, 80)  # randn 创建指定形状的随机数；此示例原本按 [B, T, F] 排列。
    print(model(x).shape)  # 修正示例参数和输入顺序后，预期输出形状为 [1, 12]。
