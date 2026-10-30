# 学习目标：理解“文件里的声音”怎样变成“模型能处理的频谱”。
# 先读 AudioTransform（统一声音格式），再读 DbMelSpec（计算频谱），最后跟踪形状。
# 波形是随时间变化的一串数；频谱描述各个时刻有哪些频率、强度多大。
# 形状记号：B=一批样本数，N=声音采样点数，F=频率格数，T=时间格数。
# 完整路线：不同格式的波形 → [1, N] → 加载器组批 [B, 1, N] → 频谱 [B, 1, F, T]。
import torch
from torch import nn
import torch.nn.functional as F
import torchaudio
from .configs import Config

import torchaudio.functional as AF


# ======================================================================
# 1. 创建频谱转换工具：准备窗口、频率转换表和可选增强
# ======================================================================
class DbMelSpec(nn.Module):
    def __init__(self, cfg: Config, augment: bool = False):
        super().__init__()
        self.cfg = cfg
        # nn.Module 也能表示转换工具；调用对象(x) 时，PyTorch 会进入 forward。
        # hann_window(n_fft) 生成两端较小的权重，减少截取声音小片段造成的边缘影响。
        # register_buffer 保存不需要学习的数组，使它随 .to(device) 移到同一设备。
        # persistent=False 表示不把这份数组放入 state_dict 存档，可由配置重新生成。
        self.register_buffer(
            "window", torch.hann_window(cfg.preprocess.n_fft), persistent=False
        )
        # API：melscale_fbanks 返回“普通频率格 → Mel 频率格”的权重表。
        # Mel 刻度更贴近人耳区分频率的方式；n_mels 决定最终频谱有多少行。
        fb = AF.melscale_fbanks(
            n_freqs=cfg.preprocess.n_fft // 2 + 1,
            f_min=0.0,
            f_max=cfg.preprocess.sample_rate / 2,
            n_mels=cfg.preprocess.n_mels,
            sample_rate=cfg.preprocess.sample_rate,
            norm=None,
            mel_scale="htk",
        )
        self.register_buffer("fb", fb, persistent=False)
        self.augment = augment
        if self.augment:
            # Sequential 按顺序执行；下面两个工具随机遮住一段频率 / 时间上的信息。
            # 为什么遮挡：让模型不要过度依赖某一小块频谱；参数为 0 时相应遮挡无效果。
            self.augmentations = nn.Sequential(
                torchaudio.transforms.FrequencyMasking(
                    freq_mask_param=cfg.augment.freq_mask_param
                ),
                torchaudio.transforms.TimeMasking(
                    time_mask_param=cfg.augment.time_mask_param
                ),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ------------------------------------------------------------------
        # 1.1 整理形状：把单段 [N] 或一批 [B, 1, N] 统一为 [B, N]
        # ------------------------------------------------------------------
        # ndim 是维度数量；unsqueeze 加一个大小为 1 的维度，squeeze 去掉指定的单例维度。
        if x.ndim == 1:
            x = x.unsqueeze(0)
        if x.ndim == 3:
            x = x.squeeze(1)

        x = x.contiguous()  # 保证数据在内存中连续排列，便于后续计算。

        # ------------------------------------------------------------------
        # 1.2 短时傅里叶变换：把连续声音拆成重叠小片段，查看每片的频率成分
        # ------------------------------------------------------------------
        # stft 输入 [B, N]，这里输出复数数组 [B, n_fft // 2 + 1, T]。
        # n_fft 是变换长度，hop_length 是相邻片段移动多少个采样点。
        # win_length 是窗口长度；center=True 会在两端补值，让帧围绕对应时间点居中。
        X = torch.stft(
            x,
            n_fft=self.cfg.preprocess.n_fft,
            hop_length=self.cfg.preprocess.hop_len,
            win_length=self.cfg.preprocess.n_fft,
            window=self.window,
            center=True,
            return_complex=True,
        )

        power = X.abs() ** 2  # abs 取复数的幅度，平方得到功率（各频率成分的强弱）。

        # ------------------------------------------------------------------
        # 1.3 换成 Mel 刻度，再用对数压缩较大的强弱差异
        # ------------------------------------------------------------------
        # .T 转置权重表；matmul 做矩阵乘法，将普通频率格加权汇总成 F 个 Mel 格。
        mel = torch.matmul(self.fb.T, power)
        # amplitude_to_DB 转为分贝；功率用 multiplier=10，amin 防止对 0 取对数。
        # top_db=80 限制保留的动态范围，而不是把所有数值都设为 80。
        db = AF.amplitude_to_DB(
            mel, multiplier=10, amin=1e-10, db_multiplier=0.0, top_db=80
        )
        # ------------------------------------------------------------------
        # 1.4 可选遮挡，再将每段频谱调整到相近的数值尺度
        # ------------------------------------------------------------------
        # 这里按 augment 开关执行；该开关不随外部 model.eval() 自动改变。
        if self.augment:
            db = self.augmentations(db)
        # dim=(-2, -1) 表示频率和时间两维；keepdim 保留维度，方便每个位置减均值、除标准差。
        mean = db.mean(dim=(-2, -1), keepdim=True)
        # clamp_min(1e-5) 给分母设下限，避免除以 0 或太小的数。
        std = db.std(dim=(-2, -1), keepdim=True).clamp_min(1e-5)
        # 最后加上单通道维度，返回模型需要的 [B, 1, F, T]。
        return ((db - mean) / std).unsqueeze(1)


# ======================================================================
# 2. 统一波形格式：采样率一致 → 单声道 → 长度一致
# ======================================================================
class AudioTransform(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        # 每秒采样点数 × 秒数 = 目标长度；例如 16000 × 1 秒 = 16000 个点。
        self.expected_length = int(
            self.cfg.preprocess.sample_rate * self.cfg.preprocess.clip_length
        )

    def forward(self, wav: torch.Tensor, sr: int) -> torch.Tensor:
        # 输入：波形和原采样率；输出：固定采样率、固定长度的 [1, expected_length]。
        # 为什么需要统一：同一批音频要能堆在一起，模型看到的时间尺度也要一致。
        if sr != self.cfg.preprocess.sample_rate:
            # resample(波形, 原采样率, 新采样率) 改变采样密度，保持声音时长。
            wav = torchaudio.functional.resample(
                wav, sr, self.cfg.preprocess.sample_rate
            )
        if wav.ndim != 2:
            wav = wav.unsqueeze(0)

        # 第 0 维是声道；mean 对多个声道取平均，keepdim=True 让结果仍是 [1, N]。
        if wav.shape[0] != 1:
            wav = wav.mean(dim=0, keepdim=True)

        # 太长就截取开头，太短就在结尾补零；F.pad(..., (0, pad_amt)) 表示左补 0、右补指定数量。
        n = wav.shape[1]
        if n > self.expected_length:
            wav = wav[:, : self.expected_length]
        if n < self.expected_length:
            pad_amt = self.expected_length - n
            wav = F.pad(wav, (0, pad_amt))
        return wav


# ======================================================================
# 3. 预测 → 验证 → 修改 → 自己写
# ======================================================================
# 在单独脚本中导入 torch、Config 和本文件两个类，先读取有效 cfg，再构造：
# wav = torch.randn(2, 8000)   # 假设它是 16000 Hz 的双声道、半秒音频。
# 预测：cfg 设为 16000 Hz、1 秒时，AudioTransform(cfg)(wav, 16000) 的形状是什么？
# 验证：打印处理后的 shape，再 unsqueeze(0) 组成一批，送进 DbMelSpec(cfg, augment=False)。
# 修改：只把 n_mels 从 64 改为 32，预测频谱哪一维会变，再比较输出 shape。
# 自己写：先复写“平均声道 → 裁剪或补零”，再按“STFT → 功率 → Mel → 分贝”复写频谱流程。
# 下方是原有调试示例；音频路径需要存在，Config() 也需要补齐必填配置才能运行。
if __name__ == "__main__":
    audio_example = "/Users/cameronolson/Downloads/T3.wav"
    cfg = Config()
    transform = AudioTransform(cfg)
    wav, sr = torchaudio.load(audio_example)
    # wav, sr = torch.randn(15000), 16_000

    out = transform(wav, sr)
    print(out.shape)
