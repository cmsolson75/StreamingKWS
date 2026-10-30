# 学习目标：让同一条音频每次取样时有些变化，减少模型死记固定的声音。
# 阅读路线：先看各类的 apply，再看它调用的“挑噪声、对齐长度、调整强度”工具。
# 输入、输出都是单段波形 [1, N]；增强改变声音数值，不改变标签。
import torch
import torchaudio
from pathlib import Path

from .configs import Config
from typing import Protocol


# ======================================================================
# 1. 约定共同接口：所有增强工具都有 apply(音频) 方法
# ======================================================================
# Protocol 是类型约定：描述需要什么方法，本身不实现具体增强。
class Augmentation(Protocol):
    def apply(self, x: torch.Tensor) -> torch.Tensor: ...


# ======================================================================
# 2. 随机混入噪声：先配好长度和强度，再与原音频相加
# ======================================================================
class RandomNoiseMixIn(Augmentation):
    """
    Mixes random noise clips into audio at random SNR
    Expects mono audio tensors of shape (1, num_samples)
    """

    def __init__(self, cfg: Config, eps: float = 1e-8):
        # __init__ 先将噪声文件读进内存；eps 是避免除零的小数。
        self.cfg = cfg
        self.path = Path(cfg.augment.noise_path)
        self.data: list[torch.Tensor] = self.load_data()
        if len(self.data) == 0:
            raise ValueError("Random noise folder empty")

        self.eps = eps
        self.generator = torch.Generator().manual_seed(self.cfg.env.seed)  # 固定随机序列的起点。

    # ---------- 2.1 统一噪声的采样率和声道，便于与语音相加 ----------
    def load_data(self) -> list[torch.Tensor]:
        data = []
        for file in self.path.rglob("*.wav"):
            if not file.exists() or file.is_dir():
                continue
            audio, sr = torchaudio.load(file)
            # torchaudio.load 返回 (波形, 采样率)；resample 将每秒采样点数调成目标值。
            if sr != self.cfg.preprocess.sample_rate:
                audio = torchaudio.functional.resample(
                    audio, sr, self.cfg.preprocess.sample_rate
                )
            # unsqueeze 补出声道维；多声道用 mean(dim=0, keepdim=True) 平均成 [1, N]。
            if audio.ndim != 2:
                audio = audio.unsqueeze(0)
            if audio.shape[0] != 1:
                audio = audio.mean(dim=0, keepdim=True)
            data.append(audio)
        return data

    def _get_random_noise_file(self) -> torch.Tensor:
        # randint(0, 数量, (1,)) 随机选一个合法编号，上限不包含；item() 取出普通整数。
        idx = torch.randint(0, len(self.data), (1,), generator=self.generator).item()
        return self.data[idx]

    # ---------- 2.2 把噪声凑成与输入相同长度 ----------
    def _random_shape_match(self, noise: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        # 太短就重复拼接；较长就随机截取。cat(..., dim=1) 沿时间方向接起来。
        duration = x.shape[1]
        if noise.shape[1] < duration:
            tiles = [noise]
            remaining = duration - noise.shape[1]
            while remaining > 0:
                chunk = noise[:, :remaining]
                tiles.append(chunk)
                remaining -= chunk.shape[1]
            return torch.cat(tiles, dim=1)

        # 当前这一分支要求噪声严格长于输入，等长时 randint 的取值范围为空。
        start_loc = torch.randint(
            0, noise.shape[1] - duration, (1,), generator=self.generator
        )
        return noise[:, start_loc : start_loc + duration]

    # ---------- 2.3 用信噪比决定“噪声应该有多响” ----------
    def _get_gain_factor(self, noise: torch.Tensor, x: torch.Tensor) -> float:
        # SNR 是语音功率与噪声功率之比的分贝表示：越大，语音相对越明显。
        # empty(1).uniform_(下限, 上限) 创建一个数并填入指定范围的随机值。
        snr = torch.empty(1).uniform_(
            self.cfg.augment.min_snr_db,
            self.cfg.augment.max_snr_db,
            generator=self.generator,
        )
        noise_power = torch.mean(noise**2) + self.eps  # 平方后求平均，衡量声音功率。
        inp_power = torch.mean(x**2)

        # 功率与振幅平方成正比，因此最后用 sqrt 开平方，得到乘在噪声上的振幅倍率。
        target_power = inp_power / (10 ** (snr / 10.0))
        return torch.sqrt(target_power / noise_power).item()

    def _loudness_normalization(self, x: torch.Tensor) -> torch.Tensor:
        # 为什么限制幅度：相加后的数值可能超出通常的音频范围 [-1, 1]。
        # 先按最大绝对值等比缩小，再用 clip 限定最终上下界。
        peak_amp = torch.max(torch.abs(x)).item()
        if peak_amp > 1.0:
            x = x / peak_amp
        return torch.clip(x, -1.0, 1.0)

    # ---------- 2.4 按概率决定是否执行整套加噪操作 ----------
    def apply(self, x: torch.Tensor) -> torch.Tensor:
        # rand 给出 [0, 1) 中的数；例如阈值为 0.75，约 75% 的调用会进入加噪分支。
        p = torch.rand(1, generator=self.generator).item()
        if p < self.cfg.augment.prob_noise_mix_in:
            noise = self._get_random_noise_file()
            noise = self._random_shape_match(noise, x)

            gain_factor = self._get_gain_factor(noise, x)
            output = x + noise * gain_factor
            return self._loudness_normalization(output)
        else:
            return x


# ======================================================================
# 3. 随机调音量：按概率抽一个分贝增益，再应用到整条波形
# ======================================================================
class RandomGain(Augmentation):
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.generator = torch.Generator().manual_seed(self.cfg.env.seed)

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        p = torch.rand(1, generator=self.generator).item()
        if p < self.cfg.augment.prob_gain_aug:
            gain_db = torch.empty(1).uniform_(
                self.cfg.augment.min_db_gain,
                self.cfg.augment.max_db_gain,
                generator=self.generator,
            )
            # gain(音频, 增益分贝)：正值通常变响，负值变轻，波形长度保持不变。
            x = torchaudio.functional.gain(x, gain_db)
            return x
        return x


# ======================================================================
# 4. 把多个增强串起来：上一项的输出成为下一项的输入
# ======================================================================
class AugmentationPipeline:
    def __init__(self, augmentations: list[Augmentation]):
        self.augmentations = augmentations

    def apply(self, x: torch.Tensor) -> torch.Tensor:
        for augmentation in self.augmentations:
            x = augmentation.apply(x)
        return x


def build_augmentation_pipeline(
    augmentations: list[Augmentation],
) -> AugmentationPipeline:
    return AugmentationPipeline(augmentations)  # 只创建工具，取样本时调用 apply 才改变音频。


# 学习练习（在单独脚本里准备 cfg 和非零波形 x）：
# 预测：把 prob_gain_aug 设为 0，RandomGain(cfg).apply(x) 还会改变声音吗？
# 验证：打印输出 shape，并用 torch.equal(x, 输出) 比较数值是否完全相同。
# 修改：设 prob_gain_aug=1、min_db_gain=max_db_gain=6，观察输出绝对值是否约为输入的两倍。
# 自己写：先实现一个固定音量倍率的 apply，再加入概率开关，最后放入 AugmentationPipeline。
