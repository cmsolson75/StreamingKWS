# 学习目标：自己写出一个 Dataset，按编号返回 (一段音频, 类别编号)。
# 阅读顺序：先看 SpeechCommands 的 __init__、__len__、__getitem__，再追踪辅助方法。
# 输入：文件目录、配置、样本编号。输出：单声道音频 [1, N] 和一个整数标签。
# N 是采样点数；Tensor（张量）可以先理解成数组，shape 表示每个方向的大小。
# 先用“预测 / 验证”追踪一个样本，最后按底部“自己写”重建最小版本。
import torchaudio
import torch
from torch.utils.data import Dataset
from .configs import Config
from pathlib import Path
from typing import Literal
import json
from .transforms import AudioTransform
from .augmentations import AugmentationPipeline


# ======================================================================
# 1. 建立类别表：把文字类别变成模型可以使用的整数编号
# ======================================================================
def load_labels(
    cfg: Config,
    write_json: bool = False,
    labels_path: str = "labels.json",
):
    # 输入：cfg 中的数据目录、关键词列表；write_json 决定是否另存类别名称。
    # 输出：(名称列表, 名称到整数的对应表, 类别总数)。
    # 为什么要编号：人读 "yes"，模型训练用整数标签；所有数据集必须共用同一张对应表。
    # API：Path 把字符串变成路径对象；set 创建集合，方便判断一个词是否在 subset 中。
    path = Path(cfg.data.path)
    subset = set(cfg.data.subset) if cfg.data.subset is not None else set()
    # iterdir() 查看目录下一层；is_dir() 判断是否为文件夹；name 是最后一级名称。
    # 只留下选中的关键词目录，排除以下划线开头的目录；sorted(...) 按名字排序。
    labels = sorted(
        [
            p.name
            for p in path.iterdir()
            if p.is_dir() and p.name[0] != "_" and p.name in subset
        ]
    )
    # append(...) 在列表末尾加一项：非目标词统一算 UNKNOWN，静音和背景噪声算 SILENCE。
    labels.append("<UNKNOWN>")
    labels.append("<SILENCE>")
    num_classes = len(labels)
    # enumerate(labels) 同时给出编号和名称，再用 {名称: 编号} 生成字典。
    # 预测：若只选中 "yes" 和 "no" 且目录齐全，排序、追加两类后，四个编号分别是什么？
    # 验证：拿到 cfg 后执行 print(load_labels(cfg))，对照名称顺序和编号。
    label_to_idx = {w: i for i, w in enumerate(labels)}

    if write_json:
        # open("w") 打开文件用于写入；with 结束时自动关闭；json.dumps 转成 JSON 文本。
        labels_path = Path(labels_path)
        with labels_path.open("w") as f:
            f.write(json.dumps(labels))

    return labels, label_to_idx, num_classes  # 名称列表、名称到编号的对应表、类别总数。


# ======================================================================
# 2. 关键词数据集：从 wav 文件中读取模型需要识别的词
# ======================================================================
# (Dataset) 表示继承 PyTorch 的数据集基类；self 表示当前这个数据集对象。
class SpeechCommands(Dataset):
    # ---------- 2.1 子目标：建立“样本编号 → 路径和类别”的清单 ----------
    def __init__(
        self,
        cfg: Config,
        split: str,
        augmentation_pipeline: AugmentationPipeline | None = None,
    ):
        # 输入：配置 cfg、数据划分 split，以及可省略的增强工具。
        # 结果：准备文件清单和处理工具；真正读某条音频，要等到 __getitem__。
        # __init__ 创建对象时执行；self.xxx 保存的内容可以被这个对象的其他方法使用。
        self.cfg: Config = cfg
        self.subset = (
            set(self.cfg.data.subset) if self.cfg.data.subset is not None else set()
        )
        self.path: Path = Path(self.cfg.data.path)
        self.split: str = split
        # 数据集自带文本清单，用来区分哪些音频属于验证集、测试集。
        self.val_split = self.open_split("val")
        self.test_split = self.open_split("test")
        self.dataset = self._load_dataset()  # 此时主要收集文件路径，还没有读取所有音频。
        # AudioTransform 是本项目的工具：统一采样率、单声道和音频长度。
        self.transform = AudioTransform(cfg)

        self.augmentation_pipeline = augmentation_pipeline

        self.cache = {}  # 缓存：用文件路径记住处理结果，下次取同一文件可少读一次磁盘。

    # ---------- 2.2 收集路径：每一项保存 (文件路径, 类别名称) ----------
    def _load_dataset(self):
        # 结果是一张清单，例如 [(".../yes/a.wav", "yes"), ...]，不存全部音频数值。
        # 为什么先存路径：构建数据集时不必一次读完所有关键词音频。
        # self.stoi 是名称到编号的字典；stoi 可读作 string to integer（文字转整数）。
        self.labels, self.stoi, self.num_classes = load_labels(
            cfg=self.cfg, write_json=True, labels_path="labels.json"
        )
        dataset = []
        for label in self.labels:
            # Path 对象用 / 拼接路径；例如 数据目录 / "yes"。
            label_path = self.path / label
            # rglob("*.wav") 在当前目录和子目录里查找所有 wav 文件。
            for file in label_path.rglob("*.wav"):
                file_id = f"{label}/{file.name}"
                if self.in_split(file_id):  # 只保留当前 train / val / test 应该使用的文件。
                    dataset.append((str(file), label))
        return dataset

    # ---------- 2.3 子目标：知道哪些文件留给验证和测试 ----------
    def open_split(self, split: Literal["val", "test"]) -> set:
        # Literal[...] 是类型提示；字典 .get(split) 用 val / test 找到对应清单名。
        split_files = {"val": "validation_list.txt", "test": "testing_list.txt"}
        split_file = split_files.get(split)
        split_path = self.path / split_file
        with split_path.open("r") as file:
            # read() 读全文；splitlines() 按行拆开；set(...) 方便后面判断文件是否在清单里。
            data = set(sorted(list(file.read().splitlines())))
        return data

    # ---------- 2.4 子目标：只收集本次划分应该使用的文件 ----------
    def in_split(self, file_id: str):
        # 输入是类似 "yes/a.wav" 的标识，输出 True（保留）或 False（排除）。
        # 为什么要分开：检查模型时应使用没有参与训练的文件。
        # 预测：文件在 val 清单里、self.split="train" 时，会走到哪个 return？
        if (
            self.split == "train"
            and file_id not in self.val_split
            and file_id not in self.test_split
        ):
            return True
        if self.split == "test" and file_id in self.test_split:
            return True
        if self.split == "val" and file_id in self.val_split:
            return True
        return False

    # ---------- 2.5 子目标：回答“有多少个样本可以取” ----------
    def __len__(self):
        # API 对应关系：len(sc) 会进入这里，返回文件清单的长度。
        return len(self.dataset)

    # ---------- 2.6 子目标：把一个文件编号变成可训练的样本 ----------
    def __getitem__(self, idx: int):
        # 输入：idx，例如 sc[0] 会让 idx=0；输出：(音频 [1, N], 整数标签)。
        # 按四步追踪：找路径 → 读文件并统一格式 → 可选增强 → 把名称换成编号。
        audio_path, label = self.dataset[idx]
        if audio_path not in self.cache:
            # torchaudio.load(路径) 返回声音数值 wav 和采样率 sr（每秒多少个采样点）。
            wav, sr = torchaudio.load(audio_path)
            # 为什么统一格式：原文件长度和采样率可能不同，凑成一批前需要一致。
            audio = self.transform(wav, sr)
            self.cache[audio_path] = audio  # 缓存增强前的版本，后续取样仍可产生随机变化。
        else:
            audio = self.cache.get(audio_path)  # 字典 .get(key) 取出之前保存的值。

        if (
            self.cfg.augment.use_augmentations
            and self.augmentation_pipeline is not None
        ):
            # apply(audio) 是本项目的接口，依次执行已经准备好的增强操作。
            audio = self.augmentation_pipeline.apply(audio)

        # 验证（已创建 sc 且数据非空后）：audio, label = sc[0]，再打印 audio.shape, label。
        # 修改练习：不传增强工具，在 torchaudio.load 处设断点，连续取两次 sc[0]，看断点命中几次。
        return audio, self.stoi[label]


# ======================================================================
# 3. 非目标词数据集：读取没有被选为关键词的词，统一标为 UNKNOWN
# ======================================================================
# OOV 是 out of vocabulary 的缩写，在这里表示目标关键词列表以外的词。
# 学习重点：和 SpeechCommands 比较，找出“选择哪些目录”和“返回什么标签”的区别。
class OOVDataset(Dataset):
    # ---------- 3.1 准备配置、文件清单和缓存；过程与关键词数据集相同 ----------
    def __init__(
        self,
        cfg: Config,
        split: str,
        augmentation_pipeline: AugmentationPipeline | None = None,
    ):
        self.cfg: Config = cfg
        self.split = split
        self.subset = (
            set(self.cfg.data.subset) if self.cfg.data.subset is not None else set()
        )
        self.path: Path = Path(self.cfg.data.path)
        # 多个返回值可以用 _ 接住不需要的部分；这里只需要类别编号表。
        _, self.stoi, _ = load_labels(self.cfg)
        self.val_split = self.open_split("val")
        self.test_split = self.open_split("test")
        self.dataset = self._load_dataset()
        self.transform = AudioTransform(self.cfg)
        self.augmentation_pipeline = augmentation_pipeline
        self.cache = {}

    # ---------- 3.2 读取划分清单，再判断某个文件属于哪一组 ----------
    def open_split(self, split: Literal["val", "test"]) -> set:
        split_files = {"val": "validation_list.txt", "test": "testing_list.txt"}
        split_file = split_files.get(split)
        split_path = self.path / split_file
        with split_path.open("r") as file:
            data = set(sorted(list(file.read().splitlines())))
        return data

    def in_split(self, file_id: str):
        if (
            self.split == "train"
            and file_id not in self.val_split
            and file_id not in self.test_split
        ):
            return True
        if self.split == "test" and file_id in self.test_split:
            return True
        if self.split == "val" and file_id in self.val_split:
            return True
        return False

    # ---------- 3.3 子目标：把所有非目标词合并成一个 UNKNOWN 类 ----------
    def _load_dataset(self):
        # 为什么合并：模型只需知道它不是目标词，不必认出它具体说了哪个词。
        # 预测：一个词从 subset 中移除后，在文件划分不变时，会进入哪个数据集？
        subset = set(self.subset) if self.subset is not None else set()
        labels = sorted(
            [
                p.name
                for p in self.path.iterdir()
                if p.is_dir() and p.name[0] != "_" and p.name not in subset
            ]
        )
        dataset = []
        for label in labels:
            label_path = self.path / label
            for file in label_path.rglob("*.wav"):
                file_id = f"{label}/{file.name}"
                if self.in_split(file_id):
                    dataset.append((str(file), "<UNKNOWN>"))  # 原来是什么词，都归到同一类。
        return dataset

    # ---------- 3.4 提供数量和单个样本，供 len(oov) / oov[idx] 调用 ----------
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 与关键词数据一样：第一次读取并缓存，下次复用，再按设置做增强。
        audio_path, label = self.dataset[idx]
        if audio_path not in self.cache:
            wav, sr = torchaudio.load(audio_path)
            audio = self.transform(wav, sr)
            self.cache[audio_path] = audio
        else:
            audio = self.cache.get(audio_path)
        if (
            self.cfg.augment.use_augmentations
            and self.augmentation_pipeline is not None
        ):
            audio = self.augmentation_pipeline.apply(audio)

        return audio, self.stoi[label]


# ======================================================================
# 4. 合成静音数据集：不读音频文件，每次现场生成很小的随机声音
# ======================================================================
# 与文件数据集的区别：size 决定“能访问多少个编号”，编号不对应固定音频文件。
class SyntheticSilenceDataset(Dataset):
    # ---------- 4.1 size 决定数据集报告多少个样本，音频长度由配置计算 ----------
    def __init__(
        self,
        cfg: Config,
        size: int,
        augmentation_pipeline: AugmentationPipeline | None = None,
    ):
        self.cfg = cfg
        # 例如时长 1 秒 × 每秒 16000 个采样点 = 每段音频 16000 个数。
        self.num_samples = int(
            self.cfg.preprocess.clip_length * self.cfg.preprocess.sample_rate
        )
        _, self.stoi, _ = load_labels(self.cfg)
        self.size = size
        self.augmentation_pipeline = augmentation_pipeline

    # ---------- 4.2 创建接近静音的声音 ----------
    def generate_silence(self, noise_floor=-60):
        # 输入：noise_floor（分贝）；输出：[1, N] 的很小随机声音，近似安静环境。
        # 为什么不全填零：这里用微弱噪声模拟接近静音、但并非绝对无声的情况。
        # 将分贝换算成振幅倍率；-60 对应 0.001。
        amplitude = 10 ** (noise_floor / 20)
        # API：randn((1, N)) 生成一行 N 个正态分布随机数，再乘倍率缩小声音。
        # 预测：从 -60 改成 -40 后更响还是更轻？验证：比较两次生成结果的 std().item()。
        # std() 衡量这些数的波动大小；item() 将单个数从 Tensor 中取成普通数值。
        return torch.randn((1, self.num_samples)) * amplitude

    # ---------- 4.3 提供数量和样本；相同编号也会重新生成声音 ----------
    def __len__(self):
        return self.size

    def __getitem__(self, _) -> torch.Tensor:
        # _ 表示收到的编号不需要使用；实际返回的是 (音频, 标签)，不只是 Tensor。
        silence = self.generate_silence()
        if (
            self.cfg.augment.use_augmentations
            and self.augmentation_pipeline is not None
        ):
            silence = self.augmentation_pipeline.apply(silence)
        return silence, self.stoi["<SILENCE>"]


# ======================================================================
# 5. 背景噪声数据集：先读取噪声文件，取样时随机选一段，标签也是 SILENCE
# ======================================================================
# 为什么随机截取：一段长噪声可以提供不同短片段，不必每次都用同一个位置。
class BackgroundNoiseDataset(Dataset):
    # ---------- 5.1 准备噪声、随机数生成器和目标音频长度 ----------
    def __init__(
        self,
        cfg: Config,
        size: int,
        augmentation_pipeline: AugmentationPipeline | None = None,
    ):
        self.cfg = cfg
        self.size = size
        self.path = Path(cfg.augment.noise_path)
        self.data: list[torch.Tensor] = self._load_noise()  # 这里会把噪声音频读进内存。
        _, self.stoi, _ = load_labels(self.cfg)
        if len(self.data) == 0:
            raise ValueError("Random noise folder empty")  # 没有可用噪声就主动报错。

        self.augmentation_pipeline = augmentation_pipeline
        # Generator 是专用随机数生成器；manual_seed 固定随机抽取的起点。
        self.generator = torch.Generator().manual_seed(self.cfg.env.seed)

        self.expected_len = int(
            self.cfg.preprocess.sample_rate * self.cfg.preprocess.clip_length
        )

    # ---------- 5.2 读取噪声，并统一成目标采样率、单声道 ----------
    def _load_noise(self) -> list[torch.Tensor]:
        data = []
        for file in self.path.rglob("*.wav"):
            if not file.exists() or file.is_dir():
                continue
            audio, sr = torchaudio.load(file)
            # resample(音频, 原采样率, 目标采样率)：改变每秒采样点数，保持声音时长。
            if sr != self.cfg.preprocess.sample_rate:
                audio = torchaudio.functional.resample(
                    audio, sr, self.cfg.preprocess.sample_rate
                )
            # ndim 是维度个数；unsqueeze(0) 在最前面加一维，例如 [N] → [1, N]。
            if audio.ndim != 2:
                audio = audio.unsqueeze(0)
            # shape[0] 是声道数；mean(dim=0) 对声道求平均，keepdim=True 保留这一维。
            if audio.shape[0] != 1:
                audio = audio.mean(dim=0, keepdim=True)
            data.append(audio)
        return data

    # ---------- 5.3 随机选一个文件，再从中截出固定长度的片段 ----------
    def _random_crop(self, noise: torch.Tensor) -> torch.Tensor:
        # 输入：较长的噪声 [1, L]；输出：固定长度片段 [1, expected_len]。
        # randint(下限, 上限, 形状)：随机取整数，上限不包含在内；(1,) 表示取一个数。
        # 这里要求原噪声比目标片段更长，才能随机选择截取起点。
        start_loc = torch.randint(
            0, noise.shape[1] - self.expected_len, (1,), generator=self.generator
        )
        # [:, 起点:终点]：保留全部声道，只取这段时间；终点位置不包含在内。
        return noise[:, start_loc : start_loc + self.expected_len]

    def _get_random_noise(self) -> torch.Tensor:
        # .item() 把只有一个数的 Tensor 转成普通 Python 数值，用作列表编号。
        idx = torch.randint(0, len(self.data), (1,), generator=self.generator).item()
        return self.data[idx]

    # ---------- 5.4 子目标：用相同接口提供随机噪声样本 ----------
    def __len__(self):
        return self.size

    def __getitem__(self, _: int) -> torch.Tensor:
        # 编号同样不参与选取；返回的是 (音频, SILENCE 类别编号)。
        noise = self._get_random_noise()
        noise = self._random_crop(noise)
        if (
            self.cfg.augment.use_augmentations
            and self.augmentation_pipeline is not None
        ):
            noise = self.augmentation_pipeline.apply(noise)
        return noise, self.stoi["<SILENCE>"]


# ======================================================================
# 6. 从读懂到自己写：用一个样本验证，再重建最小数据集
# ======================================================================
# 验证：在下面的 breakpoint 停下后，先看 sc.dataset[0]，再看 sc[0]，比较二者内容。
# 修改：用无增强的 SyntheticSilenceDataset(cfg, size=3)，检查 len(...) 与音频长度。
# 自己写（单独练习文件）：创建一个只读 wav 文件的 Dataset，先不加缓存和增强。
# 只看这三步复写：__init__ 保存路径清单 → __len__ 返回条数 → __getitem__ 返回音频和标签。
# 完成后接入 DataLoader，检查“一个样本”和“一批样本”的形状有什么区别。
# 以下是原有调试入口：作为入口运行时执行，导入本文件时不执行。
if __name__ == "__main__":
    # Config.from_yaml(路径)：读取 YAML 配置文件并创建配置对象。
    cfg = Config.from_yaml("configs/config.yaml")
    split = "test"
    sc = SpeechCommands(cfg, split)
    unknown = OOVDataset(cfg, split)
    silence_pool_size = int((len(sc) + len(unknown)) * cfg.sampler.silence_weight)
    silence = SyntheticSilenceDataset(cfg, silence_pool_size)

    breakpoint()  # 暂停进入 Python 调试器，可以查看 len(sc)、sc[0] 等结果。
