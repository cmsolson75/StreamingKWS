# 学习目标：把“能取一个样本的数据集”变成“每次能取一批数据的加载器”。
# 先读底部 load_speech_cmds 的两步写法，再读主函数的增强、混合与抽样。
# 读每段时先预测结果，再用“验证”注释检查；最后按底部“自己写”完成练习。
# 三个角色：Dataset 提供样本，Sampler 提供编号，DataLoader 按编号取样本并凑成一批。
# 一个样本是 (音频, 类别编号)；batch（一批）是多个样本的集合。
# DataLoader、ConcatDataset 来自 PyTorch；from .xxx 导入本项目的 xxx.py。
from torch.utils.data import DataLoader, ConcatDataset
from .configs import Config
from .dataset import (
    SpeechCommands,
    OOVDataset,
    SyntheticSilenceDataset,
    BackgroundNoiseDataset,
)
from .weighted_sampler import WeightedSampler, WeightedSamplerFinite
from .augmentations import build_augmentation_pipeline, RandomNoiseMixIn, RandomGain


# ======================================================================
# 完整流程：准备增强 → 创建数据集 → 合并 → 选择抽样方式 → 返回加载器
# ======================================================================
def load_dataloader(cfg: Config, split: str, start_step: int | None = None):
    # 输入：cfg 提供路径和参数；split 选择 train / val / test；start_step 可省略。
    # 输出：一个 DataLoader；遍历它会得到一批音频和对应的类别编号。
    # cfg.train.batch_size 是读取配置中的“每批样本数”；冒号后的类型是阅读提示。
    # int | None 表示整数或空值；= None 表示没传这个参数时使用空值。

    # ------------------------------------------------------------------
    # 1. 准备增强：让同一段音频产生变化，减少模型死记训练样本
    # ------------------------------------------------------------------
    # API：RandomNoiseMixIn(cfg) 创建加噪操作，RandomGain(cfg) 创建音量调整操作。
    # build_augmentation_pipeline([...]) 接收操作列表，返回按顺序执行它们的工具。
    # 这里没有改变音频；Dataset 取样本时才按 use_augmentations 设置决定是否增强。
    augmentation_pipeline = build_augmentation_pipeline(
        [RandomNoiseMixIn(cfg), RandomGain(cfg)]
    )
    # 背景噪声只调整音量。
    bg_augmentation_pipeline = build_augmentation_pipeline([RandomGain(cfg)])

    # ------------------------------------------------------------------
    # 2. 准备四类数据：关键词、非目标词、静音、背景噪声
    # ------------------------------------------------------------------
    # API：SpeechCommands / OOVDataset(cfg, split, 增强工具) 返回各自的数据集。
    # 结果：sc[0] 取一个 (音频, 类别编号)，len(sc) 得到样本总数。
    # 为什么分开建：后面才能控制关键词、非目标词等数据各自出现的比例。
    sc = SpeechCommands(cfg, split, augmentation_pipeline)
    oov = OOVDataset(cfg, split, augmentation_pipeline)

    # int(...) 转成整数：若语音共 10000 个、silence_weight=0.1，两个池各有 1000 个位置。
    # 池大小是可访问的编号数量；实际选中哪类数据，由第 4 步的 weights 决定。
    pool_size = int((len(sc) + len(oov)) * cfg.sampler.silence_weight)
    # API：下面两个数据集接收 (配置, 大小, 增强工具)。
    # 前者取样时生成很小的声音，后者从已有噪声中随机截取一段。
    silence = SyntheticSilenceDataset(cfg, pool_size, augmentation_pipeline)
    background = BackgroundNoiseDataset(cfg, pool_size, bg_augmentation_pipeline)

    # ------------------------------------------------------------------
    # 3. 合成一个数据集：用一套编号访问四类数据
    # ------------------------------------------------------------------
    # API：ConcatDataset([A, B, ...]) 返回合并后的数据集，不会自动打乱。
    # 若 A 有 100 个、B 有 50 个，编号 0~99 来自 A，100~149 来自 B。
    # 为什么合并：DataLoader 只需要面对一个数据集，采样器也能统一使用样本编号。
    # 预测：dataset 的长度是多少？验证：比较 len(dataset) 与四个数据集长度之和。
    dataset = ConcatDataset([sc, oov, silence, background])

    # ------------------------------------------------------------------
    # 4. 如果是训练：按比例挑样本，再凑成一批
    # ------------------------------------------------------------------
    if split == "train":
        # 没传 start_step 就从头取；传 100 表示跳过前 100 次取样，不是 100 批。
        if start_step is None:
            start_step = 0
        # API：WeightedSampler 返回一个不断提供样本编号的工具，可重复抽到同一个样本。
        # weights 按关键词、非目标词、静音、背景噪声排列；[0.6, 0.2, 0.1, 0.1] 是一个例子。
        # 为什么用权重：避免数量最多的那类数据占据大多数训练机会。
        # seed 固定随机抽取的起点；数据集大小相同时，相同 seed 会产生相同编号顺序。
        # 抽样没有终点；停止条件写在 train_utils.py 的训练循环里。
        sampler = WeightedSampler(
            dataset,
            weights=[
                cfg.sampler.keyword_weight,
                cfg.sampler.oov_weight,
                cfg.sampler.silence_weight,
                cfg.sampler.background_weight,
            ],
            seed=cfg.env.seed,
            start_step=start_step,
        )
        # API：DataLoader(dataset, batch_size, sampler, ...) 将单个样本凑成一批。
        # 输入是数据集和取样规则；输出的一批音频形如 [B, 1, N]，标签形如 [B]。
        # B 是 batch_size，1 是单声道，N 是每段音频的采样点数。
        # sampler 已决定取样顺序，所以不再同时设置 shuffle=True（随机打乱）。
        loader = DataLoader(
            dataset,  # 去哪里取：上面合并好的数据集
            batch_size=cfg.train.batch_size,  # 一次取多少个样本
            sampler=sampler,  # 取哪些编号：由上面的采样器决定
            num_workers=cfg.train.num_workers,  # 用几个子进程帮忙加载；0 表示当前进程自己做
            pin_memory=cfg.train.pin_memory,  # 使用便于向 GPU 传数据的内存；不会自动传到 GPU
            persistent_workers=cfg.train.persistent_workers,  # 保留这些子进程，要求 num_workers > 0
        )

    # ------------------------------------------------------------------
    # 5. 如果是验证或测试：只挑指定数量的样本
    # ------------------------------------------------------------------
    else:
        # API：WeightedSamplerFinite 的抽样方法同上，达到 total_iterations 就结束。
        # 这里不传 total_iterations，默认取 1000 个；可能重复，并非遍历整个数据集。
        # 为什么要结束：验证循环需要取完数据后汇总结果。
        # 实现细节：当前接收 start_step，但没有用它跳过样本。
        sampler = WeightedSamplerFinite(
            dataset,
            weights=[
                cfg.sampler.keyword_weight,
                cfg.sampler.oov_weight,
                cfg.sampler.silence_weight,
                cfg.sampler.background_weight,
            ],
            seed=cfg.env.seed,
            start_step=start_step,
        )
        # 未写的参数采用默认设置，例如 num_workers=0。
        # 最后一批不够 batch_size 也会返回，例如剩 8 个就返回 8 个。
        loader = DataLoader(dataset, sampler=sampler, batch_size=cfg.train.batch_size)

    # ------------------------------------------------------------------
    # 6. 把准备好的加载器交给调用这个函数的代码
    # ------------------------------------------------------------------
    # 预测：batch_size 改为 8，音频和标签形状的哪一维会改变？
    # 验证（有可用数据和 cfg 后，在调试器或单独脚本中运行）：
    # loader = load_dataloader(cfg, "val")
    # audio, labels = next(iter(loader))  # iter 创建迭代器，next 只拿下一批。
    # print(audio.shape, labels.shape)    # shape 查看数组各个方向的大小。
    return loader


# ======================================================================
# 基础写法与复写练习：先掌握“一个数据集 → 一个加载器”
# 只做两件事：创建数据集 → 用 DataLoader 每次拿一批。
# 不传 sampler 时，shuffle=False 按 0、1、2……取；True 则每轮打乱顺序。
# 静音和背景噪声是在取样时随机产生的，所以编号相同，声音也可能不同。
# 修改练习：在单独脚本里把 batch_size 从 8 改为 4，比较第一批的形状。
# 自己写：遮住函数体，按“创建关键词数据集 → 设置批大小 → 返回加载器”重写。
# ======================================================================

# ------------------------- 关键词 -------------------------
def load_speech_cmds(cfg: Config, split: str):
    sc = SpeechCommands(cfg, split)
    return DataLoader(sc, shuffle=False, batch_size=cfg.train.batch_size)


# ------------------------- 非目标词 -----------------------
def load_oov_loader(cfg: Config, split: str):
    oov = OOVDataset(cfg, split)
    return DataLoader(oov, shuffle=False, batch_size=cfg.train.batch_size)


# ------------------------- 静音（4000 个样本）-------------
def load_silence_loader(cfg: Config):
    synth_silence_dataset = SyntheticSilenceDataset(cfg, 4000)
    return DataLoader(
        synth_silence_dataset, shuffle=False, batch_size=cfg.train.batch_size
    )


# ------------------------- 背景噪声（4000 个样本）---------
def load_background_loader(cfg: Config):
    background_dataset = BackgroundNoiseDataset(cfg, 4000)
    return DataLoader(
        background_dataset, shuffle=False, batch_size=cfg.train.batch_size
    )
