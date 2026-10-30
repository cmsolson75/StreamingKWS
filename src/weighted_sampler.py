# 学习目标：控制每类数据“被拿到的机会”，而不是只按文件数量取样。
# 输入是 ConcatDataset 和每类权重；输出是一个个样本编号，不直接返回音频。
# 阅读顺序：先看 pools 怎样划分编号，再看 __iter__ 怎样“选池 → 选样本”。
import torch
from torch.utils.data import ConcatDataset, Sampler


# ======================================================================
# 1. 无限抽样：持续提供编号，由外部训练循环决定何时停止
# ======================================================================
class WeightedSampler(Sampler):
    def __init__(
        self,
        dataset: ConcatDataset,
        weights: list[float],
        seed: int = 42,
        start_step: int = 0,
    ):
        self.weights = torch.tensor(weights)  # tensor 把 Python 列表转成 PyTorch 数组。
        self.seed = seed
        self.start_step = start_step

        # cumulative_sizes 是每个子数据集结束时的总长度；[0] 补上第一个起点。
        # 若两个子集各长 3、2，边界是 [0, 3, 5]，两个池就是 [0,1,2] 和 [3,4]。
        # range(起点, 终点) 不包含终点；池顺序必须与 weights 相同。
        cumulative = [0] + dataset.cumulative_sizes
        self.pools = [
            list(range(cumulative[i], cumulative[i + 1])) for i in range(len(weights))
        ]

    def __iter__(self):
        # iter(sampler) 会进入这里；每次创建迭代器都用同一 seed 从头生成随机序列。
        g = torch.Generator()
        g.manual_seed(self.seed)

        step = 0
        while True:
            # multinomial(weights, 1) 按相对权重挑一个池；权重会归一化，不要求加起来为 1。
            # randint 从池中均匀选一个位置；item() 把只有一个数的 Tensor 转成普通整数。
            pool_idx = torch.multinomial(self.weights, 1, generator=g).item()
            pool = self.pools[pool_idx]
            idx = pool[torch.randint(len(pool), (1,), generator=g).item()]

            step += 1
            if step <= self.start_step:  # 恢复取样时，重放并跳过旧编号；单位是样本，不是 batch。
                continue
            yield idx  # yield 交出一个编号并暂停；下一次 next(...) 从这里继续。


# ======================================================================
# 2. 有限抽样：方法相同，但抽够 total_iterations 个样本就结束
# ======================================================================
class WeightedSamplerFinite(Sampler):
    def __init__(
        self,
        dataset: ConcatDataset,
        weights: list[float],
        seed: int = 42,
        start_step: int = 0,
        total_iterations: int = 1000,
    ):
        self.weights = torch.tensor(weights)
        self.seed = seed
        self.start_step = start_step  # 当前有限版本仅保存此值，未用来跳过编号。
        self.total_iterations = total_iterations

        cumulative = [0] + dataset.cumulative_sizes
        self.pools = [
            list(range(cumulative[i], cumulative[i + 1])) for i in range(len(weights))
        ]

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        # 可重复抽到同一个样本；这不是把数据集全部遍历一遍。
        for _ in range(self.total_iterations):
            pool_idx = torch.multinomial(self.weights, 1, generator=g).item()
            pool = self.pools[pool_idx]
            idx = pool[torch.randint(len(pool), (1,), generator=g).item()]
            yield idx


# ======================================================================
# 3. 预测 → 验证 → 修改 → 自己写
# ======================================================================
# 不依赖音频的练习：在单独脚本中创建 ConcatDataset([range(3), range(2)])。
# 预测：权重 [0., 1.] 时可能出现哪些编号？用有限采样器取 8 次并 list(...) 验证。
# 修改：换成 [0.5, 0.5]，比较两个池出现的比例；短序列不保证严格各占一半。
# 自己写：先让采样器依次 yield 0、1、2，再改成随机编号，最后加入分池权重。
# 以下是原有音频调试入口；无限采样器应配合有步数上限的循环使用。
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from .configs import Config
    from .dataset import SyntheticSilenceDataset, SpeechCommands, OOVDataset
    from .transforms import DbMelSpec

    cfg = Config.from_yaml("configs/config.yaml")
    speech_commands = SpeechCommands(cfg, "train")
    silence = SyntheticSilenceDataset(cfg, 1000)
    unknown = OOVDataset(cfg, "train")
    dataset = ConcatDataset([speech_commands, unknown, silence])
    sampler = WeightedSampler(dataset, [0.0, 0, 1.0])

    db_mel_spec = DbMelSpec(cfg)

    loader = DataLoader(dataset, 32, sampler=sampler)

    it = iter(loader)
    max_steps = 1
    for step in range(max_steps):
        x, y = next(it)
        x = db_mel_spec(x)
        breakpoint()
