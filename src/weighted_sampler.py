import torch
from torch.utils.data import ConcatDataset, Sampler


class WeightedSampler(Sampler):
    def __init__(
        self,
        dataset: ConcatDataset,
        weights: list[float],
        seed: int = 42,
        start_step: int = 0,
    ):
        self.weights = torch.tensor(weights)
        self.seed = seed
        self.start_step = start_step

        cumulative = [0] + dataset.cumulative_sizes
        self.pools = [
            list(range(cumulative[i], cumulative[i + 1])) for i in range(len(weights))
        ]

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed)

        step = 0
        while True:
            pool_idx = torch.multinomial(self.weights, 1, generator=g).item()
            pool = self.pools[pool_idx]
            idx = pool[torch.randint(len(pool), (1,), generator=g).item()]

            step += 1
            if step <= self.start_step:
                continue
            yield idx


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from .configs import Config
    from .dataset import SpeechCommands, Silence, UnknownDS

    cfg = Config.from_yaml("configs/config.yaml")
    speech_commands = SpeechCommands(cfg, "train")
    silence = Silence(cfg)
    unknown = UnknownDS(cfg)
    dataset = ConcatDataset([speech_commands, unknown, silence])
    sampler = WeightedSampler(dataset, [0.7, 0.2, 0.1])

    loader = DataLoader(dataset, 32, sampler=sampler)

    it = iter(loader)
    max_steps = 100
    for step in range(max_steps):
        sample = next(it)
        print(sample)
