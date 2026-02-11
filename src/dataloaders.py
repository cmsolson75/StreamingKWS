import torch
from torch.utils.data import DataLoader
from .configs import Config
from .transforms import AudioTransform
from .dataset import SpeechCommands


def load_dataloader(cfg: Config, split: str):
    if split == "train":
        dataset = SpeechCommands(cfg, split)
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistant_workers,
            drop_last=True,
        )
    else:
        dataset = SpeechCommands(cfg, split)
        loader = DataLoader(dataset, shuffle=False, batch_size=cfg.batch_size, drop_last=True)
    return loader


def infinite_dataloader(loader: DataLoader):
    while True:
        yield from loader


if __name__ == "__main__":
    import time

    loader = load_dataloader("configs/config.yaml", "train")
    it = iter(infinite_dataloader(loader))
    for _ in range(10):
        next(it)
    num_steps = 500
    start = time.perf_counter()
    for step in range(1, num_steps + 1):
        x, y = next(it)

    end = time.perf_counter()
    t = (end - start) / num_steps
    print(1 / t)
