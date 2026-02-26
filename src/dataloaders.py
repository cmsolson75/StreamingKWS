from torch.utils.data import DataLoader, ConcatDataset
from .configs import Config
from .dataset import SpeechCommands, OOVDataset, SyntheticSilenceDataset
from .weighted_sampler import WeightedSampler, WeightedSamplerFinite


def load_dataloader(cfg: Config, split: str, start_step: int | None = None):
    sc = SpeechCommands(cfg, split)
    unknown = OOVDataset(cfg, split)

    silence_pool_size = int((len(sc) + len(unknown)) * cfg.silence_weight)
    silence = SyntheticSilenceDataset(cfg, silence_pool_size)

    dataset = ConcatDataset([sc, unknown, silence])
    if split == "train":
        if start_step is None:
            start_step = 0
        sampler = WeightedSampler(
            dataset,
            weights=[cfg.keyword_weight, cfg.unknown_weight, cfg.silence_weight],
            seed=cfg.seed,
            start_step=start_step,
        )
        loader = DataLoader(
            dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            num_workers=cfg.num_workers,
            pin_memory=cfg.pin_memory,
            persistent_workers=cfg.persistent_workers,
        )
    else:
        sampler = WeightedSamplerFinite(
            dataset,
            weights=[cfg.keyword_weight, cfg.unknown_weight, cfg.silence_weight],
            seed=cfg.seed,
            start_step=start_step,
        )
        loader = DataLoader(dataset, sampler=sampler, batch_size=cfg.batch_size)
    return loader


def infinite_dataloader(loader: DataLoader):
    while True:
        yield from loader


def load_speech_cmds(cfg: Config, split: str):
    sc = SpeechCommands(cfg, split)
    return DataLoader(sc, shuffle=False, batch_size=cfg.batch_size)


def load_oov_loader(cfg: Config, split: str):
    oov = OOVDataset(cfg, split)
    return DataLoader(oov, shuffle=False, batch_size=cfg.batch_size)


def load_silence_loader(cfg: Config):
    synth_silence_dataset = SyntheticSilenceDataset(cfg, 8000)
    return DataLoader(synth_silence_dataset, shuffle=False, batch_size=cfg.batch_size)


# if __name__ == "__main__":
#     import time

#     loader = load_dataloader("configs/config.yaml", "train")
#     it = iter(infinite_dataloader(loader))
#     for _ in range(10):
#         next(it)
#     num_steps = 500
#     start = time.perf_counter()
#     for step in range(1, num_steps + 1):
#         x, y = next(it)

#     end = time.perf_counter()
#     t = (end - start) / num_steps
#     print(1 / t)
