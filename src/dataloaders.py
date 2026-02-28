from torch.utils.data import DataLoader, ConcatDataset
from .configs import Config
from .dataset import (
    SpeechCommands,
    OOVDataset,
    SyntheticSilenceDataset,
    BackgroundNoiseDataset,
)
from .weighted_sampler import WeightedSampler, WeightedSamplerFinite
from .augmentations import build_augmentation_pipeline, RandomNoiseMixIn


def load_dataloader(cfg: Config, split: str, start_step: int | None = None):
    augmentation_pipeline = build_augmentation_pipeline([RandomNoiseMixIn(cfg)])
    bg_augmentation_pipeline = None

    sc = SpeechCommands(cfg, split, augmentation_pipeline)
    oov = OOVDataset(cfg, split, augmentation_pipeline)

    pool_size = int((len(sc) + len(oov)) * cfg.silence_weight)
    silence = SyntheticSilenceDataset(cfg, pool_size, augmentation_pipeline)
    background = BackgroundNoiseDataset(cfg, pool_size, bg_augmentation_pipeline)

    dataset = ConcatDataset([sc, oov, silence, background])
    if split == "train":
        if start_step is None:
            start_step = 0
        sampler = WeightedSampler(
            dataset,
            weights=[
                cfg.keyword_weight,
                cfg.oov_weight,
                cfg.silence_weight,
                cfg.background_weight,
            ],
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
            weights=[
                cfg.keyword_weight,
                cfg.oov_weight,
                cfg.silence_weight,
                cfg.background_weight,
            ],
            seed=cfg.seed,
            start_step=start_step,
        )
        loader = DataLoader(dataset, sampler=sampler, batch_size=cfg.batch_size)
    return loader


def load_speech_cmds(cfg: Config, split: str):
    sc = SpeechCommands(cfg, split)
    return DataLoader(sc, shuffle=False, batch_size=cfg.batch_size)


def load_oov_loader(cfg: Config, split: str):
    oov = OOVDataset(cfg, split)
    return DataLoader(oov, shuffle=False, batch_size=cfg.batch_size)


def load_silence_loader(cfg: Config):
    synth_silence_dataset = SyntheticSilenceDataset(cfg, 4000)
    return DataLoader(synth_silence_dataset, shuffle=False, batch_size=cfg.batch_size)


def load_background_loader(cfg: Config):
    background_dataset = BackgroundNoiseDataset(cfg, 4000)
    return DataLoader(background_dataset, shuffle=False, batch_size=cfg.batch_size)


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
