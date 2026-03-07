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


def load_dataloader(cfg: Config, split: str, start_step: int | None = None):
    augmentation_pipeline = build_augmentation_pipeline(
        [RandomNoiseMixIn(cfg), RandomGain(cfg)]
    )
    bg_augmentation_pipeline = build_augmentation_pipeline([RandomGain(cfg)])

    sc = SpeechCommands(cfg, split, augmentation_pipeline)
    oov = OOVDataset(cfg, split, augmentation_pipeline)

    pool_size = int((len(sc) + len(oov)) * cfg.sampler.silence_weight)
    silence = SyntheticSilenceDataset(cfg, pool_size, augmentation_pipeline)
    background = BackgroundNoiseDataset(cfg, pool_size, bg_augmentation_pipeline)

    dataset = ConcatDataset([sc, oov, silence, background])
    if split == "train":
        if start_step is None:
            start_step = 0
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
        loader = DataLoader(
            dataset,
            batch_size=cfg.train.batch_size,
            sampler=sampler,
            num_workers=cfg.train.num_workers,
            pin_memory=cfg.train.pin_memory,
            persistent_workers=cfg.train.persistent_workers,
        )
    else:
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
        loader = DataLoader(dataset, sampler=sampler, batch_size=cfg.train.batch_size)
    return loader


def load_speech_cmds(cfg: Config, split: str):
    sc = SpeechCommands(cfg, split)
    return DataLoader(sc, shuffle=False, batch_size=cfg.train.batch_size)


def load_oov_loader(cfg: Config, split: str):
    oov = OOVDataset(cfg, split)
    return DataLoader(oov, shuffle=False, batch_size=cfg.train.batch_size)


def load_silence_loader(cfg: Config):
    synth_silence_dataset = SyntheticSilenceDataset(cfg, 4000)
    return DataLoader(
        synth_silence_dataset, shuffle=False, batch_size=cfg.train.batch_size
    )


def load_background_loader(cfg: Config):
    background_dataset = BackgroundNoiseDataset(cfg, 4000)
    return DataLoader(
        background_dataset, shuffle=False, batch_size=cfg.train.batch_size
    )
